# arch-opt-2026 Phase 1: the CUDA-machine runbook

The dev box has g++ 12.2, no cmake, no CUDA, no GPU. Everything here runs on a
**Linux CUDA machine with a real card**. It measures the three things Phase 1
added and leaves unmeasured on the dev box: the expert-cache RAM budget arbiter,
`FNX_PF_STAGE_ROWS`, and the ZFS ARC floor under `device_free_bytes()`.

The spec is in `.monkeycode/specs/arch-optimization-2026/`. Nothing below invents a
number; an empty cell in the README is a cell this machine did not print.

## 0. What the code does (so the run measures the right switch)

- `FNX_ARBITER` (`src/split/src/engine.cu`, ctor): **off by default**; set
  `FNX_ARBITER=1` to turn the arbiter on (any value but `0` turns it on). Watermarks
  `FNX_ARBITER_LOW`/`FNX_ARBITER_HIGH` (percent of the smallest card's total, defaults
  15/40). `Engine::arbiter_tick()` runs at the `window()` and
  `step()` entries (`engine_common.cu`) and yields a `BudgetPlan` (expert target //
  KV pages) through `MemoryBudgetArbiter::tick()` (`src/core/memory_arbiter.cpp`).
  `DegradationStateMachine::apply_level()` maps the plan to levels 0..5.
- `FNX_PF_STAGE_ROWS` (default 2048): the prompt-path expert reads in place when the
  prompt is shorter than this, because no staging buffer was filled with experts
  yet. **It only does anything when `w.ec.n_slots` is nonzero** (experts spilled to
  RAM, the expert cache carrying slots). With no spill the variable is inert and a
  row that says otherwise is measuring noise.
- ZFS: `device_free_bytes()` (`src/core/expert_cache.cpp`) on Linux, non-GFX906,
  gfx906 excluded, takes `max(MemAvailable, arc_reclaimable)` where
  `arc_reclaimable = 0.8 * arcstats size` (`zfs_arc_reclaimable_bytes`, in
  `src/platform/memory.cpp`). `FNX_NO_ZFS_ARC` turns the ARC half off when it parses
  to a **nonzero integer** (`atoi(...) != 0`), so use `FNX_NO_ZFS_ARC=1`.

##  Code build (CUDA on a Linux host with cmake+ninja)

Use the same configure the shipped CUDA image uses (from `Dockerfile:41`), plus
two switches: the split engine, and tests (for the two CPU unit tests).

```sh
# a llama.cpp checkout the split engine can see its ggml-cuda headers in
git clone --depth 1 https://github.com/ggml-org/llama.cpp /src/llama.cpp

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_SPLIT=ON \
  -DSTRATA_GGML_DIR=/src/llama.cpp \
  -DSTRATA_SPLIT_CARDS=2 \
  -DSTRATA_SPLIT_ARCH=86 \
  -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86-real;89-real;120-real;120-virtual"

# the server + the two split targets (strata-split-2 for a 2-card box)
cmake --build build --target strata strata-split-2 memory_arbiter_test zfs_arc_test -j "$(nproc)"
```

`STRATA_SPLIT_CARDS` takes 2, 3 or 4 and gates `strata-split-{2,3,4}`;
`STRATA_SPLIT_ARCH` must match the card (86 for 3090/A6000, 89 for 4090,
120 for 5090 and newer). The split engine builds only on Linux + CUDA (or Linux
+ HIP gfx1201); the CMake in `src/split/CMakeLists.txt:16-30` says so and skips
otherwise. Adjust `CUDA_ARCHITECTURES` and the two `STRATA_SPLIT_*` values to the
box you are on.

First, the CPU tests the dev box already ran, here again to prove the box agrees:

```sh
ctest --test-dir build -R 'memory_arbiter_test|zfs_arc_test' --output-on-failure
python -m pytest serve/test_server.py -k ModelAliases -q
```

## 1. Arbiter on / off at 32K

`tools/ab_engine.py` starts `serve/server.py --engine strata --config CFG` once
per arm, alternates the arms over rounds, and prints the server's own
`timings.predicted_per_second` / `timings.prompt_per_second` medians. A non-`exe`,
non-`xargs` arm key is set **in the server process's environment**, and the engine
child inherits it (`child_env()` starts from `os.environ`), so `FNX_ARBITER`
reaches the split engine. The 32K window comes from the config's `args`
(`--max-context 32768`, and the model those args load). Arms are the same `exe`,
so only the env differs:

```sh
python tools/ab_engine.py strata-q2_0.json \
  A:FNX_ARBITER=0  B:FNX_ARBITER=1 \
  --rounds 3 --max-new 256 --long-words 22000 \
  --log bench/results/2026-11-09-arch-arbiter/arbiter.jsonl
```

`--long-words 22000` fills the long request with 22,000 words of the fixed
vocabulary in `long_prompt()`, which tokenises to roughly a 25-30K-token prompt;
raise it until the server reports the 32K `prompt_n` you mean to test (the
`--long-words` count is words, not tokens). The short chat prompts and the long one
each get an off and an on median over 3 rounds. On is only interesting if it does
not lose decode speed while it keeps more experts resident: record decode,
prompt, and the engine's `expert cache ... hit` share from each `done:` line.

## 2. `FNX_PF_STAGE_ROWS` on a short prompt

Only with the expert cache carrying slots (experts spilled to RAM, `w.ec.n_slots`
nonzero, the same `strata-q2_0.json` on a card smaller than the experts). Use the
same harness; the arms pin the staging rows, and only the ~200-token chat requests
are read (delete the long row when filling the table, it is not this test):

```sh
python tools/ab_engine.py strata-q2_0.json \
  A:FNX_ARBITER=0,FNX_PF_STAGE_ROWS=2048  B:FNX_ARBITER=0,FNX_PF_STAGE_ROWS=8 \
  --rounds 3 --max-new 64 \
  --log bench/results/2026-11-09-arch-arbiter/pf_rows.jsonl
```

If the engine's INFO on start says `expert_slots=0`, stop: no spill, the variable
is inert, and the table keeps "no spill" in every note.

## 3. ZFS ARC floor (a ZFS host only)

On a host whose model lives on datasets/ARC (ZFS root or a pool under load), start
the engine the normal way twice, once with the ARC half on and once off, and read
the `vram_free`/free-bytes figure the engine logged at start, next to the kernel's:

```sh
# ARC size the engine will treat as reclaimable (0.8 * size)
grep -E '^size ' /proc/spl/kstat/zfs/arcstats
mem_available() { awk '/MemAvailable/{print $2" KiB"}' /proc/meminfo; }

# on (default): the floor may sit above MemAvailable
# off interest: turn the ARC half off
FNX_NO_ZFS_ARC=1 <same strata launch / same engine start>
```

`device_free_bytes()` is the engine's own free figure; it is not printed by the
server, so read it from the engine's start line (the split engine logs its free
and reserve MiB before READY; `grep -i 'free.*mib\|reserve' engine.log`). The
table's point: with ARC on, the engine sees more free memory than `MemAvailable`
says (the page cache ZFS will hand back), and both the same and a ZFS arm still
start clean. If the box has no ZFS, write "not run: no ZFS host" and move on; do
not fake an arcstats.

## 4. Sanitizers (correctness, not speed)

Optional, and a sanitizer build runs slower, so these results never enter the
tok/s tables. Build the split engine once with the flag and drive it with the same
`ab_engine.py` (or `strata launch --plan` + a few requests):

```sh
# ASAN: leaks and overflows in the arbiter / cache / ARC path
cmake -G Ninja -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_SPLIT=ON -DSTRATA_GGML_DIR=/src/llama.cpp \
  -DSTRATA_SPLIT_CARDS=2 -DSTRATA_SPLIT_ARCH=86 \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DCMAKE_CUDA_FLAGS="-Xcompiler -fsanitize=address"
cmake --build build-asan --target strata-split-2 -j "$(nproc)"

# TSAN: races in the arbiter tick on the window()/step() path
cmake -G Ninja -S . -B build-tsan ...same, -fsanitize=thread...
```

ASAN/TSAN do not see device memory; they cover the host side of the arbiter,
`device_free_bytes`, and the ARC parse. Say so in the README's sanitizer line.

## Fill the results

Copy what `ab_engine.py` and the engine log print into
`bench/results/2026-11-09-arch-arbiter/README.md`. This folder is hand-filled;
there is no summariser to run for it. Then update the
"opt-in and unmeasured" sentences in `docs/DETAILS.md:242-260` with the measured
medians and the box line (GPUs, driver, commit), and delete the word "unmeasured"
only where a row now carries a number. Leave it where nothing was run.

Keep one sentence in `docs/DETAILS.md` that the arbiter is still **opt-in**
(`FNX_ARBITER=1`), whatever the run shows; the default path must stay off.
