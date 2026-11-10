# arch-opt-2026 Phase 1: ZFS ARC, model detail API, budget arbiter

Status: **not yet run**. This file is the template a CUDA machine fills in.
Spec: `.monkeycode/specs/arch-optimization-2026/` (requirements, design, tasklist).
Numbers already in `docs/DETAILS.md` for the arbiter / ZFS path are marked
"opt-in and unmeasured"; replace them only with what this run prints.

What is measured here (the parts that need a GPU):

1. The expert-cache RAM budget arbiter, on vs off (`FNX_ARBITER`), 32K context.
2. `FNX_PF_STAGE_ROWS` on a short prompt (where it is meant to do nothing).
3. The ZFS ARC lower bound on `device_free_bytes()`, on a ZFS host.

What is measured without a GPU (already green on the dev box, do not repeat here):

- ZFS arcstats parsing: `tests/platform/zfs_arc_test.cpp` (`zfs_arc_test`, CPU-only).
- Arbiter / degradation watermarks and monotonicity: `tests/core/memory_arbiter_test.cpp`
  (`memory_arbiter_test`, CPU-only).
- `GET /v1/models/{id}` + alias + 404: `serve/test_server.py`, class `ModelAliases`
  (`python -m pytest serve/test_server.py -k ModelAliases`).

## Hardware this was run on

<!-- fill after the run; copy the shape of bench/results/2026-09-29-rtx3090-epyc-milan/hw.txt -->

- GPUs:
- Driver / CUDA:
- CPU / RAM:
- Disk (the model and the filesystem matter for the ZFS arm):
- Strata commit:
- `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`:
- build: `cmake -LAH` grep of `STRATA_ENABLE_CUDA|STRATA_BUILD_SPLIT|STRATA_SPLIT_ARCH|
  STRATA_SPLIT_CARDS|STRATA_MMQ_KQUANTS|STRATA_EXPERIMENTAL_SM60`:

## How the run went

<!-- one honest paragraph: what would not build, what a card refused, what was
     not tested at all tonight. Keep the failed and the passed apart. -->

## Results

<!-- one table per experiment; medians over rounds; the round count stated.
     Empty cells stay empty with the reason, never a guess. -->

### 1. Arbiter on / off

`tools/ab_engine.py strata-q2_0.json A:FNX_ARBITER=0 B:FNX_ARBITER=1 --rounds 3 --max-new 256`
(A: `FNX_ARBITER=0`, B: `FNX_ARBITER=1`, same exe; 32K context from the config)

| request | arm | decode tok/s | prompt tok/s | prompt_n | notes |
|---|---|---|---|---|---|
| short   | A off | | | | |
| short   | B on  | | | | |
| long    | A off | | | | |
| long    | B on  | | | | |

### 2. `FNX_PF_STAGE_ROWS` on a short prompt

<!-- needs the expert cache carrying slots (w.ec.n_slots nonzero, experts spilled
     to RAM); with no spill this variable is inert and the row says "no spill". -->

| FNX_PF_STAGE_ROWS | prompt tokens | decode tok/s | prompt tok/s | notes |
|---|---|---|---|---|
| 2048 (default) | ~200 | | | |
| 8             | ~200 | | | |

### 3. ZFS ARC on the free-memory floor

<!-- a ZFS host only. `free_b` the engine logged, then arc size from /proc/spl/kstat/zfs/arcstats -->

| field | non-ZFS | ZFS | ZFS, `FNX_NO_ZFS_ARC=1` |
|---|---|---|---|
| arc `size` (bytes) | n/a | | n/a |
| `device_free_bytes()` (MiB) | | | |
| below `MemAvailable` by (MiB) | n/a | | |
| engines that still start clean | | | |

### Sanitizers

<!-- optional; a sanitizer build of the engine changes speed, so these runs are
     for memory and race *cleanliness*, not the tok/s tables above. -->

- ASAN: build config, what was run, any leak/overflow: (none found / list) / not run
- TSAN: build config, which requests raced: (none found / list) / not run

## What this does not show

<!-- e.g. one card model, CPU-only hosts, no AMD. Keep it short and true. -->
