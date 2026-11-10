# Phase 2 notes: what to carry over, and the Issue #5440 gap

Companion to `tasklist.md` and `design.md` in this folder. This file is short on
purpose: it names the open questions Phase 1 hands to Phase 2, and the one input
(Issue #5440, the Strata loader/scheduler) that is not in this checkout.

Issue numbers are the Strata issue tracker's; Phase 1's own tracked item is not
reproduced here.

## What Phase 1 leaves for Phase 2 (measured on the CUDA box, see
`bench/results/2026-11-09-arch-arbiter/README.md`)

- The **arbiter is opt-in** (`FNX_ARBITER`, on unless `0`; watermarks
  `FNX_ARBITER_LOW`/`FNX_ARBITER_HIGH`, percent of the smallest card, defaults 15/40).
  Phase 2 turns it on by default only after the Phase 1 table shows it does not cost
  decode speed at 32K on an spilled-expert config. Until then the default path is
  byte-identical with `FNX_ARBITER=0`.
- **`FNX_PF_STAGE_ROWS`** is a pure prompt-path knob (`w.ec.n_slots` nonzero). The
  Phase 1 row for 200 tokens is the input to the Phase 2 decision on whether
  a short prompt should ever stage; do not change its default (2048) without that
  row.
- **ZFS ARC floor** (`device_free_bytes()` = max(MemAvailable, 0.8 * ARC size),
  `FNX_NO_ZFS_ARC` non-zero to turn off) is a free-memory input the arbiter now
  sees. Phase 2's `ResidencyManager` byte target inherits it.

## The Issue #5440 gap (the loader/scheduler issue)

Phase 2 is **event-driven expert load + KV pipeline**
(`ExpertEventSync`, `KvPipeline`, `ResidencyManager`; tasklist 2.1-2.4). The
Phase 1 design references "Issue #5440" (a Strata issue about the model
loader/scheduler) as the reason to keep the scheduler's hand-off points narrow and
to keep a single arbitration tick per window. **That issue's text is not in this
checkout, and a web lookup was not possible from this session**, so the design
records it as the assumed constraint below, not as the issue's words.

### TODO before Phase 2 starts: fill these from the issue (do not guess)

- [ ] Issue number/URL as written in Phase 1 design.md ("#5440"): confirm it is the
  Strata tracker and not a llama.cpp upstream number. `TODO: owner`
- [ ] Title and one-line problem, quoted verbatim. `TODO`
- [ ] Which module the issue names: loader (`gguf.cpp` Stager), scheduler
  (`engine_common.cu` window/step), or both? `TODO`
- [ ] The acceptance the issue asks for (a log line, a latency budget, a bug
  reproduction): quote it; if it asks a number, it belongs in tasklist 2.4's
  short-conversation +10-20%. `TODO`
- [ ] Any hardware it assumes (CPU-only, one card, NVLink) `TODO`

### Constraint Phase 2 carries regardless (from design.md, not the issue)

- One arbitration tick per `window()`/`step()` entry, as Phase 1 already wired in
  `engine_common.cu`; Phase 2's residency and KV shrink hang off that same tick so
  there is never a second, racing budget decision per step.
- Every Phase 2 feature is opt-in and its off path is the Phase 1 path:
  `FNX_NO_EVENT_SYNC=1`, no `FNX_KV_RAM_TIER`, `ResidencyManager` not constructed.
- Greedy consistency is per-bit against the Phase 1 build
  (tasklist TASK-3.3.2); a Phase 2 change that moves a logit is a bug, not a win.

## Order

1. Run the Phase 1 CUDA table; fill `bench/results/2026-11-09-arch-arbiter/README.md`.
2. Update `docs/DETAILS.md:242-260`: replace "opt-in and unmeasured" only where a row
   now has a number; keep "opt-in" throughout.
3. Fill the Issue #5440 TODOs above from the issue text; only then start 2.1.
