# Strata architecture v2.0: predictive, event-driven, resilient

Version 2.0. Repository Niko1221/Strata. Scope: Phase 2 and Phase 3.
Phase 1 (adaptive runtime, ZFS ARC awareness, API endpoints) is on disk and
green under CPU-only tests. The whole-engine CUDA build and the 32K
benchmarks are still open; nothing here is a measured speedup of the current
tree until that run prints it.

Reconnaissance the spec asked for (answers inline, §2):

- (a) `STRATA_KV_PREFETCH` exists and is merged.  `src/prefill/prefill.cpp:3344`
  and `sycl/src/prefill/prefill.cpp` read it; `docs/KV_PREFETCH.md` specifies it
  (independent copy stream, a release event protecting the staging pool's last
  reader, a ready event ordering the next attention consumer, `drain()` on early
  return / relayout / destruct).  `src/split/` does not use it; its offload
  waits are `cudaDeviceSynchronize()` (`engine_common.cu:428`, and `engine.cu`'s
  init/ring-clear sites).  So KvPipeline is a policy + adapter over a
  transport that already exists, not a second transport.
- (b) The offload sync points a host-wide wait today: `engine_common.cu:428`
  (`cudaDeviceSynchronize()` inside the window loop) and the `ck(cudaDeviceSynchronize(), ...)`
  calls in `engine.cu` (init, ring clear, arena idle).  `prefill.cu` already
  keeps its own per-copy events (`ev_r`/`ev_m`, `cudaEventQuery`).  ExpertEventSync
  replaces the host-wide wait on the expert path with per-expert events.
- (c) The staging pool's event lifecycle is reusable: `stage_ev[2]` and the
  `ev_r`/`ev_m` pair in `engine_impl.hpp`/`prefill.cu` are exactly the
  "release protects the last reader, ready orders the next consumer" shape.
  The new core modules do NOT re-create an event pool; they take a
  `BackendEvent`/`BackendStream` interface whose CUDA adapter in `src/split`
  wraps these existing handles.

## 1. Principles (binding)

R1 I/O must lead. An expert weight or KV page move SHALL NOT sit on the decode
   critical path. Prefill is the only phase that may spend compute to hide I/O.
R2 Sync granularity is the expert. No `stream.synchronize()` and no whole-layer
   wait on the decode path. The compute stream waits only the ready events of
   the experts the current token actually routed to.
R3 Smooth over peak. When I/O queue depth or P99 latency crosses a threshold, the
   prefetch aggressiveness SHALL fall at once (DegradationStateMachine), never
   letting the system produce a random long stall while over-subscribed.
R4 Expert and KV share one VRAM budget. A single MemoryBudgetArbiter allocates
   sub-budgets (Expert / KV / Staging) with hysteretic rebalance on a
   statistical window; neither pool manages the "last few MB" alone.

## 2. Requirements (EARS, Phase 2 acceptance-bearing)

R2-1 (Event abstraction) The system SHALL expose a BackendEvent / BackendStream
  interface that reports `record`, non-blocking `query`, and `wait`.
R2-2 (CUDA adapter) When the CUDA backend is built, the BackendEvent adapter
  SHALL use `cudaEventCreateWithFlags(cudaEventDisableTiming)` and
  `cudaStreamWaitEvent`, and SHALL reuse a fixed-ring slot only after waiting
  its previous event.
R2-3 (No-event fallback) When a backend has no fine-grained event, BackendEvent
  SHALL become a lazy "record is sync" object, and the engine SHALL fall back to
  layer synchronization for that backend. It SHALL NOT block the CUDA path to
  chase uniformity.
R2-4 (Expert-granular wait) While the expert event sync is enabled
  (`STRATA_EXPERT_EVENT_SYNC=1`), the decode path SHALL issue a non-blocking H2D
  copy per needed expert on the copy stream, record one ready event per copy,
  and the compute stream SHALL wait only those ready events. A host-wide
  `cudaDeviceSynchronize()` SHALL NOT appear on that path.
R2-5 (Recycle) The expert event sync SHALL recycle a ring slot only after
  `query()` on that slot's event is true, so a ring of N never waits on an event
  it has not recorded.
R2-6 (KV pipeline opt-in) When `STRATA_KV_PREFETCH` is already on in the tree,
  KvPipeline SHALL NOT add a second transport; it SHALL gate the existing
  opt-in on the DegradationStateMachine level (on at L0/L1, off at L2+).
R2-7 (KV drain coverage) The KvPipeline drain SHALL run on early return,
  relayout, and capture/destruct; a borrowed slot SHALL NOT be refilled before
  its release event is Drain-complete.
R2-8 (KV benefit honesty) KvPipeline's verified benefit SHALL be stated as prefill
  only, with the measured configuration. Decode SHALL be "no promised gain".
R2-9 (Degradation levels) The DegradationStateMachine SHALL have five levels
  (Normal, ShrinkPrefetch, HighConfOnly, NoNgram, NoPrefetch, FreezeLowPrio, as
  the Phase 1 enum) and SHALL enter a level with hysteresis N.
R2-10 (Level recovery) It SHALL leave a level only after N consecutive windows
  satisfy the level's "not triggered" condition, to avoid oscillation.
R2-11 (Window scale) The prefetch window SHALL shrink by a configurable scale at
  L1+, and low-confidence prefetch SHALL be refused at L2+.
R2-12 (Off switch) Every new module SHALL have an off switch that restores the
  pre-Phase-2 path: `STRATA_EXPERT_EVENT_SYNC=0`, `STRATA_KV_PIPELINE=0`,
  `STRATA_PREFETCH=0`, `STRATA_RESIDENCY=0`, `STRATA_CLUSTER_MMAP=0`. Default off.

## 3. Scope: not being done (anti-scope-creep)

- No Rust migration. New modules stay C++.
- No KV spill to SSD. KV sub-budget only moves the VRAM / pinned-RAM residency
  ratio; it does not add SSD I/O.
- No llama.cpp / ggml offload rewrite. Decisions are injected through the
  interfaces already in the tree (expert_source, expert_cache, staging pool).
- No fine-grained event for every backend. Non-event backends fall back to
  layer sync and are not slowed for uniformity.
- No invented numbers. A benefit without a measurement is written
  "opt-in, expected, unmeasured".

## 4. Module map (files added in this change)

Header-only / CPU-testable, no CUDA type in `include/strata/core`:

- `include/strata/core/backend_event.hpp` - BackendEvent / BackendStream /
  NullBackendEvent / CountingBackendStream test doubles; capability probe.
- `include/strata/core/expert_event_sync.hpp` + `src/core/expert_event_sync.cpp`
  - per-expert ready-event ring; `issue_transfer`, `wait_needed`, `poll_and_recycle`;
  `FNX_NO_EVENT_SYNC` / `STRATA_EXPERT_EVENT_SYNC=0` fall back to layer sync.
- `include/strata/core/kv_pipeline.hpp` + `src/core/kv_pipeline.cpp`
  - cross-layer KV stage; `on_kv_proj_done`, `wait_kv_ready`, `drain`;
  `STRATA_KV_PIPELINE` default off, gated by level; detects existing
  `STRATA_KV_PREFETCH` and defers to it.
- `include/strata/core/expert_prefetcher.hpp` + `src/core/expert_prefetcher.cpp`
  - P0 (prefill heuristic: sliding-window route frequency + cluster mmap),
  P1 (layer lookahead, off by default), P2 (ngram/MTP, L0 + spare I/O only);
  `set_window`, `set_confidence_threshold`, gated by DegradationStateMachine;
  P1 SHALL be a separate switch `STRATA_PREFETCH_LOOKAHEAD`.
- `include/strata/core/residency_manager.hpp` + `src/core/residency_manager.cpp`
  - Greedy-Dual score = `frequency_decayed * miss_cost_estimate`;
  `miss_cost` is a callback supplied by ClusterMmapReader, not a constant;
  Anchor Expert Set is a `protected_set` (evict skips it); batch eviction.
  Hit-rate falls below threshold => degrade to LRU (caller switches).
- `include/strata/core/cluster_mmap.hpp` + `src/core/cluster_mmap.cpp`
  - `cluster_manifest.json` (`{"version":1,"clusters":[{"id":N,"experts":[[layer,expert],...],"bytes":B}],...}`)
  and `{"global":[[l,e],...]}` for anchors; `read_cluster`, `prefetch_cluster`,
  `has_manifest`; `latency_model(cluster_id)` returns the bytes and a per-cluster
  cost proxy for miss_cost. No manifest => `has_manifest()==false`, caller falls
  back to the original layout.
- `tools/cluster_experts.py` - input a co-activation matrix (npz/csv), output
  `cluster_manifest.json` + `global` anchor list; Louvain via `networkx`'s
  `louvain_communities` when present, else a deterministic connected-component
  fallback (says so). `--layers` filter.
- Extend `include/strata/core/memory_arbiter.hpp` + `src/core/memory_arbiter.cpp`
  with `SubBudgets {expert_bytes, kv_bytes, staging_bytes}`,
  `initial_budgets(total_vram, total_ram)` (default 60/30/10), and
  `rebalance(UsageReport, UsageReport, MetricsSnapshot)` with per-window
  migration cap (default 5% of total) and hysteresis (direction must hold N
  windows). KV never spills to SSD.

## 5. Corrected facts carried from Phase 1

- The "32K -5.3% (16.989s -> 16.082s)" figure is from `docs/KV_PREFETCH.md`, a
  HISTORICAL PROTOTYPE measurement, labeled in that doc "NOT a measured speedup
  of this pure-upstream port". It may not be quoted as the current tree's gain.
- The merged `STRATA_KV_PREFETCH` arm measured in this repo
  (`bench/results/2026-10-06-community-rtx4090-opt-ins/`) is **-4.3% prompt tok/s
  at 32K** and **-2.1% at 128K** with decode unchanged, on a config WITHOUT
  `--kv-resident` - i.e. there was nothing in RAM to overlap, and the extra
  streams cost more. The mechanism can only pay when an appreciable KV prefix is
  host-resident. Acceptance text below therefore requires `--kv-resident`.

## 6. Phase 2 acceptance (all rows unmeasured until a CUDA run prints them)

- Short conversation (<8K) decode **+10%-20%** over the previous build. Opt-in,
  expected, unmeasured.
- 32K prefill **+3%-5%** over the previous build, ON a `--kv-resident` config
  (the prototype's 5.3% was that config; the non-resident arm lost 4.3%).
  Opt-in, expected, unmeasured.
- 32K decode does not regress more than 10%; P99 latency spread down >=20%.
  Opt-in, expected, unmeasured.
- Greedy decode token-for-token identical to the previous build (per-bit logits).
  Requirement, not a target.
- `STRATA_EXPERT_EVENT_SYNC=0` and `STRATA_KV_PIPELINE=0` restore the old path.

## 7. Phase 3 (later; not in this change)

ExpertPrefetcher P0 is delivered here (prefill heuristic + cluster mmap) since
Phase 2's KvPipeline and Phase 3's prefetcher share the event pool; P1 lookahead
is behind `STRATA_PREFETCH_LOOKAHEAD` and unmeasured. Anchor set and Greedy-Dual
are delivered here as `ResidencyManager` (host-testable). Long-context
"+20%-50%" and "I/O stall share -30%" remain Phase 3 acceptance targets at
"opt-in, expected, unmeasured" until a CUDA run fills them.
