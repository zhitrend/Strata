// include/strata/core/kv_pipeline.hpp - cross-layer asynchronous KV staging over BackendEvent (arch v2.0 3.2.3).
//
// **WHAT THIS IS AND IS NOT.**  The tree already ships an opt-in KV prefetch, STRATA_KV_PREFETCH, in
// src/prefill/prefill.cpp (#732): the next streamed layer's KV prefix is copied on its own stream, a release event
// protects the staging pool's last reader, a ready event orders the next attention consumer, and drain() runs on early
// return, relayout and destruction.  This class does NOT implement a second transport.  It is the POLICY that
// decides, from the degradation level, whether that existing prefetch is on, and it adapts the same release/ready
// event lifecycle when a backend wants to drive it here directly.
//
// **HONEST ABOUT THE BENEFIT.**  The verified gain is prefill-side and, in this repository's one measurement of the
// merged option (bench/results/2026-10-06-community-rtx4090-opt-ins/), it was a small LOSS on a config with no host KV
// to overlap ("nothing in RAM").  The mechanism pays only when an appreciable KV prefix is host-resident
// (--kv-resident).  The 16.989 s -> 16.082 s (-5.3%) figure in docs/KV_PREFETCH.md is a historical prototype and that
// doc itself says it is not a measured speedup of this port.  Decode gain: none promised.
//
// **DRAIN IS THE CONTRACT (R2-7).**  A slot borrowed from the staging pool may not be refilled until its release event
// completed.  drain() waits every in-flight release; it must run on early return, relayout, and destruction.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "strata/core/backend_event.hpp"

namespace strata::core {

enum class KvPipelineGate {
    On,       // L0/L1: the prefetch runs (when the backend supports it)
    L1,       // window halved
    Off       // L2+: no cross-layer KV prefetch; the layer reads its KV from where it is
};

struct KvPipelineConfig {
    bool enabled = false;          // STRATA_KV_PIPELINE; off by default (R2-12)
    bool upstream_opt_in = false;  // STRATA_KV_PREFETCH already on in this build: defer to it (R2-6)
    int ring_slots = 2;            // cross-layer buffers in flight (L and L+1)
    BackendEventCapabilities caps;
};

/// Cross-layer KV staging.  Holds up to `ring_slots` borrowed staging buffers, each with a release event (the last
/// reader of the buffer done) and a ready event (the next attention consumer may read it).
class KvPipeline {
public:
    struct Slot {
        int layer = -1;
        BackendEvent* release = nullptr;  // the last reader of the buffer done
        BackendEvent* ready = nullptr;    // the next attention consumer may read the buffer
        bool in_flight = false;
    };

    explicit KvPipeline(KvPipelineConfig cfg, std::string* err = nullptr);

    /// True when this pipeline drives the cross-layer prefetch itself.  False when upstream STRATA_KV_PREFETCH is on
    /// (it is left to run, this does not double it), the module is off, or the backend has no fine-grained events.
    bool active() const { return active_; }

    /// The degradation level -> gate.  L0/L1 on, L2+ off.  Called by the engine each statistical window.
    void set_level(int degradation_level);

    KvPipelineGate gate() const { return gate_; }

    /// Stage layer `layer`'s KV onto the copy stream: record its ready event when the copy is submitted; the release
    /// event is the previous reader's.  Returns the slot, or -1 (off, full, or upstream owns it).
    int on_kv_proj_done(int layer, BackendEvent* ready, BackendStream& copy_stream);

    /// The compute stream waits `layer`'s ready event before it reads that layer's KV.
    bool wait_kv_ready(int layer, BackendStream& compute);

    /// Free a slot whose release event completed.  Its buffer may be refilled only after this returned true.
    bool release_done(int layer) const;

    /// Block until every in-flight release completes, and clear the ring.  MUST run on early return, relayout,
    /// and destruction (R2-7).
    void drain();

    size_t in_flight() const;

private:
    int slot_for(int layer) const;

    KvPipelineConfig cfg_;
    std::vector<Slot> slots_;
    KvPipelineGate gate_ = KvPipelineGate::On;
    bool active_ = false;
};
}  // namespace strata::core
