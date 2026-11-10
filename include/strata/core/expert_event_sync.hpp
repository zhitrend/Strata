// include/strata/core/expert_event_sync.hpp - per-expert ready-event ring over BackendEvent (arch v2.0 §3.2.2).
//
// **WHY THIS EXISTS.**  The split engine's offload waits the WHOLE device at the window edge
// (`cudaDeviceSynchronize()` in engine_common.cu), so every layer pays for the slowest expert of every layer.  A token
// routes to a handful of experts per layer; the rest of the copy work could still be in flight.  This ring records one
// ready event per issued H2D copy and lets the compute stream wait ONLY the experts the token actually routed to
// (principle R2).  It reuses the staging pool's lifecycle (release protects the last reader, ready orders the next
// consumer, a slot is reused only after its previous event completed) rather than inventing a second pool.
//
// **THE BACKEND IS BEHIND AN INTERFACE.**  This translation unit names no CUDA/Metal/SYCL type: it takes BackendEvent
// and BackendStream factories.  src/split supplies the CUDA ones.  A backend with no fine-grained event passes a
// capabilities struct with supports_per_expert_sync()==false, and this class becomes a no-op that leaves the caller on
// layer sync (R2-3) - it never fakes a per-expert wait.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/backend_event.hpp"

namespace strata::core {

struct ExpertEventSyncConfig {
    int ring_slots = 0;          // 0: one slot per (layer, expert) the caller may need in flight; set by the caller
    bool enabled = true;         // STRATA_EXPERT_EVENT_SYNC; false = layer-sync fallback
    BackendEventCapabilities caps;
};

/// Per-expert ready-event ring.  One instance per backend that has fine-grained events (in practice: CUDA).
///
/// Lifecycle per (layer, expert) that will be needed this step:
///   issue_transfer(layer, expert, ready)  -> non-blocking H2D on the copy stream + record ready event
///   wait_needed({{layer, expert}, ...}, compute_stream)   -> compute stream waits only those ready events
///   poll_and_recycle()                                    -> free ring slots whose ready event completed
///
/// `issue_transfer` has no device bytes here; the CUDA adapter (src/split) makes `bytes` the staging slot and does the
/// copy.  This class only carries the slot/quarantine bookkeeping and the wait-on-ready events decision.
class ExpertEventSync {
public:
    struct Slot {
        int layer = -1;
        int expert = -1;
        BackendEvent* ready = nullptr;  // recorded by issue_transfer on the copy stream
        bool in_flight = false;
    };

    explicit ExpertEventSync(ExpertEventSyncConfig cfg, std::string* err = nullptr);

    /// True when the per-expert path is live.  False: the caller keeps layer sync and must not call record/wait here.
    bool enabled() const { return enabled_; }
    bool supports_per_expert_sync() const { return cfg_.caps.supports_per_expert_sync(); }

    /// Records a ready event for `(layer, expert)` on the copy stream and returns its slot, or -1 when disabled or
    /// when the ring has no slot whose previous event completed (the caller must poll_and_recycle first).  `ready`
    /// is the backend event the CUDA adapter recorded after submitting the H2D copy; the ring takes ownership of the
    /// pointer for its lifetime.
    int issue_transfer(int layer, int expert, BackendEvent* ready);

    /// The compute stream waits exactly the events of `needed` (a contiguous run of (layer, expert) pairs), skipping
    /// any that are not in flight.  Counts the waits actually issued (for the profiler and the tests).
    size_t wait_needed(const std::vector<std::pair<int, int>>& needed, BackendStream& compute);

    /// Recycle every slot whose ready event completed (query()==true).  Returns how many were reused.
    size_t poll_and_recycle();

    /// Slots still in flight (not yet recycled).  The tests read this to check recycling.
    size_t in_flight() const;
    int ring_size() const { return (int) cfg_.ring_slots; }

private:
    int slot_for(int layer, int expert) const;

    ExpertEventSyncConfig cfg_;
    std::vector<Slot> slots_;
    bool enabled_ = false;
};
} // namespace strata::core
