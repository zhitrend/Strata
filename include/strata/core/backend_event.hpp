// include/strata/core/backend_event.hpp - backend-agnostic event/stream primitives for per-expert sync and the KV
// pipeline (arch v2.0 §3.2.1).  No CUDA / Metal / SYCL type appears here: the core and its CPU tests build without
// a GPU toolkit, and each backend supplies an adapter (the CUDA one in src/split wraps cudaEvent_t + cudaStream_t).
//
// **WHY AN ABSTRACTION AND NOT cudaEvent_t EVERYWHERE.**  The decode path must wait only the ready events of the
// experts the current token routed to (principle R2), and a backend that has no fine-grained event must fall back to
// layer sync without blocking the CUDA path (R2-3).  A thin interface lets ExpertEventSync and KvPipeline be written
// once against it and unit-tested, and lets `NullBackendEvent` stand for the no-event backends.
#pragma once

#include <cstdint>
#include <memory>

namespace strata::core {

/// One backend stream.  The only operation the sync layer needs is "run after this event".
class BackendEvent;

class BackendStream {
public:
    virtual ~BackendStream() = default;
    /// Make this stream wait for `e` to complete before the work after this call runs.  Maps to cudaStreamWaitEvent.
    virtual void wait_event(BackendEvent& e) = 0;
};

/// One backend event.  record/wait are ordered on their stream; query is a non-blocking poll (R2-1).
class BackendEvent {
public:
    virtual ~BackendEvent() = default;
    /// Record this event on `s`; its completion is ordered after all work already submitted to `s`.
    virtual void record(BackendStream& s) = 0;
    /// Non-blocking: true once the work this event was recorded after has completed.
    virtual bool query() const = 0;
    /// Make `s` wait for this event.  (Explicit form; BackendStream::wait_event is the usual one.)
    virtual void wait(BackendStream& s) = 0;
    /// Block the calling thread until the event completes.  Diagnostic / fallback path only (R2-3).
    virtual void sync() = 0;
};

/// **THE LAZY EVENT FOR A BACKEND WITH NO FINE-GRAINED EVENTS.**  `record` is a no-op, `query` is always true,
/// `wait` and `sync` do nothing.  With this in place the engine is allowed only after it has detected the capability
/// gap and switched ExpertEventSync to layer synchronization; the object itself never pretends a per-expert wait
/// happened.
class NullBackendEvent final : public BackendEvent {
public:
    void record(BackendStream&) override {}
    bool query() const override { return true; }
    void wait(BackendStream&) override {}
    void sync() override {}
};

/// A BackendStream that records nothing and waits nothing: the stand-in the layer-sync fallback passes around in
/// tests.
class NullBackendStream final : public BackendStream {
public:
    void wait_event(BackendEvent&) override {}
};

/// What the current backend can do, so the engine can pick per-expert sync or layer sync once at start (R2-3).
struct BackendEventCapabilities {
    bool fine_grained_events = false;  // false: use NullBackendEvent + layer sync
    bool can_query = false;         // false: poll_and_recycle cannot run; recycle per completed layer
    /// True when this backend may use the per-expert path.  Both flags off means layer sync.
    bool supports_per_expert_sync() const { return fine_grained_events && can_query; }
};

}  // namespace strata::core
