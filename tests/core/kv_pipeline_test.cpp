// KvPipeline on a plain CPU: deferral to an upstream STRATA_KV_PREFETCH, the degradation gate (On at L0/L1, Off at
// L2+), that wait_kv_ready only fires for a staged layer, and that drain() waits every in-flight ready event before it
// clears the ring.  No GPU: needs none.
#include "strata/core/kv_pipeline.hpp"

#include <cstdio>

using strata::core::BackendEvent;
using strata::core::BackendEventCapabilities;
using strata::core::BackendStream;
using strata::core::KvPipeline;
using strata::core::KvPipelineConfig;
using strata::core::KvPipelineGate;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

struct Stream : BackendStream {
    int waits = 0;
    void wait_event(BackendEvent&) override { ++waits; }
};

struct Event : BackendEvent {
    bool done = false;
    int records = 0, syncs = 0;
    void record(BackendStream&) override { ++records; }
    bool query() const override { return done; }
    void wait(BackendStream& s) override { static_cast<Stream*>(&s)->waits += 1; }
    void sync() override { syncs += 1; done = true; }
};

KvPipelineConfig on_cfg(bool upstream = false) {
    KvPipelineConfig c;
    c.enabled = true;
    c.upstream_opt_in = upstream;
    c.ring_slots = 2;
    c.caps.fine_grained_events = true;
    c.caps.can_query = true;
    return c;
}

}  // namespace

int main() {
    // 1. upstream STRATA_KV_PREFETCH owns the transport: this pipeline stays inactive and does nothing
    {
        KvPipeline p(on_cfg(true));
        check(!p.active(), "upstream opt-in leaves this pipeline inactive");
        Event e;
        Stream cs;
        check(p.on_kv_proj_done(0, &e, cs) == -1, "inactive pipeline stages nothing");
        check(e.records == 0, "inactive pipeline recorded nothing");
    }

    // 2. no fine-grained events: inactive (the caller keeps a whole-layer wait)
    {
        KvPipelineConfig c;
        c.enabled = true;
        KvPipeline p(c);
        check(!p.active(), "no events leaves the pipeline inactive");
    }

    // 3. level gate: On at L0, L1 at 1, Off at 2+
    {
        KvPipeline p(on_cfg(false));
        check(p.active(), "enabled + events is active");
        p.set_level(0);
        check(p.gate() == KvPipelineGate::On, "L0 on");
        p.set_level(1);
        check(p.gate() == KvPipelineGate::L1, "L1 halves");
        p.set_level(3);
        check(p.gate() == KvPipelineGate::Off, "L2+ off");
        p.set_level(0);
        Event e;
        Stream cs;
        check(p.on_kv_proj_done(0, &e, cs) == 0, "L0 stages a layer");
        check(e.records >= 1, "the ready event was recorded");
        p.set_level(2);
        Event e2;
        check(p.on_kv_proj_done(1, &e2, cs) == -1, "L2+ stages nothing");
        check(e2.records == 0, "L2+ recorded nothing");
    }

    // 4. wait_kv_ready: waits the staged layer, not an unstaged one
    {
        KvPipeline p(on_cfg(false));
        Event e0;
        Stream cs, compute;
        p.set_level(0);
        p.on_kv_proj_done(5, &e0, cs);
        check(p.wait_kv_ready(5, compute), "a staged layer is waited");
        check(compute.waits == 1, "one wait");
        check(!p.wait_kv_ready(6, compute), "an unstaged layer is not waited");
        check(compute.waits == 1, "still one wait");
    }

    // 5. ring full: stage two, the third is refused; release_done gates the free
    {
        KvPipeline p(on_cfg(false));
        Event e0, e1, e2;
        Stream cs;
        p.set_level(0);
        check(p.on_kv_proj_done(0, &e0, cs) == 0, "stage 0");
        check(p.on_kv_proj_done(1, &e1, cs) == 1, "stage 1");
        check(p.on_kv_proj_done(2, &e2, cs) == -1, "ring full refuses 2");
        check(p.in_flight() == 2, "two in flight");
        check(!p.release_done(0), "0 not released while running");
        e0.done = true;
        check(p.release_done(0), "0 released once its event completed");
    }

    // 6. drain waits every in-flight event, then clears
    {
        KvPipeline p(on_cfg(false));
        Event e0, e1;
        Stream cs;
        p.set_level(0);
        p.on_kv_proj_done(0, &e0, cs);
        p.on_kv_proj_done(1, &e1, cs);
        p.drain();
        check(e0.syncs == 1 && e1.syncs == 1, "drain synced both in-flight events");
        check(p.in_flight() == 0, "drain cleared the ring");
    }

    if (g_fail == 0) std::puts("kv_pipeline_test: ok");
    return g_fail == 0 ? 0 : 1;
}
