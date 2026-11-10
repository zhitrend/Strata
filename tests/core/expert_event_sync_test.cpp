// ExpertEventSync on a plain CPU: the ring hands out free slots, refuses to overwrite an in-flight one, waits only
// the needed experts, recycles a slot only after its ready event's query() is true, and turns itself off (layer sync)
// when the backend has no fine-grained event.  No GPU: needs none.
#include "strata/core/expert_event_sync.hpp"

#include <cstdio>

using strata::core::BackendEvent;
using strata::core::BackendEventCapabilities;
using strata::core::BackendStream;
using strata::core::ExpertEventSync;
using strata::core::ExpertEventSyncConfig;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

// A stream that counts the events it was made to wait for (wait_event calls).
struct CountingStream : BackendStream {
    int waits = 0;
    void wait_event(BackendEvent&) override { ++waits; }
};

// A fake ready event whose query() the test flips.
struct FakeEvent : BackendEvent {
    bool done = false;
    CountingStream* recorder = nullptr;
    void record(BackendStream& s) override { recorder = static_cast<CountingStream*>(&s); }
    bool query() const override { return done; }
    void wait(BackendStream& s) override { static_cast<CountingStream*>(&s)->waits += 1; }
    void sync() override { done = true; }
};

ExpertEventSyncConfig on_config(int ring) {
    ExpertEventSyncConfig c;
    c.ring_slots = ring;
    c.enabled = true;
    c.caps.fine_grained_events = true;
    c.caps.can_query = true;
    return c;
}

}  // namespace

int main() {
    // 1. a backend with no fine-grained event: disabled, and every method is a no-op (the caller keeps layer sync)
    {
        ExpertEventSyncConfig c;
        c.ring_slots = 4;
        c.enabled = true;
        c.caps.fine_grained_events = false;
        c.caps.can_query = false;
        ExpertEventSync s(c);
        check(!s.enabled(), "no-event backend leaves the sync disabled");
        check(s.issue_transfer(0, 0, nullptr) == -1, "disabled sync issues no transfer");

        CountingStream cs;
        FakeEvent e;
        check(s.wait_needed({{0, 0}}, cs) == 0 && cs.waits == 0, "disabled sync waits nothing");
        check(s.poll_and_recycle() == 0, "disabled sync recycles nothing");
    }

    // 2. enabled but ring_slots 0: refuses to come up, says why
    {
        std::string err;
        ExpertEventSync s(on_config(0), &err);
        check(!s.enabled(), "ring_slots 0 leaves the sync disabled");
        check(!err.empty(), "ring_slots 0 reports an error");
    }

    // 3. the ring hands out slots in order and refuses to overwrite an in-flight slot
    {
        FakeEvent e0, e1;
        ExpertEventSync s(on_config(2));
        check(s.enabled(), "two slots enable the sync");
        check(s.issue_transfer(0, 1, &e0) == 0, "first transfer takes slot 0");
        check(s.issue_transfer(0, 2, &e1) == 1, "second transfer takes slot 1");
        check(s.in_flight() == 2, "two in flight");

        FakeEvent e2;
        check(s.issue_transfer(0, 3, &e2) == -1, "a full ring refuses a third transfer, no overwrite");
        check(s.in_flight() == 2, "still two in flight after the refusal");

        // same (layer, expert) again: reuses its slot, records nothing new
        check(s.issue_transfer(0, 1, &e2) == 0, "a repeated pair reuses its slot");
        check(s.in_flight() == 2, "no new slot for a repeated pair");
    }

    // 4. wait_needed waits only the pairs asked for; unknown pairs are skipped
    {
        FakeEvent e0, e1;
        ExpertEventSync s(on_config(4));
        s.issue_transfer(1, 10, &e0);
        s.issue_transfer(1, 11, &e1);

        CountingStream compute;
        check(s.wait_needed({{1, 10}}, compute) == 1, "one needed pair: one wait");
        check(compute.waits == 1, "the stream saw one wait");
        check(s.wait_needed({{1, 10}, {1, 11}, {9, 99}}, compute) == 2, "two resident + one unknown: two waits");
        check(compute.waits == 3, "three waits total");
        check(s.wait_needed({{9, 99}}, compute) == 0, "an unknown pair waits nothing");
        check(compute.waits == 3, "no extra wait for the unknown pair");
    }

    // 5. recycle only after the ready event completes; the slot becomes free afterwards
    {
        FakeEvent e0, e1;
        ExpertEventSync s(on_config(2));
        s.issue_transfer(2, 20, &e0);
        s.issue_transfer(2, 21, &e1);

        check(s.poll_and_recycle() == 0, "nothing recycles while both events run");
        check(s.in_flight() == 2, "both still in flight");

        e0.done = true;
        check(s.poll_and_recycle() == 1, "the completed event is recycled, the running one kept");
        check(s.in_flight() == 1, "one left in flight");

        FakeEvent e2;
        check(s.issue_transfer(2, 22, &e2) == 0, "the freed slot is handed out again");

        e1.done = true;
        e2.done = true;
        check(s.poll_and_recycle() == 2, "the rest recycle");
        check(s.in_flight() == 0, "nothing in flight");
    }

    if (g_fail == 0) std::puts("expert_event_sync_test: ok");
    return g_fail == 0 ? 0 : 1;
}
