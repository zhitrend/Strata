// src/core/expert_event_sync.cpp
#include "strata/core/expert_event_sync.hpp"

namespace strata::core {

ExpertEventSync::ExpertEventSync(ExpertEventSyncConfig cfg, std::string* err) : cfg_(cfg) {
    // The per-expert path is live only when the caller enabled it AND the backend can record and query a fine-grained
    // event (R2-3).  A backend without that capability leaves enabled_ false and the caller on layer sync; the error
    // string says so once rather than pretending a wait happened.
    enabled_ = cfg_.enabled && cfg_.caps.supports_per_expert_sync();
    if (enabled_ && cfg_.ring_slots <= 0) {
        enabled_ = false;
        if (err) *err = "ExpertEventSync: enabled but ring_slots <= 0";
        return;
    }
    if (enabled_) slots_.resize((size_t) cfg_.ring_slots);
}

int ExpertEventSync::slot_for(int layer, int expert) const {
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].in_flight && slots_[i].layer == layer && slots_[i].expert == expert) return (int) i;
    }
    return -1;
}

int ExpertEventSync::issue_transfer(int layer, int expert, BackendEvent* ready) {
    if (!enabled_ || ready == nullptr) return -1;
    // Already in flight for this pair: nothing new to record; the caller may wait the same slot.
    int existing = slot_for(layer, expert);
    if (existing >= 0) return existing;
    // First free slot, else -1: the caller must poll_and_recycle.  Never overwrite an in-flight slot.
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i].in_flight) {
            slots_[i] = Slot{layer, expert, ready, true};
            return (int) i;
        }
    }
    return -1;
}

size_t ExpertEventSync::wait_needed(const std::vector<std::pair<int, int>>& needed, BackendStream& compute) {
    if (!enabled_) return 0;
    size_t waits = 0;
    for (const auto& [layer, expert] : needed) {
        int i = slot_for(layer, expert);
        if (i >= 0) {
            compute.wait_event(*slots_[i].ready);
            ++waits;
        }
    }
    return waits;
}

size_t ExpertEventSync::poll_and_recycle() {
    if (!enabled_) return 0;
    size_t recycled = 0;
    for (auto& s : slots_) {
        if (!s.in_flight) continue;
        if (s.ready != nullptr && !s.ready->query()) continue;  // still running: keep the slot
        s.in_flight = false;
        s.layer = -1;
        s.expert = -1;
        s.ready = nullptr;
        ++recycled;
    }
    return recycled;
}

size_t ExpertEventSync::in_flight() const {
    size_t n = 0;
    for (const auto& s : slots_) n += s.in_flight ? 1 : 0;
    return n;
}

}  // namespace strata::core
