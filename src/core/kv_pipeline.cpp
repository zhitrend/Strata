// src/core/kv_pipeline.cpp
#include "strata/core/kv_pipeline.hpp"

namespace strata::core {

KvPipeline::KvPipeline(KvPipelineConfig cfg, std::string* err) : cfg_(cfg) {
    // R2-6: an upstream STRATA_KV_PREFETCH owns the transport.  This pipeline stays inactive so it cannot double it.
    if (cfg_.upstream_opt_in) {
        active_ = false;
        return;
    }
    // R2-12 / R2-3: off unless asked for AND the backend can record and query a fine-grained event.
    active_ = cfg_.enabled && cfg_.caps.supports_per_expert_sync();
    if (!active_) {
        if (err) {
            *err = cfg_.enabled ? "KvPipeline: backend has no fine-grained event, layer sync"
                                : "KvPipeline: disabled";
        }
        return;
    }
    if (cfg_.ring_slots <= 0) {
        active_ = false;
        if (err) *err = "KvPipeline: ring_slots <= 0";
        return;
    }
    slots_.resize((size_t) cfg_.ring_slots);
}

void KvPipeline::set_level(int degradation_level) {
    // L0/L1 the prefetch runs; L1 the window halves; L2+ off (R2-11).  The gate is what the engine reads.
    if (degradation_level <= 0) {
        gate_ = KvPipelineGate::On;
    } else if (degradation_level == 1) {
        gate_ = KvPipelineGate::L1;
    } else {
        gate_ = KvPipelineGate::Off;
    }
}

int KvPipeline::slot_for(int layer) const {
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].in_flight && slots_[i].layer == layer) return (int) i;
    }
    return -1;
}

int KvPipeline::on_kv_proj_done(int layer, BackendEvent* ready, BackendStream& copy_stream) {
    if (!active_ || gate_ == KvPipelineGate::Off || ready == nullptr) return -1;
    // Layer already staged this window: hand back its slot, do not stage twice.
    int existing = slot_for(layer);
    if (existing >= 0) return existing;
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i].in_flight) {
            slots_[i] = Slot{layer, nullptr, ready, true};
            ready->record(copy_stream);
            return (int) i;
        }
    }
    return -1;  // ring full: the caller drains or skips this layer this window
}

bool KvPipeline::wait_kv_ready(int layer, BackendStream& compute) {
    if (!active_ || gate_ == KvPipelineGate::Off) return false;
    int i = slot_for(layer);
    if (i < 0 || slots_[i].ready == nullptr) return false;
    compute.wait_event(*slots_[i].ready);
    return true;
}

bool KvPipeline::release_done(int layer) const {
    int i = slot_for(layer);
    if (i < 0) return false;
    const Slot& s = slots_[i];
    return s.ready == nullptr || s.ready->query();
}

void KvPipeline::drain() {
    if (!active_) return;
    // Wait every in-flight consumer event, then clear: the next refill cannot start before this returns (R2-7).
    for (auto& s : slots_) {
        if (s.in_flight && s.ready != nullptr) s.ready->sync();
    }
    for (auto& s : slots_) {
        s = Slot{};
    }
}

size_t KvPipeline::in_flight() const {
    size_t n = 0;
    for (const auto& s : slots_) n += s.in_flight ? 1 : 0;
    return n;
}

}  // namespace strata::core
