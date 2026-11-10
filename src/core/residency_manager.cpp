// src/core/residency_manager.cpp
#include "strata/core/residency_manager.hpp"

#include <algorithm>
#include <cmath>

namespace strata::core {

ResidencyManager::ResidencyManager(const ResidencyConfig& cfg) : cfg_(cfg) {}

void ResidencyManager::set_anchors(const std::vector<ExpertKey>& anchors) {
    for (ExpertKey k : anchors) protected_.insert(k);
}

void ResidencyManager::set_miss_cost_fn(MissCostFn fn) { miss_cost_fn_ = std::move(fn); }

void ResidencyManager::protect(ExpertKey key, bool on) {
    if (on) protected_.insert(key);
    else protected_.erase(key);
}

bool ResidencyManager::is_protected(ExpertKey key) const { return protected_.count(key) != 0; }

void ResidencyManager::refresh(ExpertKey key, Entry& e) {
    // Age the heat up to the current window first, then apply the injected miss cost (Greedy-Dual score).
    if (e.last_touched >= 0.0 && now_ > e.last_touched) {
        e.freq_decayed *= std::pow(cfg_.heat_decay, now_ - e.last_touched);
        e.last_touched = now_;
    }
    double mc = e.miss_cost;
    if (miss_cost_fn_) mc = miss_cost_fn_(key);
    e.miss_cost = mc;
    e.score = e.freq_decayed * mc;
}

void ResidencyManager::touch(ExpertKey key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        Entry e;
        e.freq_decayed = 1.0;
        e.last_touched = now_;
        refresh(key, e);
        entries_[key] = e;
        return;
    }
    Entry& e = it->second;
    if (e.last_touched >= 0.0 && now_ > e.last_touched) {
        e.freq_decayed *= std::pow(cfg_.heat_decay, now_ - e.last_touched);
    }
    e.freq_decayed += 1.0;
    e.last_touched = now_;
    refresh(key, e);
}

void ResidencyManager::advance_window(double now) {
    now_ = now;
    for (auto& [key, e] : entries_) refresh(key, e);
}

double ResidencyManager::score(ExpertKey key) const {
    auto it = entries_.find(key);
    return it == entries_.end() ? 0.0 : it->second.score;
}

bool ResidencyManager::should_fall_back_to_lru(double window_hit_rate) const {
    return window_hit_rate < cfg_.lru_fallback_hit_rate;
}

std::vector<ExpertKey> ResidencyManager::evict(double target_bytes,
                                               const std::function<std::int64_t(ExpertKey)>& bytes_of,
                                               std::int64_t* evicted_bytes) {
    std::int64_t freed = 0;
    std::vector<ExpertKey> out;

    // Candidate list: everything tracked, not protected.  Cheap to build for the sizes this runs at (experts per
    // window, hundreds to low thousands).
    std::vector<std::pair<double, ExpertKey>> cand;
    cand.reserve(entries_.size());
    for (const auto& [key, e] : entries_) {
        if (is_protected(key)) continue;
        cand.emplace_back(e.score, key);
    }
    // Cheapest first: the score Greedy-Dual would evict soonest.
    std::sort(cand.begin(), cand.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (const auto& [s, key] : cand) {
        if ((int) out.size() >= cfg_.max_batch) break;
        if (freed >= (std::int64_t) target_bytes) break;
        std::int64_t b = bytes_of ? bytes_of(key) : 0;
        out.push_back(key);
        freed += b;
    }
    charged_bytes_ += freed;
    if (evicted_bytes) *evicted_bytes = freed;
    return out;
}

}  // namespace strata::core
