// include/strata/core/residency_manager.hpp - Greedy-Dual residency scoring with an Anchor Expert protected set
// (arch v2.0 3.4).
//
// **WHY GREEDY-DUAL AND NOT A SECOND LRU.**  The cache already decays heat exponentially.  Greedy-Dual adds the two
// things a fixed LRU cannot: a per-object miss cost (so an expert that costs a cluster mmap is charged more
// than an expert that is a page hit) and a protected set for the globally hottest experts that must never be evicted
// (the Anchor Expert Set).  Keeping both in one scorer avoids two eviction policies fighting over the same slots.
//
// **miss_cost IS INJECTED, NOT A CONSTANT.**  For an SSD-resident expert it is the ClusterMmapReader's cluster-read
// latency model; for a RAM-resident expert it is the H2D transfer time.  A fixed constant would be a guess dressed as a
// measurement.
//
// **WHERE THIS RUNS.**  Host only: it scores (layer, expert) entries and names the ones to evict.  The eviction itself
// (unmap/free) is the caller's; this class owns no device memory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

namespace strata::core {

using ExpertKey = int64_t;  // (layer << 32) | expert; -1 reserved

inline ExpertKey make_expert_key(int layer, int expert) {
    return ((ExpertKey) layer << 32) | (ExpertKey) (expert & 0xffffffff);
}
inline int expert_layer_of(ExpertKey k) { return (int) (k >> 32); }
inline int expert_id_of(ExpertKey k) { return (int) (k & 0xffffffff); }

struct ResidencyConfig {
    double heat_decay = 0.9;          // per eviction-time unit; matches the cache's exponential decay
    double cost_estimate = 1.0;      // one miss: the injected miss_cost is added at this weight
    int max_batch = 64;              // cap on entries evicted in one window (batch eviction, not a per-token dribble)
    double lru_fallback_hit_rate = 0.30;  // hit rate under which the caller is told to fall back to plain LRU
};

/// One resident entry's counters, owned by the caller (the cache) and fed here each window.
struct ResidencyEntry {
    ExpertKey key = -1;
    double frequency = 0.0;        // decayed activation frequency
    double miss_cost = 1.0;        // injected: cluster-read latency or H2D time, in the caller's unit
    double last_touched = 0.0;      // the window at this entry's last touch (its decay origin)
};

class ResidencyManager {
public:
    explicit ResidencyManager(const ResidencyConfig& cfg = {});

    /// Fed by ClusterMmapReader (SSD-resident) or the transfer path (RAM-resident); never a constant.
    using MissCostFn = std::function<double(ExpertKey key)>;

    /// The Anchor Expert Set: top-frequency experts, loaded at start and never evicted (3.3.2).
    void set_anchors(const std::vector<ExpertKey>& anchors);
    void set_miss_cost_fn(MissCostFn fn);   // ClusterMmapReader::latency_model adapter

    /// One activation of `key` this window: bump its frequency, refresh its score.
    void touch(ExpertKey key);

    /// Charge every resident entry's decayed frequency by `miss_cost(key)` for one window since its last touch.
    /// One pass per statistical window.
    void advance_window(double now);

    /// **EVICT `target_bytes` WORTH, CHEAPEST FIRST, ANCHORS SKIPPED.**  Lower score = evict first.  Entries
    /// in protected_set (anchors) are never returned.  Batch: one call plans a whole window's evictions, not a per-token
    /// dribble.  Returns the keys chosen, and their total bytes.
    std::vector<ExpertKey> evict(double target_bytes, const std::function<std::int64_t(ExpertKey)>& bytes_of,
                                 std::int64_t* evicted_bytes = nullptr);

    /// Current Greedy-Dual score of one entry (frequency_decayed * miss_cost); 0 when not tracked.
    double score(ExpertKey key) const;

    /// Protected set membership (anchors; also any key added with protect()).
    bool is_protected(ExpertKey key) const;
    void protect(ExpertKey key, bool on = true);

    /// The hit-rate signal from the cache: below lru_fallback_hit_rate() the caller should swap to plain LRU (R2-2).
    bool should_fall_back_to_lru(double window_hit_rate) const;
    double lru_fallback_hit_rate() const { return cfg_.lru_fallback_hit_rate; }

    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        double freq_decayed = 0.0;
        double miss_cost = 1.0;
        double last_touched = 0.0;
        double score = 0.0;
    };

    ResidencyConfig cfg_;
    double now_ = 0.0;                          // Greedy-Dual running clock (one per window advance)
    std::unordered_map<ExpertKey, Entry> entries_;
    std::unordered_set<ExpertKey> protected_;
    MissCostFn miss_cost_fn_;
    std::int64_t charged_bytes_ = 0;            // cumulative charged bytes, kept for tests

    void refresh(ExpertKey key, Entry& e);
};
}  // namespace strata::core
