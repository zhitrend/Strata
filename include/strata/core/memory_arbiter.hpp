// src/core/memory_arbiter.hpp - unified VRAM/RAM budget arbitration between Expert Cache and KV Pool.
#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>

namespace strata::core {

struct ExpertCacheCost {
    size_t working_set_bytes = 0;   // currently resident expert bytes
    size_t target_bytes = 0;        // desired resident bytes (from budget)
    double pressure = 0.0;          // working_set / target (1.0 = full)
};

struct KvPagesCost {
    int working_set_pages = 0;      // currently allocated KV pages
    int target_pages = 0;           // desired KV pages (from budget)
    double pressure = 0.0;          // working_set / target (1.0 = full)
};

struct BudgetPlan {
    size_t target_expert_bytes = 0; // this round's expert budget
    int target_kv_pages = 0;        // this round's KV page budget
    int degradation_level = 0;      // 0..5 (DegradationLevel)
    bool kv_ram_tier_enabled = false;
};

/// Sub-budgets the arbiter hands the three pools (arch v2.0 3.1.2).  Expert and KV are the zero-sum pair; Staging is
/// the prefetch pool's reservation (the prefetcher asks here, it does not take from Expert ad hoc).  They sum to the
/// usable VRAM (free less headroom).
struct SubBudgets {
    size_t expert_bytes = 0;
    size_t kv_bytes = 0;
    size_t staging_bytes = 0;
    size_t total() const { return expert_bytes + kv_bytes + staging_bytes; }
};

/// What the arbiter sees each window to decide whether to move budget between Expert and KV ajax.  The miss cost is a
/// sliding average in the caller's unit; comparing the two decides the direction (3.1.2).  `pressure_high_windows` is
/// the hysteresis counter: a move happens only when the same direction held for the configured N windows.
struct UsageReport {
    double miss_cost = 0.0;         // freq x miss_cost sliding average, this pool
    int pressure_high_windows = 0;  // consecutive windows this pool is the one under pressure
};

/// The arbiter's initial split and its per-window migration policy (arch v2.0 3.1.2).
struct RebalanceConfig {
    double expert_default = 0.60;   // Expert 60% / KV 30% / Staging 10%
    double kv_default = 0.30;
    double staging_default = 0.10;
    double max_migration_fraction = 0.05;  // one window moves at most 5% of total
    int hysteresis_windows = 3;            // a direction must hold this many windows to move
};

struct ArbiterConfig {
    double headroom_ratio = 0.10;       // reserve 10% of free VRAM
    double expert_weight = 1.0;         // expert pressure weight
    double kv_weight = 1.2;             // KV pressure weight (slightly higher priority)
    int low_tick_threshold = 3;         // consecutive low-water ticks to degrade
    int high_tick_threshold = 5;        // consecutive high-water ticks to recover
    size_t min_expert_bytes = 2ull << 30; // minimum expert budget (2 GiB)
    int min_kv_pages = 1000;            // minimum KV pages
};

class MemoryBudgetArbiter {
public:
    explicit MemoryBudgetArbiter(const ArbiterConfig& cfg = {});

    BudgetPlan tick(size_t free_vram_bytes,
                    const ExpertCacheCost& expert_cost,
                    const KvPagesCost& kv_cost);

    void set_water_marks(size_t low_watermark_bytes, size_t high_watermark_bytes);

    int current_degradation_level() const { return degradation_level_; }

    /// Phase 2 (3.1.2): the initial Expert / KV / Staging split from the hardware and model shape.  Default 60/30/10;
    /// the default is a starting point, not a measured optimum - size it on the target card (tasklist 2.4).
    SubBudgets initial_budgets(size_t total_vram, size_t total_ram) const;

    /// Phase 2 (3.1.2): one statistical window's rebalance.  Moves budget toward the pool whose miss cost is higher,
    /// at most `rcfg.max_migration_fraction` of total this window, and only when that direction held
    /// `rcfg.hysteresis_windows` windows (anti-flap).  KV never spills to SSD: kv_bytes only decides VRAM vs pinned
    /// RAM residency; staging moves come from the same usable VRAM pool.
    SubBudgets rebalance(const SubBudgets& current, const UsageReport& expert_usage,
                         const UsageReport& kv_usage, const RebalanceConfig& rcfg = {});

private:
    ArbiterConfig cfg_;
    std::atomic<int> degradation_level_{0};
    int low_tick_count_ = 0;
    int high_tick_count_ = 0;
    size_t low_watermark_ = 0;
    size_t high_watermark_ = 0;
};

}  // namespace strata::core