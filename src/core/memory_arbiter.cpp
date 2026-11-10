// src/core/memory_arbiter.cpp
#include "strata/core/memory_arbiter.hpp"

#include <algorithm>

namespace strata::core {

MemoryBudgetArbiter::MemoryBudgetArbiter(const ArbiterConfig& cfg) : cfg_(cfg) {}

void MemoryBudgetArbiter::set_water_marks(size_t low_watermark_bytes, size_t high_watermark_bytes) {
    low_watermark_ = low_watermark_bytes;
    high_watermark_ = high_watermark_bytes;
}

BudgetPlan MemoryBudgetArbiter::tick(size_t free_vram_bytes,
                                     const ExpertCacheCost& expert_cost,
                                     const KvPagesCost& kv_cost) {
    BudgetPlan plan;

    // Calculate available budget (free VRAM minus headroom)
    size_t available_budget = 0;
    if (free_vram_bytes > 0) {
        size_t headroom = static_cast<size_t>(free_vram_bytes * cfg_.headroom_ratio);
        if (free_vram_bytes > headroom) {
            available_budget = free_vram_bytes - headroom;
        }
    }

    // Demand: what each pool would like to hold this round.  When a pool reports no target, its share among the
    // weights is the fallback; when it reports one, the demand drives the split (a pool asking for nothing gets the
    // floor below).
    size_t expert_demand = expert_cost.target_bytes ? expert_cost.target_bytes : expert_cost.working_set_bytes;
    size_t kv_demand = static_cast<size_t>(kv_cost.target_pages ? kv_cost.target_pages : kv_cost.working_set_pages) * 4096;
    size_t total_demand = expert_demand + kv_demand;

    // Determine degradation level based on water marks
    if (low_watermark_ > 0 && free_vram_bytes < low_watermark_) {
        low_tick_count_++;
        high_tick_count_ = 0;
        if (low_tick_count_ >= cfg_.low_tick_threshold) {
            degradation_level_ = std::min(5, degradation_level_ + 1);
            low_tick_count_ = 0; // reset after triggering
        }
    } else if (high_watermark_ > 0 && free_vram_bytes > high_watermark_) {
        high_tick_count_++;
        low_tick_count_ = 0;
        if (high_tick_count_ >= cfg_.high_tick_threshold) {
            degradation_level_ = std::max(0, degradation_level_ - 1);
            high_tick_count_ = 0; // reset after triggering
        }
    } else {
        // In between water marks - no change
        low_tick_count_ = 0;
        high_tick_count_ = 0;
    }

    plan.degradation_level = degradation_level_;

    // Allocate budget: proportional to the weights, but nudged by demand - a pool asking for more (relative to its
    // weight) gets more, together moves at most 15% off its weight share.  With no demand on either side this is the
    // plain weight split.
    double w_exp = cfg_.expert_weight, w_kv = cfg_.kv_weight;
    if (total_demand > 0) {
        const double d_exp = (double) expert_demand / (double) total_demand;
        const double d_kv = (double) kv_demand / (double) total_demand;
        const double wsum = w_exp + w_kv;
        w_exp = w_exp * 0.85 + wsum * d_exp * 0.15;
        w_kv = w_kv * 0.85 + wsum * d_kv * 0.15;
    }
    double total_weight = w_exp + w_kv;
    double expert_share = w_exp / total_weight;
    double kv_share = w_kv / total_weight;

    // Apply degradation level scaling
    double degradation_scale = 1.0;
    switch (degradation_level_) {
        case 0: degradation_scale = 1.0; break;
        case 1: degradation_scale = 0.9; break;  // ShrinkPrefetch
        case 2: degradation_scale = 0.8; break;  // HighConfOnly
        case 3: degradation_scale = 0.7; break;  // NoNgram
        case 4: degradation_scale = 0.6; break;  // NoPrefetch
        case 5: degradation_scale = 0.5; break;  // FreezeLowPrio
        default: degradation_scale = 0.5; break;
    }

    size_t scaled_budget = static_cast<size_t>(available_budget * degradation_scale);

    // Allocate to expert and KV
    plan.target_expert_bytes = static_cast<size_t>(scaled_budget * expert_share);
    if (plan.target_expert_bytes < cfg_.min_expert_bytes) {
        plan.target_expert_bytes = cfg_.min_expert_bytes;
    }

    plan.target_kv_pages = static_cast<int>((scaled_budget * kv_share) / 4096);
    if (plan.target_kv_pages < cfg_.min_kv_pages) {
        plan.target_kv_pages = cfg_.min_kv_pages;
    }

    // Enable KV RAM tier at higher degradation levels
    plan.kv_ram_tier_enabled = (degradation_level_ >= 3);

    return plan;
}

SubBudgets MemoryBudgetArbiter::initial_budgets(size_t total_vram, size_t total_ram) const {
    // Default 60 / 30 / 10 of the VRAM the pools share.  total_ram is accepted for a later expert-to-RAM tier and is
    // not used in the VRAM split here; naming it keeps the signature the spec's (3.1.2) rather than a bare size_t.
    (void) total_ram;
    SubBudgets b;
    b.expert_bytes = (size_t) ((double) total_vram * 0.60);
    b.kv_bytes = (size_t) ((double) total_vram * 0.30);
    b.staging_bytes = (size_t) ((double) total_vram * 0.10);
    const size_t used = b.expert_bytes + b.kv_bytes + b.staging_bytes;
    if (used < total_vram) b.expert_bytes += total_vram - used;  // rounding goes to the expert cache
    return b;
}

SubBudgets MemoryBudgetArbiter::rebalance(const SubBudgets& current, const UsageReport& expert_usage,
                                          const UsageReport& kv_usage, const RebalanceConfig& rcfg) {
    SubBudgets next = current;
    const size_t total = current.total();
    if (total == 0) return next;

    const int need = rcfg.hysteresis_windows > 0 ? rcfg.hysteresis_windows : 1;
    const size_t cap = (size_t) ((double) total * rcfg.max_migration_fraction);
    if (cap == 0) return next;

    if (expert_usage.miss_cost > kv_usage.miss_cost && expert_usage.pressure_high_windows >= need) {
        size_t move = cap;
        size_t from_kv = std::min(move, next.kv_bytes);
        next.kv_bytes -= from_kv;
        move -= from_kv;
        size_t from_staging = std::min(move, next.staging_bytes);
        next.staging_bytes -= from_staging;
        move -= from_staging;
        next.expert_bytes += cap - move;
    } else if (kv_usage.miss_cost > expert_usage.miss_cost && kv_usage.pressure_high_windows >= need) {
        size_t move = cap;
        size_t from_expert = std::min(move, next.expert_bytes);
        next.expert_bytes -= from_expert;
        move -= from_expert;
        size_t from_staging = std::min(move, next.staging_bytes);
        next.staging_bytes -= from_staging;
        move -= from_staging;
        next.kv_bytes += cap - move;
    }
    return next;
}

}  // namespace strata::core