// MemoryBudgetArbiter::tick and DegradationStateMachine on a plain CPU: hysteresis, the proportional budget,
// and the degradation levels wired into the prefetch queries.  No GPU: needs none.
#include "strata/core/degradation.hpp"
#include "strata/core/memory_arbiter.hpp"

#include <cstdio>

using strata::core::ArbiterConfig;
using strata::core::BudgetPlan;
using strata::core::DegradationConfig;
using strata::core::DegradationLevel;
using strata::core::DegradationStateMachine;
using strata::core::ExpertCacheCost;
using strata::core::KvPagesCost;
using strata::core::MemoryBudgetArbiter;
using strata::core::RebalanceConfig;
using strata::core::SubBudgets;
using strata::core::UsageReport;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

}  // namespace

int main() {
    // 1. no pressure: the level stays Normal, the budget is handed out with the headroom taken off, the RAM tier off
    {
        ArbiterConfig cfg;
        cfg.headroom_ratio = 0.10;
        MemoryBudgetArbiter a(cfg);
        a.set_water_marks(4ull << 30, 8ull << 30);
        ExpertCacheCost e;
        e.working_set_bytes = 2ull << 30;
        e.target_bytes = 2ull << 30;
        KvPagesCost k;
        k.working_set_pages = 100;
        k.target_pages = 100;
        for (int i = 0; i < 10; ++i) {
            (void) a.tick(16ull << 30, e, k);
        }
        check(a.current_degradation_level() == 0, "above the low mark stays Normal");
        const BudgetPlan p = a.tick(16ull << 30, e, k);
        check(p.target_expert_bytes > 0, "a budget is handed out");
        check(p.target_kv_pages > 0, "KV pages too");
        check(!p.kv_ram_tier_enabled, "the RAM tier is off at Normal");
    }

    // 2. sustained low water: three ticks raise the level (hysteresis), and it stops at 5
    {
        ArbiterConfig cfg;
        cfg.low_tick_threshold = 3;
        MemoryBudgetArbiter a(cfg);
        a.set_water_marks(4ull << 30, 8ull << 30);
        ExpertCacheCost e;
        e.working_set_bytes = 1ull << 30;
        e.target_bytes = 1ull << 30;
        KvPagesCost k;
        k.working_set_pages = 10;
        k.target_pages = 10;
        for (int i = 0; i < 2; ++i) (void) a.tick(1ull << 30, e, k);
        check(a.current_degradation_level() == 0, "two low ticks are not enough");
        (void) a.tick(1ull << 30, e, k);
        check(a.current_degradation_level() == 1, "the third low tick steps down one");
        for (int i = 0; i < 30; ++i) (void) a.tick(0, e, k);
        check(a.current_degradation_level() == 5, "it stops at 5");
    }

    // 3. one high tick resets the low count: a low run interrupted never fires
    {
        ArbiterConfig cfg;
        cfg.low_tick_threshold = 3;
        MemoryBudgetArbiter a(cfg);
        a.set_water_marks(4ull << 30, 8ull << 30);
        ExpertCacheCost e;
        KvPagesCost k;
        (void) a.tick(1ull << 30, e, k);
        (void) a.tick(1ull << 30, e, k);
        (void) a.tick(16ull << 30, e, k);   // the high water resets the low count
        (void) a.tick(1ull << 30, e, k);
        (void) a.tick(1ull << 30, e, k);
        check(a.current_degradation_level() == 0, "a high tick clears the low count");
    }

    // 4. the top level has the floored budget
    {
        ArbiterConfig cfg;
        cfg.min_expert_bytes = 1ull << 30;
        cfg.min_kv_pages = 1;
        MemoryBudgetArbiter a(cfg);
        a.set_water_marks(1, ~0ull);
        ExpertCacheCost e;
        KvPagesCost k;
        BudgetPlan p = a.tick(0, e, k);
        for (int i = 0; i < 40; ++i) p = a.tick(0, e, k);
        check(p.degradation_level == 5, "five levels");
        check(p.kv_ram_tier_enabled, "the RAM tier is on by level 3");
        check(p.target_expert_bytes == cfg.min_expert_bytes, "the dynamic expert budget is floored");
    }

    // 5. DegradationStateMachine: the query meanings per level
    {
        DegradationConfig dcfg;
        dcfg.shrink_prefetch_window_scale = 0.5;
        dcfg.high_conf_threshold = 0.8;
        DegradationStateMachine d(dcfg);
        d.apply_level(DegradationLevel::Normal);
        check(d.prefetch_enabled() && d.ngram_prefetch_enabled(), "Normal: both prefetchers run");
        check(d.prefetch_window_scale() == 1.0, "Normal: the window is whole");
        d.apply_level(DegradationLevel::ShrinkPrefetch);
        check(d.prefetch_window_scale() == 0.5, "Shrink: half a window");
        d.apply_level(DegradationLevel::HighConfOnly);
        check(d.prefetch_confidence_threshold() == 0.8, "HighConf: the 0.8 threshold");
        d.apply_level(DegradationLevel::NoNgram);
        check(d.prefetch_enabled() && !d.ngram_prefetch_enabled(), "NoNgram: the model prefetch stays, the n-gram goes");
        d.apply_level(DegradationLevel::NoPrefetch);
        check(!d.prefetch_enabled() && !d.ngram_prefetch_enabled(), "NoPrefetch: neither");
        d.apply_level(DegradationLevel::FreezeLowPrio);
        check(d.freeze_low_priority_sessions(), "Freeze: the sessions freeze");
    }

    // 4. Phase 2: initial_budgets splits the VRAM 60/30/10 and the rounding goes to the expert cache
    {
        MemoryBudgetArbiter a;
        const size_t vram = 12ull << 30;   // a 12 GB card
        SubBudgets b = a.initial_budgets(vram, 32ull << 30);
        check(b.expert_bytes + b.kv_bytes + b.staging_bytes == vram, "initial budgets sum to the VRAM");
        check(b.expert_bytes > b.kv_bytes && b.kv_bytes > b.staging_bytes, "expert > kv > staging");
        check(b.expert_bytes >= (size_t) ((double) vram * 0.60), "expert at least 60%");
    }

    // 5. rebalance: with the direction confirmed and the expert under pressure, budget moves toward the expert, at
    //    most the migration cap this window, and the split stays whole
    {
        MemoryBudgetArbiter a;
        RebalanceConfig rc;
        rc.max_migration_fraction = 0.05;  // 5% of total per window
        rc.hysteresis_windows = 3;
        const size_t total = 10ull << 30;
        SubBudgets cur;
        cur.expert_bytes = total * 60 / 100;
        cur.kv_bytes = total * 30 / 100;
        cur.staging_bytes = total - cur.expert_bytes - cur.kv_bytes;

        UsageReport hot_expert;
        hot_expert.miss_cost = 5.0;
        hot_expert.pressure_high_windows = 3;
        UsageReport cool_kv;
        cool_kv.miss_cost = 1.0;
        cool_kv.pressure_high_windows = 0;

        SubBudgets nx = a.rebalance(cur, hot_expert, cool_kv, rc);
        check(nx.expert_bytes > cur.expert_bytes, "the expert budget grew under expert pressure");
        check(nx.total() == cur.total(), "the split stays whole");
        check(nx.expert_bytes - cur.expert_bytes <= (size_t) ((double) total * 0.05) + 1,
              "the move is capped by the migration fraction");
    }

    // 6. rebalance: a direction under the hysteresis count does not move
    {
        MemoryBudgetArbiter a;
        RebalanceConfig rc;
        rc.hysteresis_windows = 3;
        SubBudgets cur{3, 3, 3};
        UsageReport hot_expert; hot_expert.miss_cost = 5.0; hot_expert.pressure_high_windows = 1;  // unconfirmed
        UsageReport cool_kv;    cool_kv.miss_cost = 1.0;
        SubBudgets nx = a.rebalance(cur, hot_expert, cool_kv, rc);
        check(nx.expert_bytes == cur.expert_bytes && nx.kv_bytes == cur.kv_bytes, "unconfirmed direction does not move");
    }

    // 7. rebalance: equal miss cost moves nothing
    {
        MemoryBudgetArbiter a;
        SubBudgets cur{100, 100, 100};
        UsageReport e; e.miss_cost = 2.0; e.pressure_high_windows = 99;
        UsageReport k; k.miss_cost = 2.0; k.pressure_high_windows = 99;
        SubBudgets nx = a.rebalance(cur, e, k, RebalanceConfig{});
        check(nx.expert_bytes == 100 && nx.kv_bytes == 100 && nx.staging_bytes == 100, "ties do not move");
    }

    if (g_fail) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    std::puts("memory_arbiter_test: ok");
    return 0;
}