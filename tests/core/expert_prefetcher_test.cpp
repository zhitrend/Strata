// ExpertPrefetcher on a plain CPU: P0 ranks the prefill window, the degradation machine gates it cooperatively,
// lookahead and ngram levels stay off until asked.  No GPU: needs none.
#include "strata/core/degradation.hpp"
#include "strata/core/expert_prefetcher.hpp"

#include <cstdio>

using strata::core::DegradationLevel;
using strata::core::DegradationStateMachine;
using strata::core::ExpertPrefetcher;
using strata::core::PrefetchConfig;

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
    // P0 ranks the prefill window by frequency and keeps at most top_k
    {
        PrefetchConfig cfg;
        cfg.top_k = 4;
        ExpertPrefetcher p(cfg);
        p.on_prefill_end({{0, 1, 2, 2}, {0, 0, 0, 0, 1}});
        const auto plan = p.plan_p0();
        check(!plan.empty(), "P0 plans something after prefill");
        check(plan.size() <= 4, "P0 keeps at most top_k");
        if (!plan.empty()) check(plan[0].expert == 2, "P0 puts the hottest expert first (expert 2, twice)");
        for (const auto& r : plan) check(r.level == 0, "P0 requests are level 0");
    }

    // P0 is empty once decoding has rolled past the prefill phase
    {
        ExpertPrefetcher p;
        p.on_prefill_end({{1, 1}, {1, 0}});
        check(!p.plan_p0().empty(), "P0 plans right after prefill");
        p.on_layer_routed({{0}, {0}});
        check(p.plan_p0().empty(), "P0 stops after the phase leaves AfterPrefill");
    }

    // The degradation machine winds the window down: ShrinkPrefetch scales it thyme, NoPrefetch drops the plan whole
    {
        ExpertPrefetcher p;
        p.on_prefill_end({{1, 1, 1, 2, 3}});
        check(p.window() == 32, "full window before the machine is wired");
        DegradationStateMachine d;
        p.set_degradation(&d);
        d.apply_level(DegradationLevel::ShrinkPrefetch);
        check(p.window() == 16, "ShrinkPrefetch halves the window");
        const auto shrunk = p.plan_p0();
        check(!shrunk.empty() && shrunk.size() <= 16, "the shrunk plan honours the half window");
        d.apply_level(DegradationLevel::NoPrefetch);
        check(p.plan_p0().empty(), "NoPrefetch plans nothing");
    }

    // Lookahead (P1) is off by default and has no planner in this CPU build
    {
        PrefetchConfig cfg;
        cfg.lookahead_enabled = true;
        ExpertPrefetcher p(cfg);
        p.on_prefill_end({{1, 0}, {0, 1}});
        check(p.plan_p1(0).empty(), "P1 has no planner in this build");
    }

    // Ngram (P2) passes only high confidence candidates, and only with the machine's ngram level on
    {
        ExpertPrefetcher p;
        DegradationStateMachine d;
        p.set_degradation(&d);
        std::vector<std::pair<int, double>> cands = {{7, 0.95}, {8, 0.2}};
        const auto hi = p.plan_p2(cands);
        check(hi.size() == 1 && hi[0].expert == 7, "P2 keeps the high confidence expert");
        d.apply_level(DegradationLevel::HighConfOnly);
        check(p.plan_p2(cands).size() == 1, "P2 at HighConf keeps 0.95 above 0.8");
        d.apply_level(DegradationLevel::NoNgram);
        check(p.plan_p2(cands).empty(), "P2 with NoNgram plans nothing");
    }

    // frequency() is the raw normalised count of the current window
    {
        ExpertPrefetcher p;
        p.on_prefill_end({{2, 2}});
        check(p.frequency(0, 1) > 0.49, "frequency of the two-count expert is near 0.5");
        check(p.frequency(0, 1) > 0.49 && p.frequency(0, 1) < 0.51, "frequency is half of a two-count window");
        check(p.frequency(0, 99) == 0.0, "a never-routed expert has frequency 0");
    }

    if (g_fail) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    std::puts("expert_prefetcher_test: ok");
    return 0;
}
