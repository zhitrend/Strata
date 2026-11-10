// include/strata/core/expert_prefetcher.hpp - predictive expert prefetch, three levels gated by the degradation
// state machine (arch v2.0 3.2.4).
//
// **WHY PREFETCH AT PREFILL.**  Prefill is the only phase that may spend compute to hide I/O (principle R1).  The
// experts a decode step routes to are strongly suggested by the experts prefill routed to, so P0 - "after prefill, at
// the first decode token, queue the top-K of the prefill route-frequency window" - is the cheapest win and needs no
// new model.  P1 (layer lookahead) and P2 (ngram/MTP) are progressively more speculative and are off by default.
//
// **EVERY LEVEL FOLLOWS THE DEGRADATION MACHINE.**  The machine yields: L0 runs P0 and P1
// if there is I/O headroom, L1 halves the per-cycle prefetch window, L2 keeps P0 at high confidence only, L3+ stops
// planning and returns 0.  The prefetcher never issues more than its window; the arbiter's staging sub-budget is the
// hard byte ceiling the caller enforces on the returned plan.
//
// **HOLD ONLY KEYS, NOT THE I/O.**  This plans "prefetch these (layer, expert)"; the caller (src/split) turns a plan into
// mmap/copy on a real stream.  Keeping the planner free of device types is what lets it be unit-tested without a GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "strata/core/degradation.hpp"

namespace strata::core {

struct PrefetchConfig {
    int top_k = 32;                 // P0: experts queued per prefill heuristic
    int history_windows = 8;        // route-frequency sliding window size (windows)
    double p0_confidence = 0.0;     // P0 needs none
    double high_confidence = 0.8;   // L2's "high confidence only"
    bool lookahead_enabled = false; // P1 (STRATA_PREFETCH_LOOKAHEAD); off by default
};

enum class PrefetchPhase {
    Idle,
    AfterPrefill,  // on_prefill_end ran; the next step may issue P0
    Decode
};

class ExpertPrefetcher {
public:
    struct Request {
        int layer = -1;
        int expert = -1;
        double confidence = 0.0;  // 0..1
        int level = 0;            // which prefetch level asked (0 = P0, 1 = P1, 2 = P2)
    };

    explicit ExpertPrefetcher(const PrefetchConfig& cfg = {}, std::string* err = nullptr);

    /// Wire the degradation machine; the prefetcher reads it on each issue (set_window is the explicit override).
    void set_degradation(const DegradationStateMachine* deg) { deg_ = deg; }

    // --- feeding the routing history (P0's only input) ---
    /// Prefill finished with this per-layer route-frequency table (layer -> expert -> count).
    void on_prefill_end(const std::vector<std::vector<double>>& prefill_counts);
    /// One decode step routed these experts per layer (layer -> list of expert ids).  Rolls the window.
    void on_layer_routed(const std::vector<std::vector<int>>& routed);

    // --- planning ---
    /// P0 candidates from the most recent route-frequency window when phase == AfterPrefill.  Empty otherwise.
    std::vector<Request> plan_p0() const;
    /// P1 (layer lookahead): experts adjacent layers routed alongside `layer`'s set; empty unless lookahead_enabled.
    std::vector<Request> plan_p1(int layer) const;
    /// P2 (ngram/MTP indexed list): the caller passes candidate ids + confidence; only those above the current
    /// threshold and only at L0 are returned.
    std::vector<Request> plan_p2(const std::vector<std::pair<int, double>>& candidates) const;

    /// The plan this step actually issues: P0 (and P1/P2 if enabled), filtered by the degradation machine's window
    /// and confidence; empty when the machine has prefetch off (L3+).  Called once per step by the engine.
    std::vector<Request> issue_prefetch() const;

    void set_window(int tokens) { window_override_ = tokens; }
    void set_confidence_threshold(double t) { confidence_override_ = t; }

    PrefetchPhase phase() const { return phase_; }
    int window() const;

    /// A frequency number in [0,1] for `(layer, expert)` from the current window; 0 when never routed.
    double frequency(int layer, int expert) const;

private:
    struct Stats { double total = 0.0; std::unordered_map<int, double> per_expert; };

    bool prefetch_enabled() const;           // false at L3+ (machine) - nothing is planned
    double confidence_threshold() const;    // machine's high-confidence bar, or the config's when unattached
    std::vector<Request> rank_window() const;   // P0: rank the current window's experts by frequency


    PrefetchConfig cfg_;
    const DegradationStateMachine* deg_ = nullptr;
    std::deque<double> window_time_;              // sliding window lengths, back = newest
    std::deque<std::vector<Stats>> layers_;       // per-window, per-layer route counts
    PrefetchPhase phase_ = PrefetchPhase::Idle;
    int window_override_ = -1;
    double confidence_override_ = -1.0;
};
}  // namespace strata::core
