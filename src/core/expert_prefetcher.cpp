#include "strata/core/expert_prefetcher.hpp"

#include <algorithm>

namespace strata::core {

namespace {
void set_err(std::string* err, const char* msg) {
    if (err != nullptr) *err = msg;
}
}  // namespace

ExpertPrefetcher::ExpertPrefetcher(const PrefetchConfig& cfg, std::string* err) : cfg_(cfg) {
    if (cfg_.top_k <= 0) {
        cfg_.top_k = 1;
        set_err(err, "expert_prefetcher: top_k <= 0, clamped to 1");
    }
    if (cfg_.history_windows < 1) cfg_.history_windows = 1;
    if (cfg_.high_confidence < 0.0) cfg_.high_confidence = 0.0;
    if (cfg_.high_confidence > 1.0) cfg_.high_confidence = 1.0;
}

void ExpertPrefetcher::on_prefill_end(const std::vector<std::vector<double>>& prefill_counts) {
    // A fresh route-frequency window from the prefill: per layer, per expert raw counts, kept for frequency().
    layers_.clear();
    window_time_.clear();
    std::vector<Stats> win(prefill_counts.size());
    for (size_t l = 0; l < prefill_counts.size(); ++l) {
        for (size_t e = 0; e < prefill_counts[l].size(); ++e) {
            const double c = prefill_counts[l][e];
            if (c <= 0.0) continue;
            win[l].total += c;
            win[l].per_expert[(int) e] += c;
        }
    }
    layers_.push_back(std::move(win));
    window_time_.push_back(1.0);
    phase_ = PrefetchPhase::AfterPrefill;
}

void ExpertPrefetcher::on_layer_routed(const std::vector<std::vector<int>>& routed) {
    std::vector<Stats> win(routed.size());
    for (size_t l = 0; l < routed.size(); ++l) {
        for (int e : routed[l]) {
            if (e < 0) continue;
            win[l].total += 1.0;
            win[l].per_expert[e] += 1.0;
        }
    }
    layers_.push_back(std::move(win));
    window_time_.push_back(1.0);
    while ((int) layers_.size() > cfg_.history_windows) {
        layers_.pop_front();
        window_time_.pop_front();
    }
    phase_ = PrefetchPhase::Decode;
}

double ExpertPrefetcher::frequency(int layer, int expert) const {
    if (layer < 0 || expert < 0) return 0.0;
    double hit = 0.0, tot = 0.0;
    for (const auto& win : layers_) {
        const size_t li = (size_t) layer;
        if (li >= win.size()) continue;
        const auto it = win[li].per_expert.find(expert);
        if (it != win[li].per_expert.end()) hit += it->second;
        tot += win[li].total;
    }
    return tot > 0.0 ? hit / tot : 0.0;
}

bool ExpertPrefetcher::prefetch_enabled() const {
    if (deg_ != nullptr) return deg_->prefetch_enabled();
    return true;
}

double ExpertPrefetcher::confidence_threshold() const {
    if (confidence_override_ >= 0.0) return confidence_override_;
    if (deg_ != nullptr) return deg_->prefetch_confidence_threshold();
    return cfg_.p0_confidence;
}

int ExpertPrefetcher::window() const {
    if (window_override_ >= 0) return window_override_;
    const int base = cfg_.top_k;
    if (deg_ != nullptr) {
        const double scale = deg_->prefetch_window_scale();
        int w = (int) (scale * (double) base);
        if (w < 1) w = 1;
        return w;
    }
    return base;
}

std::vector<ExpertPrefetcher::Request> ExpertPrefetcher::rank_window() const {
    // Rank (layer, expert) over the most recent window by frequency, keep the top `window()`.
    std::vector<Request> out;
    if (layers_.empty() || !prefetch_enabled()) return out;
    const auto& win = layers_.back();
    const int keep = window();
    double window_total = 0.0;
    for (const auto& st : win) window_total += st.total;
    if (window_total <= 0.0) return out;
    for (size_t l = 0; l < win.size(); ++l) {
        for (const auto& kv : win[l].per_expert) {
            out.push_back(Request{(int) l, kv.first, kv.second / window_total, 0});
        }
    }
    std::sort(out.begin(), out.end(), [](const Request& a, const Request& b) {
        if (a.confidence != b.confidence) return a.confidence > b.confidence;
        return a.expert < b.expert;
    });
    if ((int) out.size() > keep) out.resize((size_t) keep);
    return out;
}

std::vector<ExpertPrefetcher::Request> ExpertPrefetcher::plan_p0() const {
    if (phase_ != PrefetchPhase::AfterPrefill) return {};
    return rank_window();
}

std::vector<ExpertPrefetcher::Request> ExpertPrefetcher::plan_p1(int layer) const {
    (void) layer;  // the lookahead planner is a CPU-build stub; the key is consumed here
    return {};
    return {};  // P1 (layer lookahead) is a planner stub in this CPU build
}

std::vector<ExpertPrefetcher::Request> ExpertPrefetcher::plan_p2(
    const std::vector<std::pair<int, double>>& candidates) const {
    std::vector<Request> out;
    if (!prefetch_enabled() || !deg_ || !deg_->ngram_prefetch_enabled()) return out;
    const double bar = confidence_threshold();
    for (const auto& c : candidates) {
        if (c.first < 0 || c.second < bar) continue;
        out.push_back(Request{-1, c.first, c.second, 2});
    }
    return out;
}

std::vector<ExpertPrefetcher::Request> ExpertPrefetcher::issue_prefetch() const {
    std::vector<Request> plan = plan_p0();
    if (!cfg_.lookahead_enabled || plan.empty()) return plan;
    const double bar = confidence_threshold();
    plan.erase(std::remove_if(plan.begin(), plan.end(),
                              [bar](const Request& r) { return r.confidence < bar; }),
               plan.end());
    return plan;
}

}  // namespace strata::core
