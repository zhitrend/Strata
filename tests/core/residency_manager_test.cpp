// ResidencyManager on a plain CPU: Greedy-Dual scoring (frequency_decayed * injected miss_cost), the Anchor Expert
// protected set never evicted, batch eviction capped to max_batch, the LRU-degrade signal, and exponential decay.
// No GPU, no device memory.
#include "strata/core/residency_manager.hpp"

#include <cstdio>
#include <map>

using namespace strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}
}  // namespace

int main() {
    ResidencyConfig cfg;
    cfg.heat_decay = 0.5;   // halve per window, easy to reason about
    cfg.max_batch = 8;
    ResidencyManager rm(cfg);

    std::map<ExpertKey, std::int64_t> bytes;
    auto bytes_of = [&](ExpertKey k) -> std::int64_t {
        return bytes.count(k) ? bytes[k] : (std::int64_t) 100;
    };

    // 1. a touched expert has a positive score; one entry tracked
    ExpertKey a = make_expert_key(0, 1);
    ExpertKey b = make_expert_key(0, 2);
    rm.touch(a);
    check(rm.score(a) > 0.0, "a touched expert has a positive score");
    check(rm.size() == 1, "one entry tracked");
    check(rm.score(b) == 0.0, "an untouched expert scores 0");

    // 2. repeated touch raises a's score; after the decay a cold b sits below a hot a
    for (int i = 0; i < 5; ++i) rm.touch(a);
    rm.touch(b);          // b touched once
    rm.advance_window(4.0);
    rm.touch(a);          // a hot again at window 4
    check(rm.score(a) > rm.score(b), "the frequently touched expert outscores the cold one");

    // 3. anchors are the protected set: a never evicted, even though it could be the cheapest
    rm.set_anchors({a});
    check(rm.is_protected(a), "anchor a is protected");
    std::int64_t freed = 0;
    bytes[a] = 64; bytes[b] = 128;
    auto gone = rm.evict(200, bytes_of, &freed);
    for (ExpertKey k : gone) check(k != a, "the anchor is never evicted");
    check(freed >= 128, "eviction frees at least the non-anchor bytes");
    check(gone.size() >= 1, "the non-anchor was chosen");
    check(gone.at(0) == b, "the coldest non-anchor (b) is chosen first");

    // 4. batch eviction capped to max_batch
    ResidencyManager rm2(cfg);
    rm2.set_miss_cost_fn(nullptr);  // default cost 1.0
    for (int i = 0; i < 20; ++i) rm2.touch(make_expert_key(1, i));
    std::int64_t f2 = 0;
    auto gone2 = rm2.evict(1 << 20, [](ExpertKey) { return (std::int64_t) 1; }, &f2);
    check((int) gone2.size() == cfg.max_batch, "batch eviction stops at max_batch");

    // 5. LRU-degrade signal
    check(rm.should_fall_back_to_lru(0.10), "10% hit rate falls back to LRU");
    check(!rm.should_fall_back_to_lru(0.50), "50% hit rate keeps Greedy-Dual");

    // 6. decay over unwatched windows lowers the score
    ResidencyManager rm3(cfg);
    ExpertKey c = make_expert_key(2, 0);
    rm3.touch(c);
    double s0 = rm3.score(c);
    rm3.advance_window(3.0);
    double s1 = rm3.score(c);
    check(s1 < s0, "two windows of decay lower the score");
    check(s1 > 0.0, "decay never reaches exactly zero");

    // 7. the top-up on first touch: nothing negative from a fresh entry
    rm3.touch(make_expert_key(9, 9));
    check(rm3.score(make_expert_key(9, 9)) > 0.0, "a fresh entry stays positive");

    if (g_fail == 0) std::puts("residency_manager_test: ok");
    return g_fail == 0 ? 0 : 1;
}
