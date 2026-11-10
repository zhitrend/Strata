// src/core/degradation.hpp - five-level degradation state machine for memory pressure.
#pragma once

#include <cstdint>
#include <atomic>

namespace strata::core {

enum class DegradationLevel : uint8_t {
    Normal = 0,           // full functionality
    ShrinkPrefetch = 1,   // prefetch window 50%
    HighConfOnly = 2,     // only confidence > 0.8
    NoNgram = 3,          // disable ngram/MTP prefetch
    NoPrefetch = 4,       // disable all prefetch
    FreezeLowPrio = 5     // freeze low-priority sessions
};

struct DegradationConfig {
    double shrink_prefetch_window_scale = 0.5;
    double high_conf_threshold = 0.8;
    int freeze_priority_threshold = 50;  // sessions with priority < this are frozen
};

class DegradationStateMachine {
public:
    explicit DegradationStateMachine(const DegradationConfig& cfg = {});

    void apply_level(DegradationLevel level);  // atomic apply
    DegradationLevel current() const { return level_.load(std::memory_order_acquire); }

    // Runtime queries for Prefetcher/ResidencyManager/SessionManager
    bool prefetch_enabled() const;
    double prefetch_confidence_threshold() const;
    double prefetch_window_scale() const;
    bool ngram_prefetch_enabled() const;
    bool freeze_low_priority_sessions() const;
    int freeze_priority_threshold() const;

private:
    std::atomic<DegradationLevel> level_{DegradationLevel::Normal};
    DegradationConfig cfg_;
};

}  // namespace strata::core