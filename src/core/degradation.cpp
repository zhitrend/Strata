// src/core/degradation.cpp
#include "strata/core/degradation.hpp"

namespace strata::core {

DegradationStateMachine::DegradationStateMachine(const DegradationConfig& cfg) : cfg_(cfg) {}

void DegradationStateMachine::apply_level(DegradationLevel level) {
    level_.store(level, std::memory_order_release);
}

bool DegradationStateMachine::prefetch_enabled() const {
    auto lvl = current();
    return lvl != DegradationLevel::NoPrefetch && lvl != DegradationLevel::FreezeLowPrio;
}

double DegradationStateMachine::prefetch_confidence_threshold() const {
    auto lvl = current();
    switch (lvl) {
        case DegradationLevel::Normal: return 0.3;
        case DegradationLevel::ShrinkPrefetch: return 0.4;
        case DegradationLevel::HighConfOnly: return cfg_.high_conf_threshold;
        case DegradationLevel::NoNgram: return cfg_.high_conf_threshold;
        case DegradationLevel::NoPrefetch: return 1.0;
        case DegradationLevel::FreezeLowPrio: return 1.0;
        default: return 0.3;
    }
}

double DegradationStateMachine::prefetch_window_scale() const {
    auto lvl = current();
    switch (lvl) {
        case DegradationLevel::ShrinkPrefetch: return cfg_.shrink_prefetch_window_scale;
        default: return 1.0;
    }
}

bool DegradationStateMachine::ngram_prefetch_enabled() const {
    auto lvl = current();
    return lvl != DegradationLevel::NoNgram &&
           lvl != DegradationLevel::NoPrefetch &&
           lvl != DegradationLevel::FreezeLowPrio;
}

bool DegradationStateMachine::freeze_low_priority_sessions() const {
    auto lvl = current();
    return lvl == DegradationLevel::FreezeLowPrio;
}

int DegradationStateMachine::freeze_priority_threshold() const {
    return cfg_.freeze_priority_threshold;
}

}  // namespace strata::core