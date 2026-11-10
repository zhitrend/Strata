// include/strata/core/cluster_mmap.hpp - cluster-clustered expert layout and its read latency model (arch v2.0 3.3.1).
//
// **WHY A CLUSTER LAYOUT.**  Experts that fire together are read together.  Louvain clustering of the calibration
// co-activation matrix (tools/cluster_experts.py) groups them, and the runtime mmaps a whole cluster sequentially
// (MADV_SEQUENTIAL) instead of seeking across the file per expert.  The reader also exposes the per-cluster cost that
// ResidencyManager's Greedy-Dual scorer charges a miss - so the eviction decision is driven by a measured I/O model,
// not a constant.
//
// **NO JSON PARSER IN THE CORE THE TREE.**  The core and its CPU tests carry no third-party dependency, so the manifest
// this reads is a compact binary `STRCM` (written by tools/cluster_experts.py next to the human JSON it also emits).  The
// binary mirrors read_expert_profile's shape: magic, version, uint32 header, raw arrays.  A missing or foreign-version
// manifest leaves has_manifest() false, and the caller uses the original per-expert layout (R2-12).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// One cluster: the experts it holds, contiguous in the rearranged weights file.
struct ExpertCluster {
    int32_t id = -1;
    int64_t byte_offset = 0;   // where the cluster starts in the rearranged weights file
    int64_t byte_length = 0;   // its bytes (the sum of its experts' blobs)
    std::vector<std::pair<int32_t, int32_t>> experts;  // (layer, expert)
};

/// Where an expert reads from: which cluster and where inside it.
struct ClusterLocation {
    int32_t cluster = -1;
    int64_t offset = 0;   // from the file start; -1 when unknown
};

class ClusterMmapReader {
public:
    ClusterMmapReader() = default;

    /// Reads `path` (a STRCM manifest).  False with err on a missing/foreign file; has_manifest() stays false and the
    /// caller falls back to the original layout.
    bool open(const std::string& path, std::string& err);

    /// True once a version-1 STRCM manifest was read.
    bool has_manifest() const { return !clusters_.empty(); }
    int cluster_count() const { return (int) clusters_.size(); }

    const ExpertCluster& cluster(int id) const;

    /// (layer, expert) -> the cluster it lives in and its offset from the file start, or cluster -1.
    ClusterLocation locate(int layer, int expert) const;

    /// Estimated bytes read to bring `cluster_id` in: the cluster's length.
    int64_t cluster_bytes(int cluster_id) const;

    /// **THE MISS-COST MODEL Greedy-Dual CHARGES (R2-2, 3.4).**  Per-cluster: a fixed read latency plus the bytes over a
    /// bandwidth.  In nanoseconds.  This is a MODEL - it is sized from the caller's calibration (set_latency_model),
    /// not measured here - and Greedy-Dual only needs its ORDER to be right.  An unknown cluster returns `default_cost`.
    void set_latency_model(double fixed_ns, double ns_per_byte, double default_cost_ns);
    double latency_ns(int cluster_id) const;

    /// The anchors the offline tool wrote into this manifest (the global top-frequency set), (layer, expert) pairs.
    const std::vector<std::pair<int32_t, int32_t>>& global_anchors() const { return anchors_; }

private:
    std::vector<ExpertCluster> clusters_;
    std::vector<std::pair<int32_t, int32_t>> anchors_;
    std::vector<ClusterLocation> by_layer_expert_;  // n_layers * n_expert_ dense lookup, cluster_-1 / offset-1 where none
    int n_layers_ = 0, n_expert_ = 0;
    double fixed_ns_ = 0.0, ns_per_byte_ = 0.0, default_cost_ns_ = 0.0;
};
}  // namespace strata::core
