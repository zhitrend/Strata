// src/core/cluster_mmap.cpp
#include "strata/core/cluster_mmap.hpp"

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {

// STRCM (little-endian), written by tools/cluster_experts.py:
//   char[5] magic = "STRCM"
//   u32 version (= 1)
//   u32 n_layers, n_expert, n_clusters, n_anchors
//   n_clusters x { i32 id, u64 byte_offset, u64 byte_length, u32 n_members }
//       then n_members x { u16 layer, u16 expert }
//   n_anchors x { u16 layer, u16 expert }
bool take(std::FILE* f, void* p, size_t n) { return std::fread(p, 1, n, f) == n; }

}  // namespace

bool ClusterMmapReader::open(const std::string& path, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "ClusterMmapReader: cannot open " + path; return false; }

    char magic[5] = {0};
    uint32_t hdr[4] = {0, 0, 0, 0};

    const bool head_ok = take(f, magic, 5) && std::memcmp(magic, "STRCM", 5) == 0 && take(f, hdr, 16);
    if (!head_ok) {
        std::fclose(f);
        err = "ClusterMmapReader: " + path + " is not a STRCM manifest";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], nc = hdr[3];
    if (version != 1) {
        std::fclose(f);
        err = "ClusterMmapReader: " + path + " is version " + std::to_string(version) + ", this reads 1";
        return false;
    }

    std::vector<ExpertCluster> clusters(nc);
    for (uint32_t i = 0; i < nc; ++i) {
        int32_t id = 0;
        uint64_t off = 0, len = 0;
        uint32_t nm = 0;
        if (!take(f, &id, 4) || !take(f, &off, 8) || !take(f, &len, 8) || !take(f, &nm, 4)) {
            std::fclose(f);
            err = "ClusterMmapReader: truncated cluster table";
            return false;
        }
        clusters[i].id = id;
        clusters[i].byte_offset = (int64_t) off;
        clusters[i].byte_length = (int64_t) len;
        clusters[i].experts.reserve(nm);
        for (uint32_t m = 0; m < nm; ++m) {
            uint16_t l = 0, e = 0;
            if (!take(f, &l, 2) || !take(f, &e, 2)) {
                std::fclose(f);
                err = "ClusterMmapReader: truncated member list";
                return false;
            }
            clusters[i].experts.emplace_back(l, e);
        }
    }

    uint32_t na = 0;
    // n_anchors follows the cluster records (the header's 4th u32 is n_clusters)
    if (!take(f, &na, 4)) { std::fclose(f); err = "ClusterMmapReader: truncated anchor count"; return false; }
    std::vector<std::pair<int32_t, int32_t>> anchors;
    anchors.reserve(na);
    for (uint32_t i = 0; i < na; ++i) {
        uint16_t l = 0, e = 0;
        if (!take(f, &l, 2) || !take(f, &e, 2)) { std::fclose(f); err = "ClusterMmapReader: truncated anchor list"; return false; }
        anchors.emplace_back(l, e);
    }
    std::fclose(f);

    clusters_ = std::move(clusters);
    anchors_ = std::move(anchors);
    n_layers_ = (int) nl;
    n_expert_ = (int) ne;

    const size_t cells = (size_t) (n_layers_ < 0 ? 0 : n_layers_) * (size_t) (n_expert_ < 0 ? 0 : n_expert_);
    by_layer_expert_.assign(cells, ClusterLocation{});
    for (int ci = 0; ci < (int) clusters_.size(); ++ci) {
        for (const auto& [l, x] : clusters_[ci].experts) {
            if (l < 0 || l >= n_layers_ || x < 0 || x >= n_expert_) continue;
            by_layer_expert_[(size_t) l * n_expert_ + x] =
                ClusterLocation{clusters_[ci].id, clusters_[ci].byte_offset};
        }
    }
    return true;
}

const ExpertCluster& ClusterMmapReader::cluster(int id) const { return clusters_[(size_t) id]; }

ClusterLocation ClusterMmapReader::locate(int layer, int expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return ClusterLocation{};
    return by_layer_expert_[(size_t) layer * n_expert_ + expert];
}

int64_t ClusterMmapReader::cluster_bytes(int cluster_id) const {
    if (cluster_id < 0 || cluster_id >= (int) clusters_.size()) return 0;
    return clusters_[cluster_id].byte_length;
}

void ClusterMmapReader::set_latency_model(double fixed_ns, double ns_per_byte, double default_cost_ns) {
    fixed_ns_ = fixed_ns;
    ns_per_byte_ = ns_per_byte;
    default_cost_ns_ = default_cost_ns;
}

double ClusterMmapReader::latency_ns(int cluster_id) const {
    if (cluster_id < 0 || cluster_id >= (int) clusters_.size()) return default_cost_ns_;
    return fixed_ns_ + ns_per_byte_ * (double) clusters_[cluster_id].byte_length;
}

}  // namespace strata::core
