// ClusterMmapReader on a plain CPU: reads a STRCM manifest it writes to a temp file, maps (layer, expert) to its
// cluster and offset, reports cluster bytes, refuses a foreign-version file (has_manifest stays false), and answers the
// latency model (an unknown cluster returns the default).  No GPU, no real weights file.
#include "strata/core/cluster_mmap.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <vector>

using strata::core::ClusterMmapReader;
using strata::core::ClusterLocation;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}
void w32(std::FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
void w64(std::FILE* f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
void w16(std::FILE* f, uint16_t v) { std::fwrite(&v, 2, 1, f); }
void wi32(std::FILE* f, int32_t v) { std::fwrite(&v, 4, 1, f); }
}  // namespace

int main() {
    char path[] = "/tmp/strcmXXXXXX";
    int fd = mkstemp(path);
    check(fd >= 0, "mkstemp");
    if (fd < 0) return 1;

    std::FILE* f = fdopen(fd, "wb");
    check(f != nullptr, "fdopen");

    // header: magic, version 1, n_layers 2, n_expert 3, n_clusters 2
    std::fwrite("STRCM", 1, 5, f);
    w32(f, 1);
    w32(f, 2);
    w32(f, 3);
    w32(f, 2);

    // cluster 0: id 0, offset 0, len 4096, members (0,0) (0,1)
    wi32(f, 0); w64(f, 0); w64(f, 4096); w32(f, 2);
    w16(f, 0); w16(f, 0);
    w16(f, 0); w16(f, 1);

    // cluster 1: id 1, offset 4096, len 2048, members (1,2)
    wi32(f, 1); w64(f, 4096); w64(f, 2048); w32(f, 1);
    w16(f, 1); w16(f, 2);

    // anchors: (0,0)
    w32(f, 1);
    w16(f, 0); w16(f, 0);

    std::fclose(f);

    ClusterMmapReader r;
    std::string err;
    check(r.open(path, err), "opens a STRCM manifest");
    check(r.has_manifest(), "has_manifest true after open");
    check(r.cluster_count() == 2, "two clusters");

    check(r.cluster_bytes(0) == 4096, "cluster 0 has 4096 bytes");
    check(r.cluster_bytes(1) == 2048, "cluster 1 has 2048 bytes");

    ClusterLocation a = r.locate(0, 1);
    check(a.cluster == 0 && a.offset == 0, " (0,1) lives in cluster 0");
    ClusterLocation b = r.locate(1, 2);
    check(b.cluster == 1 && b.offset == 4096, "(1,2) lives in cluster 1 at offset 4096");
    ClusterLocation n = r.locate(9, 9);
    check(n.cluster == -1, "an unknown expert locates nowhere");

    check(r.global_anchors().size() == 1, "one global anchor");

    r.set_latency_model(1000.0, 0.5, 7777.0);
    check(r.latency_ns(0) == 1000.0 + 0.5 * 4096, "cluster 0 latency");
    check(r.latency_ns(5) == 7777.0, "unknown cluster returns the default");

    // a foreign version must be refused, and has_manifest stays false on a fresh reader
    {
        char bad[] = "/tmp/strcmXXXXXX";
        int bfd = mkstemp(bad);
        std::FILE* bf = fdopen(bfd, "wb");
        std::fwrite("STRCM", 1, 5, bf); w32(bf, 99); for (int i = 0; i < 3; ++i) w32(bf, 0);
        std::fclose(bf);
        ClusterMmapReader badr;
        std::string be;
        check(!badr.open(bad, be), "a version-99 manifest is refused");
        check(!badr.has_manifest(), "refused manifest leaves has_manifest false");
        check(!be.empty(), "refusal says why");
        std::remove(bad);
    }

    std::remove(path);
    if (g_fail == 0) std::puts("cluster_mmap_test: ok");
    return g_fail == 0 ? 0 : 1;
}
