// zfs_arc_reclaimable_bytes_from(): the /proc/spl/kstat/zfs/arcstats parser on fixture files.  No real ZFS, no GPU.
// A data line is "size 4 <value>"; the header ("name type data" / "size 4 hash") must not match.  The public reader
// honors FNX_NO_ZFS_ARC=1; the path reader only parses.
#include "strata/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using strata::platform::zfs_arc_reclaimable_bytes;
using strata::platform::zfs_arc_reclaimable_bytes_from;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

// a fixture file under a temp directory that the process owns; removed when done
struct Fixture {
    fs::path dir;
    fs::path file;
    explicit Fixture(const std::string& body) {
        dir = fs::temp_directory_path() / ("zfs_arc_test_" + std::to_string(::getpid()));
        fs::create_directories(dir);
        file = dir / "arcstats";
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out << body;
    }
    ~Fixture() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

}  // namespace

int main() {
    // 1. a real-shaped arcstats: two header lines, then data lines.  Only `size` matters; 80% is returned.
    {
        Fixture f(
            "13 1 0x01 88 1 0 0 0\n"
            "name                            type data\n"
            "hits                            4    12345678\n"
            "size                            4    10737418240\n"   // 10 GiB
            "misses                          4    42\n");
        const size_t want = (size_t) (10737418240ull * 0.8);
        check(zfs_arc_reclaimable_bytes_from(f.file.string().c_str()) == want, "10 GiB -> 8 GiB reclaimable");
    }

    // 2. `size` before other lines still wins; a trailing header does not confuse it
    {
        Fixture f(
            "size 4 5368709120\n"    // 5 GiB
            "name type data\n"
            "size 4 hash\n");
        check(zfs_arc_reclaimable_bytes_from(f.file.string().c_str()) == (size_t) (5368709120ull * 0.8),
              "5 GiB found on the first data line");
    }

    // 3. no size line at all: 0
    {
        Fixture f("name type data\nhits 4 7\nmisses 4 9\n");
        check(zfs_arc_reclaimable_bytes_from(f.file.string().c_str()) == 0, "no size line -> 0");
    }

    // 4. size 0: 0 (a positive size is required)
    {
        Fixture f("name type data\nsize 4 0\n");
        check(zfs_arc_reclaimable_bytes_from(f.file.string().c_str()) == 0, "size 0 -> 0");
    }

    // 5. missing file, null path: 0, no crash
    {
        check(zfs_arc_reclaimable_bytes_from("/nonexistent/zfs/arcstats") == 0, "missing file -> 0");
        check(zfs_arc_reclaimable_bytes_from(nullptr) == 0, "null path -> 0");
    }

    // 6. FNX_NO_ZFS_ARC=1 turns the reader off (0) and off-again the flag it is the reader's own that matters)
    {
        const char* name = "FNX_NO_ZFS_ARC";
#if defined(_WIN32)
        _putenv_s(name, "1");
#else
        setenv(name, "1", 1);
#endif
        check(zfs_arc_reclaimable_bytes() == 0, "FNX_NO_ZFS_ARC=1 -> 0");
#if defined(_WIN32)
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
        // with the flag cleared the reader reads the real /proc file (0 off Linux where it is absent)
        (void) zfs_arc_reclaimable_bytes();
    }

    if (g_fail) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    std::puts("zfs_arc_test: ok");
    return 0;
}