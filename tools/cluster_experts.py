#!/usr/bin/env python3
"""tools/cluster_experts.py - build a STRCM expert-cluster manifest from co-activation data.

Reads a co-activation matrix (CSV, or a .npz holding `counts`) and clusters the experts that fire together, then
writes two files:

    <out>.strcm   the binary manifest src/core/cluster_mmap.cpp reads (magic "STRCM", version 1)
    <out>.json    a human-readable copy of the same clusters and anchors

Layout bytes (little-endian), kept byte-for-byte in step with ClusterMmapReader::open:
    char[5]  "STRCM"
    u32      version (= 1)
    u32      n_layers, n_expert, n_clusters, n_anchors
    n_clusters x { i32 id, u64 byte_offset, u64 byte_length, u32 n_members, n_members x { u16 layer, u16 expert } }
    n_anchors x { u16 layer, u16 expert }

Clustering is community structure on the co-activation graph: Louvain when networkx is installed, connected components
otherwise.  Each cluster's byte_length is the summed expert byte size (--expert-bytes per expert) so the reader's latency
model (fixed + ns_per_byte * length) sees a real per-cluster cost.  Anchors are the members of the largest cluster, held
resident whatever the policy does.

Deterministic: members are sorted, clusters are numbered by their first sorted member, so the same input writes the same
bytes and a rebuild diffs to nothing.
"""

import argparse
import csv
import json
import struct
import sys

MAGIC = b"STRCM"
VERSION = 1
DEFAULT_EXPERT_BYTES = 6 * 1024 * 1024


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Build a STRCM expert-cluster manifest from co-activation data.")
    ap.add_argument("input", help="co-activation CSV or .npz")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.strcm and <out>.json")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--expert-bytes", type=int, default=DEFAULT_EXPERT_BYTES)
    ap.add_argument("--min-weight", type=float, default=0.0)
    return ap.parse_args(argv)


def read_csv(path):
    with open(path, newline="") as fh:
        return [row for row in csv.reader(fh) if row]


def load_coactivation(path, n_layers, n_experts, npz_key):
    n_total = n_layers * n_experts
    edges = []
    if path.endswith(".npz"):
        try:
            import numpy as np
        except ImportError:
            sys.exit("cluster_experts: .npz input needs numpy (pip install numpy)")
        data = np.load(path)[npz_key]
        flat = data.reshape(n_total, n_total)
        for a in range(n_total):
            for b in range(a + 1, n_total):
                w = float(flat[a, b])
                if w > 0.0:
                    edges.append((a, b, w))
        return edges
    rows = read_csv(path)
    if rows and rows[0][0].strip().lower() == "layer":
        for l1, e1, l2, e2, w in rows[1:]:
            n = float(w)
            if n > 0.0:
                edges.append((int(l1) * n_experts + int(e1), int(l2) * n_experts + int(e2), n))
        return edges
    if len(rows) != n_total:
        sys.exit(f"cluster_experts: matrix has {len(rows)} rows, expected {n_total}")
    for a in range(n_total):
        for b in range(a + 1, n_total):
            w = float(rows[a][b])
            if w > 0.0:
                edges.append((a, b, w))
    return edges


def cluster_nodes(n_total, edges, min_weight):
    """Partition the experts into clusters.

    Louvain when networkx is installed: it maximises modularity on the weighted co-activation graph, which is the
    heuristic the design names.  Otherwise connected components: co-firing peers land in the same cluster, singletons
    alone.  The fallback is weaker but never wrong, and the writer keeps the format identical for both.
    """
    adj = {i: {} for i in range(n_total)}
    for a, b, w in edges:
        if w < min_weight:
            continue
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w

    try:
        import networkx as nx
    except ImportError:
        nx = None

    if nx is not None:
        g = nx.Graph()
        g.add_nodes_from(range(n_total))
        for a, nb in adj.items():
            for b, w in nb.items():
                g.add_edge(a, b, weight=w)
        parts = nx.community.louvain_communities(g, weight="weight", seed=0)
        clusters = [sorted(part) for part in parts]
        clusters.sort(key=lambda c: (c[0] if c else n_total))
        if clusters:
            return clusters
    # connected-components fallback
    seen = set()
    clusters = []
    for start in range(n_total):
        if start in seen:
            continue
        stack = [start]
        comp = []
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comp.sort()
        clusters.append(comp)
    clusters.sort(key=lambda c: (c[0] if c else n_total))
    return clusters


def write_strcm(path, n_layers, n_expert, clusters, expert_bytes, anchors):
    data = bytearray()
    data += MAGIC
    data += struct.pack("<IIII", VERSION, n_layers, n_expert, len(clusters))
    offset = 0
    for cid, members in enumerate(clusters):
        length = expert_bytes * len(members)
        data += struct.pack("<iQQI", cid, offset, length, len(members))
        for flat in members:
            layer = flat // n_expert
            expert = flat % n_expert
            data += struct.pack("<HH", layer, expert)
        offset += length
    data += struct.pack("<I", len(anchors))
    for flat in anchors:
        data += struct.pack("<HH", flat // n_expert, flat % n_expert)
    with open(path, "wb") as fh:
        fh.write(data)


def write_json(path, n_layers, n_experts, clusters, expert_bytes, anchors):
    cluster_docs = []
    byte_offset = 0
    for cid, members in enumerate(clusters):
        cluster_docs.append({
            "id": cid,
            "byte_offset": byte_offset,
            "byte_length": expert_bytes * len(members),
            "members": [[flat // n_experts, flat % n_experts] for flat in members],
        })
        byte_offset += expert_bytes * len(members)
    doc = {
        "format": "STRCM",
        "version": VERSION,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "expert_bytes": expert_bytes,
        "clusters": cluster_docs,
        "anchors": [[flat // n_experts, flat % n_experts] for flat in anchors],
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def main(argv=None):
    args = parse_args(argv)
    edges = load_coactivation(args.input, args.layers, args.experts, "counts")
    clusters = cluster_nodes(args.layers * args.experts, edges, args.min_weight)
    anchors = max(clusters, key=len) if clusters else []
    anchors = sorted(anchors)
    write_strcm(args.out + ".strcm", args.layers, args.experts, clusters, args.expert_bytes, anchors)
    write_json(args.out + ".json", args.layers, args.experts, clusters, args.expert_bytes, anchors)
    print(f"wrote {len(clusters)} clusters, {len(anchors)} anchors to {args.out}.strcm and {args.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

