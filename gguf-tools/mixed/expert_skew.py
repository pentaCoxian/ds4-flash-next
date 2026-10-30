#!/usr/bin/env python3
"""How concentrated is expert routing?  Reads ds4 expert hotlists
(DS4_EXPERT_HOTLIST=file, one per workload) and prints per layer the share of
selections, by count and by routing weight, that the k most used experts
cover, the normalized routing entropy, and how much the top-128 sets of the
workloads agree (Jaccard) and how their counts rank alike (Spearman).

  expert_skew.py NAME=hotlist.txt [NAME=hotlist.txt ...]

A per-expert precision split (hot experts at a higher quant) is only worth
building when few experts carry most of the routing and the same experts do
so on every workload: the summary prints that gate.
"""

from __future__ import annotations

import math
import statistics
import sys

KS = (64, 128, 256)
GATE_COVER = 0.55     # median weighted share of the top 128 experts
GATE_JACCARD = 0.6    # lowest cross-workload Jaccard of the top-128 sets


def read_hotlist(path: str) -> dict[int, list[tuple[int, float]]]:
    """layer -> [(hits, weight)] indexed by expert."""
    n_expert = None
    rows: list[tuple[int, int, int, float]] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("# experts "):
                n_expert = int(line.split()[2])
            if line.startswith("#") or not line.strip():
                continue
            layer, expert, hits, weight = line.split()
            rows.append((int(layer), int(expert), int(hits), float(weight)))
    if n_expert is None:
        raise ValueError(f"{path}: not a ds4 expert hotlist")
    layers: dict[int, list[tuple[int, float]]] = {}
    for layer, expert, hits, weight in rows:
        layers.setdefault(layer, [(0, 0.0)] * n_expert)[expert] = (hits, weight)
    return {il: v for il, v in layers.items() if sum(h for h, _ in v) > 0}


def cover(values: list[float], k: int) -> float:
    total = sum(values)
    return sum(sorted(values, reverse=True)[:k]) / total if total else 0.0


def entropy(values: list[float]) -> float:
    total = sum(values)
    if not total:
        return 0.0
    h = -sum(v / total * math.log(v / total) for v in values if v > 0)
    return h / math.log(len(values))


def top(values: list[float], k: int) -> set[int]:
    return set(sorted(range(len(values)), key=lambda e: -values[e])[:k])


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda e: values[e])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def main(argv: list[str]) -> int:
    if not argv or any("=" not in a for a in argv):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    runs = {}
    for arg in argv:
        name, path = arg.split("=", 1)
        runs[name] = read_hotlist(path)
    names = list(runs)
    layers = sorted(set.intersection(*(set(r) for r in runs.values())))
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    head = "layer " + " ".join(f"{n}:c{k}w" for n in names for k in KS) + " entropy"
    if pairs:
        head += " " + " ".join(f"J128:{a}/{b} rho:{a}/{b}" for a, b in pairs)
    print(head)
    weighted128, jaccards = [], []
    for il in layers:
        cols = []
        ent = []
        for n in names:
            w = [weight for _hits, weight in runs[n][il]]
            cols += [f"{cover(w, k):.3f}" for k in KS]
            weighted128.append(cover(w, 128))
            ent.append(entropy(w))
        cols.append(f"{statistics.fmean(ent):.3f}")
        for a, b in pairs:
            ca = [float(h) for h, _w in runs[a][il]]
            cb = [float(h) for h, _w in runs[b][il]]
            ta, tb = top(ca, 128), top(cb, 128)
            j = len(ta & tb) / len(ta | tb)
            jaccards.append(j)
            cols += [f"{j:.3f}", f"{spearman(ca, cb):.3f}"]
        print(f"{il:5d} " + " ".join(cols))

    med = statistics.median(weighted128)
    print(f"\nmedian weighted share of the top 128 experts: {med:.3f} (gate {GATE_COVER})")
    if jaccards:
        print(f"lowest cross-workload Jaccard of the top-128 sets: {min(jaccards):.3f} (gate {GATE_JACCARD})")
    uniform = 128 / len(next(iter(runs.values()))[layers[0]]) if layers else 0
    print(f"uniform routing would give {uniform:.3f}")
    go = med >= GATE_COVER and (not jaccards or min(jaccards) >= GATE_JACCARD)
    print("per-expert precision split:", "worth building" if go else "not worth building")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
