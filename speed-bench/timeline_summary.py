#!/usr/bin/env python3
"""Summarize a DS4_METAL_ENCODER_TIMELINE file: GPU time per kernel.

  timeline_summary.py TIMELINE [--last N] [--top K]

--last N keeps the last N command buffers (for example the buffers of the
final decode tokens, after warm-up and prefill).  Prints per kernel the
number of passes, the summed GPU microseconds, its share of the busy time,
and the summed gaps before its passes, then the busy and wall time of the
selected buffers.  The timeline stamps every pass, so its gaps and idle time
overstate the host's share; time host work with DS4_QWEN4_TIMING=1 instead.
"""

from __future__ import annotations

import argparse
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("timeline")
    ap.add_argument("--last", type=int, default=0, help="only the last N command buffers")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    buffers: list[tuple[int, float, float]] = []
    passes: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    with open(args.timeline, encoding="utf-8", errors="replace") as fp:
        for line in fp:
            f = line.split()
            if not f or f[0] == "#":
                continue
            if f[0] == "B" and len(f) >= 5:
                buffers.append((int(f[1]), float(f[3]), float(f[4])))
            elif f[0] == "E" and len(f) >= 12:
                passes[int(f[1])].append((float(f[5]), float(f[6]), " ".join(f[11:])))
    if args.last:
        buffers = buffers[-args.last:]
    keep = {seq for seq, _s, _e in buffers}

    dur = defaultdict(float)
    gap = defaultdict(float)
    count = defaultdict(int)
    busy = 0.0
    for seq in keep:
        for d, g, k in passes.get(seq, ()):
            dur[k] += d
            gap[k] += g
            count[k] += 1
            busy += d
    wall = sum((e - s) / 1e3 for _seq, s, e in buffers)

    print(f"{'kernel':60s} {'passes':>7s} {'gpu_us':>10s} {'share':>6s} {'gaps_us':>9s}")
    for k in sorted(dur, key=lambda k: -dur[k])[: args.top]:
        print(f"{k[:60]:60s} {count[k]:7d} {dur[k]:10.1f} {100 * dur[k] / busy if busy else 0:5.1f}% {gap[k]:9.1f}")
    print(f"\n{len(buffers)} command buffers: kernels busy {busy / 1e3:.2f} ms,"
          f" buffers span {wall / 1e3:.2f} ms, gaps {sum(gap.values()) / 1e3:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
