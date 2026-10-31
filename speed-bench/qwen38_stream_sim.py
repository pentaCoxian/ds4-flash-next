#!/usr/bin/env python3
"""Replay ds4 expert traces against an SSD-streamed expert cache.

  qwen38_stream_sim.py TENSORS.tsv SSD.tsv TRACE [TRACE ...]
      [--budgets 11,16,20,26,32] [--policies lru,hot] [--model MODEL.gguf]
      [--layer-ms 0.6] [--resident-ms 34.5] [--sync-ms 3]

TENSORS.tsv is `gguf_mix.py list MODEL.gguf`: it gives every layer's expert
bytes (gate + up + down of one expert).  SSD.tsv is the output of
speed-bench/ssd_expert_read_bench on the target Mac: the "miss" rows at 9
threads give the time of one layer's k missing experts, and a "layer_8MiB"
row the sequential rate of a prefill page-in.  TRACE is a DS4_EXPERT_TRACE
file of one workload.

The cache is a byte budget shared by all layers, as the streaming cache's
slabs are.  Forwards of at most 64 rows (decode, MTP verify, a short tool
result) go through it: a layer's distinct missing experts are read before the
layer's experts run, costing t(k).  Longer prefill chunks page whole layers
in and seed the cache with their experts for free; their I/O is reported as
the full layers, and as only the experts the cache lacked with the time that
takes at the sequential rate.

Policies: "lru", and "hot", the streaming cache's: a selection count per
expert, misses included, halved every 16 decode tokens; the coldest, then the
least recently used, is evicted.

With --model and a trace written with DS4_EXPERT_TRACE_INPUT=1, the decode
rows' router inputs are scored: layer L's own router must reproduce the
traced selection (a check of the replay), and layer L+1's router applied to
layer L's input is the prefetch predictor.  Its top-P ids are read during the
layer's compute (--layer-ms), so a predicted miss costs only what overruns
that window.  --resident-ms and --sync-ms (the per-token router readbacks)
turn the I/O into tokens per second.  Needs numpy for --model only.
"""

from __future__ import annotations

import argparse
import heapq
import re
import struct
import sys
from collections import OrderedDict, defaultdict

GIB = 1024 ** 3
SMALL_FORWARD = 64


def expert_bytes(tsv: str) -> tuple[dict[int, int], dict[int, tuple[int, int, int]]]:
    """layer -> bytes of one expert; layer -> (router offset, n_in, n_expert)."""
    size: dict[int, int] = defaultdict(int)
    router: dict[int, tuple[int, int, int]] = {}
    for line in open(tsv, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) != 5 or line.startswith("#"):
            continue
        name, typ, dims, off, nbytes = f
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
        if m:
            size[int(m.group(1))] += int(nbytes) // int(dims.split(",")[-1])
        m = re.fullmatch(r"blk\.(\d+)\.ffn_gate_inp\.weight", name)
        if m and typ == "BF16":
            n_in, n_exp = (int(x) for x in dims.split(","))
            router[int(m.group(1))] = (int(off), n_in, n_exp)
    return dict(size), router


def ssd_model(tsv: str, threads: int) -> tuple[list[tuple[int, float]], float]:
    """Measured (k, ms) points of a layer's misses and the sequential GiB/s."""
    points, seq = [], 0.0
    for line in open(tsv, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) != 8 or f[4] == "-" or f[0] == "mode":
            continue
        if f[0] == "miss" and int(f[1]) == threads:
            points.append((int(f[2]), float(f[4])))
        elif f[0].startswith("layer_") and int(f[1]) == threads:
            seq = max(seq, float(f[7]))
    if not points:
        sys.exit(f"{tsv}: no miss rows at {threads} threads")
    return sorted(points), seq or 6.0


def miss_ms(points: list[tuple[int, float]], k: int) -> float:
    """Piecewise linear in k, extended with the last slope."""
    if k <= 0:
        return 0.0
    for (k0, t0), (k1, t1) in zip(points, points[1:]):
        if k <= k1:
            return t0 + (t1 - t0) * (k - k0) / (k1 - k0) if k >= k0 else t0 * k / k0
    (k0, t0), (k1, t1) = points[-2], points[-1]
    return t1 + (t1 - t0) / (k1 - k0) * (k - k1)


def read_trace(path: str, keep_input: bool):
    """[(pos0, rows, [(layer, ids[rows][n_used], input or None)])] per forward."""
    data = open(path, "rb").read()
    if data[:8] != b"DS4XTRC1":
        sys.exit(f"{path}: not a ds4 expert trace")
    n_layer, n_expert, n_used, n_embd = struct.unpack_from("<4I", data, 8)
    o, forwards, cur = 24, [], None
    while o < len(data):
        pos0, layer, rows, flag = struct.unpack_from("<IHHB", data, o)
        o += 12
        ids = struct.unpack_from(f"<{rows * n_used}h", data, o)
        o += rows * n_used * 6   # i16 ids, then f32 weights
        inp = None
        if flag & 1:
            if keep_input:
                inp = data[o:o + rows * n_embd * 2]
            o += rows * n_embd * 2
        if layer == 0 or cur is None:
            cur = (pos0, rows, [])
            forwards.append(cur)
        cur[2].append((layer, [ids[r * n_used:(r + 1) * n_used] for r in range(rows)], inp))
    return forwards, n_embd


class Cache:
    def __init__(self, budget: int, size: dict[int, int], policy: str):
        self.budget, self.size, self.policy = budget, size, policy
        self.used = 0
        self.lru: OrderedDict = OrderedDict()
        self.hot: dict = defaultdict(float)   # scaled by 2**epoch: halving is a new epoch
        self.version: dict = {}
        self.heap: list = []
        self.scale = 1.0
        self.clock = 0

    def __contains__(self, key) -> bool:
        return key in self.lru

    def note(self, key, count: float = 1.0) -> None:
        """A selection (hit or miss) of key."""
        self.clock += 1
        if self.policy == "hot":
            self.hot[key] += count * self.scale
            if key in self.lru:
                self._push(key)
        if key in self.lru:
            self.lru.move_to_end(key)

    def halve(self) -> None:
        self.scale *= 2.0

    def _push(self, key) -> None:
        v = self.version.get(key, 0) + 1
        self.version[key] = v
        heapq.heappush(self.heap, (self.hot[key], self.clock, v, key))

    def insert(self, key) -> None:
        if key in self.lru:
            return
        need = self.size[key[0]]
        while self.used + need > self.budget and self.lru:
            self._evict()
        if self.used + need > self.budget:
            return
        self.lru[key] = True
        self.used += need
        if self.policy == "hot":
            self._push(key)

    def _evict(self) -> None:
        if self.policy == "lru":
            key, _ = self.lru.popitem(last=False)
        else:
            while True:
                _, _, v, key = heapq.heappop(self.heap)
                if key in self.lru and self.version.get(key) == v:
                    break
            del self.lru[key]
        self.used -= self.size[key[0]]


def predictions(forwards, router, model: str, n_embd: int, widths: list[int]):
    """Per decode forward and layer L+1: predicted ids from layer L's input,
    and the share of traced selections that layer L's own router reproduces."""
    import numpy as np
    w = {}
    with open(model, "rb") as f:
        for layer, (off, n_in, n_exp) in router.items():
            f.seek(off)
            raw = np.frombuffer(f.read(n_in * n_exp * 2), dtype=np.uint16)
            w[layer] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(n_exp, n_in)
    pred, same, total = {}, 0, 0
    for fi, (_pos0, rows, layers) in enumerate(forwards):
        if rows > 3:
            continue
        for layer, ids, inp in layers:
            if inp is None:
                continue
            x = np.frombuffer(inp, dtype=np.float16).astype(np.float32).reshape(rows, n_embd)
            for r in range(rows):
                if layer in w:
                    top = set(np.argpartition(-(w[layer] @ x[r]), 10)[:10].tolist())
                    same += len(top & set(ids[r]))
                    total += len(ids[r])
                if layer + 1 in w:
                    logits = w[layer + 1] @ x[r]
                    order = np.argsort(-logits)
                    for p in widths:
                        pred[(fi, layer + 1, r, p)] = set(order[:p].tolist())
    return pred, (same / total if total else 0.0)


def simulate(forwards, size, budget, policy, points, seq_gibs, pred, width, layer_ms):
    cache = Cache(budget, size, policy)
    st = defaultdict(float)
    for fi, (_pos0, rows, layers) in enumerate(forwards):
        if rows > SMALL_FORWARD:
            for layer, ids, _ in layers:
                counts: dict[int, int] = defaultdict(int)
                for row in ids:
                    for e in row:
                        counts[e] += 1
                layer_bytes = size[layer] * 512
                missing = sum(size[layer] for e in counts if (layer, e) not in cache)
                st["prefill_full_gib"] += layer_bytes / GIB
                st["prefill_missing_gib"] += missing / GIB
                for e, c in sorted(counts.items(), key=lambda kv: kv[1]):
                    cache.note((layer, e), c)
                    cache.insert((layer, e))
            st["prefill_chunks"] += 1
            continue
        decode = rows <= 3
        for layer, ids, _ in layers:
            want = {e for row in ids for e in row}
            miss = [e for e in want if (layer, e) not in cache]
            exposed = miss_ms(points, len(miss))
            if pred is not None and decode and miss:
                guessed = set()
                for r in range(rows):
                    guessed |= pred.get((fi, layer, r, width), set())
                early = [e for e in miss if e in guessed]
                late = len(miss) - len(early)
                # The predicted reads start a layer earlier and overlap its compute.
                exposed = max(0.0, miss_ms(points, len(early)) - layer_ms) + miss_ms(points, late)
                wasted = len([e for e in guessed if e not in want and (layer, e) not in cache])
                st["wasted_mb"] += wasted * size[layer] / 1e6
                st["predicted_misses"] += len(early)
            if decode:
                st["misses"] += len(miss)
                st["miss_mb"] += len(miss) * size[layer] / 1e6
                st["io_ms"] += miss_ms(points, len(miss))
                st["exposed_ms"] += exposed
            for row in ids:
                for e in row:
                    cache.note((layer, e))
            for e in miss:
                cache.insert((layer, e))
        if decode:
            st["decode_rows"] += rows
            if int(st["decode_rows"]) // 16 != int(st["decode_rows"] - rows) // 16:
                cache.halve()
    st["prefill_missing_s"] = st["prefill_missing_gib"] / seq_gibs
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tensors")
    ap.add_argument("ssd")
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--budgets", default="11,16,20,26,32", help="expert cache GiB")
    ap.add_argument("--policies", default="lru,hot")
    ap.add_argument("--threads", type=int, default=9)
    ap.add_argument("--model", help="GGUF of TENSORS.tsv, for the router predictor")
    ap.add_argument("--widths", default="10,16", help="predicted ids per layer")
    ap.add_argument("--layer-ms", type=float, default=0.6, help="compute window of one decode layer")
    ap.add_argument("--resident-ms", type=float, default=0.0, help="resident ms per token, for tok/s")
    ap.add_argument("--sync-ms", type=float, default=0.0, help="router readback ms per token")
    args = ap.parse_args()

    size, router = expert_bytes(args.tensors)
    points, seq = ssd_model(args.ssd, args.threads)
    budgets = [float(b) for b in args.budgets.split(",")]
    policies = args.policies.split(",")
    widths = [int(w) for w in args.widths.split(",")]
    print("trace\tpolicy\tbudget_gib\tpredict\tdecode_rows\tmiss_per_tok\tmiss_mb_per_tok\tio_ms_per_tok"
          "\texposed_ms_per_tok\tpredicted_share\twasted_mb_per_tok\tprefill_gib_per_chunk\tprefill_missing_gib"
          "\tprefill_missing_s\ttok_s")
    for path in args.traces:
        forwards, n_embd = read_trace(path, args.model is not None)
        pred, agree = (None, 0.0)
        if args.model:
            pred, agree = predictions(forwards, router, args.model, n_embd, widths)
            print(f"# {path}: layer L's router reproduces {agree * 100:.2f}% of the traced selections")
        name = path.rsplit("/", 1)[-1].removesuffix(".trace")
        for policy in policies:
            for budget in budgets:
                for width in ([0] + widths if pred else [0]):
                    st = simulate(forwards, size, int(budget * GIB), policy, points, seq,
                                  pred if width else None, width, args.layer_ms)
                    n = st["decode_rows"] or 1
                    chunks = st["prefill_chunks"] or 1
                    exposed = st["exposed_ms"] / n
                    tok_s = (1e3 / (args.resident_ms + args.sync_ms + exposed)) if args.resident_ms else 0.0
                    print(f"{name}\t{policy}\t{budget:g}\t{width}\t{int(st['decode_rows'])}"
                          f"\t{st['misses'] / n:.1f}\t{st['miss_mb'] / n:.1f}\t{st['io_ms'] / n:.2f}"
                          f"\t{exposed:.2f}\t{(st['predicted_misses'] / st['misses'] if st['misses'] else 0):.2f}"
                          f"\t{st['wasted_mb'] / n:.1f}\t{st['prefill_full_gib'] / chunks:.2f}"
                          f"\t{st['prefill_missing_gib'] / chunks:.2f}\t{st['prefill_missing_s'] / chunks:.2f}"
                          f"\t{tok_s:.1f}")
                    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
