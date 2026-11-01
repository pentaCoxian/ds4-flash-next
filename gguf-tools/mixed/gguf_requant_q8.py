#!/usr/bin/env python3
"""Requantize selected F32/F16/BF16 tensors of a GGUF to Q8_0.

  gguf_requant_q8.py --base B --out OUT [--tensors REGEX]

Every tensor whose name REGEX matches (re.fullmatch) is rewritten as Q8_0
blocks of 32 with ggml's reference rounding (quantize_row_q8_0_ref: the
scale is the block's f32 amax / 127, the codes x / scale rounded half away
from zero, the scale stored as f16); every other tensor is copied byte for
byte.  The default selection is the Qwen3.8 hyper-connection mixers'
low-rank down and up projections, 194 BF16 tensors: in iq3m they are 1.2
of the 4.3 GiB a decode token reads, and as Q8_0 they save 0.57 GiB with
a mean KL of 0.011 against BF16 on code, English and Japanese.  The output
records the base file name and the selection as general.requant.* strings.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_mix as gm  # noqa: E402

HC_TENSORS = r"blk\.[0-9]+\.hc_(attn|ffn)_(down|up)\.weight|output_hc_(down|up)\.weight"
GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30


def to_f32(raw: bytes, ggml_type: int) -> np.ndarray:
    if ggml_type == GGML_BF16:
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if ggml_type == GGML_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    return np.frombuffer(raw, dtype=np.float32)


def q8_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref over consecutive blocks of 32."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1) / np.float32(127.0)
    inv = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d != 0)
    v = b * inv[:, None]
    q = np.clip(np.trunc(v + np.copysign(np.float32(0.5), v)), -127, 127).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def q8_0_dequant(blob: bytes) -> np.ndarray:
    blk = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 34)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32)
    return (blk[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tensors", default=HC_TENSORS, help="re.fullmatch over tensor names")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    base = gm.parse_gguf(a.base)
    sel = re.compile(a.tensors)
    plan, rel = [], 0
    for t in base.tensors:
        requant = sel.fullmatch(t.name) is not None
        if requant:
            if t.ggml_type not in (GGML_F32, GGML_F16, GGML_BF16) or t.dims[0] % 32:
                raise ValueError(f"{t.name}: type {t.ggml_type}, rows of {t.dims[0]}; need F32/F16/BF16 rows of 32k")
            n = t.n_bytes // (4 if t.ggml_type == GGML_F32 else 2) // 32 * 34
        else:
            n = t.n_bytes
        plan.append((t, requant, rel, n))
        rel += gm.pad_to(n, base.alignment)
    n_requant = sum(1 for p in plan if p[1])
    if not n_requant:
        raise ValueError("the selection matches no tensor")

    tmp = a.out.with_name(a.out.name + ".tmp")
    if a.out.exists() and not a.force:
        raise FileExistsError(f"{a.out} exists; pass --force")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    prov = [("general.requant.base", a.base.name), ("general.requant.type", "q8_0"),
            ("general.requant.tensors", a.tensors)]
    err_num = err_den = 0.0
    with a.base.open("rb") as src, tmp.open("wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", base.version))
        out.write(struct.pack("<Q", base.tensor_count))
        out.write(struct.pack("<Q", base.kv_count + len(prov)))
        out.write(base.kv_blob)
        out.write(gm.provenance_kv(prov))
        for t, requant, off, _n in plan:
            out.write(gm.pack_string(t.name))
            out.write(struct.pack("<I", len(t.dims)))
            for d in t.dims:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<I", GGML_Q8_0 if requant else t.ggml_type))
            out.write(struct.pack("<Q", off))
        gm.write_padding(out, out.tell(), base.alignment)
        for t, requant, _off, n in plan:
            src.seek(t.data_offset)
            if requant:
                x = to_f32(gm.read_exact(src, t.n_bytes), t.ggml_type)
                blob = q8_0(x)
                y = q8_0_dequant(blob)
                err_num += float(np.sum((y - x) ** 2))
                err_den += float(np.sum(x.astype(np.float64) ** 2))
                out.write(blob)
            else:
                gm.copy_exact(src, out, n)
            gm.write_padding(out, n, base.alignment)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, a.out)
    saved = sum(t.n_bytes - n for t, requant, _o, n in plan if requant)
    print(f"{a.out}: {n_requant} tensors to Q8_0, {saved / 2**30:.3f} GiB smaller, "
          f"relative RMS error {np.sqrt(err_num / err_den):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
