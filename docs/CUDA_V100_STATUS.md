# ds4 on 2x V100: two 262K slots and decode speed vs llama.cpp

[README](../README.md) | [NVIDIA V100](CUDA_V100.md)

Status as of 2026-11-04.

## Summary

Two 262K-token slots now run side by side on the 2x V100 node. Single-stream
decode is 9.4% slower than llama.cpp at temp 0.8 and 5.7% slower greedy, so
the speed goal is not met yet.

- **Two slots:** done. Both fit in 32 GB per GPU with an indexer key ring, a
  shared prefill workspace and `--gpu-vram full`.
- **Decode:** at parity on prose, japanese and math. Long-context (-16% at
  32K, -22% at 128K) and summarize (-15%) trail most.
- **Prefill:** within 7% of llama.cpp at every prompt size, and slightly
  faster at 2.6K tokens.
- **Two requests at once:** 176 tok/s total sampled and 190 greedy, 1.34x and
  1.41x a single stream.
- **Code:** three commits on `cuda-v100`, pushed on 2026-11-04.

All numbers are from build 318b984, measured on 2026-11-03.

## Setup

ds4 runs Qwen3.8 Flash Next (iq3m8) with MTP speculative decoding, split by
layer across two V100s.

```
 2x V100-SXM2 32 GB; PCIe x8, no peer-to-peer; 200 W power limit

 +--------------------------------+    +--------------------------------+
 | GPU0: embedding + layers 0-24  |    | GPU1: layers 25-47 + out head  |
 |                                |    |                                |
 | Front half of each 1-4 row     |    | Back half of the window, then  |
 | window. Runs ahead on the next |    | verify.                        |
 | window while GPU1 verifies     |    | +----------------------------+ |
 | (lookahead).                   |    | | MTP draft block            | |
 |                                |    | | Up to 3 drafts, ~0.75 ms   | |
 |                                |    | | each; on the critical path | |
 |                                |    | | of every window            | |
 |                                |    | +----------------------------+ |
 +--------------------------------+    +--------------------------------+
     |  activations   ^ tokens            ^ activations    |  logits
     v                | + drafts          | (PCIe x8)      v  + drafts
 +-------------------------------------------------------------------------+
 | Host CPU: samples tokens, accepts or rejects drafts, launches each      |
 | window's kernels. No peer-to-peer: activations go GPU0 -> host -> GPU1. |
 +-------------------------------------------------------------------------+
```

Each window verifies the current token plus up to three MTP drafts. GPU0
starts the next window on the token it predicts while GPU1 is still
verifying.

The benchmark is llama.cpp's own 11-task suite, run against `ds4-server` on
the same node as the llama.cpp reference:

- **Sampling:** llama-server's defaults (temp 0.8, top_k 40, top_p 0.95,
  min_p 0.05) and the same seeds, plus a greedy pass.
- **Decode speed:** (generated tokens - 1) / (time after the first token),
  measured client-side over streaming.
- **Runs:** 2 per task, 1 for the two long-context tasks (34K and 142K-token
  prompts).
- **Server:** 262K context, `--mtp`, 3,072-token prefill chunks, multilingual
  draft vocabulary.

## Two 262K slots

Two 262K slots started 11.5 GiB over the VRAM budget. Three changes made them
fit, with a peak of 30.6 and 30.9 GB of 32 on the two GPUs.

| Change | VRAM effect | Notes |
|---|---|---|
| Indexer key ring on CUDA | -1.5 GB per slot | Keys live in a fixed ring of rows, as on Metal. Greedy output is byte-identical to the full cache at 238K tokens. |
| Shared prefill workspace | -5.7 GB for two slots | Slots borrow one chunk-sized set of prefill buffers per GPU. Each keeps its own 16-row decode buffers. |
| `--gpu-vram full` | Frees ~2.5 GiB | Drops the automatic reserve (2 GiB or 5%) and the 512 MiB margin. Two 262K slots need it. |

Measured with both slots holding ~240K-token prompts:

- **Prefill:** 2,215 tok/s for both together.
- **Decode:** 72 + 75 tok/s at that depth; 188 tok/s total for two
  short-context streams.
- **Correctness:** greedy output matches single-stream decode. The HANDOFF
  server tests (`req2.py`, `req3.py`), `ds4_test --server`, the kernel tests
  with a new ring test and the CPU-only build all pass.

In two-slot mode, prefill runs in 12-chunk quanta when idle and 1-chunk
quanta beside generation. The old 2,048-token quanta had cut a 34K-token
prompt to 1,380 tok/s; it now prefills at 2,290. The CUDA graph "islands"
tried earlier were 3% slower and were removed.

## Benchmark results

Single-stream decode is 9.4% behind llama.cpp at temp 0.8, and the
long-context tasks trail most. Tasks are sorted by the temp 0.8 gap.

| Decode, tok/s | llama.cpp | ds4 temp 0.8 | vs llama.cpp | ds4 greedy | vs llama.cpp |
|---|---|---|---|---|---|
| longctx-128k | 130.3 | 101.8 | -22% | 112.3 | -14% |
| longctx-32k | 134.7 | 113.3 | -16% | 123.9 | -8% |
| summarize | 132.2 | 112.5 | -15% | 115.9 | -12% |
| code-edit | 188.5 | 168.4 | -11% | 172.6 | -8% |
| translate | 156.4 | 140.2 | -10% | 146.0 | -7% |
| explain | 130.2 | 119.1 | -9% | 126.7 | -3% |
| json | 187.0 | 171.3 | -8% | 177.8 | -5% |
| codegen | 173.7 | 160.2 | -8% | 168.8 | -3% |
| math-think | 146.2 | 141.6 | -3% | 148.3 | +1% |
| prose | 93.5 | 90.7 | -3% | 91.0 | -3% |
| japanese | 91.5 | 95.5 | +4% | 90.6 | -1% |
| **geomean** | | | **-9.4%** | | **-5.7%** |

ds4 values are the mean of 2 runs per task (1 for longctx). The llama.cpp
reference is `/workspace/qwen38/task-bench.json`, measured on the same node
at temp 0.8. Greedy is compared against that same reference. The previous run
on this build gave -8.8% and -5.2%, so the difference is run-to-run
variation from sampling.

Prefill is within 7% of llama.cpp at every prompt size (first request, no
cache):

| Prompt | ds4 (tok/s) | llama.cpp (tok/s) |
|---|---|---|
| code-edit, 1.4K tokens | 1,162-1,211 | 1,244 |
| translate, 1.1K tokens | 1,107 | 1,176 |
| summarize, 2.6K tokens | 1,533 | 1,519 |
| json, 2.6K tokens | 1,512 | 1,484 |
| longctx, 34K tokens | 2,379-2,399 | 2,433 |
| longctx, 142K tokens | 2,418-2,428 | 2,558 |

Two requests at once give 1.34-1.41x a single stream. These are the 9 short
tasks, each sent twice at the same time to the `--batched-session 2` server:

| Sampling | Each stream (tok/s) | Both streams (tok/s) | vs one ds4 stream |
|---|---|---|---|
| temp 0.8 | 70-107 | 131-214, mean 176 | 1.34x |
| greedy | 77-107 | 154-214, mean 190 | 1.41x |

Both long-context answers passed their correctness checks.

## What closed the decode gap

The gap to llama.cpp shrank from -15.2% to -9.4% at temp 0.8, and from
-10.4% to -5.7% greedy.

| Change | Where | Measured effect |
|---|---|---|
| 4-row verify windows: up to 3 MTP drafts per window, chosen per cycle from draft probabilities and the measured 4-row cost | 09dc6da | code-edit 160 -> 166 tok/s, json 158 -> 166 (greedy). Greedy output stays identical to plain decode. |
| Drafting from the sampled token: the token after the accepted drafts is sampled in settle, and the next drafts follow it | 09dc6da | Drafts are no longer discarded when the sample is not the argmax |
| 64K multilingual draft vocabulary (`draftvocab-64k-ml.ids`) | Data file, documented in 09dc6da | translate +13%, japanese +12%. The default 98,304-row prefix misses many CJK tokens. |
| Interleaved lookahead: GPU1's half of the window being verified and GPU0's half of the next window are encoded alternately | 318b984 | GPU0 no longer idles for the first ~5 ms of each window. code-edit 168 -> 173 tok/s, json 165 -> 171 (greedy). |

Tried with no gain, so not worth retrying:

- CUDA graph islands (3% slower).
- GPU top-k sampling: exact, but the host draw is not on the critical path.
- Matvec variants: paired rows, staged loads.
- GDN scan with 2 rows per warp.
- A smaller draft-chain head: it lowers acceptance.
- Forcing depth 3, shifting the layer split, async lookahead state copies.

## Why the gap remains, and next steps

The remaining gap comes from four costs. The first two are on the critical
path of every window.

- **Sampling breaks the run-ahead.** GPU0 starts the next window on the token
  it predicts. When the sample differs, that work is discarded and the window
  runs without overlap. explain shows it: 126.7 tok/s greedy, 119.1 sampled.
- **GPU1 carries the MTP draft chain.** Each draft step takes ~0.75 ms, up to
  four per window, before the next window can start.
- **Matvecs are latency-bound.** At 4 rows they reach ~350-400 GB/s of the
  V100's 900 GB/s. No profiler on this node supports Volta, so this is
  measured from timings.
- **Long-context attention.** Top-k block selection takes ~56 us per call
  with one thread block per row, and attention ~72 us per call. This is why
  128K trails by 22%.

Next steps, in order of expected gain:

- [ ] Run the MTP draft chain beside GPU1's back half on a second CUDA
  stream. This needs per-GPU temporary buffers that are safe across streams.
  The alternative is moving the MTP block to GPU0.
- [ ] Spread long-context top-k block selection over several thread blocks
  per row.
- [ ] Requantize the MTP block to cut its matvec time.

## How to run and reproduce

Build, then start the two-slot server with the model files below.

```bash
PATH=/usr/local/cuda-12.9/bin:$PATH make -j64 ds4 ds4-server CUDA_ARCH=sm_70 CUDA_HOME=/usr/local/cuda-12.9

M=/workspace/models/flash-next
export DS4_QWEN_NGRAM_GGUF=$M/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
export DS4_QWEN_MTP_GGUF=$M/Qwen3.8-Flash-Next-MTP-block.gguf
export DS4_QWEN4_MTP_DRAFT_VOCAB=$M/draftvocab-64k-ml.ids
W=$M/IQ3_XXS_M64_HC8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64_HC8-00001-of-00002.gguf

./ds4-server --cuda --gpu-devices 0,1 --gpu-vram full --prefill-chunk 3072 -m "$W" \
    --ctx 262144 --mtp --batched-session 2 --host 127.0.0.1 --port 8010
```

The single-stream benchmark used `--gpu-vram 31,31` and no
`--batched-session`. Only one ds4 process can own the GPUs, so stop any
server before running another.

The benchmark drives the llama.cpp suite's own prompts
(`/workspace/src/llama.cpp/deploy/qwen38-v100/bench/tasks.py`) with
llama-server's default sampling and seeds. The llama.cpp reference is
`/workspace/qwen38/task-bench.json`. The runner scripts (`tasks_ds4.py`,
`tasks_pair.py`) are not in the repo yet. [Running](CUDA_V100.md#running)
says where the draft list comes from.

Switches for experiments, all documented in [CUDA_V100.md](CUDA_V100.md):

| Variable | Effect |
|---|---|
| `DS4_QWEN4_MTP_DRAFT_VOCAB=<file>` | Draft over the listed vocabulary rows |
| `DS4_QWEN4_MTP_DEPTH=1/2/3/4` | Force the MTP verify depth |
| `DS4_QWEN4_LA_INTERLEAVE=0` | Encode the lookahead's two halves one after the other |
| `DS4_QWEN4_SPLIT_SHIFT=N` | Start the layer split N layers later |
| `DS4_QWEN4_IK_FULL=1` | Keep indexer keys per position instead of in a ring |
| `DS4_QWEN4_PRIVATE_SCRATCH=1` | Give each session its own prefill buffers |
| `DS4_SERVER_NO_PIPE=1` | Turn off pipelined two-slot decode |

## Commits and open items

The work is in three commits on `cuda-v100` from 2026-11-03, pushed on
2026-11-04.

| Commit | Change |
|---|---|
| 318b984 | Interleave the lookahead's back half with the window run ahead (CUDA) |
| 09dc6da | Verify up to three MTP drafts on CUDA, and draft from the sampled token |
| c75b658 | Two 262K MTP slots on the V100 pair; int8 block keys and faster prefill (CUDA) |

- [ ] Decide whether the benchmark runner scripts should move into the repo,
  so the suite can be rerun after this session.
- [ ] Close the remaining decode gap (see next steps above).
