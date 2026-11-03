# Qwen3.8 Flash Next on NVIDIA V100

[README](../README.md) | [CUDA GPUs](CUDA_MULTI_GPU.md) | [Qwen3.8 Flash Next](QWEN38_FLASH_NEXT.md)

iq3m8 runs resident on two 32 GB V100s (Volta, `sm_70`), its layers split
over the two GPUs. The CUDA path for Qwen3.8 has Volta-specific kernels for
decode, MTP verification and prefill, and it pipelines work over the two GPUs.
Everything on this page is CUDA-only; the Metal path is unchanged.

The reference machine has two Tesla V100-SXM2-32GB and CUDA 12.9. Its two
GPUs are linked by NVLink. The intended deployment is two
PCIe cards (x8/x8) without NVLink or peer access. That case was measured
by sending every cross-GPU copy through host memory (`DS4_FORCE_HOST_BOUNCE=1`);
the numbers below stay within 1 to 2% of the NVLink ones.

## Speed

iq3m8, two V100s, `--gpu-vram auto`. Decode is 256 generated tokens, greedy
unless noted; "with MTP" means `--mtp`, which includes the lookahead
described below.

| Decode, tok/s | Code | Prose | Tech |
|---|---|---|---|
| plain | 72 | 72 | 72 |
| with MTP, greedy | 142 | 116 | 128 |
| with MTP, temperature 0.7 (mean of 3 seeds) | 127 | 95 | 119 |

The code, prose and tech prompts are three short questions: a linked-list
function, why the sky is blue, and TCP against UDP. MTP speed depends on how
predictable the answer is. A long prompt costs little: a 12K-token prompt
decodes at 142 tok/s with MTP (66 plain), and a 34K-token prompt at a 262K
context also at 142.

| Prefill | tok/s |
|---|---|
| 630-token prompt | 1000 |
| 2.4K-token prompt | 1315 |
| 12K-token prompt | 1900 |
| 34K-token prompt, 262K context | 1680 |

For comparison, the first working CUDA build decoded at 23 tok/s (33 with
MTP) and prefilled at 270 tok/s on the same machine.

## Build

Volta needs CUDA 12.x: CUDA 13 dropped `sm_70`. Install the 12.x toolkit
beside a newer one if necessary, and point the build at it:

```sh
PATH=/usr/local/cuda-12.9/bin:$PATH \
    make cuda CUDA_ARCH=sm_70 CUDA_HOME=/usr/local/cuda-12.9
```

Install the toolkit package (`cuda-toolkit-12-9`), not the `cuda`
metapackage, which also replaces the driver.

## Files

The layout is the one `dstar pull --quant iq3m8` creates:

```text
~/models/flash-next/
  IQ3_XXS_M64_HC8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64_HC8-00001-of-00002.gguf   weights, 42.4 GiB
  IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf                   n-gram table
  Qwen3.8-Flash-Next-MTP-block.gguf                                               MTP block
```

The n-gram table (26.8 GiB) stays on disk and is read row by row for each
prompt chunk and token. Keep it on a local SSD and leave the host page cache
some room.

## Running

```sh
M=~/models/flash-next
export DS4_QWEN_NGRAM_GGUF=$M/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
export DS4_QWEN_MTP_GGUF=$M/Qwen3.8-Flash-Next-MTP-block.gguf
W=$M/IQ3_XXS_M64_HC8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64_HC8-00001-of-00002.gguf

./ds4 --cuda --gpu-devices 0,1 --gpu-vram auto -m $W --ctx 32768 --mtp

./ds4-server --cuda --gpu-devices 0,1 --gpu-vram auto -m $W \
    --ctx 262144 --mtp --host 127.0.0.1 --port 8010
```

`--gpu-vram auto` budgets 29 GB a GPU and fits 262K of context with MTP. The prefill chunk defaults to 2048 tokens, and to 1024 at 128K and above
to keep the transients within those budgets. The speeds above were measured
with `ds4`; the server runs the same session code.

## What runs where

**Layer placement.** For Qwen3.8 the CUDA placement gives each GPU an equal
share of the trunk layers when the bytes fit (24/24 at 32K). A prefill layer
costs about the same whatever its quant, and the two halves run in parallel
during prefill and lookahead. At 262K the KV cache moves the split to 25/23.
`DS4_QWEN4_BYTE_SPLIT=1` restores the byte-proportional split.

**Decode and verify rows.** Matvecs read their activations as int8 with a
scale per 32 values and use `dp4a`. The weight decoders take whole groups of
8 or 32 weights, with the IQ codebooks staged in shared memory. Routed
experts shared by several rows of a verify are read once. The kernels are
written so that a row's sums do not depend on how many rows share the call:
a 3-row MTP verify gives each row the bits it gets alone, so greedy MTP output
equals plain greedy decode. That includes attention, whose split-key
partials are cut by each row's own key count.

**K/V cache.** The attention K and V rows are 8-bit with an FP16 scale per
32 values, as on Metal, on GPUs without sm_80's tensor-core attention tiles.
That is 47% of the FP16 bytes, about 3 GB less at 262K. The indexer's raw keys
stay FP32.

**MTP.** MTP drafts up to two tokens, verified as a 2- or 3-row pass chosen
per cycle by expected tokens per second. After a prefill the MTP layer's
cache rows for the prompt are filled from the trunk, so drafts attend over
the real prompt. On a prompt that restates part of its own source, first-draft
acceptance went from 79% to 87%.

**Lookahead.** Over a layer split each GPU used to idle while the other ran
its half of a verify. Now, while GPU1 verifies a 3-row window, the MTP chain
drafts the next three tokens, and GPU0 runs the next window's layers with
them.

- If the window is accepted whole and its next token is the first of those
  three, the next window is half done.
- If not, GPU0's state goes back to a copy kept for that purpose, and the
  window is settled as before.

Windows chain within one decode call (up to 17 tokens), and everything is
settled before the call returns. It covers greedy decode and default
sampling. With sampling, the engine draws the token after a window with the
caller's settings and passes that draw to the next `ds4_session_sample`, so
nothing is drawn twice. It gains 13 to 18% on code, 4 to 8% on prose.

**Prefill.**

- Routed experts run on Volta tensor cores (`wmma`, FP16 operands, FP32
  sums), 64 rows by 64 tokens a block.
- Dense projections are FP16 GEMMs over rows packed under power-of-two
  scales.
- The hyper-connection mixes run as GEMMs.
- The two GPUs are pipelined: GPU0 runs chunk k+1 while GPU1 finishes chunk
  k. The pipeline drains every 8 chunks, before reporting progress, so a
  saved session always sees consistent state.
- One GPU's half of a chunk is thousands of kernel launches, more than the
  driver's default launch queue holds, so the build sets
  `CUDA_SCALE_LAUNCH_QUEUES=4x` (your own value wins).

A prompt of one chunk or less has nothing to overlap, which is why short
prompts prefill slower.

## Precision

The CUDA path trades a little precision for speed in three places. KL
divergence is measured against FP32 token-by-token logits on a 696-token
test prompt (for prefill, at the ends of 32-token chunks):

| Setting | Mean KL | Off switch |
|---|---|---|
| int8 decode activations | 0.016 | `DS4_QWEN4_F32_ACT=1` |
| FP16 prefill (tiles and GEMMs) | 0.004 | `DS4_CUDA_NO_TF32=1` |
| 8-bit K/V, on top of int8 activations | 0.039 (median 0.007) | `DS4_QWEN4_KV_F16=1` |

On Metal the same 8-bit K/V format measured 0.006 against FP16 K/V over six
2048-token sequences. The CUDA figure comes from one prompt, so treat it as
an upper estimate. Top-1 agreement with the reference stays at 94 to 96%.

## Switches

| Variable | Effect |
|---|---|
| `DS4_QWEN4_LOOKAHEAD=0` | no lookahead |
| `DS4_QWEN4_MTP_FILL=0` | no MTP cache rows for the prompt |
| `DS4_QWEN4_MTP_DEPTH=1/2/3` | force the MTP verify depth (`window` for the older rule) |
| `DS4_QWEN4_KV_F16=1` | FP16 K/V rows |
| `DS4_QWEN4_KV_Q8=1` | 8-bit K/V rows on newer GPUs too |
| `DS4_QWEN4_F32_ACT=1` | FP32 decode activations |
| `DS4_CUDA_NO_TF32=1` | FP32 prefill GEMMs and expert tiles |
| `DS4_QWEN4_NO_VOLTA_TILES=1` | FP32 expert tiles on Volta |
| `DS4_QWEN4_NO_PIPE=1` | no prefill pipeline over the GPUs |
| `DS4_QWEN4_PIPE_GROUP=N` | drain the prefill pipeline every N chunks (default 8) |
| `DS4_QWEN4_BYTE_SPLIT=1` | byte-proportional layer split |
| `DS4_FORCE_HOST_BOUNCE=1` | send cross-GPU copies through host memory, as without peer access |
| `DS4_QWEN4_TIMING=1` / `2` | decode timing; `2` also times prefill stages (and turns the pipeline off) |
| `DS4_QWEN4_SPEC_TRACE=1` | log every MTP cycle and lookahead hit |

`--mtp-timing` prints the MTP acceptance at exit, but adds synchronization,
so compare its speeds only with each other.

## Testing

```sh
PATH=/usr/local/cuda-12.9/bin:$PATH \
    make test-qwen4-cuda CUDA_ARCH=sm_70 CUDA_HOME=/usr/local/cuda-12.9
```

The test runs the kernel tests twice, with FP32 and with int8 activations.
The references round as the GPU does (int8 activations, half operands,
8-bit K/V), and the verify-row tests require bit-identical rows. At the
model level, two checks catch most regressions:

- Greedy `--mtp` output must equal plain greedy output for the same prompt.
- `ds4 --first-token-test` with `DS4_QWEN4_GPU=1 DS4_QWEN4_GPU_ONLY=1`,
  `DS4_QWEN4_GPU_CHUNK=N` and `DS4_QWEN4_FT_OUT=file` dumps logits for KL
  against an FP32 run.

## Limits

- `--mtp-exact-sampling` has no lookahead. That mode still verifies 2-row
  windows only.
- With sampling, the same seed gives different text with and without
  lookahead. After a lookahead hit the next window carries the chain's drafts,
  and default sampling takes the argmax wherever a draft matches it.
- Tensor parallelism over the two GPUs measured slower than the layer split
  for this model and is not used.
- Everything was run on the V100 machine only. Newer GPUs share the prefill
  pipeline, the FP16 dense GEMMs, the lookahead and the attention change, and
  these are untested there. The Volta expert tiles stay Volta-only, and 8-bit
  K/V is not the default on sm_80 and later.

For the same model on two PCIe V100s, the patched llama.cpp in
[qwen38-v100-patches](https://github.com/Warabii1248/qwen38-v100-patches)
reports 1100 to 2600 tok/s prefill and 92 to 185 tok/s decode, depending on
prompt length and how well the MTP drafts are accepted. That is a different
quant on a different machine; the ideas for the prefill pipeline and the
lookahead come from there.
