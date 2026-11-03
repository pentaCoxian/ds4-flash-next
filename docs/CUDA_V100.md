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

### Two PCIe x8/x8 cards

Measured on the deployment layout: two V100-SXM2-32GB on PCIe 3.0 x8 each,
on different sockets, no peer access, 200 W power limit, a 2013 Xeon host.
262K context, `--gpu-vram 31,31 --prefill-chunk 3072` (see Running).

| Prefill, tok/s | |
|---|---|
| 1.5K-token prompt | 1300 |
| 2.5K-token prompt | 1520 |
| 34K-token prompt | 2490-2550 |
| 143K-token prompt | 2550-2560 |
| 448K-token prompt, 524K context (YaRN 2, 1024-row chunks) | 1850 |

Decode with MTP, greedy, 512 tokens: code 135, tech 110, prose 99 tok/s; a
copy-heavy edit of a 1.4K-token file 157-161; 102 after a 148K-token prompt
and 83 after a 448K-token one.  Plain decode 67 tok/s, 62 at 143K.

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

`--gpu-vram auto` leaves each card about 2.5 GB more than it needs. An
explicit `--gpu-vram 31,31` hands it to the planner, and then 262K takes
3072-row prefill chunks, which fill the routed experts' tiles far better:

```sh
./ds4-server --cuda --gpu-devices 0,1 --gpu-vram 31,31 --prefill-chunk 3072 -m $W \
    --ctx 262144 --mtp --host 127.0.0.1 --port 8010
```

524K of context takes YaRN and 1024-row chunks (2048 does not fit):

```sh
DS4_QWEN4_YARN_FACTOR=2 ./ds4-server --cuda --gpu-devices 0,1 --gpu-vram 31,31 \
    --prefill-chunk 1024 -m $W --ctx 524288 --mtp --host 127.0.0.1 --port 8010
```

### Two slots

`--batched-session 2` serves two conversations at once, each with MTP and
its own 262K context:

```sh
./ds4-server --cuda --gpu-devices 0,1 --gpu-vram full --prefill-chunk 3072 -m $W \
    --ctx 262144 --mtp --batched-session 2 --host 127.0.0.1 --port 8010
```

`--gpu-vram full` budgets each card's free memory with no reserve and no
safety margin. Two 262K slots need it: the planner places 25.7 and 26.5 GB
of layers and caches, and the cards peaked at 30.6 and 30.9 GB of 32 with
both slots holding 240K-token prompts.

The two slots' decode cycles are pipelined over the GPUs: GPU0 runs one
stream's first-half layers while GPU1 verifies the other stream's window.
Each stream keeps MTP, and greedy output is token-identical to the stream
decoding alone. A slot generating alone takes the single-stream cycle with
its lookahead.

| Two slots, 2x V100 x8/x8 | Stream 1 | Stream 2 | Total |
|---|---|---|---|
| Engine, alone (code / TCP prompt, `--nothink`, 256 tok) | 145 | 118 | |
| Engine, both pipelined | 104-106 | 94-95 | 188-189 |
| Server chat, greedy, both at once | 102 | 85 | 187 |
| Server chat, T=0.7, both at once | 92 | 71 | 163 |
| Server, both after ~240K-token prompts | 72 | 75 | 147 |

Prefill and decode take turns, in time shares of about half each: a waiting
prefill gets the model for one quantum between decode cycles, and decode
then keeps it as long again. A quantum is one prefill chunk while another
slot generates (2048 tokens at least; `--mixed-prefill-quantum` sets it)
and twelve chunks while none does. Each quantum is a sync of its own and
pipelines only its own chunks, so 2048-token quanta prefilled a 34K prompt
at 1380 tok/s against 2290 with these. Two 240K-token prompts sent together
prefilled at 2215 tok/s in all. `DS4_SERVER_NO_PIPE=1` turns the pipelined
decode off; the slots then decode one after another without MTP.

Two slots fit at 262K because of two memory changes:

- **Indexer key ring.** The raw indexer keys are read only when a block of
  four tokens completes, so they live in a ring of a chunk and eight rows
  (`pos % rows`), as on Metal. That is 1.5 GB less per slot at 262K.
- **Shared prefill transients.** The sessions borrow one set of
  chunk-sized transients per GPU in turn (2.8 GB a GPU at 262K with
  3072-row chunks). Each keeps 16-row sets of its own for decode, verify
  and lookahead windows, which two pipelined cycles use at the same time.
  That is 5.7 GB less than private sets for both slots.
  `DS4_QWEN4_PRIVATE_SCRATCH=1` gives each session its own.

Two bugs in multi-session CUDA use were fixed on the way. Both made the
second session's output wrong:

- The cache of the last `cudaSetDevice` was global, but the device is per
  host thread. The server prefills on a slot's thread and decodes on its
  coordinator's, so the worker could skip a needed switch. It now caches
  per thread.
- A graph's same-tier "switch" returned without selecting the device. After
  another session left the device on GPU1, a fresh graph ran its GPU0 layers
  on GPU1 through peer access, unordered against GPU0's queue.

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

**MTP.** MTP drafts up to three tokens, verified as a 2-, 3- or 4-row pass
chosen per cycle by expected tokens per second: the draft probabilities,
calibrated per session, against the measured cost of each depth. A 4-row
verify snapshots the recurrent state after rows 0, 1 and 2, and its
attention runs as two 2-row sub-batches where the rows' block sets differ,
so greedy output stays identical to plain decode. After a prefill the MTP
layer's cache rows for the prompt are filled from the trunk, so drafts
attend over the real prompt. On a prompt that restates part of its own
source, first-draft acceptance went from 79% to 87%.

With default sampling, a draft is accepted when it is the verify row's
argmax, and the token after the accepted ones is drawn at once, with the
caller's settings, and the next drafts follow it (it is the caller's next
sample). Drafted from the argmax instead, a sample other than it found no
draft, and 48 of 237 cycles of an explanation at temperature 0.8 ran as
plain steps.

The drafts score a prefix of the output head, the first 98,304 vocabulary
rows, which leaves out many CJK tokens. Pass a list of rows instead:
`DS4_QWEN4_MTP_DRAFT_VOCAB=<file>` (token ids, one a line; the rows are
copied beside the MTP block). With a 64K list that adds the Japanese,
Chinese and Korean tokens (`draftvocab-64k-ml`), translation into Japanese
went from 123 to 140 tok/s and Japanese prose from 84 to 94, with English
unchanged.

**Lookahead.** Over a layer split each GPU used to idle while the other ran
its half of a verify. Now, while GPU1 verifies a 3- or 4-row window, the MTP
chain drafts as many next tokens, and GPU0 runs the next window's layers
with them.

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
  k. The pipeline drains about every 64K tokens (at least every 8 chunks),
  before reporting progress, so a saved session always sees consistent
  state; a drain idles each GPU for about half a chunk.
- The host keeps both GPUs fed: the hand-off between them goes through
  pinned host memory without the host waiting on it, the next chunk's
  n-gram and input rows are read while the current chunk runs, and the
  staging buffers are page-locked.  On the x8/x8 node these waits had left
  both GPUs idle for 240 ms of every 3072-row chunk.
- Prompts are cut into equal chunks, and from 1536 tokens into at least
  two, so even a mid-size prompt overlaps the GPUs.
- The indexer scores a chunk's rows on the tensor cores (the queries as
  half plus a half residual, the pooled keys as stored).
- Sparse attention over 8-bit K/V rows runs on Volta's tensor cores too
  (`attention_group_q8`): one block per row and KV group, the int8 values
  exact as half and their scales applied in FP32.  It reads each selected
  key once for the group's 12 query heads, where the per-head kernel read
  it 12 times: 2.5x faster, and closer to the FP64 reference.
- The GDN scan of prefill rows gives a state row eight lanes (three-shuffle
  reductions, the next token's operands loaded ahead): 2.2x faster than
  the per-row scan.  Every chunk and tail size sums alike, so a split
  prompt keeps the bits of a whole one.
- One GPU's half of a chunk is thousands of kernel launches, more than the
  driver's default launch queue holds, so the build sets
  `CUDA_SCALE_LAUNCH_QUEUES=4x` (your own value wins).

**Indexer at depth.** Decode and verify rows score an int8 copy of the
pooled block keys (an f16 scale per 32 values, beside the f16 keys that
prefill and checkpoints use) with `dp4a`, reading each key once for all the
rows: 16 us a layer at 148K for a 3-row verify, against 168.  A verify's
argmax per row is taken on the GPU, and only the row the session keeps is
copied back.

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

The int8 copy of the indexer's block keys only ranks blocks; it matters past
2K tokens, where the 696-token test does not reach.  Teacher-forced decode of
128 tokens of C source, int8 against f16 keys (`DS4_QWEN4_IDX_F16=1`): after
34K tokens NLL 0.0784 against 0.0784, after 143K 0.0723 against 0.0695, top-1
the same (97.7%, 96.9%); plain decode at 143K 62.0 against 59.9 tok/s.  The
tensor-core prefill scorer agrees with an FP64 reference to 6e-7 of the
scores' scale.

## Switches

| Variable | Effect |
|---|---|
| `DS4_QWEN4_LOOKAHEAD=0` | no lookahead |
| `DS4_QWEN4_MTP_FILL=0` | no MTP cache rows for the prompt |
| `DS4_QWEN4_MTP_DEPTH=1/2/3/4` | force the MTP verify depth (`window` for the older rule) |
| `DS4_QWEN4_MTP_DRAFT_VOCAB=<file>` | draft over the listed vocabulary rows (see MTP) |
| `DS4_QWEN4_KV_F16=1` | FP16 K/V rows |
| `DS4_QWEN4_KV_Q8=1` | 8-bit K/V rows on newer GPUs too |
| `DS4_QWEN4_F32_ACT=1` | FP32 decode activations |
| `DS4_CUDA_NO_TF32=1` | FP32 prefill GEMMs and expert tiles |
| `DS4_QWEN4_NO_VOLTA_TILES=1` | FP32 expert tiles on Volta |
| `DS4_QWEN4_NO_PIPE=1` | no prefill pipeline over the GPUs |
| `DS4_QWEN4_PIPE_GROUP=N` | drain the prefill pipeline every N chunks (default: about 64K tokens, at least 8) |
| `DS4_QWEN4_IDX_F16=1` | decode rows score the f16 block keys (no int8 copy) |
| `DS4_QWEN4_IDX_SCORE_FP32=1` | prefill rows score on the FP32 kernel |
| `DS4_QWEN4_NO_ATTN_MM=1` | per-head attention kernel for prefill (also on Volta's 8-bit rows) |
| `DS4_QWEN4_GDN_SCAN_PLAIN=1` | the per-row GDN scan for prefill too |
| `DS4_QWEN4_VROWS_HOST=1` | a lookahead window's rows read and scanned on the host |
| `DS4_QWEN4_BYTE_SPLIT=1` | byte-proportional layer split |
| `DS4_QWEN4_SPLIT_SHIFT=N` | start the layer split N layers later (more layers on the first GPU) |
| `DS4_QWEN4_LA_INTERLEAVE=0` | encode the lookahead's two halves one after the other |
| `DS4_QWEN4_IK_FULL=1` | raw indexer keys kept per position, not in a ring |
| `DS4_QWEN4_PRIVATE_SCRATCH=1` | each session its own prefill transients (several sessions) |
| `DS4_SERVER_NO_PIPE=1` | no pipelined two-slot MTP decode in the server |
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
