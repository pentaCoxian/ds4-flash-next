# Handover

[README](../README.md) | [The fork in detail](FLASH_NEXT_FORK.md) | [Testing](TESTING.md)

Where this fork stands on 2026-11-22, and how to carry on from a fresh
clone. The last code changes are from 2026-11-03 on `max-64gb` and
2026-11-04 on `cuda-v100`. Notes that lived only on the development Mac are
summarized here; what could not be kept is listed under
[Not in the repository](#not-in-the-repository).

## Start

```sh
git clone https://github.com/pentaCoxian/ds4-flash-next.git
cd ds4-flash-next && make          # max-64gb, the default branch
git remote add upstream https://github.com/paperniuk/ds4.git && git fetch upstream
```

`git switch cuda-v100` or `git switch stream-32gb` checks out the other
lines of work. Read [AGENT.md](../AGENT.md) (the engine's rules, from
upstream) and the [README](../README.md) (what the default branch ships)
before changing code.

## Branches

| Branch | What it holds | Layout |
|---|---|---|
| `max-64gb` (default) | Qwen3.8-Flash-Next on 64 GB M1 to M3 Max: iq3m8, its Metal kernels, `dstar`, the `/dashboard` page. 53 commits on upstream's `m1-flash-next`. | sources in `src/`, objects in `build/` |
| `cuda-v100` | `max-64gb` plus 22 commits: iq3m8 on two 32 GB V100s, two 262K slots. Kept off the default branch on purpose. | as `max-64gb` |
| `stream-32gb` | 4 commits measuring SSD streaming of routed experts for 32 to 48 GB Macs: tools and numbers, no streaming code yet. | **old layout**, sources in the root; see [Moving stream-32gb](#moving-stream-32gb-onto-max-64gb) |
| `m1max-deploy`, tag `m1max-deploy-2026-10-31` | The first deploy point, an ancestor of `max-64gb`. History only. | old layout |
| `upstream/m1-flash-next` | paperniuk's Flash-Next fork of antirez/ds4, the base of this one. 319 commits not merged here (Hy4 Flash, ISTA's expert-pruned Qwen3.8 Coder, a Neural Engine speed check for Qwen prefill, and more). | old layout |

## Models

| File | Where | SHA-256 |
|---|---|---|
| iq3m8, IQ3_XXS_M64_HC8 (45.5 GB) | [pentacoxian-dev/Qwen3.8-Flash-Next-GSQ-RCO-M64-GGUF](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-GSQ-RCO-M64-GGUF) | `84bc308c…` |
| iq3m, IQ3_XXS_M64 (46.1 GB) | [pentacoxian-dev/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64-GGUF](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64-GGUF) | `9c533e06…` |
| n-gram table, ISTA's second shard (28.8 GB) | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | |
| MTP block (1.5 GB) | [paperniuk/Qwen3.8-Flash-Next-MTP-block](https://huggingface.co/paperniuk/Qwen3.8-Flash-Next-MTP-block) | |
| IQ3_S, the KL reference | ISTA's repo | |

`./dstar pull --quant iq3m8` downloads iq3m8, the n-gram table, the MTP
block and the vision encoder, and checks the SHA; `--quant iq3s` fetches the
reference. Each model card has the full SHA list and the commands that
rebuild the file from ISTA's. Change a card with
`hf upload <repo> README.md README.md`; never upload the GGUF again for
that. Uploading a spliced file is cheap anyway: the Hub's Xet storage
deduplicates it against ISTA's files (iq3m8's 45.5 GB went up as about 0.5
GB).

## Goals and fixed decisions

For `max-64gb`:

- Qwen3.8-Flash-Next only, on 64 GB M1, M2 and M3 Max.
- A 56 GiB GPU budget (`iogpu.wired_limit_mb=57344`), sized for the full
  262K context with MTP, vision and the Neural Engine.
- Priorities, in order: decode speed, prefill and time to first token in
  agent sessions, quality. Workloads: coding agents, English prose,
  Japanese.
- Quants are chosen by measured speed and KL, never by size: Q2_0 is not
  always faster. Requantizing tensors ourselves is fine when it measures
  better (iq3m8's Q8_0 mixers). Quality starts to drop around IQ3_XXS, so
  keep the sensitive tensors high. Per GiB saved, attention costs the most
  KL, then GDN; the hyper-connection mixers almost none.
- Changes for the M3 only are gated on `g_metal_apple_family == 9`. M1 and
  M2 defaults change only with data from those chips.
- One commit per change, with its test and numbers in the message.

## Where it stands

**Mac (`max-64gb`).** At 262K with MTP, vision and the Neural Engine,
iq3m8 plans 51.4 GiB. On the M3 Max at 40K it decodes 46 to 48 tok/s plain
and 52 to 53 with MTP, and prefills 489 to 500 tok/s ([README](../README.md)).
Those numbers are from the build of 2026-11-01. Two decode changes came after
and have not been timed on a quiet machine: the fused decode down rows and
MoE reduce (`7e1a3eb`) and the 98304-row first-draft head (`a8935a7`). K/V
rows are 8-bit on Metal by default (`DS4_QWEN4_KV_F16=1` keeps f16). Only
the M3 Max was measured. On M1 and M2 the Apple9 paths are off and the text
matched upstream's in the M1 path check; the Q8_0 mixer kernels run there
too, untimed.

**V100 (`cuda-v100`).** Two 262K slots fit on two 32 GB V100s. One stream
decodes 9.4% slower than llama.cpp when sampling and 5.7% slower greedy; long
context trails most (-16% at 32K, -22% at 128K), and prefill is within 7%.
Setup, numbers and the remaining gaps are on that branch:
[CUDA_V100.md](https://github.com/pentaCoxian/ds4-flash-next/blob/cuda-v100/docs/CUDA_V100.md)
and
[CUDA_V100_STATUS.md](https://github.com/pentaCoxian/ds4-flash-next/blob/cuda-v100/docs/CUDA_V100_STATUS.md).

**32 to 48 GB (`stream-32gb`).** Phase 0 of the plan below is done and
passed its gate (M3 Max, iq3m):

- **SSD:** one expert read costs 0.41 ms, ten 2.54 ms (5.7 GiB/s). A layer
  with any miss costs at least 0.4 ms.
- **Locality:** decode is local. With a 16 GiB expert cache the agent
  workload misses 38 MB a token, against 151 MB for a fixed hot set of the
  same size. Plain LRU beats the engine's hotness eviction at every size.
- **Prefetch:** layer L+1's router applied to layer L's input finds 54 to
  68% of the misses and halves the exposed I/O.
- **Router round trip:** waiting on the host for each layer's router costs
  11.5 ms a token, so the wait must be pipelined.

SSD read time per decode token, in ms, without / with prefetch (16
predicted ids):

| Expert cache | 11 GiB | 16 GiB | 20 GiB | 26 GiB | 32 GiB |
|---|---|---|---|---|---|
| agent | 16.8 / 9.1 | 8.4 / 4.2 | 5.0 / 2.5 | 2.6 / 1.2 | 0.8 / 0.4 |
| agent edit | 31.0 / 18.8 | 17.2 / 9.4 | 10.4 / 5.5 | 5.0 / 2.5 | 1.6 / 0.8 |
| Japanese | 11.1 / 5.4 | 4.8 / 2.3 | 3.1 / 1.5 | 1.5 / 0.7 | 0.4 / 0.2 |

## Open work

### Mac

1. **Spend the memory left at 262K on better weights.** iq3m8 plans 51.4
   of the 56 GiB. The plan was to spend about 3 GiB on IQ3_S tensors:
   - **Candidates:** routed down experts Q2_0 to IQ4_NL (0.22 GiB a
     layer), and the lowest gate/up expert layers one type up. IQ2_XXS
     layers moving up also get faster, onto the codebook grid kernel.
   - **Scoring:** the change in KL against IQ3_S and in NLL, with
     `DS4_QWEN_TENSOR_MIX`.
   - **Choosing:** greedy by quality per GiB, losing at most 1% of decode
     speed.
   - **Building:** `gguf_mix.py splice`.

   Check the room with `./dstar doctor` first. Not started.
2. **Time the current build** on a quiet machine and refresh the README
   tables.
3. **Make iq3m8 the default of `dstar pull`** on 64 GB Macs. Without
   `--quant` it still downloads q2.
4. **Merge upstream.** It has 319 commits not here, including the Neural
   Engine speed check for Qwen prefill (`b57ac99`) and `reasoning_effort`
   `no_think` (`98dfbeb`). Upstream keeps its sources in the root: git
   follows most of the moves, so expect conflicts in the Makefile and in
   files both sides changed.
5. **Deeper MTP.** A 3-row verify costs about 1.35x a 2-row one and commits
   2.8 tokens against about 1.9: only about 3% better per millisecond. An
   extra verify row costs about 8.75 ms. Its largest parts:
   - the routed experts, 2.7 ms;
   - the dense projections, 1.4 ms;
   - the draft head's argmax over the full vocabulary, 1.1 ms.

   Also open: exact fast 3-token kernels for IQ4_XS (it spills registers)
   and the K-quants, and per-row block universes so the attention core
   needs no 2+1 split. Untried: activations in half registers for the
   3-token kernels (this changes multi-token numerics), and a loop order
   that keeps decoded weights.
6. **Measure M1 and M2** on real machines.
7. **A prebuilt release.** Upstream's `install.sh` ships upstream's
   binaries, without these kernels.

### 32 to 48 GB

First move `stream-32gb` onto `max-64gb` ([below](#moving-stream-32gb-onto-max-64gb)).
Then the plan's phases:

1. **Streaming decode, correctness first.**
   - Use plain LRU for Qwen; the hotness policy stays for DeepSeek and GLM.
   - Pipeline the router round trip from the start. Encode and commit the
     experts, and the next layer up to its router, before the wait, behind
     an `MTLSharedEvent` that the I/O worker signals. Prototype this in
     `DS4_QWEN4_LAYER_SYNC` and measure its floor first.
   - Lift the Qwen refusal of `--ssd-streaming` on Metal only. Keep
     refusing it with batching and with `DS4_QWEN_TENSOR_MIX`.
   - Generalize the Metal expert cache to one slab set per (gate, up, down)
     size class; iq3m has about 8 classes. The Qwen expert kernels address
     experts through `selected`, so rewriting `selected` to cache slot ids
     needs no kernel change.
   - Map only the non-routed spans, and mlock them.
2. **Streaming prefill:** layer by layer with prepare-ahead page-in, seeding
   the decode cache as it goes.
3. **Hide the I/O.** Each lever stays only if measured faster:
   - next-layer prefetch;
   - pre-encoded command buffers;
   - hits before misses;
   - buffered against `F_NOCACHE` reads;
   - MTP under streaming.
4. **Quant per RAM class.** On 32 and 36 GB, moving all down experts to
   Q2_0 cuts misses by 20 to 25%, at a KL cost (layer 47 alone adds 0.07 on
   Japanese). On 48 GB, compare streamed iq3m with Q2_0 fully resident.
5. **M5 Neural Accelerator prefill tiles** for the ISTA types (IQ2_XS to
   IQ4_NL, Q2_0). This is independent of streaming, and speed needs a real
   M5.
6. **dstar and docs:**
   - a streaming plan below 60 GB, from the real GPU limit;
   - the cache share in `dstar doctor`;
   - plan tests for 32, 36 and 48 GB.

**Target Macs:** 32 GB M5 (10-core GPU), 32 GB M1 Pro, 36 GB M3 Max, 48 GB
M5 Pro and 48 GB M4 Pro.

**Expected speeds:**
- 30 tok/s on 48 GB, and perhaps on prose on the 36 GB M3 Max.
- 11 to 21 tok/s on the 32 GB M5s. Even fully resident they read 4.7 GB a
  token over 153 GB/s.

**CPU:** it does only I/O, polling and bookkeeping, because a busy
performance core slows the GPU.

**Hardware for testing:**
- Develop on a large Mac that emulates the smaller memory sizes.
- For real chips, Scaleway rents bare-metal M4 (32 GB) and M4 Pro (64 GB)
  Macs by the day. No cloud rents an M5.
- Avoid virtual Macs: they have no `iogpu` sysctl and their disks are
  virtual.

### V100

See the status page on `cuda-v100`.

## Moving stream-32gb onto max-64gb

This was done once as a dry run on 2026-11-22, and the result was thrown
away:

```sh
git switch stream-32gb
git rebase max-64gb
```

**The one conflict.** The rebase stops once, on the first commit
(`91c68fd`), in the Makefile. To resolve it:
- keep `max-64gb`'s lines (paths under `src/`, objects under
  `$(OBJDIR)`);
- add the `speed-bench/ssd_expert_read_bench` target and its
  `ssd-expert-read-bench` alias above `tests/test_mxfp4_metal.o`;
- add the binary to the `clean` list.

**The rest.** The other three commits apply cleanly; git carries their
`ds4.c` changes into `src/ds4.c`. The result builds without warnings.

**Not yet run with a model.** Before building on it, repeat the checks in
its commit messages:
- replaying the traces reproduces 99.99% of the expert selections;
- `DS4_QWEN4_LAYER_SYNC=1` leaves the tokens unchanged.

Don't merge the branch as it is: its Makefile and paths follow the old
layout.

## Testing

Before each commit on the Mac:

```sh
make
make test-quant-types test-qwen4-kernels test-qwen4-q2
DS4_TEST_MODEL=<Q2_0 shard 1> make test-qwen4-ik-ring test-qwen4-prompt-anchor
tests/test_dstar_plan.sh
./ds4_test --server
```

**Session tests.** With `DS4_TEST_MODEL` (and `DS4_TEST_MTP`) pointing at
Qwen3.8 files, also run `./ds4_test --session-rewind`, `--session-snapshot`
and `--qwen4-restore-reuse`. They need the same `DS4_QWEN_NGRAM_GGUF` that
`./dstar serve --dry-run` prints.

**Decode and MTP changes.** Greedy decoding (`--temp 0`) of a long agent
prompt must give byte-identical text with and without `--mtp`.

**Vision.** Run `make test-qwen4-vision-lazy`; it prints the variables it
needs.

**Other paths.** AGENT.md asks that the SSD streaming, CUDA and distributed
paths be checked whenever a change could touch them. `cuda-v100` holds the
V100 kernels, so check that it still builds after a change to shared code.

## Measuring

- **Emulating 64 GB.** The development Mac was an M3 Max with 128 GB. It
  emulated 64 GB with `--simulate-used-memory` plus the `iogpu` sysctl.
  Under the emulation the 28.8 GB n-gram table no longer fits the page
  cache, and SSD reads add noise. Time kernel changes without the emulation;
  run end-to-end checks under it separately.
- **Other apps.** Close every app that uses the GPU first: browsers,
  Discord, Lightroom, QuickTime, Music. They can halve decode speed.
- **Heat.** A laptop slows down within seconds of load. Alternate the
  variants (ABBA), take the best of several rounds, and compare ratios
  measured inside the same run. The first run of a series is slow.
- **Snapshot the build.** Metal sources load from the working directory at
  run time. Copy `./ds4` and `metal/` to a scratch folder before long A/B
  runs, so that edits made meanwhile don't leak into them.
- **English prompts.** On long English summarize or explain prompts, the
  model ends its turn at once (every quant does this). Time English decode
  with a short essay request instead.
- **MTP policies.** Greedy output is the same with and without MTP. So
  compare MTP depth policies by their mix of cycles, priced at costs
  measured in the same run, not by wall tok/s.
- **Timelines.** `DS4_METAL_ENCODER_TIMELINE` inflates the host gaps,
  because it timestamps every dispatch. For host time use
  `DS4_QWEN4_TIMING=1` and `DS4_QWEN4_SPEC_TIMING=1`. A depth-2 cycle
  spends only about 0.4 ms outside GPU work.
- **Kernels on cold weights:**
  - `QWEN4_COLD_BENCH` and `QWEN4_MOE_COLD_BENCH` in
    `tests/test_qwen4_kernels`;
  - `DS4_QUANT_BENCH=<type> DS4_QUANT_BENCH_T=<tokens>` in
    `tests/test_quant_types`;
  - `make qwen4-mv-bench`.

  The cold benches can disagree with the model, so confirm a kernel change
  with real decode.
- **GPU counters.** Use `xcrun xctrace record` with Metal System Trace and
  the Performance Limiters counter set. The counters are sampled over time
  (about 36 µs windows), not per kernel, so profile a microbenchmark loop
  rather than the model.
- **Quality:**
  - `DS4_QWEN4_FT_LIST` with `DS4_QWEN4_FT_REF=write`, then `=compare`,
    gives the KL against IQ3_S, top-1 agreement and NLL over six 2048-token
    sequences (two each of C code, English prose and Japanese prose).
  - `DS4_QWEN_TENSOR_MIX_GGUF` with `DS4_QWEN_TENSOR_MIX` scores a mix
    without writing a file.
  - `DS4_EXPERT_PROFILE` profiles the expert routing.

More switches are listed in
[The fork in detail](FLASH_NEXT_FORK.md#switches-for-measuring).

## Tried and rejected

On the M3 Max. Don't repeat any of these without a new idea:

- **Multi-token K-quant matvecs** in every form tried: the existing
  multi-token kernels, one row per simdgroup, token-outer with the weights
  read once, and weights unpacked to half2 once. All lose to running the
  one-token grid once per token, because these kernels are bound by
  per-thread latency and ALU.
- **One row per simdgroup at 3 tokens** for IQ4_XS and IQ4_NL: 20% slower.
- **Codebook tables in threadgroup memory** for the expert kernels: faster
  in the cold bench, 8 to 10% slower in the model.
- **More rows per simdgroup or threadgroup, and XOR sign flips** in the
  expert kernels: no gain.
- **Grouped codebook MoE for 2 and 3-row verifies:** it wins only when the
  rows share all their experts, and adjacent tokens share 30 to 48%.
- **4 rows per group in the fused hc norm and down projection** at one
  token: within the noise.
- **Overlapping host and GPU work, or periodic flushes:** there is no host
  time left to hide.
- **Forcing MTP depth 3:** it loses everywhere. The policy picks depth 3
  only on confident cycles.
- **4-bit hc mixers:** KL 0.2585 against IQ3_S, broken. Q8_0 is free.
- **Recall-based drafts:** declined by the owner.

## Not in the repository

- **The benchmark prompts** were kept in a temporary folder and are gone:
  - agent_8k, a coding-agent session of about 8K tokens;
  - ja_8k, Japanese prose;
  - en_essay, a short English essay request;
  - en5_4k, documentation plus a friendly question, with thinking on;
  - the six KL sequences.

  Rebuild a set before comparing with the numbers in commit messages.
  Absolute speeds won't carry over; ratios measured within one run will.
  `speed-bench/qwen38_quant_matrix.sh` expects them in one folder, as
  `<workload>.txt` and `<workload>_<size>.txt`.
- **The expert traces for the streaming simulator.** Regenerate them on
  `stream-32gb` with `DS4_EXPERT_TRACE=file`; add
  `DS4_EXPERT_TRACE_INPUT=1` to score prefetching.
- **The Instruments template used for GPU counters.** Make a new one from
  Metal System Trace with the Performance Limiters counter set.

## Commit rules

- No Co-Authored-By or other attribution trailers. The history was
  rewritten once to remove them.
- One change per commit, with what was measured and the numbers. `git log`
  shows the style.
- `.env` stays out of git; it is in `.gitignore`.
- Don't run two large model processes at once; the instance lock is
  intentional. Don't time anything while another heavy job runs.
