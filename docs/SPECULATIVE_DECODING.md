# Speculative Decoding

[README](../README.md)

Speculation proposes future tokens with a smaller draft block, then checks
them with the main model. An accepted prefix advances generation by several
tokens in one verification pass. It does not accelerate prefill.

It is opt-in. Gains depend on the prompt, model, backend, and context length;
poor acceptance can make it slower. Measure your workload rather than assuming
that a draft model always helps.

## DeepSeek Flash: DSpark

DSpark is a separate support GGUF, not a standalone language model. It proposes
up to five future tokens. Match its checkpoint to the main model:

| Main checkpoint | Download | Support file |
| --- | --- | --- |
| Flash 0731 | `ds4f-dspark` | `gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` |
| Flash Vision Experimental | `ds4f-vision-dspark` | `gguf/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf` |

For the 0731 Q2 model:

```sh
./scripts/download_model.sh ds4f-q2
./scripts/download_model.sh ds4f-dspark
./ds4 --dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

For Vision Experimental, substitute its matching main model and support file.
Do not mix the two checkpoints. DSpark is not supported for PRO.
The same flags work in `ds4-agent` and non-batched `ds4-server` requests.

The support file adds about 5.6 GiB of weights plus runtime state. On Metal,
the main model can be resident or SSD-streamed. DSpark replaces the legacy
one-stage MTP drafter for that run; the two are not stacked.

Resident M5 paths batch supported verifier expert rows, including two-Mac TP.
On DGX Spark, resident Q2 also batches the seed with longer drafts and uses
small-batch Q8 and expert kernels. No extra flags are needed.
The scheduler can back off when drafting is unproductive. Defaults select the
fast paths; diagnostic environment variables are not needed for normal use.
Recorded comparisons are in [the QA guide](QA_BEFORE_RELEASES.md).

For the tested Strix Halo coding configuration, use `--dspark --dspark-confidence 0.7` with the default five-token draft cap and scheduler. Client sampling is temperature `1.0`, `top_p=0.95`, `min_p=0`, and `top_k=0`; high reasoning was also checked on coding and tool-use requests. This uses opportunistic sampling as described below; exact-mode throughput is not qualified by these measurements. `--mtp-draft` controls legacy autoregressive MTP, not the DSpark draft width.

## GLM: built-in MTP

GLM's draft block is already in its main GGUF:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --mtp
```

`--mtp-timing` also enables it and prints acceptance and timing counters.
The current GLM cycle commits up to two tokens. No external support file is
needed, and ordinary decode remains the default.

## Qwen3.8: built-in MTP

Both Qwen downloads include MTP and native BF16 n-grams in the main GGUF:

```sh
./scripts/download_model.sh qwen38-q4k
./ds4 --mtp
```

Ordinary decode uses the same file with `--mtp` omitted. For non-zero
temperature, add `--mtp-exact-sampling` to preserve the target sampling
distribution. See [Qwen setup](QWEN38_FLASH_NEXT.md) for Metal and CUDA.

The cycle drafts one token ahead and can verify a **second, chained draft**
(one extra nextn-layer step conditioned on the predictor's own stream) in a
3-row pass. The depth is chosen per cycle to maximize committed tokens per
second: an action pays when its expected tokens beat what its time would
commit at the session's recent rate. Each draft comes with the predictor's
probability of it. The first draft's is calibrated (a draft of probability p
is accepted about p of the time); the chained draft's is overconfident (about
p2^1.5 after an accepted first draft), so each session calibrates it from its
own outcomes.

- The session measures a 3-row cycle and a chained draft against the 2-row
  cost of the same moment (1.37 and 0.044 of a 2-row cycle on an M3 Max at
  8K): the ratios hold while the absolute times drift.
- After the first draft, a second is chained when p1 times the chained
  drafts' acceptance at that p1 beats the rate times the extra verify time
  plus the chain, by 10%.
- Once chained, the cycle verifies three rows when p1 times the calibrated
  p2 beats the rate times the extra verify time, by 10% (the chain is paid
  for by then).
- The chained draft scores the head's first 98304 rows, the frequent tokens
  (`DS4_QWEN4_MTP_CHAIN_ROWS=N` changes it); the first draft and the verify
  rows use the whole head.

Greedy output is the same under every policy, so policies compare exactly by
their cycle mix priced at those costs. Against depth 2 on an M3 Max at 8K:
a copy-heavy edit +6.2%, a coding agent turn +0.5%, Japanese prose -0.1%
(depth 3 throughout: +6.3%, -5.7%, -10.3%).
`DS4_QWEN4_MTP_DEPTH=1`, `=2` or `=3` fixes the depth (1 decodes without
verifying drafts), `=window` keeps the window policy (three rows after eight
accepted first drafts in a row), `DS4_QWEN4_SPEC_POLICY=ratio` the previous
probability rule, and `DS4_QWEN4_SPEC_TRACE=1` prints every cycle's
probabilities and cost.

## Sampling and reproducibility

At temperature zero, accepted drafts must match the target's greedy
continuation. At non-zero temperature, the default mode is opportunistic:
ordinary tokens use the requested sampling settings, but matching greedy
drafts are accepted directly. Sampling resumes when the proposed suffix does
not match. This is deliberately more deterministic than ordinary sampling.

Use `--mtp-exact-sampling` to preserve the ordinary target sampling
distribution. Exact mode accepts greedy proposals with their target
probability and samples from the remaining distribution on rejection.

When a verified block crosses a tool sampling-mode boundary (for example
entering tool-call syntax during server decoding), the server rewinds to the
block start and re-evaluates the boundary token so the next sample uses the
new mode. Under exact sampling that rewind restores a pre-verify snapshot of
the recurrent state instead of resetting the graph, so long retained
contexts are not replayed at every boundary.

Accepted tokens keep the state produced by the batched verifier. Floating-point
reduction order can differ from one-token decode, so long greedy continuations
need not be byte-identical. For DeepSeek comparisons against the ordinary
target-only path, use `--quality` or `--dspark-strict`; these disable the
speculative acceptance path. They do not promise identical output across
different hardware or execution configurations.

Session-batched serving uses ordinary target decoding instead of combining
DSpark/MTP with the session batch. See [serving](SERVER.md#multiple-sessions).
