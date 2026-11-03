# Qwen3.8-Flash-Next on a 64 GB Mac

Run **Qwen3.8-Flash-Next** on a 64 GB M1 to M3 Max with the **full 262K
context**, MTP speculative decoding, vision and Neural Engine prefill, as a
local OpenAI/Anthropic compatible server for OpenCode, Claude Code, DSH, Pi
or any other agent.

It is built around one file, **iq3m8**
([IQ3_XXS_M64_HC8 on Hugging Face](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-GSQ-RCO-M64-GGUF)):
ISTA-DASLab's GSQ-RCO IQ3_XXS quant, reshaped to fit 64 GB with room to
spare and to decode faster on the M3 Max, at the same quality.

This repository is a fork of [paperniuk/ds4](https://github.com/paperniuk/ds4),
the Flash-Next fork of [antirez/ds4](https://github.com/antirez/ds4)
(DwarfStar). Upstream's prebuilt release does not have iq3m8 or the kernels
that make it fast; build this one from source.

## Quick start

```sh
git clone https://github.com/pentaCoxian/ds4-flash-next.git
cd ds4-flash-next && make
./dstar doctor                  # what your Mac fits: memory, files, contexts
./dstar pull --quant iq3m8      # 76 GB, resumable
sudo sysctl iogpu.wired_limit_mb=57344   # once per boot, for 262K
./dstar serve                   # http://127.0.0.1:8010/v1
```

Open `http://127.0.0.1:8010/dashboard` to watch prefill, decode, MTP and GPU
memory live. Then point your agent at the server:

```sh
./dstar opencode   # provider block for OpenCode
./dstar dsh        # model route for the DeepSeek Harness (dsh)
./dstar link       # make dstar a command that works from any directory
```

[Coding agent clients](docs/CLIENTS.md) covers Claude Code, Codex CLI and Pi
too; use port 8010 and the context the server started with.

## The model

iq3m8 is two changes to ISTA-DASLab's
[GSQ-RCO IQ3_XXS](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):

1. **The routed down experts of layers 29, 30, 31 and 35 come from ISTA's
   Q2_0 file**, copied byte for byte. Of the 18 layers IQ3_XXS keeps in
   IQ4_NL, these four moved the output least when swapped. That saves 0.9
   GiB, which on 64 GB makes room at 262K for the Neural Engine. This step
   alone is `iq3m`.
2. **The 194 low-rank hyper-connection mixer projections are Q8_0 instead of
   BF16.** They are 1.2 of the 4.3 GiB a decode token reads, and with this
   fork's Q8_0 kernels the token takes 20.6 ms of GPU time instead of 22.4.

KL divergence against IQ3_S, ISTA's best file, over 2048-token sequences
of C code, English prose and Japanese prose:

| File | Weights | Mean KL vs IQ3_S | Top-1 agreement |
|---|---|---|---|
| Q2_0 (ISTA) | 35.0 GiB | 0.438 | 78.9% |
| IQ3_XXS (ISTA) | 43.8 GiB | 0.1834 | 85.7% |
| iq3m | 42.9 GiB | 0.1835 | 85.5% |
| **iq3m8** | **42.4 GiB** | **0.1795** | **85.4%** |

ISTA's LiveCodeBench score for IQ3_XXS is 86.3 (Q2_0 81.1, IQ3_S 86.9).
No benchmarks were run on iq3m or iq3m8; by KL they sit with IQ3_XXS, well
above Q2_0, and the gap is largest in Japanese.

### Speed

M3 Max, 40-core GPU, 40K context, 8K prompts (a coding agent / Japanese
prose), 256 greedy tokens, best of three alternating rounds:

| File | Decode | Decode with MTP | Prefill |
|---|---|---|---|
| iq3m | 43.9 / 42.0 tok/s | 50.6 / 50.9 tok/s | 497 / 485 tok/s |
| **iq3m8** | **46.1 / 47.8 tok/s** | **53.0 / 52.4 tok/s** | **489 / 500 tok/s** |

Measured with the build of 2026-11-01; the decode changes since then are
not in these numbers. At 262K under an emulation of a 64 GB Mac, iq3m
prefilled 585 to 607 tok/s with the Neural Engine against 450 to 505 for
IQ3_XXS, which has no room for it there. Greedy output is the same with
and without MTP. Only the M3 Max has been measured with iq3m8; M1 and M2
run the same Q8_0 kernels, untimed.

### Memory

At 262K with MTP and the Neural Engine, ds4 plans 51.4 GiB: the weights and
MTP block 43.8, the 8-bit KV cache 3.7, the prefill buffers 3.1 and the
Neural Engine 0.9. The vision encoder maps only while it encodes an image.

macOS lets the GPU use three quarters of the memory by default, 48 GiB of
64. That holds iq3m8 at 32K only. `dstar serve` takes the largest context
that fits and prints the line that unlocks more:

| GPU limit | iq3m8 runs with |
|---|---|
| default (48 GiB) | 32K |
| `sudo sysctl iogpu.wired_limit_mb=53248` | 262K, without the Neural Engine |
| `sudo sysctl iogpu.wired_limit_mb=57344` | 262K, with the Neural Engine (prefill about a quarter faster) |

The setting is lost at every reboot. At 57344 the system keeps 8 GB, so
close the browser and other GPU-heavy apps first.

## Requirements

- An Apple Silicon Mac with 64 GB or more; tuned for the M1 to M3 Max.
- 76 GB of disk: 45.5 GB for iq3m8, 28.8 GB for ISTA's n-gram table, 1.5
  GB for the MTP block and 0.6 GB for the vision encoder. A fast internal
  SSD, because the n-gram table is read from disk on every token.
- Below 64 GB nothing runs yet: Qwen3.8 has no SSD streaming in ds4.
  paperniuk's [Splash M1 port](https://github.com/paperniuk/splash) with
  Qwen3.8-27B (21 GB) is the choice there.

## Using it

`dstar serve` picks the context that fits the memory of the Mac and the GPU
limit, and turns on MTP, vision and the Neural Engine when they fit. Once
iq3m8 is on disk, it is what `dstar serve` runs.

| Command | What it does |
|---|---|
| `dstar pull --quant iq3m8` | download iq3m8, the n-gram table, the MTP block and the vision encoder into `~/models/flash-next`, resumable; `--no-vision` skips the encoder |
| `dstar serve` | start the server; `--ctx 131k/262k/400k/524k`, `--port N`, `--lan`, `--no-mtp`, `--no-vision`, `--no-ane`, `--power N` |
| `dstar serve --dry-run` | print the `ds4-server` command and environment instead of running it |
| `dstar chat` | talk to the model in the terminal |
| `dstar doctor` | check the machine, the files, the GPU limit and which contexts fit |
| `dstar opencode` | print the provider block for `~/.config/opencode/opencode.json` |
| `dstar dsh` | print the model route for a DeepSeek Harness profile ([details](docs/CLIENTS.md#deepseek-harness-dsh)) |
| `dstar link` | link the launcher into `~/.local/bin` |

`--power N` keeps the GPU busy N percent of the time, for a cooler and
quieter Mac; the output does not change. `DSTAR_MODELS` changes the model
directory, `PORT` and `HOST` the address. After a `git pull`, run `make` and
restart the server.

Tips for agents:

- Pick the reasoning effort once per session. Qwen3.8 writes it at the top of
  the system prompt, so switching it mid-session prefills the whole
  conversation again.
- The server keeps one live conversation. Switching to another one saves the
  current state to the disk cache, and switching back restores it in about a
  second. A retried or edited turn reuses the cache instead of prefilling the
  prompt again.

## Other quants and models

`--quant` in `dstar pull` and `dstar serve` picks another file:

| `--quant` | File | Weights | When |
|---|---|---|---|
| `iq3m8` | this README | 42.4 GiB | 64 GB, with this build |
| `iq3m` | iq3m8 with BF16 mixers ([Hugging Face](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64-GGUF)) | 42.9 GiB | with upstream ds4, which has no fast Q8_0 mixer kernels |
| `q2` | ISTA Q2_0 | 35.0 GiB | 64 GB at the default GPU limit, or contexts to 524K; about 10% faster than iq3m, clearly worse answers |
| `iq3` | ISTA IQ3_XXS | 43.8 GiB | 262K without the Neural Engine on 64 GB |
| `iq3s` | ISTA IQ3_S, the best file | 51.0 GiB | 96 GB or more |

Without `--quant`, `dstar pull` downloads `q2`. The launcher also runs
DeepSeek V4 and V4.1 Flash (streamed from the SSD on 64 GB), GLM and the
upstream Qwen3.8 files: `dstar models` lists them, `dstar pull NAME` and
`dstar serve NAME` run one. See [The fork in detail](docs/FLASH_NEXT_FORK.md)
for their numbers and [Models](docs/MODELS.md) for what fits where.

## Status

Experimental, one developer, measured on one M3 Max. Everything else in ds4
(other models, CUDA, distributed inference, SSD streaming) is inherited from
upstream and should keep working, but is not the focus here.

The weights are under the Qwen Community License 1.0; read it before any
commercial use. The code is MIT, like upstream.

## More

- [The fork in detail](docs/FLASH_NEXT_FORK.md): what changed, which Macs gain, the larger quants, contexts beyond 262K, running `ds4-server` by hand.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [NVIDIA V100](docs/CUDA_V100.md): iq3m8 on two 32 GB V100s, its speeds, switches, and how the CUDA path gets there.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): Qwen MTP, the depth policy, and sampling.
- [Serving](docs/SERVER.md): APIs, the live dashboard, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): OpenCode, DSH, Pi, Codex CLI, and Claude Code.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.
- [The upstream README](https://github.com/antirez/ds4#readme): what DwarfStar is, its other models and hardware.
