#!/usr/bin/env bash
# dstar memory plan: the context it picks and the memory it holds back when
# planning for a smaller Mac. Uses empty placeholder model files and
# --dry-run, so no model and no GPU are needed. Run from the repo root.
set -uo pipefail

cd "$(dirname "$0")/.."
[ "$(uname -s)" = Darwin ] || { echo "skip dstar plan: not macOS"; exit 0; }

PASS=0
FAIL=0
MODELS=$(mktemp -d)
trap 'rm -rf "$MODELS"' EXIT

ok()   { PASS=$((PASS+1)); echo "ok $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

for t in Q2_0 IQ3_XXS; do
    mkdir -p "$MODELS/$t"
    for s in 00001 00002; do : > "$MODELS/$t/Qwen3.8-Flash-Next-GSQ-RCO-$t-$s-of-00002.gguf"; done
done
: > "$MODELS/Qwen3.8-Flash-Next-MTP-block.gguf"

REAL_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))

# serve ENV... [-- ARGS]: dry-run output (stdout and stderr) in $OUT
serve() {
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    OUT=$(env DSTAR_MODELS="$MODELS" "${envs[@]}" ./dstar serve --dry-run "$@" 2>&1)
}

expect() {
    # $1 = name, $2 = pattern that must be in $OUT
    if grep -q -- "$2" <<<"$OUT"; then ok "$1"; else
        fail "$1 (missing: $2)"
        sed 's/^/    /' <<<"$OUT"
    fi
}

reject() {
    if grep -q -- "$2" <<<"$OUT"; then fail "$1 (unexpected: $2)"; sed 's/^/    /' <<<"$OUT"; else ok "$1"; fi
}

# A 64 GB Mac at its default limit (three quarters of the RAM) holds q2 at
# 262K in 2048-row chunks with the Neural Engine (8-bit K/V rows); iq3 needs
# the limit raised.
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=0
expect "64 GB default limit holds q2 at 262K" "--ctx 262144 "
expect "64 GB default limit prefills q2 in 2048 rows" "--prefill-chunk 2048 "
expect "64 GB default limit runs q2 with the Neural Engine" "DS4_ANE=1"
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=0 DSTAR_QUANT=iq3
expect "64 GB default limit says how to fit iq3" "sudo sysctl iogpu.wired_limit_mb="

# The raised limit of the README holds the native window, with 2048-row
# prefill chunks and the Neural Engine for q2.
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=57344
expect "64 GB at 57344 MiB picks 262K" "--ctx 262144 "
expect "q2 at 262K prefills 2048 rows" "--prefill-chunk 2048 "
expect "q2 at 262K uses the Neural Engine" "DS4_ANE=1"
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=57344 -- --no-ane
reject "--no-ane keeps the Neural Engine off" "DS4_ANE=1"

# iq3 holds 262K too, now in 2048-row chunks with the Neural Engine.
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=57344 DSTAR_QUANT=iq3
expect "iq3 at 57344 MiB picks 262K" "--ctx 262144 "
expect "iq3 at 262K prefills 2048 rows" "--prefill-chunk 2048 "
expect "iq3 at 262K uses the Neural Engine" "DS4_ANE=1"
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=57344 DSTAR_QUANT=iq3 -- --ctx 131k
expect "iq3 at 131K uses the Neural Engine" "DS4_ANE=1"

# iq3m, the mix, makes room at 262K for the Neural Engine, and a plain
# serve prefers it once it is on disk.
mkdir -p "$MODELS/IQ3_XXS_M64"
: > "$MODELS/IQ3_XXS_M64/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64-00001-of-00002.gguf"
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=57344
expect "a built iq3m is the default" "IQ3_XXS_M64-00001-of-00002.gguf"
expect "iq3m at 57344 MiB picks 262K" "--ctx 262144 "
expect "iq3m at 262K uses the Neural Engine" "DS4_ANE=1"
expect "iq3m at 262K prefills 2048 rows" "--prefill-chunk 2048 "

# iq3m8, iq3m with 8-bit mixers, wins over iq3m once it is on disk and
# fits the same plan; the n-gram shard of any quant serves it.
mkdir -p "$MODELS/IQ3_XXS_M64_HC8"
: > "$MODELS/IQ3_XXS_M64_HC8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64_HC8-00001-of-00002.gguf"
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=57344
expect "a built iq3m8 is the default" "IQ3_XXS_M64_HC8-00001-of-00002.gguf"
expect "iq3m8 at 57344 MiB picks 262K" "--ctx 262144 "
expect "iq3m8 at 262K uses the Neural Engine" "DS4_ANE=1"
expect "iq3m8 reads a shared n-gram shard" "DS4_QWEN_NGRAM_GGUF=.*-00002-of-00002.gguf"
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=57344 DSTAR_QUANT=iq3m
expect "DSTAR_QUANT=iq3m still picks iq3m" "IQ3_XXS_M64-00001-of-00002.gguf"
rm -r "$MODELS/IQ3_XXS_M64_HC8"
rm -r "$MODELS/IQ3_XXS_M64"

if [ "$REAL_GB" -gt 64 ]; then
    expect "a larger Mac locks the difference" "--simulate-used-memory $(( REAL_GB - 64 ))GB"
else
    reject "a 64 GB Mac locks nothing" "--simulate-used-memory"
fi

# Without the override the plan is for the real machine.
serve DSTAR_WIRED_LIMIT_MB=57344
reject "no override locks nothing" "--simulate-used-memory"

# An override above the real RAM is ignored.
serve DSTAR_RAM_GB=$(( REAL_GB + 64 )) DSTAR_WIRED_LIMIT_MB=57344
reject "a larger override locks nothing" "--simulate-used-memory"

echo "dstar plan: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
