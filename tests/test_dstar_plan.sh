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

REAL_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))

# serve NAME ENV... -- ARGS: dry-run output (stdout and stderr) in $OUT
serve() {
    OUT=$(env DSTAR_MODELS="$MODELS" "$@" ./dstar serve --dry-run 2>&1)
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

# A 64 GB Mac at its default limit (three quarters of the RAM) holds 131K.
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=0
expect "64 GB default limit picks 131K" "--ctx 131072 "
expect "64 GB default limit says how to reach 262K" "sudo sysctl iogpu.wired_limit_mb="

# The raised limit of the README holds the native window.
serve DSTAR_RAM_GB=64 DSTAR_WIRED_LIMIT_MB=57344
expect "64 GB at 57344 MiB picks 262K" "--ctx 262144 "

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
