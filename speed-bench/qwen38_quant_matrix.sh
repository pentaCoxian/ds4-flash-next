#!/usr/bin/env bash
# Qwen3.8-Flash-Next speed per quant as on a 64 GB Mac.
#
#   speed-bench/qwen38_quant_matrix.sh OUT_DIR PROMPT_DIR [Q2_0 IQ3_XXS IQ3_S]
#
# For each quant: ds4-bench prefill and teacher-forced decode at context
# frontiers of every corpus, then plain and MTP decode of every raw prompt,
# each with the peak wired memory of the run. PROMPT_DIR holds the corpora
# <workload>.txt and the prompts <workload>_<size>.txt.
#
# Environment: DSTAR_MODELS (~/models/flash-next), RAM_GB (64: the rest of
# the RAM is locked with --simulate-used-memory), CTX_MAX (131072), SIZES
# ("8k 32k"), WORKLOADS ("agent en ja"), GEN (256), CTX_ALLOC (context of
# the decode runs, 262144), EXTRA_ENV (assignments added to every run, for
# A/B variants), TAG (suffix of the result files), BENCH (0 skips the
# ds4-bench sweeps), DS4_ROOT (the directory
# with ds4, ds4-bench and metal/ to run: a frozen copy keeps rebuilds out of
# a long run).
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${DS4_ROOT:-$REPO}"
[ $# -ge 2 ] || { sed -n '2,15p' "$0"; exit 2; }
OUT=$1 PROMPTS=$2
shift 2
QUANTS=${*:-Q2_0 IQ3_XXS IQ3_S}
MODELS=${DSTAR_MODELS:-$HOME/models/flash-next}
RAM_GB=${RAM_GB:-64}
CTX_MAX=${CTX_MAX:-131072}
SIZES=${SIZES:-8k 32k}
WORKLOADS=${WORKLOADS:-agent en ja}
GEN=${GEN:-256}
CTX_ALLOC=${CTX_ALLOC:-262144}
EXTRA_ENV=${EXTRA_ENV:-}
TAG=${TAG:+_$TAG}
mkdir -p "$OUT"

REAL_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
SIM=()
[ "$REAL_GB" -gt "$RAM_GB" ] && SIM=(--simulate-used-memory "$(( REAL_GB - RAM_GB ))GB")
NGRAM=$(ls "$MODELS"/*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf 2>/dev/null | head -1)
MTP=$(ls "$MODELS"/Qwen3.8-Flash-Next-MTP-block.gguf "$MODELS"/mtp/Qwen3.8-Flash-Next-MTP-block.gguf 2>/dev/null | head -1)
[ -n "$NGRAM" ] || { echo "no n-gram shard under $MODELS" >&2; exit 1; }

# Peak of "Pages wired down" while the command runs, in GiB above the start.
with_wired_peak() {
    local log=$1; shift
    local page base peak=0 w pid
    page=$(sysctl -n hw.pagesize)
    base=$(vm_stat | awk '/wired down/ { gsub(/\./, "", $4); print $4 }')
    "$@" > "$log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        w=$(vm_stat | awk '/wired down/ { gsub(/\./, "", $4); print $4 }')
        [ "$w" -gt "$peak" ] && peak=$w
        sleep 1
    done
    wait "$pid"
    local rc=$?
    awk -v p="$peak" -v b="$base" -v s="$page" 'BEGIN { printf "peak_wired_gib %.2f\n", (p - b) * s / 1073741824 }' >> "$log"
    return $rc
}

meta() {
    local commit
    commit=$(cat COMMIT 2>/dev/null | head -1 || true)
    [ -n "$commit" ] || commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
    echo "# $(date -u +%FT%TZ) commit $commit" \
         "chip $(sysctl -n machdep.cpu.brand_string) macOS $(sw_vers -productVersion)" \
         "wired_limit $(sysctl -n iogpu.wired_limit_mb) ram_plan ${RAM_GB}GB env: $EXTRA_ENV"
}

for q in $QUANTS; do
    S1=$MODELS/$q/Qwen3.8-Flash-Next-GSQ-RCO-$q-00001-of-00002.gguf
    [ -f "$S1" ] || { echo "skip $q: no $S1" >&2; continue; }
    RES=$OUT/$q$TAG.tsv
    meta > "$RES"
    echo -e "kind\tworkload\tsize\tprompt_tps\tgen_tps\tmtp_cycles\tmtp_accept_pct\tpeak_wired_gib" >> "$RES"

    for w in $WORKLOADS; do
        [ "${BENCH:-1}" = 1 ] && [ -f "$PROMPTS/$w.txt" ] || continue
        log=$OUT/$q${TAG}_bench_$w.log
        with_wired_peak "$log" env $EXTRA_ENV DS4_QWEN_NGRAM_GGUF="$NGRAM" DS4_QWEN4_MOE_TAILS=1 \
            ./ds4-bench -m "$S1" --prompt-file "$PROMPTS/$w.txt" --ctx-start 4096 --ctx-max "$CTX_MAX" \
            --step-mul 2 --gen-tokens 128 --teacher-forced-decode --prefill-chunk 2048 ${SIM[@]+"${SIM[@]}"} \
            --csv "$OUT/$q${TAG}_bench_$w.csv"
        echo "$q $w bench: $(tail -1 "$log")"
    done

    for w in $WORKLOADS; do
        for sz in $SIZES; do
            p=$PROMPTS/${w}_$sz.txt
            [ -f "$p" ] || continue
            for mode in plain mtp; do
                [ "$mode" = mtp ] && [ -z "$MTP" ] && continue
                log=$OUT/$q${TAG}_${mode}_${w}_$sz.log
                args=()
                envs=(DS4_QWEN_NGRAM_GGUF="$NGRAM" DS4_QWEN4_MOE_TAILS=1)
                [ "$mode" = mtp ] && { args=(--mtp --mtp-timing); envs+=(DS4_QWEN_MTP_GGUF="$MTP"); }
                with_wired_peak "$log" env $EXTRA_ENV "${envs[@]}" ./ds4 -m "$S1" --raw-prompt --prompt-file "$p" \
                    --ctx "$CTX_ALLOC" --prefill-chunk 2048 --temp 0 -n "$GEN" ${args[@]+"${args[@]}"} ${SIM[@]+"${SIM[@]}"}
                pt=$(sed -nE 's/.*prefill: ([0-9.]+) t\/s, generation: ([0-9.]+) t\/s.*/\1/p' "$log" | tail -1)
                gt=$(sed -nE 's/.*prefill: ([0-9.]+) t\/s, generation: ([0-9.]+) t\/s.*/\2/p' "$log" | tail -1)
                cy=$(sed -nE 's/.*mtp: ([0-9]+) verify cycles.*/\1/p' "$log" | tail -1)
                ac=$(sed -nE 's/.*drafts accepted \(([0-9.]+)%\).*/\1/p' "$log" | tail -1)
                pk=$(sed -nE 's/^peak_wired_gib (.*)/\1/p' "$log")
                echo -e "$mode\t$w\t$sz\t${pt:-}\t${gt:-}\t${cy:-}\t${ac:-}\t$pk" >> "$RES"
                echo "$q $mode $w $sz: prefill ${pt:-?} gen ${gt:-?} accept ${ac:--} wired $pk"
            done
        done
    done
done
