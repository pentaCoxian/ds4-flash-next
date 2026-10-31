#!/usr/bin/env bash
# A/B of Qwen3.8 kernel knobs as on a 64 GB Mac: agent-turn prefill (512-token
# suffixes on a warm 30K context), long prefill, plain and MTP decode.
#
#   speed-bench/qwen38_knob_sweep.sh OUT_DIR PROMPT_DIR QUANT 'NAME ENV=V...' ...
#
# Each argument after QUANT is one variant: a name and the environment it
# adds ('base' with no assignments is the reference). Variants run in turn,
# PAIRS times over (ABAB...), so drift of the machine spreads over all of
# them. PROMPT_DIR holds agent.txt (corpus) and agent_8k.txt (raw prompt).
# Environment: DSTAR_MODELS, RAM_GB (64), PAIRS (3), DS4_ROOT (frozen copy
# of ds4, ds4-bench and metal/).
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "${DS4_ROOT:-$REPO}"
[ $# -ge 4 ] || { sed -n '2,13p' "$0"; exit 2; }
OUT=$1 PROMPTS=$2 Q=$3
shift 3
MODELS=${DSTAR_MODELS:-$HOME/models/flash-next}
RAM_GB=${RAM_GB:-64}
PAIRS=${PAIRS:-3}
mkdir -p "$OUT"
REAL_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
SIM=()
[ "$REAL_GB" -gt "$RAM_GB" ] && SIM=(--simulate-used-memory "$(( REAL_GB - RAM_GB ))GB")
S1=$MODELS/$Q/Qwen3.8-Flash-Next-GSQ-RCO-$Q-00001-of-00002.gguf
NGRAM=$(ls "$MODELS"/*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf | head -1)
MTP=$(ls "$MODELS"/Qwen3.8-Flash-Next-MTP-block.gguf | head -1)
RES=$OUT/sweep_$Q.tsv
[ -f "$RES" ] || echo -e "variant\trun\tturn_prefill_tps\tlong_prefill_tps\tplain_tps\tmtp_tps\tmtp_accept" > "$RES"

for run in $(seq 1 "$PAIRS"); do
    for variant in "$@"; do
        read -r name envs <<<"$variant"
        base=(env $envs DS4_QWEN_NGRAM_GGUF="$NGRAM" DS4_QWEN4_MOE_TAILS=1)
        log=$OUT/${Q}_${name}_$run
        "${base[@]}" ./ds4-bench -m "$S1" --prompt-file "$PROMPTS/agent.txt" --ctx-start 30720 --ctx-max 32768 \
            --step-incr 512 --gen-tokens 0 --prefill-chunk 2048 ${SIM[@]+"${SIM[@]}"} --csv "$log.turn.csv" > "$log.turn.log" 2>&1
        turn=$(awk -F, 'NR > 2 { s += $3; n++ } END { if (n) printf "%.1f", s / n }' "$log.turn.csv")
        long=$(awk -F, 'NR == 2 { printf "%.1f", $3 }' "$log.turn.csv")
        "${base[@]}" ./ds4 -m "$S1" --raw-prompt --prompt-file "$PROMPTS/agent_8k.txt" --ctx 16384 --prefill-chunk 2048 \
            --temp 0 -n 256 ${SIM[@]+"${SIM[@]}"} > /dev/null 2> "$log.plain.log"
        plain=$(sed -nE 's/.*generation: ([0-9.]+) t\/s.*/\1/p' "$log.plain.log" | tail -1)
        "${base[@]}" DS4_QWEN_MTP_GGUF="$MTP" ./ds4 -m "$S1" --raw-prompt --prompt-file "$PROMPTS/agent_8k.txt" \
            --ctx 16384 --prefill-chunk 2048 --temp 0 -n 256 --mtp --mtp-timing ${SIM[@]+"${SIM[@]}"} > /dev/null 2> "$log.mtp.log"
        mtp=$(sed -nE 's/.*generation: ([0-9.]+) t\/s.*/\1/p' "$log.mtp.log" | tail -1)
        acc=$(sed -nE 's/.*drafts accepted \(([0-9.]+)%\).*/\1/p' "$log.mtp.log" | tail -1)
        echo -e "$name\t$run\t$turn\t$long\t$plain\t$mtp\t$acc" >> "$RES"
        echo "$name run $run: turn $turn long $long plain $plain mtp $mtp ($acc%)"
    done
done
