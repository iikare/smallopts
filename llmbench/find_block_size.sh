#!/usr/bin/env bash
set -euo pipefail

MODELS=(
    "RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic"
    "RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8"
    "RedHatAI/Qwen3.5-9B-FP8-dynamic"
)

OUTPUT="block_size.txt"
> "$OUTPUT"

export HF_HOME="/hweng/ebay"

source bench_venv/bin/activate 2>/dev/null || source venv/bin/activate

for MODEL in "${MODELS[@]}"; do
    echo "=== $MODEL ===" | tee -a "$OUTPUT"

    LOG=$(mktemp)

    vllm serve "$MODEL" \
        --mamba-cache-mode align --enable-prefix-caching \
        --max-model-len 8192 --gpu-memory-utilization 0.5 \
        --port 8011 > "$LOG" 2>&1 &
    VLLM_PID=$!

    until grep -qiE "Setting attention block size|Error|Traceback" "$LOG"; do
        sleep 3
    done

    grep -i "Setting attention block size" "$LOG" | tee -a "$OUTPUT"

    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    rm -f "$LOG"

    echo "" | tee -a "$OUTPUT"
done

echo "Done. Results saved to $OUTPUT"

