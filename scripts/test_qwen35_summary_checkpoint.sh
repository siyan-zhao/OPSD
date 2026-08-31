#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/DATA_A/models/Qwen3.5-2B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/DATA_B/hyh/opsd_outputs/summary_segment_0706/qwen3.5_2b_opsd_summary_segment_v1/checkpoint-500}"
DATA_FILE="${DATA_FILE:-/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${CHECKPOINT_DIR%/}/summary_eval.jsonl}"
METRICS_FILE="${METRICS_FILE:-${OUTPUT_FILE%.*}_metrics.json}"

NUM_SAMPLES="${NUM_SAMPLES:-8}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"
PRINT_LIMIT="${PRINT_LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES

ARGS=(
    --model "$MODEL_DIR"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --data_file "$DATA_FILE"
    --output_file "$OUTPUT_FILE"
    --metrics_file "$METRICS_FILE"
    --num_samples "$NUM_SAMPLES"
    --batch_size "$BATCH_SIZE"
    --enable_thinking False
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
)

if [[ -n "$PRINT_LIMIT" ]]; then
    ARGS+=(--print_limit "$PRINT_LIMIT")
fi

if [[ -n "$SAMPLE_INDICES" ]]; then
    ARGS+=(--sample_indices "$SAMPLE_INDICES")
fi

python scripts/evaluate_qwen35_summary_torch.py "${ARGS[@]}"
