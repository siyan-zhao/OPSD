#!/usr/bin/env bash
set -euo pipefail

# Override these on the 5090 box if the paths differ:
#   MODEL_DIR=/data/models/Qwen3.5_2B OUTPUT_DIR=/data/opsd bash scripts/run_opsd_qwen35_2b_5090.sh
MODEL_DIR="${MODEL_DIR:-/Users/hyh/Desktop/Qwen3.5_2B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/opsd_qwen35_2b_5090}"
OPSD_DATASET="${OPSD_DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
OPSD_DATASET_SPLIT="${OPSD_DATASET_SPLIT:-train}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_5090_zero2.yaml}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12949}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RUN_CONFIG="${RUN_CONFIG:-${WANDB_RUN_NAME:-qwen35_2b_5090_lora_nonthink_topk256}}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-2.0}"
OPSD_LOSS_WEIGHT="${OPSD_LOSS_WEIGHT:-1.0}"
SFT_LOSS_WEIGHT="${SFT_LOSS_WEIGHT:-1.0}"
STUDENT_THINKING="${STUDENT_THINKING:-False}"
TEACHER_THINKING="${TEACHER_THINKING:-False}"
CLOSE_TEACHER_THINKING_BEFORE_SCORING="${CLOSE_TEACHER_THINKING_BEFORE_SCORING:-False}"
REASON_FIRST="${REASON_FIRST:-False}"
REAPPLY_CHAT_TEMPLATE_TO_INPUT="${REAPPLY_CHAT_TEMPLATE_TO_INPUT:-True}"
TEACHER_GUIDANCE_MODE="${TEACHER_GUIDANCE_MODE:-quality}"
CORRECTOR_MODE="${CORRECTOR_MODE:-False}"
DRAFT_FIELD="${DRAFT_FIELD:-sft_draft}"
TEACHER_DRAFT_FIELD="${TEACHER_DRAFT_FIELD:-}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
if [[ "$WANDB_MODE" == "disabled" ]]; then
    REPORT_TO="${REPORT_TO:-none}"
else
    REPORT_TO="${REPORT_TO:-wandb}"
fi
export WANDB_PROJECT
if [[ -n "$WANDB_ENTITY" ]]; then
    export WANDB_ENTITY
fi

if [[ "$WANDB_MODE" == "offline" || "$WANDB_MODE" == "disabled" ]]; then
    echo "[INFO] Running with WANDB_MODE=$WANDB_MODE"
elif [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[WARN] WANDB_MODE=$WANDB_MODE but WANDB_API_KEY is not set; wandb may prompt or fail."
elif command -v wandb >/dev/null 2>&1; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null
    echo "[INFO] W&B online logging enabled: project=$WANDB_PROJECT run=$RUN_CONFIG"
else
    echo "[WARN] wandb CLI not found; Python wandb will use WANDB_API_KEY if the package is installed."
fi
echo "[INFO] Trainer report_to=$REPORT_TO"
echo "[INFO] Student rollout: max_new_tokens=$MAX_COMPLETION_LENGTH temperature=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K presence_penalty=$PRESENCE_PENALTY"
echo "[INFO] Loss weights: opsd=$OPSD_LOSS_WEIGHT sft=$SFT_LOSS_WEIGHT"
echo "[INFO] Teacher guidance mode: $TEACHER_GUIDANCE_MODE"
echo "[INFO] Corrector mode: $CORRECTOR_MODE draft_field=$DRAFT_FIELD"
echo "[INFO] Teacher-only draft field: ${TEACHER_DRAFT_FIELD:-<disabled>}"
echo "[INFO] LoRA: r=$LORA_R alpha=$LORA_ALPHA"

EXTRA_TRAINING_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
    EXTRA_TRAINING_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    EXTRA_TRAINING_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
case "$GRADIENT_CHECKPOINTING" in
    True|true|TRUE|1|yes|YES|y|Y)
        EXTRA_TRAINING_ARGS+=(--gradient_checkpointing)
        ;;
esac

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    opsd_train.py \
    --model_name_or_path "$MODEL_DIR" \
    --model_loader image_text_to_text \
    --opsd_dataset "$OPSD_DATASET" \
    --opsd_dataset_split "$OPSD_DATASET_SPLIT" \
    --input_field "${INPUT_FIELD:-input}" \
    --output_field "${OUTPUT_FIELD:-output}" \
    --draft_field "$DRAFT_FIELD" \
    --corrector_mode "$CORRECTOR_MODE" \
    --teacher_draft_field "$TEACHER_DRAFT_FIELD" \
    --problem_field "${PROBLEM_FIELD:-problem}" \
    --solution_field "${SOLUTION_FIELD:-solution}" \
    --learning_rate "${LEARNING_RATE:-5e-6}" \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-30}" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --save_steps "${SAVE_STEPS:-25}" \
    --logging_steps "${LOGGING_STEPS:-2}" \
    --report_to "$REPORT_TO" \
    --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
    --torch_dtype bfloat16 \
    --max_length "$MAX_LENGTH" \
    --beta 0 \
    --use_peft \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules \
        q_proj k_proj v_proj o_proj \
        in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj \
        gate_proj up_proj down_proj \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --presence_penalty "$PRESENCE_PENALTY" \
    --opsd_loss_weight "$OPSD_LOSS_WEIGHT" \
    --sft_loss_weight "$SFT_LOSS_WEIGHT" \
    --lmbda 1 \
    --fixed_teacher \
    --reason_first "$REASON_FIRST" \
    --student_thinking "$STUDENT_THINKING" \
    --teacher_thinking "$TEACHER_THINKING" \
    --close_teacher_thinking_before_scoring "$CLOSE_TEACHER_THINKING_BEFORE_SCORING" \
    --reapply_chat_template_to_input "$REAPPLY_CHAT_TEMPLATE_TO_INPUT" \
    --teacher_guidance_mode "$TEACHER_GUIDANCE_MODE" \
    --top_k_loss "${TOP_K_LOSS:-256}" \
    --jsd_token_clip "${JSD_TOKEN_CLIP:-1e-6}" \
    --wandb_project "$WANDB_PROJECT" \
    "${EXTRA_TRAINING_ARGS[@]}"
