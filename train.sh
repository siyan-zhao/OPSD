#!/usr/bin/env bash
set -euo pipefail

cd /DATA_B/hyh/OPSD
source .venv/bin/activate

unset WANDB_MODE
unset MAX_STEPS

export WANDB_API_KEY=""
export WANDB_PROJECT="qwen3.5_2b_opsd_summary_segment_0715_teacher_draft"
export WANDB_RUN_NAME="summary_segment_teacher_only_sft_draft_r16_1epoch_v1"
export RUN_CONFIG="summary_segment_teacher_only_sft_draft_r16_1epoch_v1"

export ACCELERATE_CONFIG="accelerate_5090_zero2.yaml"

export STUDENT_THINKING="False"
export TEACHER_THINKING="False"
export CLOSE_TEACHER_THINKING_BEFORE_SCORING="False"
export REASON_FIRST="False"

# Teacher-only SFT baseline：student 仍然只看到 input，teacher 私有看到 SFT 初稿和标准答案。
export CORRECTOR_MODE="False"
export DRAFT_FIELD="sft_draft"
export TEACHER_DRAFT_FIELD="sft_draft"

# teacher 用质量偏好，不强制逐字复制参考答案。
export TEACHER_GUIDANCE_MODE="quality"

# 保留少量 SFT/EOS 锚点，OPSD 负责主要 teacher-guided 质量纠偏。
export SFT_LOSS_WEIGHT="0.1"
export OPSD_LOSS_WEIGHT="1.0"

# LoRA rank 16。
export LORA_R="16"
export LORA_ALPHA="32"

export MAX_LENGTH="4096"
export MAX_COMPLETION_LENGTH="64"
export TEMPERATURE="0.3"
export TOP_P="0.8"
export TOP_K="10"
export PRESENCE_PENALTY="0"

export GRADIENT_CHECKPOINTING="True"
export PER_DEVICE_BATCH_SIZE="4"
export GRAD_ACCUM_STEPS="1"

export NUM_TRAIN_EPOCHS="1"
export SAVE_STEPS="500"
export LOGGING_STEPS="10"

# teacher-only draft 数据必须包含 input / sft_draft / output 三列；student 不会看到 sft_draft。
export OPSD_DATASET="/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole_with_sft_draft.jsonl"
export MODEL_DIR="/DATA_A/models/Qwen3.5-2B"
export OUTPUT_DIR="/DATA_B/hyh/opsd_outputs/summary_segment_0713"

bash scripts/run_opsd_qwen35_2b_5090.sh
