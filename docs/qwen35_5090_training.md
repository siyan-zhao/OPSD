# Qwen3.5 2B OPSD LoRA 训练说明（4x RTX 5090）

这份文档说明 Qwen3.5 2B 的 OPSD LoRA 训练脚本：

```bash
scripts/run_opsd_qwen35_2b_5090.sh
```

当前配置面向 4 张 RTX 5090，使用 DeepSpeed ZeRO-2、LoRA 训练、student non-thinking rollout，以及 non-thinking teacher scoring。

## 快速开始

建议先跑 1 step smoke test，确认环境、数据、模型加载、生成和反传都能走通：

```bash
cd /DATA_B/hyh/OPSD
source .venv/bin/activate

WANDB_MODE=offline \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
MAX_STEPS=1 \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

然后跑一个接入 W&B 的 50 step 短测试：

```bash
cd /DATA_B/hyh/OPSD
source .venv/bin/activate

export WANDB_API_KEY="your_wandb_api_key"

OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
WANDB_PROJECT=OPSD \
WANDB_RUN_NAME=qwen35_2b_opsd_lora_50step \
MAX_STEPS=50 \
SAVE_STEPS=50 \
LOGGING_STEPS=5 \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

如果 50 step 稳定，再启动 1 epoch 正式训练：

```bash
cd /DATA_B/hyh/OPSD
source .venv/bin/activate

export WANDB_API_KEY="your_wandb_api_key"

OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
WANDB_PROJECT=OPSD \
RUN_CONFIG=qwen35_2b_opsd_lora_1epoch \
NUM_TRAIN_EPOCHS=1 \
SAVE_STEPS=500 \
LOGGING_STEPS=10 \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

## W&B 配置

启动脚本通过环境变量控制 W&B。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `WANDB_API_KEY` | 未设置 | 设置后脚本会自动执行 `wandb login --relogin`，并默认启用 online logging。不要把真实 key 写进代码或提交到 Git。 |
| `WANDB_MODE` | 设置了 `WANDB_API_KEY` 时为 `online`，否则为 `offline` | 正式看训练用 `online`；只在本地保存日志用 `offline`；完全关闭用 `disabled`。 |
| `WANDB_PROJECT` | `OPSD` | W&B project 名称。 |
| `WANDB_RUN_NAME` | 未设置 | 可选的 W&B run 名别名。脚本会把它当作 `RUN_CONFIG` 使用；如果同时设置了 `RUN_CONFIG`，以 `RUN_CONFIG` 为准。 |
| `WANDB_ENTITY` | 未设置 | 可选的 W&B 用户、团队或 entity。只有你的账号需要指定时才设置。 |
| `REPORT_TO` | `wandb`，当 `WANDB_MODE=disabled` 时为 `none` | Hugging Face Trainer 的日志后端。要在 W&B 页面看到 loss/grad norm/learning rate，保持 `wandb`。 |

比较安全的输入方式：

```bash
read -s WANDB_API_KEY
export WANDB_API_KEY
```

也可以一行传入，但这种方式可能会被 shell history 记录：

```bash
WANDB_API_KEY="your_wandb_api_key" \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
WANDB_PROJECT=OPSD \
WANDB_RUN_NAME=qwen35_2b_opsd_lora \
NUM_TRAIN_EPOCHS=1 \
SAVE_STEPS=500 \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

W&B 的训练指标记录频率由 `LOGGING_STEPS` 控制，例如 `LOGGING_STEPS=10` 表示每 10 个 optimizer step 记录一次指标。默认会记录 `loss`、`on_policy_loss`、`opsd_loss`、`sft_loss`、`mixed_loss`、`grad_norm`、`learning_rate`、`epoch`，训练结束会记录 `train_loss`、`train_runtime`、`train_steps_per_second` 等。

online 模式下 W&B 会持续同步这些日志；如果看到 offline run，通常是因为 `WANDB_MODE=offline` 还留在环境变量里，需要执行：

```bash
unset WANDB_MODE
```

## 主要训练参数

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_DIR` | `/Users/hyh/Desktop/Qwen3.5_2B` | Qwen3.5 base model 的本地路径。服务器上使用 `/DATA_A/models/Qwen3.5-2B`。 |
| `OPSD_DATASET` | `siyanzhao/Openthoughts_math_30k_opsd` | Hugging Face dataset 名称，或本地 `.json/.jsonl/.csv` 文件。你当前的数据是 `input/output` JSONL。 |
| `OUTPUT_DIR` | `./outputs/opsd_qwen35_2b_5090` | 输出根目录。脚本会在下面追加 `RUN_CONFIG` 子目录。 |
| `RUN_CONFIG` | `qwen35_2b_5090_lora_nonthink_topk256` | 训练 run 名称、checkpoint 子目录名，以及 W&B run 名的一部分。 |
| `NUM_PROCESSES` | `4` | GPU/process 数量。4 卡 5090 保持 `4`。 |
| `PER_DEVICE_BATCH_SIZE` | `1` | 每张卡的 micro-batch size。 |
| `GRAD_ACCUM_STEPS` | `8` | 梯度累积步数。有效 batch size = `PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_PROCESSES`，默认是 `32`。 |
| `GRADIENT_CHECKPOINTING` | `True` | 是否开启 gradient checkpointing。显存充足时可以设为 `False`，通常会更快但更吃显存。 |
| `NUM_TRAIN_EPOCHS` | `30` | 没有设置 `MAX_STEPS` 时按 epoch 训练。你的数据量较大，建议先从 `1` 开始。 |
| `MAX_STEPS` | 未设置 | 直接限制总训练 step，适合 smoke test 和短测试。设置后会覆盖 epoch 停止逻辑。 |
| `SAVE_STEPS` | `25` | checkpoint 保存间隔。正式长训建议用 `500` 或更大，避免 checkpoint 太多。 |
| `LOGGING_STEPS` | `2` | 日志记录间隔。 |
| `LEARNING_RATE` | `5e-6` | LoRA 学习率。 |
| `LORA_R` | `16` | LoRA rank。摘要任务默认使用较小 rank，降低过拟合和风格漂移风险。 |
| `LORA_ALPHA` | `32` | LoRA alpha。默认保持 `alpha/r = 2`，和旧的 `r=64, alpha=128` 缩放比例一致。 |
| `MAX_LENGTH` | `8192` | collator 处理 prompt/context 的最大长度。 |
| `MAX_COMPLETION_LENGTH` | `1024` | student rollout 的最大新 token 数。 |
| `TEMPERATURE` | `1.0` | student rollout 采样温度。摘要任务建议显式覆盖为 `0.5` 左右。 |
| `TOP_P` | `1.0` | student rollout nucleus sampling。摘要任务建议显式覆盖为 `0.8` 左右。 |
| `TOP_K` | `20` | student rollout top-k。摘要任务建议显式覆盖为 `10` 左右。 |
| `PRESENCE_PENALTY` | `2.0` | vLLM 路径下的 presence penalty。当前 torch 训练路径基本不生效，但摘要任务建议显式覆盖为 `0`，避免未来切换生成路径时鼓励展开。 |
| `OPSD_LOSS_WEIGHT` | `1.0` | OPSD 蒸馏损失权重。它让 student 在自己的 rollout 轨迹上贴近拥有参考答案上下文的 teacher 分布。 |
| `SFT_LOSS_WEIGHT` | `1.0` | SFT 交叉熵损失权重。它直接训练 `input -> output + EOS`，用于锚定短摘要格式和停止位置。设为 `0` 可回到纯 OPSD。 |
| `TEACHER_GUIDANCE_MODE` | `quality` | teacher 使用参考答案的方式。`exact` 要求贴近标准答案原文；`quality` 把标准答案当私有语义 baseline，但只允许在强事实和格式约束内做更自然、更可读的改写。 |
| `TEACHER_DRAFT_FIELD` | 空 | 可选的 teacher-only SFT 初稿字段。设置为 `sft_draft` 后，student 仍然只看到 `input`，teacher 私有看到 `sft_draft + output`，用于把 SFT 初稿作为风格/格式/常见错误的比较基线。 |
| `CORRECTOR_MODE` | `False` | 是否启用 SFT 初稿纠错模式。开启后 student 看到 `input + sft_draft`，目标仍是 `output`，用于把 OPSD 训练成 SFT 摘要的轻修正器。 |
| `DRAFT_FIELD` | `sft_draft` | corrector 模式下读取 SFT 初稿的字段名。 |
| `TOP_K_LOSS` | `256` | 蒸馏损失只在 teacher top-k token 上计算，降低显存和计算压力。 |
| `JSD_TOKEN_CLIP` | `1e-6` | 每个 token 的 JSD clipping，用于稳定训练。 |

## Thinking 设置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `STUDENT_THINKING` | `False` | student rollout 不开启 thinking。 |
| `TEACHER_THINKING` | `False` | teacher scoring prompt 是否开启 Qwen thinking mode。短摘要任务建议先保持 `False`，做稳定 baseline。 |
| `CLOSE_TEACHER_THINKING_BEFORE_SCORING` | `False` | 如果 teacher thinking 开启，在 teacher-only 隐藏思考后补上 `</think>`，再对 student token 打分。teacher non-thinking 时保持 `False`。 |
| `REASON_FIRST` | `False` | 不让 teacher 额外显式生成 reasoning 文本；摘要任务建议保持关闭，避免把分析风格蒸馏给 student。 |
| `REAPPLY_CHAT_TEMPLATE_TO_INPUT` | `True` | 重新解析 `input` 中的 ChatML，并套当前 Qwen3.5 tokenizer 的 chat template。 |

当前短摘要任务推荐先使用 non-thinking teacher 作为稳定 baseline：

```bash
STUDENT_THINKING=False
TEACHER_THINKING=False
CLOSE_TEACHER_THINKING_BEFORE_SCORING=False
REASON_FIRST=False
```

teacher thinking 可以后续作为消融实验再打开；如果打开，应保持 `CLOSE_TEACHER_THINKING_BEFORE_SCORING=True` 和 `REASON_FIRST=False`。

## 数据集格式

你当前的 SFT-style JSONL 格式已经支持：

```json
{"input": "<|im_start|>user\n...\n<|im_end|><|im_start|>assistant\n", "output": "..."}
```

代码默认读取 `input/output` 两列。`input` 会作为 student 看到的原始请求，`output` 会同时用于两条训练信号：

- SFT loss：直接训练 `input -> output + EOS`，其中 prompt 和 padding 不参与 loss。
- OPSD loss：teacher 看到 `output` 作为私有参考答案，然后在 student rollout tokens 上给分布指导。

默认 `TEACHER_GUIDANCE_MODE=quality` 时，teacher 不要求 student 一字不差复制 `output`，而是把标准答案当语义 baseline，不当上限。但这不是鼓励自由发挥，而是“强约束下的质量改写”：

- 事实优先：保留人物、数字、时间、地点、因果、转折、语气和不确定性，不允许新增原文或参考答案没有的时间、原因、评价、例子或建议。
- 匹配任务类型：短问答或短参考答案必须保持短而直接，不能把 `You sure?` 这类输入改写成“这个短语是什么意思”的解释。
- 长度贴近参考答案：长 transcript 可以更好地综合主线，但不应为了显得高级而扩写。
- 参考答案已经好时，允许接近复制；不要为了改写而改写。
- 输出只能是最终摘要：不能有 `Summary:`、`Final Answer:`、解释、澄清问题、短语分析、meta-commentary 或 thinking 文本。
- 摘要完成后应立即输出 EOS 停止。

如果要复现实验中的严格 target-answer 版本，可以设置 `TEACHER_GUIDANCE_MODE=exact`。

### Teacher-only SFT 初稿基线

如果最终推理时仍然只有 `input`，但训练时想让 teacher 参考 SFT 初稿的风格和常见错误，可以设置：

```bash
CORRECTOR_MODE=False
TEACHER_DRAFT_FIELD=sft_draft
```

此时数据集每行需要包含：

```json
{
  "input": "<|im_start|>user\nPlease briefly summarize ...<|im_end|><|im_start|>assistant\n",
  "sft_draft": "SFT model's current summary draft.",
  "output": "Reference or better corrected summary."
}
```

训练时：

```text
student 看到：input
student rollout：当前摘要
teacher 私有看到：input + sft_draft + output + student rollout
teacher 作用：用 sft_draft 作为风格/格式/简洁度 baseline，用 output 作为事实权威，指导 student 的 token 分布
```

推理时仍然是：

```text
input -> final summary
```

这不会造成训练/推理格式错位，因为 `sft_draft` 只给 teacher 看，不给 student 看。它适合“单模型部署，但训练时让 teacher 参考 SFT baseline”的实验。

### SFT 初稿纠错模式

如果要把 OPSD 训练成 SFT 摘要的“轻修正器”，开启：

```bash
CORRECTOR_MODE=True
DRAFT_FIELD=sft_draft
```

此时数据集每行需要包含：

```json
{
  "input": "<|im_start|>user\nPlease briefly summarize ...<|im_end|><|im_start|>assistant\n",
  "sft_draft": "SFT model's current summary draft.",
  "output": "Reference or better corrected summary."
}
```

训练时 student 看到的是：

```text
Original summarization request:
{input 中的用户请求}

Current SFT draft summary:
{sft_draft}

Revise the draft only if necessary...
```

teacher 仍然私有看到 `output`，但 prompt 会要求它把模型当成 corrector，而不是第二个摘要生成器：如果 SFT 初稿已经准确、简洁、自然，就尽量保持；只修事实、逻辑、指代、遗漏、冗余或明显别扭的措辞。

这个模式适合两阶段推理：

1. SFT adapter：`input -> sft_draft`
2. OPSD corrector adapter：`input + sft_draft -> corrected_summary`

如果最终想把权重合成单个一次性摘要模型，不建议使用 corrector 模式，因为推理时没有 `sft_draft` 可以输入。

如果你的数据列名不同，可以这样覆盖：

```bash
INPUT_FIELD=prompt OUTPUT_FIELD=response bash scripts/run_opsd_qwen35_2b_5090.sh
```

## 推荐训练流程

1. `MAX_STEPS=1`：确认环境、模型加载、数据加载、生成和反传都没问题。
2. `MAX_STEPS=50`：确认 W&B online logging、速度、loss 曲线和 checkpoint 保存。
3. `NUM_TRAIN_EPOCHS=1`：进行第一次正式训练。
4. 检查 checkpoint 推理效果和磁盘占用后，再决定是否增加 epoch 或 step。

你的数据有 106,919 条样本，默认有效 batch size 是 32，所以 1 epoch 大约是 3,342 个 optimizer step。

## 加速调参建议

当前 `accelerate_5090_zero2.yaml` 默认已经关闭 CPU optimizer offload：

```yaml
offload_optimizer_device: none
```

如果每张 32GB 的 5090 只占用约 5GB，可以按下面顺序提速。每次先跑 `MAX_STEPS=50`，确认没有 OOM、速度提升明显、loss/grad_norm 正常，再用于正式训练。

第一步：增大每卡 micro-batch，同时保持有效 batch size 仍为 32。

```bash
PER_DEVICE_BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=4 \
MAX_STEPS=50 \
SAVE_STEPS=50 \
LOGGING_STEPS=5 \
RUN_CONFIG=qwen35_2b_opsd_lora_bs2_ga4_test \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

第二步：如果显存仍然充足，继续提高 micro-batch。

```bash
PER_DEVICE_BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=2 \
MAX_STEPS=50 \
SAVE_STEPS=50 \
LOGGING_STEPS=5 \
RUN_CONFIG=qwen35_2b_opsd_lora_bs4_ga2_test \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

第三步：如果显存还是很空，可以关闭 gradient checkpointing。

```bash
GRADIENT_CHECKPOINTING=False \
PER_DEVICE_BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=2 \
MAX_STEPS=50 \
SAVE_STEPS=50 \
LOGGING_STEPS=5 \
RUN_CONFIG=qwen35_2b_opsd_lora_bs4_ga2_no_ckpt_test \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

如果任务输出本来不需要 1024 token，还可以降低 `MAX_COMPLETION_LENGTH`。OPSD 每步都要先生成 student completion，生成长度越短，速度提升越明显。

```bash
MAX_COMPLETION_LENGTH=512 \
PER_DEVICE_BATCH_SIZE=4 \
GRAD_ACCUM_STEPS=2 \
MAX_STEPS=50 \
SAVE_STEPS=50 \
RUN_CONFIG=qwen35_2b_opsd_lora_len512_test \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

对当前短摘要任务，更推荐把 rollout 采样也一起收紧：

```bash
MAX_COMPLETION_LENGTH=64 \
TEMPERATURE=0.5 \
TOP_P=0.8 \
TOP_K=10 \
PRESENCE_PENALTY=0 \
OPSD_LOSS_WEIGHT=1.0 \
SFT_LOSS_WEIGHT=0.3 \
TEACHER_GUIDANCE_MODE=quality \
LORA_R=16 \
LORA_ALPHA=32 \
STUDENT_THINKING=False \
TEACHER_THINKING=False \
CLOSE_TEACHER_THINKING_BEFORE_SCORING=False \
REASON_FIRST=False \
RUN_CONFIG=summary_segment_quality_teacher_r16_len64_v1 \
OPSD_DATASET=/DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
MODEL_DIR=/DATA_A/models/Qwen3.5-2B \
OUTPUT_DIR=/DATA_B/hyh/opsd_outputs \
bash scripts/run_opsd_qwen35_2b_5090.sh
```

不建议一开始就同时增大 `PER_DEVICE_BATCH_SIZE` 和 `GRAD_ACCUM_STEPS`，因为这会改变有效 batch size。比如 `PER_DEVICE_BATCH_SIZE=4`、`GRAD_ACCUM_STEPS=8`、4 卡时有效 batch size 会变成 128，训练动态和学习率可能都要重新调。

## Checkpoint 和推理

checkpoint 会保存到：

```bash
/DATA_B/hyh/opsd_outputs/<RUN_CONFIG>/
```

使用 base model 加 LoRA checkpoint 做 torch 原生推理：

```bash
python scripts/generate_qwen35_torch.py \
  --model /DATA_A/models/Qwen3.5-2B \
  --adapter /DATA_B/hyh/opsd_outputs/qwen35_2b_opsd_lora_1epoch/checkpoint-500 \
  --enable_thinking False \
  --max_new_tokens 256 \
  --prompt "Summarize these points into one sentence: ..."
```

也可以直接从你的 jsonl 训练数据里抽一条样本推理，并打印 reference 对照：

```bash
python scripts/generate_qwen35_torch.py \
  --model /DATA_A/models/Qwen3.5-2B \
  --adapter /DATA_B/hyh/opsd_outputs/qwen35_2b_opsd_lora_1epoch/checkpoint-500 \
  --data_file /DATA_A/data/hyh/Qwen3.5/qwen3.5_segment_summary_2B_0309/train/train_0115_whole.jsonl \
  --sample_index 0 \
  --enable_thinking False \
  --max_new_tokens 256
```

批量检查摘要 checkpoint 的格式和内容，可以使用：

```bash
bash scripts/test_qwen35_summary_checkpoint.sh
```

默认会测试：

```bash
/DATA_B/hyh/opsd_outputs/summary_segment_0706/qwen3.5_2b_opsd_summary_segment_v1/checkpoint-500
```

常用覆盖参数：

```bash
CHECKPOINT_DIR=/DATA_B/hyh/opsd_outputs/summary_segment_0706/qwen3.5_2b_opsd_summary_segment_v1/checkpoint-1000 \
NUM_SAMPLES=16 \
MAX_NEW_TOKENS=80 \
bash scripts/test_qwen35_summary_checkpoint.sh
```

如果想固定看几条样本：

```bash
SAMPLE_INDICES=0,10,100 \
bash scripts/test_qwen35_summary_checkpoint.sh
```

脚本会打印 prompt、reference、prediction，并标记 `<think>`、`The user wants`、`Summary:`、`Explanation:`、`Final Answer:` 等格式污染。

关于 Qwen3.5 fast path 不可用并 fallback 到 torch implementation 的 warning，目前不阻塞训练。smoke test 已经在这个 fallback 下完整跑通。
