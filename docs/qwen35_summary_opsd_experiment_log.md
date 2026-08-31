# Qwen3.5 2B 摘要 OPSD 实验复盘

本文记录 Qwen3.5 2B 在 `input/output` 摘要数据上的 OPSD LoRA 实验过程、观察到的问题、失败原因和后续修正方向。

## 任务和数据

本次任务是短摘要生成。训练数据是标准 SFT-style JSONL：

```json
{
  "input": "<|im_start|>user\nPlease briefly summarize the following text:\n...<|im_end|><|im_start|>assistant\n",
  "output": "While walking my dog, a neighbor recognized me as a veteran."
}
```

语义上：

- `input` 是学生模型推理时应该看到的完整 prompt。
- `output` 是期望学生最终输出的答案文本。
- 目标输出应当是简短摘要句，不应包含 `<think>`、`Summary:`、`Explanation:`、`Final Answer:`、`The user wants...` 等分析或标签文本。

## 初始适配

原始 OPSD 代码主要面向数学推理数据，默认数据格式是：

```json
{"problem": "...", "solution": "..."}
```

原始算法假设是：

- student 只看到 `problem`。
- teacher 额外看到 ground-truth `solution`。
- teacher 在 student 自己生成的 on-policy completion 上给 token 分布指导。

为了支持当前摘要数据，代码增加了 `input/output` 分支：

- student prompt 从 `input` 中解析 ChatML user 内容，并重新套 Qwen3.5 chat template。
- teacher prompt 额外看到 `output` 作为参考答案。
- student 使用 non-thinking。
- teacher 使用 thinking，但在 scoring student token 前关闭 teacher thinking 区域。

推荐 thinking 设置：

```bash
STUDENT_THINKING=False
TEACHER_THINKING=True
CLOSE_TEACHER_THINKING_BEFORE_SCORING=True
REASON_FIRST=False
```

## 第一阶段问题：teacher prompt 污染输出风格

早期 `input/output` 分支为了泛化到非数学任务，teacher prompt 里加入了类似：

```text
Please analyze the user's intent, constraints, and the response strategy.
Simply analyze why the reference response works.
```

以及：

```text
make sure you understand the user's intent, constraints, and desired style
```

这对数学或通用解题任务可能还可以，但对短摘要任务有明显风险：teacher 的上下文会偏向“分析任务/解释策略”，而 OPSD 会把 teacher 在这个上下文下的 token 分布蒸馏给 student。

实际推理现象是模型输出：

```text
The user wants a brief summary of the provided text...
```

这说明 student 受到了 teacher 侧 meta-analysis 风格的影响，而不是只学习 `output` 风格的短摘要。

修正：

- 去掉 `Original prompt:`、`Here is a high-quality reference response:`、`analyze user's intent` 等包装。
- 将 teacher prompt 改为更短的 private reference 风格。
- 显式保持 `REASON_FIRST=False`，避免 teacher 额外生成显式 reasoning 文本参与训练。

对应提交：

- `27974e5 Align input-output OPSD teacher prompt`
- `08aad1e Make Qwen3.5 reason-first default explicit`

## 第二阶段结果：短期变好，但长训漂移

在修掉明显的 meta prompt 后，`checkpoint-500` 的快速评测结果明显改善：

```text
Evaluated 8 samples
Average prediction tokens: 33.4
Format-flagged samples: 2/8
```

此时没有再出现 `The user wants...`、`Summary:`、`Explanation:`、`Final Answer:` 或 `<think>`。模型基本能输出摘要，但仍有部分模板化表达：

```text
The text describes...
The text suggests...
```

继续训练到 `checkpoint-5000` 后，输出明显变差：

```text
Evaluated 8 samples
Average prediction tokens: 79.9
Format-flagged samples: 2/8
```

由于评测时 `MAX_NEW_TOKENS=80`，平均 79.9 tokens 表示模型几乎每条都打满最大生成长度，已经不会自然停止。典型坏例子包括：

```text
The text describes a situation where...
However, the text does not provide...
```

```text
**Wait, I need to check the prompt again.**
```

```text
Let's build the money. Let's build the pot. Let's build the money...
```

这表明模型出现了长解释、重复和不停止的问题。

## 失败原因分析

### 1. 纯 OPSD 不等同于 SFT

当前训练不是直接优化：

```text
input -> output
```

而是：

1. student 看到 `input`，先自己生成 completion。
2. teacher 看到 `input + output`，作为更强条件模型。
3. 将 student completion 接到 teacher prompt 后。
4. teacher 对 student completion 的 token 分布打分。
5. student 学 teacher 在这些 token 位置上的分布。

因此，如果 student rollout 开始生成长解释或重复文本，OPSD 会继续在这条 student 自己走出来的轨迹上训练。它不会像 SFT 一样强制第一个答案 token 必须等于 `output` 的第一个 token。

短摘要任务本身格式强、答案短、停止位置重要，因此比数学推理更依赖直接的 `input -> output` 监督。

### 2. `MAX_COMPLETION_LENGTH` 太长

早期正式训练中使用过：

```bash
MAX_COMPLETION_LENGTH=256
```

但摘要目标通常只有十几个到几十个 token。过长的 rollout 上限会给 student 生成长解释和重复文本的空间。纯 OPSD 又会在这些长 completion 上计算 distillation loss，容易把长尾行为强化进去。

后续摘要任务建议：

```bash
MAX_COMPLETION_LENGTH=48
# 或
MAX_COMPLETION_LENGTH=64
```

### 3. 训练采样较开放

默认脚本里 student rollout 采样参数是：

```bash
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=20
```

这对推理探索类任务可以接受，但对短摘要偏开放。student 更容易采样到解释、补充说明或重复片段。摘要任务建议先收紧：

```bash
TEMPERATURE=0.5
TOP_P=0.8
TOP_K=10
```

也可以试：

```bash
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=20
```

但不建议训练 rollout 直接用 `temperature=0`，否则 on-policy 分布过窄，OPSD 的训练信号也会变得单一。

### 4. `PRESENCE_PENALTY` 的实际影响需要区分路径

脚本默认值曾是：

```bash
PRESENCE_PENALTY=2.0
```

直觉上它不适合摘要，因为 presence penalty 会鼓励模型引入更多新内容，不利于短摘要尽快停止。

但检查代码后发现：当前不使用 vLLM、走 torch `model.generate()` 的训练路径时，`PRESENCE_PENALTY` 实际没有传入 `model.generate()`，所以它不是本次 `checkpoint-5000` 漂移的直接原因。它主要会在 vLLM 路径下生效。

尽管如此，摘要任务仍建议显式设置：

```bash
PRESENCE_PENALTY=0
```

避免未来切换生成路径时引入额外变量。

### 5. teacher 只是参考答案指导，不是硬标签

即使 teacher 看到 `output`，只要训练目标仍是 “在 student rollout 上匹配 teacher 分布”，teacher 也不是直接把 `output` token 当 label 强行喂给 student。

这就是为什么仅靠纯 OPSD 长训仍然可能漂移。对当前任务，更稳的方案是引入 SFT loss：

```text
loss = SFT_LOSS_WEIGHT * SFT loss(input -> output + EOS) + OPSD_LOSS_WEIGHT * OPSD loss
```

其中 SFT loss 锚定目标格式和停止位置，OPSD loss 再提供 teacher thinking 条件下的分布指导。

## 最新修正：Exact target answer teacher prompt

为了让 teacher 更严格地利用 `output`，teacher prompt 已从 private reference 改为 exact target answer 风格：

```text
{input}

Exact target answer:
{output}

The next assistant response must match the exact target answer above.
Do not add, remove, rephrase, label, or explain anything.
Output exactly the target answer text, then emit the end-of-message token and stop:
```

这样 teacher 的分布应更贴近 `output`，减少“泛泛判断好摘要”的空间。

随后又补充了 end-of-message 约束：teacher prompt 明确要求 target answer 后立即输出 end-of-message token 并停止；训练 loss mask 也保留每条 student completion 中第一个 EOS token 作为可学习的停止目标，避免模型只学答案内容但不学停止。

对应提交：

- `1ea9588 Make input-output teacher target exact answer`

注意：这仍不是纯 SFT。它只是让 teacher 条件分布更接近标准答案；如果 student rollout 本身偏离很远，OPSD 仍然是在偏离轨迹上做分布匹配。

## 最新修正：Quality teacher guidance

在 `SFT loss + OPSD loss` 稳住格式、EOS 和短摘要长度之后，下一阶段目标是提升人性化、可读性、流畅性和摘要美感。为此新增：

```bash
TEACHER_GUIDANCE_MODE=quality
```

该模式下，teacher 仍然看到 `output`，但不再要求 student 一字不差复制参考答案，而是把参考答案当作私有语义和覆盖度参考，偏向：

- 忠实原文，不引入无根据内容。
- 将参考答案视为语义 baseline，而不是表达上限；如果 student completion 更准确、更自然，应偏好更好的表达。
- 事实精确优先于文采：人物、指代、数字、地点、因果和转折不能错。
- 简洁，通常保持一句短摘要。
- 自然、可读、流畅，有编辑感。
- 抓主线而不是机械覆盖所有 bullet。
- 保留核心人物、数字、因果和转折，删去让句子臃肿的次要细节。
- 避免生硬从句堆叠、空泛开头和 `the text is about` 这类模板表达。
- 不输出 label、解释、meta-commentary 或 thinking 文本。
- 完成摘要后输出 end-of-message 并停止。

如果要回到严格参考答案版本，可设置：

```bash
TEACHER_GUIDANCE_MODE=exact
```

同时将 Qwen3.5 摘要脚本默认 LoRA 从 `r=64, alpha=128` 调整为：

```bash
LORA_R=16
LORA_ALPHA=32
```

这样保持 `alpha/r = 2` 不变，但降低 LoRA 容量，减少对训练集答案措辞的过拟合和风格漂移风险。

## 当前建议的下一轮实验配置

建议重新开新 run，不要从已经漂移的旧 checkpoint 继续训：

```bash
RUN_CONFIG=summary_segment_sft_opsd_len64_t03_v1
MAX_COMPLETION_LENGTH=64
TEMPERATURE=0.3
TOP_P=0.8
TOP_K=10
PRESENCE_PENALTY=0
OPSD_LOSS_WEIGHT=1.0
SFT_LOSS_WEIGHT=1.0
STUDENT_THINKING=False
TEACHER_THINKING=False
CLOSE_TEACHER_THINKING_BEFORE_SCORING=False
REASON_FIRST=False
```

保持：

```bash
LEARNING_RATE=5e-6
TOP_K_LOSS=256
JSD_TOKEN_CLIP=1e-6
```

先跑短测试：

```bash
MAX_STEPS=500
SAVE_STEPS=500
LOGGING_STEPS=10
```

评测：

```bash
CHECKPOINT_DIR=/DATA_B/hyh/opsd_outputs/<run_name>/checkpoint-500 \
MAX_NEW_TOKENS=48 \
bash scripts/test_qwen35_summary_checkpoint.sh
```

如果 `checkpoint-500` 正常，再比较 `checkpoint-1000`、`checkpoint-1500`。不要默认认为 step 越多越好；本次实验已经观察到后期 checkpoint 明显退化。

## 结论

本次失败不是数据 JSON 格式读错，而是训练目标和任务类型不完全匹配：

- OPSD 原本更适合数学推理这类长 reasoning / 多路径问题。
- 当前摘要任务是强格式、短输出、强停止约束任务。
- 纯 OPSD 没有直接 SFT label 锚定，长训容易沿 student 自己的错误 rollout 漂移。
- teacher prompt 中任何分析性文字都可能被分布蒸馏放大，污染 student 输出风格。

已经修正的部分：

- 支持 `input/output` 数据格式。
- 移除 teacher meta-analysis prompt。
- 显式关闭 `REASON_FIRST`。
- 接入 W&B 指标。
- 增加 checkpoint 快速评测脚本。
- 将 teacher prompt 改成 exact target answer 风格。
- 增加 `SFT loss + OPSD loss` 混合训练，让 `output + EOS` 直接参与 token-level 监督。

仍建议继续观察的风险：

- 如果 `sft_loss` 降低但 eval 仍然持续打满 `MAX_NEW_TOKENS`，需要进一步提高 `SFT_LOSS_WEIGHT` 或先做纯 SFT warmup。
- 对短摘要任务，teacher thinking 可能没有收益，建议先用 `TEACHER_THINKING=False` 做稳定 baseline。
