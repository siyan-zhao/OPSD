# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models


<p align="center">
<a href="https://arxiv.org/pdf/2601.18734v3"><img src="https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg"></a>
<a href="https://siyan-zhao.github.io/blog/2026/opsd/"><img src="https://img.shields.io/badge/Blog-Post-blue.svg"></a>
</p>

---
## Overview

**On-Policy Self-Distillation (OPSD)** trains a single model to act as both student and teacher by conditioning on different contexts — the student sees only the problem, while the teacher additionally sees the ground-truth solution — and performs token-level distribution matching along the student's own on-policy trajectories.


## Updates

- **Mar 18, 2026**: Released updated code. 

  (1) Fixed chat template and zero2 bugs (see [template issue](https://github.com/huggingface/trl/issues/5241)), we re-ran experiments with updated results (detailed results & ablations updated on arxiv/blog). The fixes yield improved OPSD performance, most notably on Qwen3-1.7B.

  (2) Added a new training stabilization strategy 🚀: per-token point-wise KL clipping. We find style tokens (such as 'wait', 'think') can exhibit 6–15× higher KL divergence than math-related tokens, and dominates the training signal. Clipping stablizes training and improves performance.


-  **Mar 3, 2026**: Initial code release.

## Installation


```bash
conda env create -f environment.yml
conda activate opsd
```

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```
If you encounter difficulties installing flash-attn, you can check the version matching your CUDA and PyTorch versions from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases).

The code uses `trl`'s experimental GOLD trainer as a base.

## Repository Structure

```
├── opsd_trainer.py          # OPSDTrainer: core self-distillation trainer
├── data_collator.py         # Data collator for self-distillation
├── opsd_train.py            # OPSD training entry point
├── sft_train.py             # SFT baseline training entry point
├── grpo_train.py            # GRPO baseline training entry point
├── accelerate.yaml          # Accelerate config (multi-GPU)
├── scripts/
│   ├── run_opsd.sh          # Example launch script for OPSD
│   ├── run_sft.sh           # Example launch script for SFT
│   └── run_grpo.sh          # Example launch script for GRPO
└── eval/
    ├── evaluate_math.py     # Evaluation script (vLLM)
    └── run_eval.sh          # Example evaluation script
```

## Quick Start

Reproduce results on Qwen3-1.7B (🚀 training only takes **~15 minutes** on 4×H100 and peaks within 100 steps):

```bash
bash scripts/run_opsd_1b.sh
```
Evaluation: (evaluation takes ~ 30-50 minutes on 4xh100 for each checkpoint) 
```bash
cd eval
bash run_eval.sh
```

### Evaluation Results across Tasks on Qwen3-1.7B

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 51.5% |
| 25 | 51.4% |
| 50 | 52.8% |
| 75 | 54.4% |
| 100 | 57.2% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 36.7% |
| 25 | 42.5% |
| 50 | 43.9% |
| 75 | 40.6% |
| 100 | 41.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 25 | 24.7% |
| 50 | 27.8% |
| 75 | 26.9% |
| 100 | 29.2% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, thinking mode enabled, max new tokens=38912, top-p=none, top-k disabled, min-p=0, presence penalty=0, num samples=12


## Non-Thinking Mode

OPSD can also run in non-thinking setting where both the Qwen student and teacher are enabled_thinking=False during training (`--student_thinking False --teacher_thinking False`) and evaluated with non-thinking inference (`--no_thinking`), with faster evaluation time than thinking mode.

Training:
```bash
bash scripts/run_opsd_4b_nonthink.sh
bash scripts/run_opsd_8b_nonthink.sh
```

Evaluation:
```bash
cd eval
bash run_eval_nonthink.sh
```

### Evaluation Results with Non-Thinking Mode across Models

#### Qwen3-8B (`--jsd_token_clip 1e-7`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 26.4% |
| 50 | 49.7% |
| 75 | 45.3% |
| 100 | 38.3% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 19.7% |
| 50 | 35.0% |
| 75 | 26.9% |
| 100 | 27.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 18.3% |
| 75 | 17.5% |
| 100 | 15.3% |

</td>
</tr>
</table>
</div>

#### Qwen3-4B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 50 | 20.3% |
| 75 | 27.5% |
| 100 | 31.1% |
| 150 | 32.8% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 21.4% |
| 50 | 21.4% |
| 75 | 20.8% |
| 100 | 21.1% |
| 150 | 21.9% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 11.1% |
| 75 | 13.1% |
| 100 | 16.4% |
| 150 | 14.4% |

</td>
</tr>
</table>
</div>

#### Qwen3-1.7B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 11.9% |
| 50 | 15.0% |
| 75 | 13.9% |
| 100 | 12.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 9.2% |
| 50 | 6.2% |
| 75 | 8.3% |
| 100 | 8.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 5.0% |
| 25 | 7.2% |
| 50 | 5.8% |
| 75 | 5.0% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, non-thinking mode, num samples=12.



## Key OPSD arguments

| Argument | Default | Description |
|---|---|---|
| `--fixed_teacher` | `False` | Fix the teacher to the initial policy (step 0). Requires --use_peft. Note ❗ If you disable PEFT, the teacher will keep updating at every training step, which may make training unstable. Our main results use the fixed teacher, which is currently implemented with LoRA adapter weights. |
| `--use_tinker_loss` | `False` | Use sampled-token policy-gradient objective instead of full-vocabulary JSD. More memory efficient. Currently no clipped implemented for this variant, could be unstable. |
| `--max_completion_length` | — | Student generation length for distillation. We use 1024 in our main experiments. |
| `--beta` | — | Interpolation weight for the JSD mixture distribution. Beta=0 means forward KL and 1 means reverse KL. |
| `--jsd_token_clip` | 0.05 | Clip the JSD loss for each token to a maximum value. This can improve stability by preventing stylistic tokens from dominating the training signal. Note when clipping is applied, the loss can be negative due to positive KL summand being capped. | 
| `--reason_first` | `False` | Prepend an explicit rationalization to the teacher context before distillation. |
| `--run_config` | `None` | Custom name suffix for the output directory and WandB run. |

### SFT Baseline

See [`scripts/run_sft.sh`](scripts/run_sft.sh).

### GRPO Baseline

See [`scripts/run_grpo.sh`](scripts/run_grpo.sh).

## 样本级 reflection + 正确性日志（fork 新增功能）

> 此功能为本 fork 在 `reason-first-logging` 分支新增,用于做样本级分析:把每条训练样本的**题目、teacher 的 reflection 讲解、student 的 rollout,以及这道题学生做对没做对**都写进日志,方便分析 reflection 质量和正确性之间的关系。原仓库的日志只有 `{step, prompt, completion}`,既没有 reflection 也没有任何正确性信息。

### 怎么跑

**必须带 `--reason_first True`**,否则不会触发 teacher 的 reflection 路径,日志里 `teacher_reasoning` 会全是 `null`:

```bash
bash scripts/run_opsd_1b.sh --reason_first True --run_config reason_first_v1
```

### 日志在哪、长什么样

训练时每 5 步(仅主进程)写一个文件:`<output_dir>/generations/generations_step_{N}.json`。里面每条样本:

```json
{
  "step": 50,
  "problem": "……题目……",
  "solution": "……数据集里的参考解答(默认保留,见下方说明)……",
  "teacher_reasoning": "……teacher 对参考解答的讲解(reflection)……",
  "student_completion": "……学生这一步的 rollout……",
  "gt_answer": "42",
  "predicted_answer": "42",
  "correct": true
}
```

字段说明:
- `gt_answer`:从数据集 `solution` 里抽出的 `\boxed{}` 答案。
- `predicted_answer`:从 `student_completion` 里抽出的 `\boxed{}` 答案。
- `correct`:用 `math_verify` 判 `predicted_answer` 与 `gt_answer` 是否等价;`gt_answer` 为 `None` 时该字段为 `null`。
- 判分在**存盘时**才算(每 5 步、仅主进程、`math_verify` 惰性 import),不在训练热路径上,不拖慢训练。

### 第一次跑务必核对两点

1. **索引对齐**:日志里 `problem` 和 `student_completion` 必须是**同一道题**。日志按本进程本地下标对齐,若用 vLLM 且 `tensor_parallel_size > 1`,生成阶段有跨 TP group 的 gather/slice——先跑很少的步数,打开生成的 JSON 肉眼确认对齐,再正式开跑。
2. **`gt_answer` 抽取依赖 `\boxed{}`**:`gt_answer` 来自数据集 `solution` 字段里的 `\boxed{}`。如果你用的数据集 `solution` 没有 `\boxed{}`,那么 `gt_answer`/`correct` 会全是 `null`——这时需要把 `opsd_trainer.py` 里 `_extract_boxed_answer(entry.get("solution"))` 改成指向数据集真正的答案字段。为方便排查,日志里**默认保留了 `solution` 原文**(`opsd_trainer.py` 的 `_save_generation_outputs` 里有一行被注释掉的 `entry.pop("solution", None)`,确认无误后取消注释即可让日志更精简)。

### 解读时注意

`--reason_first` 的 prompt 明确要求 teacher **只讲解参考解答、不要自己重新解题**。所以 `teacher_reasoning` 反映的是"对一个已知正确答案的讲解质量",而**不是** teacher 独立解题的能力——分析时别把它当成独立解题水平。

### Acknowledgements
Our implementation builds on [TRL GOLD Trainer](https://huggingface.co/docs/trl/gold_trainer). We sincerely thank [@simran135](https://github.com/simran135) and [@beanie00](https://github.com/beanie00) for identifying the prompt template bugs and the zero-2 issue, respectively!

## Citation
If you find this useful, please consider citing:
```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```
