# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models


[![Paper](https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg)](https://arxiv.org/abs/2601.18734)

---

## Overview

This repository contains the training code for **On-Policy Self-Distillation (OPSD)**, along with SFT and GRPO baselines used in our experiments. OPSD trains a single model to act as both student and teacher by conditioning on different contexts — the student sees only the problem, while the teacher additionally sees the ground-truth solution — and performs token-level distribution matching along the student's own on-policy trajectories.

## Installation


```bash
conda env create -f environment.yml
conda activate opsd
```

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

The code uses `trl`'s experimental GOLD trainer as a base.

## Repository Structure

```
├── opsd_trainer.py          # OPSDTrainer: core self-distillation trainer
├── data_collator.py         # Data collator for self-distillation
├── opsd_train.py            # OPSD training entry point
├── sft_train.py             # SFT baseline training entry point
├── grpo_train.py            # GRPO baseline training entry point
├── accelerate.yaml          # Accelerate config (multi-GPU)
└── scripts/
    ├── run_opsd.sh          # Example launch script for OPSD
    ├── run_opsd_ema.sh      # OPSD with EMA teacher
    ├── run_opsd_topkloss.sh # OPSD with top-k vocabulary loss
    ├── run_sft.sh           # Example launch script for SFT
    └── run_grpo.sh          # Example launch script for GRPO
```



## Training


### OPSD

See [`scripts/run_opsd.sh`](scripts/run_opsd.sh). For EMA teacher and top-k vocabulary loss variants, see [`scripts/run_opsd_ema.sh`](scripts/run_opsd_ema.sh) and [`scripts/run_opsd_topkloss.sh`](scripts/run_opsd_topkloss.sh).

#### Key OPSD arguments

| Argument | Default | Description |
|---|---|---|
| `--fixed_teacher` | `False` | Fix the teacher to the initial policy (step 0). Requires `--use_peft`. Recommended. |
| `--use_ema_teacher` | `False` | Use an EMA of the student weights as the teacher instead of a fixed snapshot. See `run_opsd_ema.sh`. |
| `--ema_decay` | — | EMA decay rate for the teacher (e.g. 0.999). Only used when `--use_ema_teacher` is set. |
| `--top_k_loss` | — | Restrict distillation loss to the top-k teacher vocabulary entries. See `run_opsd_topkloss.sh`. |
| `--use_tinker_loss` | `False` | Use sampled-token policy-gradient objective instead of full-vocabulary JSD. More memory efficient. |
| `--max_completion_length` | — | Student generation length for distillation. We use 2048 in our main experiments. |
| `--beta` | — | Interpolation weight for the JSD mixture distribution. |
| `--reason_first` | `False` | Prepend an explicit rationalization to the teacher context before distillation. |
| `--run_config` | `None` | Custom name suffix for the output directory and WandB run. |

### SFT Baseline

See [`scripts/run_sft.sh`](scripts/run_sft.sh).

### GRPO Baseline

See [`scripts/run_grpo.sh`](scripts/run_grpo.sh).

## Citation

```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```
