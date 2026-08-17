import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer

from rlsd_trainer import RLSDTrainer
from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class RLSDScriptArguments(ScriptArguments):
    run_config: str = field(
        default=None,
        metadata={"help": "Run name suffix for output_dir and WandB."},
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "WandB entity (username or team name)."},
    )
    wandb_project: str = field(
        default="RLSD",
        metadata={"help": "WandB project name."},
    )
    rlsd_lambda: float = field(
        default=0.5,
        metadata={"help": "Initial RLSD mix coefficient. Decays linearly when rlsd_lambda_decay_steps > 0."},
    )
    rlsd_lambda_decay_steps: int = field(
        default=50,
        metadata={"help": "Linearly decay rlsd_lambda to 0 over this many optimizer steps. Set <=0 to disable."},
    )
    rlsd_epsilon_w: float = field(
        default=0.2,
        metadata={"help": "Clip bound for RLSD token evidence weights: [1-eps, 1+eps]."},
    )
    teacher_max_prompt_length: int = field(
        default=None,
        metadata={"help": "Optional max length for teacher privileged prompt."},
    )


def extract_boxed_answer(text):
    if text is None:
        return None
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text
    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer):
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def reward_correctness(completions, answer=None, Answer=None, solution=None, **kwargs):
    ground_truths = answer if answer is not None else Answer
    if ground_truths is None and solution is not None:
        ground_truths = [extract_boxed_answer(item) or item for item in solution]

    rewards = []
    for completion, ground_truth in zip(completions, ground_truths):
        pred_answer = extract_boxed_answer(completion)
        reward = 0.0

        gold_parsed = parse(ground_truth)
        pred_parsed = parse(_preprocess_for_parse(pred_answer))
        if gold_parsed is not None and pred_parsed is not None:
            try:
                reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
            except Exception:
                pass

        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", ground_truth or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0

        rewards.append(reward)
    return rewards


def make_format_prompt(tokenizer):
    def format_prompt(example):
        problem = example.get("problem") or example.get("Question") or example.get("question")
        solution = example.get("solution")
        answer = example.get("answer") or example.get("Answer") or extract_boxed_answer(solution)

        student_messages = [
            {
                "role": "user",
                "content": f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            }
        ]
        teacher_messages = [
            {
                "role": "user",
                "content": (
                    f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n\n"
                    "Using the reference solution only as privileged evidence, solve the problem in your own words. "
                    "Please reason step by step, and put your final answer within \\boxed{}."
                ),
            }
        ]
        return {
            "prompt": tokenizer.apply_chat_template(student_messages, tokenize=False, add_generation_prompt=True),
            "teacher_prompt": tokenizer.apply_chat_template(teacher_messages, tokenize=False, add_generation_prompt=True),
            "answer": answer,
        }

    return format_prompt


if __name__ == "__main__":
    parser = TrlParser((RLSDScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        run_name = (
            f"RLSD_{model_name}_lr{lr_str}_bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_lambda{script_args.rlsd_lambda}"
        )

    print(f"\n{'=' * 80}")
    print("RLSD RUN CONFIGURATION")
    print(f"WandB Run Name: {run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"RLSD lambda: {script_args.rlsd_lambda}")
    print(f"RLSD lambda decay steps: {script_args.rlsd_lambda_decay_steps}")
    print(f"RLSD epsilon_w: {script_args.rlsd_epsilon_w}")
    print(f"{'=' * 80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_prompt_length": training_args.max_prompt_length,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "rlsd_lambda": script_args.rlsd_lambda,
                "rlsd_lambda_decay_steps": script_args.rlsd_lambda_decay_steps,
                "rlsd_epsilon_w": script_args.rlsd_epsilon_w,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "num_processes": num_processes,
            },
        )

    model_dtype = torch.bfloat16
    if getattr(model_args, "torch_dtype", None) is not None and isinstance(model_args.torch_dtype, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
    elif getattr(model_args, "dtype", None) is not None:
        model_dtype = model_args.dtype

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    train_dataset = dataset["train"].map(make_format_prompt(tokenizer), remove_columns=dataset["train"].column_names)

    trainer = RLSDTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        rlsd_lambda=script_args.rlsd_lambda,
        rlsd_lambda_decay_steps=script_args.rlsd_lambda_decay_steps,
        rlsd_epsilon_w=script_args.rlsd_epsilon_w,
        teacher_max_prompt_length=script_args.teacher_max_prompt_length,
    )

    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if checkpoints:
            resume_from_checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
