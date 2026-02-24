import os
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    SFTTrainer,
    SFTConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def format_example(example):
    """
    Format the dataset examples using simple Problem/Solution format.

    The OpenThoughts dataset has:
    - Question: the math problem
    - solution: the reasoning/thought process
    """
    text = f"Problem: {example['Question']}\nSolution: {example['solution']}"
    return {"text": text}


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name
    ################
    # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
    model_name = model_args.model_name_or_path.split("/")[-1]

    # Format learning rate (e.g., 2e-5 -> "2e-5" or 0.00002 -> "2e-5")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Create concise run name
    full_wandb_run_name = (
        f"SFT_{model_name}_" f"lr{lr_str}_" f"bs{effective_batch_size}_" f"ep{training_args.num_train_epochs}"
    )

    ################
    # WandB Initialization
    ################
    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity="zsyucla",
            project="sft-math-reasoning",
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_seq_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
            },
        )

    ################
    # Model & Tokenizer
    ################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",  # Use right padding for SFT
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################

    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(format_example)

    # Take 1% of train for evaluation if no eval split exists
    split_dataset = train_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
