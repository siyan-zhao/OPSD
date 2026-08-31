import os
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (matches the main OPSD setup: student rolls out without <think>)."
        },
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher when scoring student tokens. "
            "Default True. Set to False for the matched non-thinking ablation (both nonthink)."
        },
    )
    close_teacher_thinking_before_scoring: bool = field(
        default=False,
        metadata={
            "help": "If teacher_thinking=True, append a teacher-only hidden thought closure before "
            "student tokens are scored. This keeps teacher thinking enabled while aligning the "
            "scoring position with the student's final-answer region."
        },
    )
    reapply_chat_template_to_input: bool = field(
        default=True,
        metadata={
            "help": "For input/output datasets, extract the user content and apply the current tokenizer's "
            "chat template instead of using an older pre-templated input verbatim."
        },
    )
    model_loader: str = field(
        default="auto",
        metadata={
            "help": "Model auto-loader to use. Choose from: auto, causal_lm, image_text_to_text. "
            "Qwen3.5-2B uses a multimodal ForConditionalGeneration architecture, so auto will "
            "prefer AutoModelForImageTextToText when a vision_config is present."
        },
    )
    opsd_dataset: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd",
        metadata={
            "help": "Training dataset name or local path. Supports Hugging Face dataset names, "
            "local dataset directories, and local .json/.jsonl/.csv files."
        },
    )
    opsd_dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to load."},
    )
    problem_field: str = field(
        default="problem",
        metadata={"help": "Problem field for math-style datasets."},
    )
    solution_field: str = field(
        default="solution",
        metadata={"help": "Solution field for math-style datasets."},
    )
    input_field: str = field(
        default="input",
        metadata={"help": "Prompt field for SFT-style datasets."},
    )
    output_field: str = field(
        default="output",
        metadata={"help": "Reference response field for SFT-style datasets."},
    )
    draft_field: str = field(
        default="sft_draft",
        metadata={"help": "Draft summary field for corrector-style datasets."},
    )
    corrector_mode: bool = field(
        default=False,
        metadata={
            "help": "Train OPSD/SFT as a summary corrector using input + draft_field -> output. "
            "When enabled, each input/output sample must also contain draft_field."
        },
    )
    teacher_draft_field: str = field(
        default="",
        metadata={
            "help": "Optional SFT draft field shown only to the teacher as a private baseline. "
            "The student still sees only input unless corrector_mode=True."
        },
    )
    teacher_guidance_mode: str = field(
        default="exact",
        metadata={
            "help": "Teacher prompt style for input/output datasets. "
            "Use 'exact' to prefer the reference wording, or 'quality' to use the reference as "
            "private meaning guidance while preferring faithful, concise, natural summaries."
        },
    )
    opsd_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the on-policy OPSD distillation loss."},
    )
    sft_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight for supervised cross-entropy loss on input/output targets. "
            "This directly trains output tokens plus EOS and helps anchor short-answer format."
        },
    )


def load_model_for_training(model_args, model_kwargs, model_loader: str = "auto"):
    """Load text-only and image-text Qwen-family models with the right AutoModel class."""
    from transformers import AutoConfig, AutoModelForCausalLM

    loader = (model_loader or "auto").lower()
    valid_loaders = {"auto", "causal_lm", "image_text_to_text"}
    if loader not in valid_loaders:
        raise ValueError(f"model_loader must be one of {sorted(valid_loaders)}, got: {model_loader}")

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    archs = list(getattr(config, "architectures", []) or [])
    has_vision_config = getattr(config, "vision_config", None) is not None
    prefers_image_text = has_vision_config or any("ForConditionalGeneration" in arch for arch in archs)

    def get_image_text_loader():
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError as exc:
            raise ImportError(
                "AutoModelForImageTextToText is not available in this transformers install. "
                "Qwen3.5 requires a recent transformers build; the model README recommends "
                "`pip install \"transformers[serving] @ git+https://github.com/huggingface/transformers.git@main\"`."
            ) from exc

    if loader == "causal_lm":
        loader_plan = [("AutoModelForCausalLM", AutoModelForCausalLM)]
    elif loader == "image_text_to_text":
        loader_plan = [("AutoModelForImageTextToText", get_image_text_loader)]
    elif prefers_image_text:
        loader_plan = [
            ("AutoModelForImageTextToText", get_image_text_loader),
            ("AutoModelForCausalLM", AutoModelForCausalLM),
        ]
    else:
        loader_plan = [
            ("AutoModelForCausalLM", AutoModelForCausalLM),
            ("AutoModelForImageTextToText", get_image_text_loader),
        ]

    failures = []
    for loader_name, loader_or_factory in loader_plan:
        try:
            loader_cls = (
                loader_or_factory()
                if callable(loader_or_factory) and not hasattr(loader_or_factory, "from_pretrained")
                else loader_or_factory
            )
            print(f"Loading model with {loader_name}")
            model = loader_cls.from_pretrained(model_args.model_name_or_path, **model_kwargs)
            if not getattr(model.config, "_name_or_path", None):
                model.config._name_or_path = model_args.model_name_or_path
            return model
        except Exception as exc:
            failures.append(f"{loader_name}: {type(exc).__name__}: {exc}")
            if loader != "auto":
                break

    joined = "\n".join(failures)
    raise RuntimeError(f"Could not load model {model_args.model_name_or_path}.\n{joined}")


def load_opsd_dataset(script_args):
    """Load the OPSD training dataset from Hugging Face or a local data file."""
    from pathlib import Path

    dataset_ref = script_args.opsd_dataset
    dataset_path = Path(dataset_ref).expanduser()
    split = script_args.opsd_dataset_split

    if dataset_path.exists():
        if dataset_path.is_file():
            suffix = dataset_path.suffix.lower()
            if suffix in {".json", ".jsonl"}:
                return load_dataset("json", data_files=str(dataset_path), split=split)
            if suffix == ".csv":
                return load_dataset("csv", data_files=str(dataset_path), split=split)
            raise ValueError(
                f"Unsupported local dataset file extension: {suffix}. Use .json, .jsonl, or .csv."
            )
        return load_dataset(str(dataset_path), split=split)

    return load_dataset(dataset_ref, split=split)


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "opsd_loss_weight": script_args.opsd_loss_weight,
                "sft_loss_weight": script_args.sft_loss_weight,
                "teacher_guidance_mode": script_args.teacher_guidance_mode,
                "corrector_mode": script_args.corrector_mode,
                "draft_field": script_args.draft_field if script_args.corrector_mode else None,
                "teacher_draft_field": script_args.teacher_draft_field or None,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    # Keep training forwards from materializing KV cache. Generation methods
    # temporarily enable cache around model.generate() and restore this value.
    model_use_cache = False
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    train_dataset = load_opsd_dataset(script_args)

    model = load_model_for_training(model_args, model_kwargs, script_args.model_loader)
    model.config.use_cache = model_use_cache
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = model_use_cache

    trainer = OPSDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        close_teacher_thinking_before_scoring=script_args.close_teacher_thinking_before_scoring,
        reapply_chat_template_to_input=script_args.reapply_chat_template_to_input,
        problem_field=script_args.problem_field,
        solution_field=script_args.solution_field,
        input_field=script_args.input_field,
        output_field=script_args.output_field,
        draft_field=script_args.draft_field,
        corrector_mode=script_args.corrector_mode,
        teacher_draft_field=script_args.teacher_draft_field,
        opsd_loss_weight=script_args.opsd_loss_weight,
        sft_loss_weight=script_args.sft_loss_weight,
        teacher_guidance_mode=script_args.teacher_guidance_mode,
    )

    if training_args.eval_strategy != "no":
        from trl import LogCompletionsCallback

        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)
