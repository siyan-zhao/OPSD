#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoConfig, AutoTokenizer


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_model_loader(name: str):
    if name == "causal_lm":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    if name == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    raise ValueError(f"Unknown loader: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/Users/hyh/Desktop/Qwen3.5_2B")
    parser.add_argument("--loader", default="image_text_to_text", choices=["causal_lm", "image_text_to_text"])
    parser.add_argument("--load_model", action="store_true", help="Actually instantiate weights and inspect modules.")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print("config class:", type(config).__name__)
    print("model_type:", getattr(config, "model_type", None))
    print("architectures:", getattr(config, "architectures", None))
    print("has vision_config:", getattr(config, "vision_config", None) is not None)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hello."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("chat template enable_thinking works:", "<think>" in prompt)

    try:
        from transformers import AutoModelForImageTextToText

        print("AutoModelForImageTextToText available:", AutoModelForImageTextToText is not None)
    except Exception as exc:
        print("AutoModelForImageTextToText available: False")
        print("import error:", repr(exc))

    if not args.load_model:
        print("Skipped weight loading. Re-run with --load_model on the 5090 machine to inspect LoRA matches.")
        return

    loader = get_model_loader(args.loader)
    model = loader.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu",
    )
    matched = []
    for name, module in model.named_modules():
        if any(name.endswith(target) for target in TARGET_MODULES):
            matched.append(name)

    print("matched LoRA modules:", len(matched))
    for name in matched[:120]:
        print(name)
    if len(matched) > 120:
        print(f"... {len(matched) - 120} more")


if __name__ == "__main__":
    main()
