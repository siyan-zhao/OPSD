#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BAD_PHRASES = [
    "<think>",
    "</think>",
    "the user wants",
    "summary:",
    "**summary**",
    "explanation:",
    "**explanation**",
    "final answer:",
    "**final answer**",
    "the text describes",
    "the provided text",
]


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def load_model_class(model_loader: str):
    if model_loader == "causal_lm":
        return AutoModelForCausalLM
    if model_loader == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    raise ValueError(f"Unknown model_loader: {model_loader}")


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype: {dtype}")
    return mapping[dtype]


def extract_chatml_user_content(prompt: str) -> str:
    start_marker = "<|im_start|>user"
    end_marker = "<|im_end|>"
    start = prompt.find(start_marker)
    if start == -1:
        return prompt.strip()
    start += len(start_marker)
    end = prompt.find(end_marker, start)
    if end == -1:
        return prompt[start:].strip()
    return prompt[start:end].strip()


def parse_chatml_messages(prompt: str):
    if "<|im_start|>" not in prompt:
        return [{"role": "user", "content": prompt}]

    messages = []
    pattern = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)
    for match in pattern.finditer(prompt):
        role = match.group(1)
        content = match.group(2).strip()
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if role == "assistant" and match.end() == len(prompt) and not content:
            continue
        messages.append({"role": role, "content": content})
    return messages or [{"role": "user", "content": extract_chatml_user_content(prompt)}]


def parse_indices(raw_indices: str | None):
    if not raw_indices:
        return None
    indices = []
    for item in raw_indices.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def load_examples(args):
    path = Path(args.data_file)
    if not path.exists():
        raise FileNotFoundError(f"data_file does not exist: {path}")

    requested = parse_indices(args.sample_indices)
    requested_set = set(requested) if requested is not None else None
    examples = []

    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if requested_set is not None:
                if line_index not in requested_set:
                    continue
            else:
                if line_index < args.start_index:
                    continue
                if (line_index - args.start_index) % args.stride != 0:
                    continue

            example = json.loads(line)
            if args.input_field not in example:
                available = ", ".join(sorted(example.keys()))
                raise KeyError(f"Input field '{args.input_field}' not found. Available fields: {available}")

            raw_prompt = str(example[args.input_field])
            reference = str(example.get(args.output_field, ""))
            examples.append(
                {
                    "index": line_index,
                    "raw_prompt": raw_prompt,
                    "prompt": extract_chatml_user_content(raw_prompt),
                    "messages": parse_chatml_messages(raw_prompt),
                    "reference": reference,
                }
            )

            if requested_set is None and len(examples) >= args.num_samples:
                break
            if requested_set is not None and len(examples) >= len(requested_set):
                break

    if not examples:
        raise ValueError(f"No examples selected from {path}")
    return examples


def detect_bad_phrases(text: str):
    lowered = text.lower()
    return [phrase for phrase in BAD_PHRASES if phrase in lowered]


def completion_token_count(tokenizer, text: str):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def main():
    parser = argparse.ArgumentParser(description="Batch-check a Qwen3.5 summary LoRA checkpoint with Torch.")
    parser.add_argument("--model", default="/DATA_A/models/Qwen3.5-2B", help="Base model path.")
    parser.add_argument("--checkpoint_dir", "--adapter", dest="checkpoint_dir", default=None, help="LoRA checkpoint directory.")
    parser.add_argument("--data_file", required=True, help="Local input/output JSONL file.")
    parser.add_argument("--output_file", default=None, help="Optional JSONL file for predictions.")
    parser.add_argument("--metrics_file", default=None, help="Optional JSON file for aggregate metrics.")
    parser.add_argument("--model_loader", default="image_text_to_text", choices=["causal_lm", "image_text_to_text"])
    parser.add_argument("--input_field", default="input")
    parser.add_argument("--output_field", default="output")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--sample_indices", default=None, help="Comma-separated 0-based indices, e.g. 0,10,100.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--print_limit",
        type=int,
        default=None,
        help="Maximum number of per-sample predictions to print. Set 0 to print only aggregate metrics.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--enable_thinking", default="False")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    examples = load_examples(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = load_model_class(args.model_loader)
    model = model_cls.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_path}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint_path))

    model.eval()
    input_device = next(model.parameters()).device

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})

    rows = []
    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        rendered_prompts = [
            tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=str_to_bool(args.enable_thinking),
            )
            for item in batch
        ]
        inputs = tokenizer(rendered_prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        for item, ids in zip(batch, output_ids):
            completion_ids = ids[prompt_length:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
            flags = detect_bad_phrases(completion)
            rows.append(
                {
                    "index": item["index"],
                    "prompt": item["prompt"],
                    "reference": item["reference"],
                    "prediction": completion,
                    "prediction_tokens": completion_token_count(tokenizer, completion),
                    "format_flags": flags,
                }
            )

    flagged = sum(1 for row in rows if row["format_flags"])
    avg_tokens = sum(row["prediction_tokens"] for row in rows) / len(rows)
    metrics = {
        "num_samples": len(rows),
        "average_prediction_tokens": avg_tokens,
        "format_flagged_samples": flagged,
        "format_flagged_ratio": flagged / len(rows),
        "max_prediction_tokens": max(row["prediction_tokens"] for row in rows),
        "min_prediction_tokens": min(row["prediction_tokens"] for row in rows),
    }
    print(f"\nEvaluated {len(rows)} samples")
    print(f"Average prediction tokens: {avg_tokens:.1f}")
    print(f"Format-flagged samples: {flagged}/{len(rows)}")

    rows_to_print = rows if args.print_limit is None else rows[: max(0, args.print_limit)]
    for row in rows_to_print:
        print("\n" + "=" * 88)
        print(f"Index: {row['index']} | tokens: {row['prediction_tokens']} | flags: {row['format_flags'] or 'none'}")
        print("\nPrompt:")
        print(row["prompt"][:1200])
        print("\nReference:")
        print(row["reference"])
        print("\nPrediction:")
        print(row["prediction"])

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved predictions to: {output_path}")

    if args.metrics_file:
        metrics_path = Path(args.metrics_file)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
