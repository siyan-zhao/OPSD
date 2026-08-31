import re

import torch


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the prompt.
    Teacher: sees the prompt plus a reference response/solution.

    Supported dataset schemas:
    - Math OPSD: {"problem": "...", "solution": "..."}
    - SFT-style: {"input": "<chat prompt ending in assistant prefix>", "output": "..."}
    - Teacher-draft style: {"input": "...", "sft_draft": "...", "output": "..."} with teacher_draft_field set
    - Corrector-style: {"input": "...", "sft_draft": "...", "output": "..."}

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        student_thinking=False,
        teacher_thinking=True,
        close_teacher_thinking_before_scoring=False,
        reapply_chat_template_to_input=True,
        problem_field="problem",
        solution_field="solution",
        input_field="input",
        output_field="output",
        draft_field="sft_draft",
        corrector_mode=False,
        teacher_draft_field="",
        teacher_guidance_mode="exact",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.close_teacher_thinking_before_scoring = close_teacher_thinking_before_scoring
        self.reapply_chat_template_to_input = reapply_chat_template_to_input
        self.problem_field = problem_field
        self.solution_field = solution_field
        self.input_field = input_field
        self.output_field = output_field
        self.draft_field = draft_field
        self.corrector_mode = corrector_mode
        self.teacher_draft_field = teacher_draft_field
        self.teacher_guidance_mode = teacher_guidance_mode.lower()
        if self.teacher_guidance_mode not in {"exact", "quality"}:
            raise ValueError("teacher_guidance_mode must be either 'exact' or 'quality'.")
        self.teacher_thinking_prefill = (
            "I have reviewed the private reference answer, quality preferences, and end position. "
            "I will score only final-answer tokens."
        )

        # Prompt for reasoning about the solution before teaching
        self.math_reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        self.generic_reason_first_prompt = (
            "\n\nThe exact target answer above is the only correct final response. "
            "Do NOT use <think> tags. Do NOT analyze, paraphrase, or explain it. "
            "Only remember the exact target answer and that it ends immediately after its final token.\n"
        )

        # Prompt for transitioning to teaching mode after reasoning
        self.math_transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )
        self.generic_exact_transition_prompt = (
            "\n\nThe next assistant response must match the exact target answer above. "
            "Do not add, remove, rephrase, label, or explain anything. "
            "Output exactly the target answer text, then emit the end-of-message token and stop:\n"
        )
        self.generic_quality_transition_prompt = (
            "\n\nUse the reference answer as a private semantic baseline. It is the minimum meaning that the "
            "next assistant response must preserve; it is not a ceiling, but improvements are allowed only "
            "inside strict summarization constraints. Score a response higher only when it is a concise, "
            "final user-visible summary that is at least as faithful as the reference and more natural or "
            "readable. Highest priority is factual precision: preserve who did what to whom, names, numbers, "
            "dates, locations, causal links, contrasts, modality, and uncertainty; never swap subjects or "
            "pronouns, never change the answer type, and never add unsupported facts, time spans, causes, "
            "evaluations, examples, or recommendations. Match the granularity of the task and reference: if "
            "the source or reference is a short direct answer, keep the response short and direct; do not turn "
            "it into a definition, interpretation, or explanation of the phrase. If the reference is already "
            "faithful, direct, and natural, copying it or staying very close to it is preferred over creative "
            "rewriting. For longer noisy transcripts, prefer editorial synthesis: identify the central point, "
            "keep only details that sharpen it, and omit filler or details that make the sentence bulky. Keep "
            "the response length close to the reference and use one smooth sentence unless the reference "
            "clearly requires otherwise. Prefer active, specific wording over passive formulas such as "
            "'is expressed' when a direct subject is available. Reject meta-writing: no labels, no 'Summary:', "
            "no 'Final Answer:', no explanation, no clarification question, no phrase analysis, no "
            "meta-commentary, and no thinking text. "
            "After a complete summary, emit the end-of-message token and stop:\n"
        )
        self.generic_corrector_quality_transition_prompt = (
            "\n\nUse the private reference answer to judge the current SFT draft, but do not behave like a "
            "second independent summarizer. The next assistant response should be a corrected final summary. "
            "Prefer preserving the draft when it is already faithful, concise, and natural. Reward changes only "
            "when they fix factual errors, logical errors, wrong references, pronoun/subject confusion, major "
            "omissions, excessive verbosity, or clearly awkward wording. Do not rewrite just to sound different. "
            "Do not add details that are not supported by the original text or private reference. Keep the same "
            "short-summary answer type and stay close to the draft's length unless a correction requires otherwise. "
            "Return only the final corrected summary: no labels, no 'Summary:', no explanation, no clarification "
            "question, no phrase analysis, no meta-commentary, and no thinking text. "
            "After a complete corrected summary, emit the end-of-message token and stop:\n"
        )
        self.teacher_draft_guidance_text = (
            "Use this SFT baseline as a teacher-only style, format, and brevity reference. The student cannot "
            "see it, so do not require the next response to explicitly edit it. When scoring the student's "
            "next response, prefer outputs that preserve the baseline's useful concision and naturalness while "
            "fixing any factual, logical, reference, pronoun, or coverage errors using the private reference. "
            "Penalize outputs that are worse than the baseline, overlong, labeled, explanatory, repetitive, or "
            "unsupported. The private reference remains the factual authority; the SFT baseline is only a "
            "comparison point for style and common failure modes.\n"
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(
            "[DataCollator] Close teacher thinking before scoring: "
            f"{self.close_teacher_thinking_before_scoring}"
        )
        print(f"[DataCollator] Reapply chat template to input: {self.reapply_chat_template_to_input}")
        print(f"[DataCollator] Teacher guidance mode: {self.teacher_guidance_mode}")
        print(f"[DataCollator] Corrector mode: {self.corrector_mode}")
        if self.corrector_mode:
            print(f"[DataCollator] Draft field: {self.draft_field}")
        print(f"[DataCollator] Teacher draft field: {self.teacher_draft_field or '<disabled>'}")
        print(
            "[DataCollator] Supported schemas: "
            f"{self.problem_field}/{self.solution_field}, "
            f"{self.input_field}/{self.output_field}, and "
            f"{self.input_field}/{self.draft_field}/{self.output_field} when corrector_mode=True "
            f"or teacher_draft_field is set"
        )

    def _maybe_close_teacher_thinking(self, teacher_prompt):
        if not (self.teacher_thinking and self.close_teacher_thinking_before_scoring):
            return teacher_prompt
        if "</think>" in teacher_prompt.rsplit("<|im_start|>assistant", maxsplit=1)[-1]:
            return teacher_prompt
        return f"{teacher_prompt}{self.teacher_thinking_prefill}\n</think>\n\n"

    def _extract_chatml_user_content(self, prompt):
        """Best-effort extraction of the user text from a Qwen-style ChatML prompt."""
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

    def _parse_chatml_messages(self, prompt):
        """Parse simple ChatML into messages, dropping a trailing empty assistant generation prefix."""
        if "<|im_start|>" not in prompt:
            return [{"role": "user", "content": prompt}]

        messages = []
        pattern = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)
        for match in pattern.finditer(prompt):
            role = match.group(1)
            content = match.group(2)
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            if role == "assistant" and match.end() == len(prompt) and not content.strip():
                continue
            messages.append({"role": role, "content": content.strip()})

        return messages or [{"role": "user", "content": prompt.strip()}]

    def _build_corrector_user_message(self, original_prompt, draft_response):
        return (
            "Original summarization request:\n"
            f"{original_prompt}\n\n"
            "Current SFT draft summary:\n"
            f"{draft_response}\n\n"
            "Revise the draft only if necessary. Fix factual, logical, reference, pronoun, or major omission "
            "errors; remove unsupported details; and make minor wording improvements only when they clearly "
            "improve readability without changing meaning. If the draft is already faithful, concise, and "
            "natural, keep it essentially unchanged. Return only the final corrected summary, with no labels, "
            "explanation, clarification question, or thinking text."
        )

    def _build_teacher_draft_context(self, feature):
        if not self.teacher_draft_field:
            return ""
        if self.teacher_draft_field not in feature:
            available = ", ".join(sorted(feature.keys()))
            raise KeyError(
                "teacher_draft_field is set but the field is missing from this sample. "
                f"Expected '{self.teacher_draft_field}'. Available fields: {available}"
            )
        draft_response = str(feature[self.teacher_draft_field])
        return (
            "Private SFT baseline answer (teacher-only; student cannot see this):\n"
            f"{draft_response}\n\n"
            f"{self.teacher_draft_guidance_text}\n"
        )

    def _build_prompts(self, feature):
        if self.problem_field in feature and self.solution_field in feature:
            problem = feature[self.problem_field]
            solution = feature[self.solution_field]

            student_user_message = (
                f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            )
            student_messages = [{"role": "user", "content": student_user_message}]
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.student_thinking,
            )

            reasoning_user_message = (
                f"Problem: {problem}\n\n"
                f"Here is a correct reasoning to this problem:"
                f"=== Reference Reasoning Start ===\n"
                f"{solution}\n"
                f"=== Reference Reasoning End ===\n\n"
                f"{self.math_reason_first_prompt}"
            )
            teacher_user_message = (
                f"Problem: {problem}\n\n"
                f"Here is a reference solution to this problem:\n"
                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                f"{self.math_transition_prompt}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}."
            )
            transition_text = (
                f"\n{self.math_transition_prompt}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}."
            )
            return student_prompt, reasoning_user_message, teacher_user_message, transition_text, solution

        if self.input_field in feature and self.output_field in feature:
            source_prompt = str(feature[self.input_field])
            reference_response = str(feature[self.output_field])

            # Re-apply the current model's chat template by default. Older SFT data may already
            # contain a ChatML assistant prefix, but not the Qwen3/Qwen3.5 thinking markers.
            # Re-templating keeps student_thinking/teacher_thinking behavior consistent.
            source_messages = self._parse_chatml_messages(source_prompt)
            original_prompt = self._extract_chatml_user_content(source_prompt)
            teacher_draft_context = self._build_teacher_draft_context(feature)
            if self.corrector_mode:
                if self.draft_field not in feature:
                    available = ", ".join(sorted(feature.keys()))
                    raise KeyError(
                        "corrector_mode=True requires a draft summary field. "
                        f"Expected '{self.draft_field}'. Available fields: {available}"
                    )
                draft_response = str(feature[self.draft_field])
                corrector_user_message = self._build_corrector_user_message(original_prompt, draft_response)
                student_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": corrector_user_message}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.student_thinking,
                )

                if self.teacher_guidance_mode == "quality":
                    reasoning_user_message = (
                        f"{corrector_user_message}\n\n"
                        f"Private reference answer (use to decide whether the SFT draft needs correction):\n"
                        f"{reference_response}\n"
                        f"{self.generic_corrector_quality_transition_prompt}"
                    )
                    teacher_user_message = reasoning_user_message
                    transition_text = f"\n{self.generic_corrector_quality_transition_prompt}"
                else:
                    reasoning_user_message = (
                        f"{corrector_user_message}\n\n"
                        f"Exact target answer (the corrected response ends immediately after this text):\n"
                        f"{reference_response}\n"
                        f"{self.generic_reason_first_prompt}"
                    )
                    teacher_user_message = (
                        f"{corrector_user_message}\n\n"
                        f"Exact target answer (the corrected response ends immediately after this text):\n"
                        f"{reference_response}\n"
                        f"{self.generic_exact_transition_prompt}"
                    )
                    transition_text = f"\n{self.generic_exact_transition_prompt}"
                return student_prompt, reasoning_user_message, teacher_user_message, transition_text, reference_response

            if "<|im_start|>" in source_prompt and not self.reapply_chat_template_to_input:
                student_prompt = source_prompt
            else:
                student_prompt = self.tokenizer.apply_chat_template(
                    source_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.student_thinking,
                )

            if self.teacher_guidance_mode == "quality":
                reasoning_user_message = (
                    f"{original_prompt}\n\n"
                    f"{teacher_draft_context}"
                    f"Private reference answer (use for meaning and coverage, not exact wording):\n"
                    f"{reference_response}\n"
                    f"{self.generic_quality_transition_prompt}"
                )
                teacher_user_message = (
                    f"{original_prompt}\n\n"
                    f"{teacher_draft_context}"
                    f"Private reference answer (use for meaning and coverage, not exact wording):\n"
                    f"{reference_response}\n"
                    f"{self.generic_quality_transition_prompt}"
                )
                transition_text = f"\n{self.generic_quality_transition_prompt}"
            else:
                reasoning_user_message = (
                    f"{original_prompt}\n\n"
                    f"{teacher_draft_context}"
                    f"Exact target answer (the response ends immediately after this text):\n"
                    f"{reference_response}\n"
                    f"{self.generic_reason_first_prompt}"
                )
                teacher_user_message = (
                    f"{original_prompt}\n\n"
                    f"{teacher_draft_context}"
                    f"Exact target answer (the response ends immediately after this text):\n"
                    f"{reference_response}\n"
                    f"{self.generic_exact_transition_prompt}"
                )
                transition_text = f"\n{self.generic_exact_transition_prompt}"
            return student_prompt, reasoning_user_message, teacher_user_message, transition_text, reference_response

        available = ", ".join(sorted(feature.keys()))
        raise KeyError(
            "Unsupported dataset schema. Expected either "
            f"'{self.problem_field}'/'{self.solution_field}' or "
            f"'{self.input_field}'/'{self.output_field}'. Available fields: {available}"
        )

    def _target_with_eos(self, response):
        eos_token = self.tokenizer.eos_token or ""
        if eos_token and not response.endswith(eos_token):
            return f"{response}{eos_token}"
        return response

    def _build_sft_batch(self, student_prompts, reference_responses):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        prompt_encoded = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        target_texts = [self._target_with_eos(response) for response in reference_responses]
        target_encoded = self.tokenizer(
            target_texts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        input_ids_list = []
        labels_list = []
        for prompt_ids, target_ids in zip(prompt_encoded["input_ids"], target_encoded["input_ids"]):
            if len(target_ids) > self.max_length:
                target_ids = target_ids[: self.max_length]

            max_prompt_len = max(0, self.max_length - len(target_ids))
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[:max_prompt_len]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        max_sft_len = max(len(input_ids) for input_ids in input_ids_list)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_sft_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
            padded_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        return {
            "sft_input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "sft_attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "sft_labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode
        teacher_transition_texts = []
        reference_responses = []

        for feature in features:
            (
                student_prompt,
                reasoning_user_message,
                teacher_user_message,
                transition_text,
                reference_response,
            ) = self._build_prompts(feature)
            student_prompts.append(student_prompt)
            teacher_transition_texts.append(transition_text)
            reference_responses.append(reference_response)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze the solution
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                # Original teacher prompt (unchanged)
                teacher_messages = [{"role": "user", "content": teacher_user_message}]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
                )
                teacher_prompt = self._maybe_close_teacher_thinking(teacher_prompt)
                teacher_prompts.append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }
        result.update(self._build_sft_batch(student_prompts, reference_responses))

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_encoded = self.tokenizer(
                teacher_transition_texts,
                padding="longest",
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
