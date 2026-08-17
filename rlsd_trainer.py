import torch

from trl import GRPOTrainer


class RLSDTrainer(GRPOTrainer):
    """
    GRPO with RLSD token-level credit redistribution.

    The verifier reward still determines the sign of the update through the
    sequence-level GRPO advantage. A privileged teacher prompt is used only to
    reweight token-level magnitudes:

        delta_t = sg(log p_teacher(y_t) - log p_student(y_t))
        w_t = exp(sign(A) * delta_t)
        A_t = A * ((1 - lambda) + lambda * clip(w_t, 1-eps_w, 1+eps_w))
    """

    _tag_names = ["trl", "grpo", "rlsd"]
    _name = "RLSD"

    def __init__(
        self,
        *args,
        rlsd_lambda: float = 0.5,
        rlsd_lambda_decay_steps: int = 50,
        rlsd_epsilon_w: float = 0.2,
        teacher_max_prompt_length: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.use_liger_kernel:
            raise ValueError("RLSDTrainer requires use_liger_kernel=False because it supplies per-token advantages.")

        self.rlsd_lambda = rlsd_lambda
        self.rlsd_lambda_decay_steps = rlsd_lambda_decay_steps
        self.rlsd_epsilon_w = rlsd_epsilon_w
        self.teacher_max_prompt_length = teacher_max_prompt_length

    def _current_rlsd_lambda(self) -> float:
        if self.rlsd_lambda_decay_steps <= 0:
            return self.rlsd_lambda
        progress = min(float(self.state.global_step) / float(self.rlsd_lambda_decay_steps), 1.0)
        return self.rlsd_lambda * (1.0 - progress)

    def _tokenize_teacher_prompts(self, teacher_prompts: list[str], completion_ids: torch.Tensor):
        encoded = self.processing_class(
            teacher_prompts,
            padding=True,
            truncation=self.teacher_max_prompt_length is not None,
            max_length=self.teacher_max_prompt_length,
            return_tensors="pt",
        )
        teacher_prompt_ids = encoded["input_ids"].to(completion_ids.device)
        teacher_prompt_mask = encoded["attention_mask"].to(completion_ids.device)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, torch.ones_like(completion_ids)], dim=1)
        teacher_attention_mask[teacher_input_ids == self.pad_token_id] = 0
        return teacher_input_ids, teacher_attention_mask

    def _get_student_logps(self, output: dict[str, torch.Tensor], batch_size: int):
        if output.get("old_per_token_logps") is not None:
            return output["old_per_token_logps"]

        prompt_completion_ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
        attention_mask = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)
        logits_to_keep = output["completion_ids"].size(1)
        student_logps, _ = self._get_per_token_logps_and_entropies(
            self.model,
            prompt_completion_ids,
            attention_mask,
            logits_to_keep,
            batch_size=batch_size,
            num_images=output.get("num_images"),
            pixel_values=output.get("pixel_values"),
            image_grid_thw=output.get("image_grid_thw"),
            pixel_attention_mask=output.get("pixel_attention_mask"),
            image_sizes=output.get("image_sizes"),
            token_type_ids=output.get("token_type_ids"),
        )
        output["old_per_token_logps"] = student_logps
        return student_logps

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        if "teacher_prompt" not in inputs[0]:
            raise KeyError("RLSDTrainer expects each dataset row to contain a 'teacher_prompt' column.")

        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        teacher_prompts = [example["teacher_prompt"] for example in inputs]
        teacher_prompts = [prompt for prompt in teacher_prompts for _ in range(num_generations)]

        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            student_logps = self._get_student_logps(output, batch_size=batch_size)
            teacher_input_ids, teacher_attention_mask = self._tokenize_teacher_prompts(teacher_prompts, completion_ids)
            teacher_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
                batch_size=batch_size,
            )

            sequence_advantages = output["advantages"]
            delta = teacher_logps - student_logps
            weights = torch.exp(sequence_advantages.sign().unsqueeze(1) * delta)
            weights = torch.clamp(weights, 1.0 - self.rlsd_epsilon_w, 1.0 + self.rlsd_epsilon_w)

            lambda_t = self._current_rlsd_lambda()
            token_advantages = sequence_advantages.unsqueeze(1) * ((1.0 - lambda_t) + lambda_t * weights)
            token_advantages = token_advantages * completion_mask

        output["advantages"] = token_advantages

        valid_weights = weights[completion_mask.bool()]
        if valid_weights.numel() > 0:
            self._metrics[mode]["rlsd/weight_mean"].append(self.accelerator.gather(valid_weights.mean()).nanmean().item())
            self._metrics[mode]["rlsd/weight_min"].append(self.accelerator.gather(valid_weights.min()).min().item())
            self._metrics[mode]["rlsd/weight_max"].append(self.accelerator.gather(valid_weights.max()).max().item())
        self._metrics[mode]["rlsd/lambda"].append(lambda_t)

        return output
