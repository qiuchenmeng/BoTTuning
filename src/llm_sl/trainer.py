"""Hugging Face Trainer using one candidate-normalized KL objective."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer

from .data import MixedSoftLabelDataset, RatioBatchSampler, parse_ratio


def candidate_kl_loss(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute KL over candidate logits at each prompt's next-token position."""

    last_indices = attention_mask.sum(dim=1) - 1
    if torch.any(last_indices < 0):
        raise ValueError("attention_mask contains an empty prompt")
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    next_token_logits = logits[batch_indices, last_indices.to(logits.device), :]
    candidate_logits = next_token_logits.gather(
        1, candidate_ids.to(next_token_logits.device)
    )
    normalized_targets = targets.to(candidate_logits.device, candidate_logits.dtype)
    normalized_targets = normalized_targets / normalized_targets.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    return F.kl_div(
        F.log_softmax(candidate_logits, dim=-1),
        normalized_targets,
        reduction="batchmean",
    )


class SoftLabelTrainer(Trainer):
    """Train BOT and normal-QA samples with the same KL implementation."""

    def __init__(
        self,
        *args: Any,
        batch_sample_ratio: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_sample_ratio = parse_ratio(batch_sample_ratio)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        model_inputs = dict(inputs)
        candidate_ids = model_inputs.pop("candidate_ids")
        targets = model_inputs.pop("targets")
        outputs = model(**model_inputs)
        loss = candidate_kl_loss(
            outputs.logits,
            model_inputs["attention_mask"],
            candidate_ids,
            targets,
        )
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        if self.batch_sample_ratio is None:
            return super().get_train_dataloader()
        if not isinstance(self.train_dataset, MixedSoftLabelDataset):
            raise TypeError(
                "batch_sample_ratio requires MixedSoftLabelDataset as train_dataset"
            )

        bot_indices, qa_indices = self.train_dataset.group_indices()
        bot_ratio, qa_ratio = self.batch_sample_ratio
        batch_sampler = RatioBatchSampler(
            bot_indices,
            qa_indices,
            batch_size=self._train_batch_size,
            bot_ratio=bot_ratio,
            qa_ratio=qa_ratio,
            seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
            drop_last=self.args.dataloader_drop_last,
            world_size=self.args.world_size,
            process_index=self.args.process_index,
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=(
                self.args.dataloader_persistent_workers
                and self.args.dataloader_num_workers > 0
            ),
        )
        return dataloader
