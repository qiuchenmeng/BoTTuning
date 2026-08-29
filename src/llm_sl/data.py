"""Validated mixed BOT/QA data loading, sampling, and Qwen collation."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal


TaskType = Literal["bot", "binary", "4choice"]
ExpectedGroup = Literal["bot", "qa"]

TASK_ALIASES: dict[str, TaskType] = {
    "bot": "bot",
    "toy": "bot",
    "binary": "binary",
    "4choice": "4choice",
    "mcq_4way": "4choice",
}
ALLOWED_ROLES = {"system", "user", "assistant"}


def normalize_task_type(value: object) -> TaskType:
    """Normalize legacy task names to the public BOT/binary/4choice names."""

    if value is None:
        return "bot"
    if not isinstance(value, str) or value not in TASK_ALIASES:
        choices = ", ".join(TASK_ALIASES)
        raise ValueError(f"task_type must be one of: {choices}")
    return TASK_ALIASES[value]


def _row_error(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {message}")


def _validate_row(
    raw: object, path: Path, line_number: int, expected_group: ExpectedGroup
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise _row_error(path, line_number, "row must be a JSON object")

    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _row_error(path, line_number, "messages must be a non-empty list")
    normalized_messages: list[dict[str, str]] = []
    has_user = False
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise _row_error(
                path, line_number, f"messages[{position}] must be an object"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise _row_error(
                path,
                line_number,
                f"messages[{position}].role must be system, user, or assistant",
            )
        if not isinstance(content, str):
            raise _row_error(
                path, line_number, f"messages[{position}].content must be a string"
            )
        has_user = has_user or role == "user"
        normalized_messages.append({"role": role, "content": content})
    if not has_user:
        raise _row_error(path, line_number, "messages must contain a user message")

    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise _row_error(path, line_number, "candidates must be a non-empty list")
    if any(not isinstance(candidate, str) or not candidate.strip() for candidate in candidates):
        raise _row_error(
            path, line_number, "candidates must contain non-empty strings"
        )
    if len(set(candidates)) != len(candidates):
        raise _row_error(path, line_number, "candidates must be unique")

    target = raw.get("soft_label")
    if not isinstance(target, list) or len(target) != len(candidates):
        raise _row_error(
            path,
            line_number,
            "soft_label must be a list matching the candidate count",
        )
    values: list[float] = []
    for position, value in enumerate(target):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _row_error(
                path, line_number, f"soft_label[{position}] must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise _row_error(
                path,
                line_number,
                f"soft_label[{position}] must be finite and non-negative",
            )
        values.append(numeric)
    total = sum(values)
    if total <= 0:
        raise _row_error(path, line_number, "soft_label sum must be positive")

    try:
        task_type = normalize_task_type(raw.get("task_type"))
    except ValueError as exc:
        raise _row_error(path, line_number, str(exc)) from exc
    if expected_group == "bot" and task_type != "bot":
        raise _row_error(path, line_number, "BOT file rows must use task_type bot")
    if expected_group == "qa" and task_type not in {"binary", "4choice"}:
        raise _row_error(
            path, line_number, "normal-QA rows must use task_type binary or 4choice"
        )

    normalized = dict(raw)
    normalized["messages"] = normalized_messages
    normalized["candidates"] = list(candidates)
    normalized["soft_label"] = [value / total for value in values]
    normalized["task_type"] = task_type
    return normalized


def load_jsonl(
    path: str | Path, expected_group: ExpectedGroup
) -> list[dict[str, object]]:
    """Load and fully validate one BOT or QA JSONL file."""

    source = Path(path)
    rows: list[dict[str, object]] = []
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{source}: cannot open JSONL file: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _row_error(
                    source, line_number, f"invalid JSON: {exc.msg}"
                ) from exc
            rows.append(_validate_row(raw, source, line_number, expected_group))
    if not rows:
        raise ValueError(f"{source}: JSONL file contains no data rows")
    return rows


class MixedSoftLabelDataset(Sequence[dict[str, object]]):
    """BOT rows followed by normal-QA rows, all with normalized targets."""

    def __init__(
        self, soft_label_path: str | Path, normal_qa_path: str | Path
    ) -> None:
        self.bot_rows = load_jsonl(soft_label_path, expected_group="bot")
        self.qa_rows = load_jsonl(normal_qa_path, expected_group="qa")
        self.rows = self.bot_rows + self.qa_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]

    def group_indices(self) -> tuple[list[int], list[int]]:
        bot_end = len(self.bot_rows)
        return list(range(bot_end)), list(range(bot_end, len(self.rows)))


def parse_ratio(value: str | None) -> tuple[int, int] | None:
    """Parse an optional BOT:QA integer sampling ratio."""

    if value is None:
        return None
    try:
        left, right = value.split(":", maxsplit=1)
        bot_ratio, qa_ratio = int(left), int(right)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("batch_sample_ratio must use positive integers BOT:QA") from exc
    if bot_ratio <= 0 or qa_ratio <= 0:
        raise ValueError("batch_sample_ratio must use positive integers BOT:QA")
    return bot_ratio, qa_ratio


class RatioBatchSampler:
    """Create deterministic batches from a repeated BOT:QA index pattern."""

    def __init__(
        self,
        bot_indices: Sequence[int],
        qa_indices: Sequence[int],
        batch_size: int,
        bot_ratio: int,
        qa_ratio: int,
        *,
        seed: int = 42,
        drop_last: bool = False,
        world_size: int = 1,
        process_index: int = 0,
    ) -> None:
        if not bot_indices or not qa_indices:
            raise ValueError("ratio sampling requires non-empty BOT and QA groups")
        if min(batch_size, bot_ratio, qa_ratio, world_size) <= 0:
            raise ValueError("batch size, ratios, and world size must be positive")
        if not 0 <= process_index < world_size:
            raise ValueError("process_index must be within world_size")
        self.bot_indices = list(bot_indices)
        self.qa_indices = list(qa_indices)
        self.batch_size = batch_size
        self.bot_ratio = bot_ratio
        self.qa_ratio = qa_ratio
        self.seed = seed
        self.drop_last = drop_last
        self.world_size = world_size
        self.process_index = process_index
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _all_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        bot = list(self.bot_indices)
        qa = list(self.qa_indices)
        rng.shuffle(bot)
        rng.shuffle(qa)
        cycles = max(
            math.ceil(len(bot) / self.bot_ratio),
            math.ceil(len(qa) / self.qa_ratio),
        )
        mixed: list[int] = []
        bot_position = 0
        qa_position = 0
        for _ in range(cycles):
            for _ in range(self.bot_ratio):
                if bot_position and bot_position % len(bot) == 0:
                    rng.shuffle(bot)
                mixed.append(bot[bot_position % len(bot)])
                bot_position += 1
            for _ in range(self.qa_ratio):
                if qa_position and qa_position % len(qa) == 0:
                    rng.shuffle(qa)
                mixed.append(qa[qa_position % len(qa)])
                qa_position += 1
        batches = [
            mixed[start : start + self.batch_size]
            for start in range(0, len(mixed), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._all_batches()
        yield from batches[self.process_index :: self.world_size]

    def __len__(self) -> int:
        count = len(self._all_batches())
        return max(0, (count - self.process_index + self.world_size - 1) // self.world_size)


class SoftLabelCollator:
    """Render Qwen prompts and resolve every candidate to exactly one token."""

    def __init__(self, tokenizer: Any, max_length: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _candidate_token_id(self, candidate: str) -> int:
        for text in (" " + candidate, candidate):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                return int(token_ids[0])
        raise ValueError(
            f"candidate {candidate!r} is not a single Qwen token with or without "
            "a leading space"
        )

    def __call__(self, features: Sequence[dict[str, object]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        candidate_counts = {len(feature["candidates"]) for feature in features}
        if len(candidate_counts) != 1:
            raise ValueError(
                "rows in one batch have different candidate counts; use "
                "per_device_train_batch_size: 1 or bucket rows by candidate count"
            )

        import torch

        prompts = [
            self.tokenizer.apply_chat_template(
                feature["messages"], tokenize=False, add_generation_prompt=True
            )
            for feature in features
        ]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        candidate_ids = [
            [self._candidate_token_id(candidate) for candidate in feature["candidates"]]
            for feature in features
        ]
        targets = torch.tensor(
            [feature["soft_label"] for feature in features], dtype=torch.float32
        )
        targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "targets": targets,
        }
