"""Portable configuration and command-line entry point for mixed KL tuning."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml


@dataclass(frozen=True)
class TrainConfig:
    model_name_or_path: str | None = None
    soft_label_data: str | None = None
    normal_qa_data: str | None = None
    output_dir: str | None = None
    max_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-5
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    batch_sample_ratio: str | None = None
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: bool = False
    deepspeed_config: str | None = None
    use_wandb: bool = False
    wandb_project: str = "llm-soft-label"
    wandb_run_name: str = "qwen-mixed-kl"
    auto_resume: bool = True
    resume_from_checkpoint: str | None = None


def load_config(path: str | Path | None) -> dict[str, object]:
    """Load a YAML mapping and resolve its DeepSpeed path relative to the YAML."""

    if path is None:
        return {}
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config {source} must contain a YAML mapping")
    known = {field.name for field in fields(TrainConfig)}
    unknown = sorted(set(loaded) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    values = dict(loaded)
    deepspeed = values.get("deepspeed_config")
    if isinstance(deepspeed, str) and not Path(deepspeed).is_absolute():
        values["deepspeed_config"] = str((source.parent / deepspeed).resolve())
    return values


def _add_boolean(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        dest=name,
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Qwen causal LM on mixed BOT and normal-QA JSONL."
    )
    parser.add_argument("--config", default=argparse.SUPPRESS)
    string_fields = (
        "model_name_or_path",
        "soft_label_data",
        "normal_qa_data",
        "output_dir",
        "batch_sample_ratio",
        "deepspeed_config",
        "wandb_project",
        "wandb_run_name",
        "resume_from_checkpoint",
    )
    integer_fields = (
        "max_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "dataloader_num_workers",
        "seed",
    )
    float_fields = ("learning_rate", "num_train_epochs")
    boolean_fields = (
        "dataloader_pin_memory",
        "bf16",
        "fp16",
        "tf32",
        "gradient_checkpointing",
        "ddp_find_unused_parameters",
        "use_wandb",
        "auto_resume",
    )
    for name in string_fields:
        parser.add_argument(
            f"--{name.replace('_', '-')}", dest=name, default=argparse.SUPPRESS
        )
    for name in integer_fields:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=int,
            default=argparse.SUPPRESS,
        )
    for name in float_fields:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=float,
            default=argparse.SUPPRESS,
        )
    for name in boolean_fields:
        _add_boolean(parser, name)
    return parser


def _validate_config(config: TrainConfig) -> None:
    required = (
        "model_name_or_path",
        "soft_label_data",
        "normal_qa_data",
        "output_dir",
    )
    missing = [name for name in required if not getattr(config, name)]
    if missing:
        raise ValueError("missing required config values: " + ", ".join(missing))
    positive = (
        "max_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "logging_steps",
        "save_steps",
        "save_total_limit",
    )
    invalid = [name for name in positive if getattr(config, name) <= 0]
    if invalid:
        raise ValueError("config values must be positive: " + ", ".join(invalid))
    if config.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers cannot be negative")
    if config.bf16 and config.fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled")


def resolve_config(argv: Sequence[str] | None = None) -> TrainConfig:
    """Merge dataclass defaults, YAML, and explicitly supplied CLI values."""

    namespace = vars(build_parser().parse_args(argv))
    config_path = namespace.pop("config", None)
    merged: dict[str, object] = asdict(TrainConfig())
    merged.update(load_config(config_path))
    merged.update(namespace)
    config = TrainConfig(**merged)
    _validate_config(config)
    return config


def _require_optional_package(package: str, extra: str) -> None:
    if importlib.util.find_spec(package) is None:
        raise RuntimeError(
            f"{package} support was requested but is not installed. "
            f"Install with: pip install -e '.[{extra}]'"
        )


def _resolve_checkpoint(config: TrainConfig, get_last_checkpoint: Any) -> str | None:
    if config.resume_from_checkpoint:
        return config.resume_from_checkpoint
    output = Path(config.output_dir or "")
    if config.auto_resume and output.is_dir():
        return get_last_checkpoint(str(output))
    return None


def run_training(config: TrainConfig) -> None:
    """Build the validated mixed stream and run Hugging Face Trainer."""

    if config.use_wandb:
        _require_optional_package("wandb", "wandb")
    if config.deepspeed_config:
        _require_optional_package("deepspeed", "deepspeed")
        if not Path(config.deepspeed_config).is_file():
            raise ValueError(
                f"deepspeed_config does not exist: {config.deepspeed_config}"
            )

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    from .data import MixedSoftLabelDataset, SoftLabelCollator, parse_ratio
    from .trainer import SoftLabelTrainer

    parse_ratio(config.batch_sample_ratio)
    dataset = MixedSoftLabelDataset(
        config.soft_label_data or "", config.normal_qa_data or ""
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, use_fast=True
    )
    collator = SoftLabelCollator(tokenizer, max_length=config.max_length)
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    if config.use_wandb:
        import os

        os.environ["WANDB_PROJECT"] = config.wandb_project

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        seed=config.seed,
        bf16=config.bf16,
        fp16=config.fp16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        deepspeed=config.deepspeed_config,
        report_to=["wandb"] if config.use_wandb else [],
        run_name=config.wandb_run_name,
        remove_unused_columns=False,
    )
    trainer = SoftLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        batch_sample_ratio=config.batch_sample_ratio,
    )
    checkpoint = _resolve_checkpoint(config, get_last_checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    try:
        config = resolve_config(argv)
        run_training(config)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
