"""Generate deterministic bag-of-tokens (BOT) soft-label JSONL data."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

try:
    from .templates import PROMPT_TEMPLATES
except ImportError:  # Support ``python data/generate_soft_labels.py``.
    from templates import PROMPT_TEMPLATES


MODES = ("uniform", "flat", "sharp", "peaked", "hard", "dirichlet")


@dataclass(frozen=True)
class GenerationConfig:
    samples: int = 100_000
    candidates_per_sample: int = 4
    prompts_per_sample: int = 1
    mode: str = "uniform"
    seed: int = 44
    split_seed: int = 200
    train_fraction: float = 0.9
    ood_fraction: float = 0.1
    dirichlet_alpha: float = 0.2

    def validate(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if self.candidates_per_sample < 2:
            raise ValueError("candidates_per_sample must be at least 2")
        if self.prompts_per_sample < 1:
            raise ValueError("prompts_per_sample must be positive")
        if self.prompts_per_sample > len(PROMPT_TEMPLATES):
            raise ValueError(
                f"prompts_per_sample cannot exceed {len(PROMPT_TEMPLATES)}"
            )
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1")
        if not 0.0 <= self.ood_fraction <= 1.0:
            raise ValueError("ood_fraction must be between 0 and 1")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")


def project_to_bounded_simplex(
    values: np.ndarray,
    target_sum: float,
    upper_bound: float,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Project values onto a simplex with a per-coordinate upper bound."""

    size = len(values)
    if target_sum > size * upper_bound - epsilon:
        return np.full(size, upper_bound, dtype=np.float64)
    if target_sum <= 0:
        return np.zeros(size, dtype=np.float64)

    low = float(np.min(values)) - upper_bound
    high = float(np.max(values))
    midpoint = 0.0
    for _ in range(50):
        midpoint = (low + high) / 2.0
        projected = np.clip(values - midpoint, 0.0, upper_bound)
        current_sum = float(projected.sum())
        if abs(current_sum - target_sum) < epsilon:
            break
        if current_sum > target_sum:
            low = midpoint
        else:
            high = midpoint
    return np.clip(values - midpoint, 0.0, upper_bound)


def sample_probabilities(
    size: int,
    mode: str,
    rng: np.random.Generator,
    dirichlet_alpha: float = 0.2,
) -> np.ndarray:
    """Sample one probability vector using the paper's six distribution modes."""

    if size < 2:
        raise ValueError("size must be at least 2")
    if mode == "dirichlet":
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        probabilities = rng.dirichlet(
            np.full(size, dirichlet_alpha, dtype=np.float64)
        )
        return probabilities / probabilities.sum()
    if mode == "hard":
        probabilities = np.zeros(size, dtype=np.float64)
        probabilities[int(rng.integers(0, size))] = 1.0
        return probabilities

    if mode == "flat":
        raw_value = float(rng.beta(1.0, 30.0))
    elif mode == "uniform":
        raw_value = float(rng.uniform(0.0, 1.0))
    elif mode == "sharp":
        raw_value = float(rng.beta(10.0, 1.0))
    elif mode == "peaked":
        raw_value = float(rng.beta(50.0, 0.2))
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    minimum_max = 1.0 / size
    maximum = minimum_max + (1.0 - minimum_max) * raw_value
    maximum_index = int(rng.integers(0, size))
    remaining = project_to_bounded_simplex(
        rng.random(size - 1), 1.0 - maximum, upper_bound=maximum
    )
    probabilities = np.zeros(size, dtype=np.float64)
    probabilities[maximum_index] = maximum
    probabilities[np.arange(size) != maximum_index] = remaining
    probabilities /= probabilities.sum()
    return probabilities


def _natural_join(parts: Sequence[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _bag_description(distribution: dict[str, float], rng: random.Random) -> str:
    percentages = {word: round(value * 100) for word, value in distribution.items()}
    items = list(percentages.items())
    words = [word for word, _ in items]
    count_parts = [f"{value} instances of the word {word}" for word, value in items]
    percent_parts = [f"{value} ball with the word {word} on it" for word, value in items]
    mapping = ", ".join(f"{word}: {value}" for word, value in items)
    variants = [
        "{items} in bag.",
        "The collection contains {items}.",
        "The bag consists of {items}.",
        "The composition of a bag is {percents}.",
        "The proportion of words is {percents}.",
        "Words included are {words}, with counts given as {mapping}.",
        "The dataset includes {count} word types: {words}. Their counts are {mapping}.",
        "Suppose the sample is made up of {items}.",
        "In this example, the words appear as follows: {items}.",
        "You are given a bag of words containing {items}.",
    ]
    return rng.choice(variants).format(
        items=_natural_join(count_parts),
        percents=_natural_join(percent_parts),
        words=_natural_join(words),
        mapping=mapping,
        count=len(items),
    )


def generate_records(
    candidate_pool: Sequence[str], config: GenerationConfig
) -> Iterator[dict[str, object]]:
    """Yield deterministic BOT records for the supplied candidate pool."""

    config.validate()
    pool = list(candidate_pool)
    if len(pool) < config.candidates_per_sample:
        raise ValueError(
            "candidate pool is smaller than candidates_per_sample "
            f"({len(pool)} < {config.candidates_per_sample})"
        )
    if len(set(pool)) != len(pool):
        raise ValueError("candidate pool must not contain duplicates")

    python_rng = random.Random(config.seed)
    numpy_rng = np.random.default_rng(config.seed)
    templates = list(enumerate(PROMPT_TEMPLATES))
    for _ in range(config.samples):
        candidates = python_rng.sample(pool, config.candidates_per_sample)
        probabilities = sample_probabilities(
            config.candidates_per_sample,
            config.mode,
            numpy_rng,
            config.dirichlet_alpha,
        )
        targets = [float(value) for value in probabilities]
        distribution = {
            word: round(probability, 4)
            for word, probability in zip(candidates, targets)
        }
        word_list = ", ".join(candidates)
        for template_id, template in python_rng.sample(
            templates, config.prompts_per_sample
        ):
            prompt = template.format(
                distribution=distribution,
                word_list=word_list,
                bag_description=_bag_description(distribution, python_rng),
            )
            yield {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ],
                "candidates": candidates,
                "soft_label": targets,
                "template_id": template_id,
                "task_type": "bot",
            }


def read_candidates(path: str | Path) -> list[str]:
    source = Path(path)
    words = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
    return [word for word in words if word]


def _write_jsonl(path: Path, rows: Iterator[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def generate_splits(
    train_candidates: str | Path,
    test_candidates: str | Path,
    output_dir: str | Path,
    config: GenerationConfig,
) -> dict[str, int]:
    """Stream ID train/eval and OOD evaluation records to three JSONL files."""

    config.validate()
    train_pool = read_candidates(train_candidates)
    test_pool = read_candidates(test_candidates)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    sample_order = list(range(config.samples))
    random.Random(config.split_seed).shuffle(sample_order)
    train_samples = set(sample_order[: int(config.samples * config.train_fraction)])

    train_path = destination / "train.jsonl"
    eval_path = destination / "eval.jsonl"
    train_count = 0
    eval_count = 0
    with train_path.open("w", encoding="utf-8", newline="\n") as train_handle, \
        eval_path.open("w", encoding="utf-8", newline="\n") as eval_handle:
        rows_per_sample = config.prompts_per_sample
        for row_index, row in enumerate(generate_records(train_pool, config)):
            sample_index = row_index // rows_per_sample
            handle = train_handle if sample_index in train_samples else eval_handle
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if sample_index in train_samples:
                train_count += 1
            else:
                eval_count += 1

    ood_samples = int(math.ceil(config.samples * config.ood_fraction))
    ood_path = destination / "ood_eval.jsonl"
    if ood_samples == 0:
        ood_path.write_text("", encoding="utf-8")
        ood_count = 0
    else:
        ood_config = replace(config, samples=ood_samples, seed=config.seed + 1)
        ood_count = _write_jsonl(ood_path, generate_records(test_pool, ood_config))
    return {"train": train_count, "eval": eval_count, "ood_eval": ood_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate train/eval/OOD bag-of-tokens soft-label JSONL files."
    )
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--test-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--candidates-per-sample", type=int, default=4)
    parser.add_argument("--prompts-per-sample", type=int, default=1)
    parser.add_argument("--mode", choices=MODES, default="uniform")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--split-seed", type=int, default=200)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--ood-fraction", type=float, default=0.1)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = GenerationConfig(
        samples=args.samples,
        candidates_per_sample=args.candidates_per_sample,
        prompts_per_sample=args.prompts_per_sample,
        mode=args.mode,
        seed=args.seed,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        ood_fraction=args.ood_fraction,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    counts = generate_splits(
        args.train_candidates, args.test_candidates, args.output_dir, config
    )
    print(
        "Wrote "
        + ", ".join(f"{name}.jsonl={count}" for name, count in counts.items())
    )


if __name__ == "__main__":
    main()
