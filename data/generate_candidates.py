"""Build reproducible single-word candidate splits from a Qwen tokenizer."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Mapping, Sequence


QWEN_WORD_TOKEN = re.compile(r"^Ġ[A-Za-z]+$")


def filter_qwen_candidates(
    vocab: Mapping[str, int], min_zipf: float = 5.0
) -> list[str]:
    """Return Qwen word-start tokens as plain English candidate words.

    Tokens are traversed in tokenizer-ID order.  The frequency dependency is
    loaded only when filtering is requested, keeping imports and ``--help``
    lightweight.
    """

    zipf_frequency = None
    if min_zipf > 0:
        try:
            from wordfreq import zipf_frequency as _zipf_frequency
        except ImportError as exc:
            raise RuntimeError(
                "Candidate frequency filtering requires wordfreq. "
                "Install the project dependencies or pass --min-zipf 0."
            ) from exc
        zipf_frequency = _zipf_frequency

    words: list[str] = []
    for token in sorted(vocab, key=vocab.get):
        if not QWEN_WORD_TOKEN.fullmatch(token):
            continue
        word = token[1:]
        if zipf_frequency is None or zipf_frequency(word.lower(), "en") > min_zipf:
            words.append(word)
    return words


def split_candidates(
    candidates: Sequence[str], train_ratio: float = 0.9, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Shuffle candidates locally and return deterministic train/test splits."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1")
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * train_ratio)
    return shuffled[:split_at], shuffled[split_at:]


def _write_words(path: Path, words: Sequence[str]) -> None:
    path.write_text("".join(f"{word}\n" for word in words), encoding="utf-8")


def generate_candidate_files(
    model_name_or_path: str,
    output_dir: str | Path,
    train_ratio: float = 0.9,
    seed: int = 42,
    min_zipf: float = 5.0,
) -> tuple[Path, Path]:
    """Load a Qwen tokenizer and write its candidate split files."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Candidate generation requires transformers. Install project dependencies."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    candidates = filter_qwen_candidates(tokenizer.get_vocab(), min_zipf=min_zipf)
    train, test = split_candidates(candidates, train_ratio=train_ratio, seed=seed)
    if not train or not test:
        raise ValueError(
            "Candidate filtering produced an empty split; lower --min-zipf or "
            "check that the supplied tokenizer is Qwen-compatible."
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "qwen_train.txt"
    test_path = destination / "qwen_test.txt"
    _write_words(train_path, train)
    _write_words(test_path, test)
    return train_path, test_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate train/test candidate words from a Qwen tokenizer."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", default="data/candidates")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-zipf", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_path, test_path = generate_candidate_files(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_zipf=args.min_zipf,
    )
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
