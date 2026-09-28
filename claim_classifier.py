"""Validation-first claim classification for sentence-level ESG report text.

The model combines TF-IDF text features with deterministic document signals.
PDF ingestion is deliberately separate: this module expects one already-split
sentence per record, optionally carrying ``is_bold`` and ``is_bullet`` flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_AUTO_ACCEPT_THRESHOLD = 0.85
DEFAULT_REVIEW_THRESHOLD = 0.50
MIN_UNIQUE_NEGATIVES = 500
MAX_DUPLICATE_FRACTION = 0.05

DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b")
ISO_RE = re.compile(r"\bISO\s*\d{4,5}\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[,.]\d+)?")
PERCENT_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
BULLET_RE = re.compile(r"^\s*(?:[-*•‣]|\d+[.)])\s+")
TARGET_RE = re.compile(r"\b(?:target|goal|aim|commit|will|reduce|increase|achieve)\w*\b", re.IGNORECASE)


@dataclass(frozen=True)
class NegativeDatasetCheck:
    rows: int
    unique_sentences: int
    duplicate_rows: int
    duplicate_fraction: float
    blank_sentences: int
    invalid_labels: int
    categories: dict[str, int]
    training_ready: bool
    reasons: list[str]


def normalise_sentence(text: str) -> str:
    return " ".join(text.casefold().split())


def is_truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def validate_negative_csv(path: Path) -> NegativeDatasetCheck:
    """Validate a negative corpus without changing the supplied source file."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"category", "sentence", "is_claim"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Negative CSV is missing required columns: {', '.join(sorted(missing))}")

    sentences = [normalise_sentence(row["sentence"]) for row in rows]
    blank_sentences = sum(not sentence for sentence in sentences)
    unique_sentences = len(set(sentence for sentence in sentences if sentence))
    duplicate_rows = max(0, len(rows) - unique_sentences - blank_sentences)
    duplicate_fraction = duplicate_rows / len(rows) if rows else 1.0
    invalid_labels = sum(is_truthy(row["is_claim"]) for row in rows)
    categories = dict(sorted(Counter(row["category"].strip() for row in rows).items()))

    reasons: list[str] = []
    if blank_sentences:
        reasons.append(f"{blank_sentences} rows have blank sentences.")
    if invalid_labels:
        reasons.append(f"{invalid_labels} rows are labelled as claims; negatives must all be false.")
    if unique_sentences < MIN_UNIQUE_NEGATIVES:
        reasons.append(f"Only {unique_sentences} unique negatives; at least {MIN_UNIQUE_NEGATIVES} are required.")
    if duplicate_fraction > MAX_DUPLICATE_FRACTION:
        reasons.append(f"Duplicate rate is {duplicate_fraction:.1%}; maximum allowed is {MAX_DUPLICATE_FRACTION:.1%}.")
    if len(categories) < 4:
        reasons.append("Fewer than four negative categories are represented.")
    return NegativeDatasetCheck(
        rows=len(rows), unique_sentences=unique_sentences, duplicate_rows=duplicate_rows,
        duplicate_fraction=duplicate_fraction, blank_sentences=blank_sentences,
        invalid_labels=invalid_labels, categories=categories,
        training_ready=not reasons, reasons=reasons,
    )


def regex_signals(sentence: str, metadata: dict[str, Any] | None = None) -> dict[str, int]:
    """Return deterministic features available before model inference."""

    metadata = metadata or {}
    words = sentence.split()
    return {
        "has_year": int(bool(DATE_RE.search(sentence))),
        "has_iso": int(bool(ISO_RE.search(sentence))),
        "has_number": int(bool(NUMBER_RE.search(sentence))),
        "has_percentage": int(bool(PERCENT_RE.search(sentence))),
        "has_capital_acronym": int(bool(ACRONYM_RE.search(sentence))),
        "has_target_language": int(bool(TARGET_RE.search(sentence))),
        "is_bullet": int(is_truthy(metadata.get("is_bullet")) or bool(BULLET_RE.search(sentence))),
        "is_bold": int(is_truthy(metadata.get("is_bold"))),
        "word_count": min(len(words), 100),
    }


def decision_gate(
    probability: float,
    *,
    auto_accept_threshold: float = DEFAULT_AUTO_ACCEPT_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> str:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Claim probability must be between 0.0 and 1.0.")
    if not 0.0 <= review_threshold <= auto_accept_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= review <= auto-accept <= 1.")
    if probability >= auto_accept_threshold:
        return "auto_accept"
    if probability >= review_threshold:
        return "review"
    return "discard"


def _read_examples(path: Path, text_column: str, label: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or text_column not in rows[0]:
        raise ValueError(f"{path.name} must contain a '{text_column}' column.")
    examples = []
    for row in rows:
        text = row[text_column].strip()
        if text:
            examples.append({"text": text, "label": label, "metadata": row})
    return examples


def _deduplicate_examples(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, str]] = set()
    unique: list[dict[str, Any]] = []
    for example in examples:
        key = (example["label"], normalise_sentence(example["text"]))
        if key not in seen:
            seen.add(key)
            unique.append(example)
    return unique


def train(
    positive_csv: Path,
    negative_csv: Path,
    model_path: Path,
    *,
    positive_text_column: str = "claim",
    allow_insufficient_negatives: bool = False,
) -> dict[str, Any]:
    """Train and persist a probability classifier after dataset validation."""

    check = validate_negative_csv(negative_csv)
    if not check.training_ready and not allow_insufficient_negatives:
        raise ValueError("Negative corpus is not production-ready: " + " ".join(check.reasons))

    try:
        from joblib import dump
        from scipy.sparse import hstack
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise RuntimeError("Training requires scikit-learn, scipy, and joblib.") from error

    positives = _read_examples(positive_csv, positive_text_column, 1)
    negatives = _read_examples(negative_csv, "sentence", 0)
    examples = _deduplicate_examples([*positives, *negatives])
    labels = [example["label"] for example in examples]
    if len(set(labels)) != 2:
        raise ValueError("Both claim and non-claim examples are required.")

    train_examples, test_examples = train_test_split(
        examples, test_size=0.20, random_state=42, stratify=labels,
    )
    text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    signal_vectorizer = DictVectorizer(sparse=True)
    train_text = text_vectorizer.fit_transform([example["text"] for example in train_examples])
    test_text = text_vectorizer.transform([example["text"] for example in test_examples])
    train_signals = signal_vectorizer.fit_transform([regex_signals(example["text"], example["metadata"]) for example in train_examples])
    test_signals = signal_vectorizer.transform([regex_signals(example["text"], example["metadata"]) for example in test_examples])
    classifier = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)
    classifier.fit(hstack([train_text, train_signals]), [example["label"] for example in train_examples])
    probabilities = classifier.predict_proba(hstack([test_text, test_signals]))[:, 1]
    predictions = (probabilities >= review_threshold()).astype(int)
    report = classification_report([example["label"] for example in test_examples], predictions, output_dict=True, zero_division=0)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump({
        "classifier": classifier,
        "text_vectorizer": text_vectorizer,
        "signal_vectorizer": signal_vectorizer,
        "thresholds": {"auto_accept": DEFAULT_AUTO_ACCEPT_THRESHOLD, "review": DEFAULT_REVIEW_THRESHOLD},
    }, model_path)
    return {"negative_dataset_check": asdict(check), "train_rows": len(train_examples), "test_rows": len(test_examples), "metrics": report}


def review_threshold() -> float:
    return DEFAULT_REVIEW_THRESHOLD


def classify_csv(
    input_csv: Path,
    model_path: Path,
    output_csv: Path,
    *,
    sentence_column: str = "sentence_text",
) -> dict[str, int]:
    """Classify each non-blank input sentence and retain its source metadata."""

    try:
        from joblib import load
        from scipy.sparse import hstack
    except ImportError as error:
        raise RuntimeError("Classification requires scipy and joblib.") from error

    with input_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = reader.fieldnames or []
        records = list(reader)
    if sentence_column not in input_fields:
        raise ValueError(
            f"{input_csv.name} must contain a '{sentence_column}' column. "
            f"Available columns: {', '.join(input_fields)}"
        )

    result_fields = ["claim_probability", "decision", "is_claim"]
    conflicts = set(input_fields).intersection(result_fields)
    if conflicts:
        raise ValueError(
            "Input CSV already has output columns: " + ", ".join(sorted(conflicts)) + ". "
            "Use a fresh sentence CSV or remove those columns first."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = load(model_path)
    required_model_parts = {"classifier", "text_vectorizer", "signal_vectorizer", "thresholds"}
    missing_model_parts = required_model_parts.difference(model)
    if missing_model_parts:
        raise ValueError("Saved model is incomplete: " + ", ".join(sorted(missing_model_parts)))

    thresholds = model["thresholds"]
    auto_accept_threshold = float(thresholds["auto_accept"])
    review_threshold_value = float(thresholds["review"])
    nonblank_indices = [
        index for index, row in enumerate(records) if row.get(sentence_column, "").strip()
    ]
    if nonblank_indices:
        sentences = [records[index][sentence_column].strip() for index in nonblank_indices]
        text_features = model["text_vectorizer"].transform(sentences)
        signal_features = model["signal_vectorizer"].transform([
            regex_signals(sentence, records[index])
            for sentence, index in zip(sentences, nonblank_indices)
        ])
        probabilities = model["classifier"].predict_proba(hstack([text_features, signal_features]))[:, 1]
    else:
        probabilities = []

    for row in records:
        row.update({"claim_probability": "", "decision": "missing_sentence", "is_claim": ""})
    decisions = Counter()
    for index, probability in zip(nonblank_indices, probabilities):
        probability_value = float(probability)
        decision = decision_gate(
            probability_value,
            auto_accept_threshold=auto_accept_threshold,
            review_threshold=review_threshold_value,
        )
        records[index].update({
            "claim_probability": f"{probability_value:.6f}",
            "decision": decision,
            "is_claim": "True" if decision == "auto_accept" else "False" if decision == "discard" else "",
        })
        decisions[decision] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*input_fields, *result_fields])
        writer.writeheader()
        writer.writerows(records)
    return {
        "input_rows": len(records),
        "classified_rows": len(nonblank_indices),
        "missing_sentence_rows": len(records) - len(nonblank_indices),
        "auto_accept": decisions["auto_accept"],
        "review": decisions["review"],
        "discard": decisions["discard"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-negatives")
    validate_parser.add_argument("--negative-csv", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--positive-csv", type=Path, required=True)
    train_parser.add_argument("--negative-csv", type=Path, required=True)
    train_parser.add_argument("--model-path", type=Path, default=Path("Model/claim_classifier.joblib"))
    train_parser.add_argument("--positive-text-column", default="claim")
    train_parser.add_argument("--allow-insufficient-negatives", action="store_true")
    classify_parser = subparsers.add_parser(
        "classify", help="Score sentences from a CSV using a saved classifier."
    )
    classify_parser.add_argument("--input-csv", type=Path, required=True)
    classify_parser.add_argument("--model-path", type=Path, default=Path("Model/claim_classifier.joblib"))
    classify_parser.add_argument("--output-csv", type=Path, required=True)
    classify_parser.add_argument("--sentence-column", default="sentence_text")
    args = parser.parse_args()
    if args.command == "validate-negatives":
        print(json.dumps(asdict(validate_negative_csv(args.negative_csv)), indent=2))
    elif args.command == "train":
        print(json.dumps(train(
            args.positive_csv, args.negative_csv, args.model_path,
            positive_text_column=args.positive_text_column,
            allow_insufficient_negatives=args.allow_insufficient_negatives,
        ), indent=2))
    else:
        print(json.dumps(classify_csv(
            args.input_csv, args.model_path, args.output_csv,
            sentence_column=args.sentence_column,
        ), indent=2))


if __name__ == "__main__":
    main()
