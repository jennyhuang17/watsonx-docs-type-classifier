"""
Prepare a Hugging Face-friendly classification dataset.

Input:
  data/annotation/watsonx_docs_annotation_candidates_1.csv

Annotation rules:
  - Exclude suggested how-to rows with label_source == body_medium.
  - Drop rows where final_label == X.
  - Treat final_label == ? as "the other binary label".
  - Treat an empty final_label as accepting suggested_label.

Output:
  data/hf_dataset/train.csv
  data/hf_dataset/validation.csv
  data/hf_dataset/test.csv
  data/hf_dataset/dataset_info.json
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path


INPUT_PATH = Path("data/annotation/watsonx_docs_annotation_candidates_1.csv")
OUTPUT_DIR = Path("data/hf_dataset")
RANDOM_SEED = 42

LABEL_TO_ID = {
    "conceptual": 0,
    "how-to": 1,
}

SAMPLES_PER_LABEL = 200
SPLIT_COUNTS = {
    "train": 140,
    "validation": 30,
    "test": 30,
}


def normalize_label(value):
    label = (value or "").strip().lower()
    if label in {"conceptual", "concept", "reference"}:
        return "conceptual"
    if label in {"how-to", "howto", "task", "troubleshooting"}:
        return "how-to"
    if label == "x":
        return "X"
    if label == "?":
        return "?"
    if label == "":
        return ""
    raise ValueError(f"Unknown label value: {value!r}")


def flip_label(label):
    if label == "conceptual":
        return "how-to"
    if label == "how-to":
        return "conceptual"
    raise ValueError(f"Cannot flip label: {label!r}")


def resolved_label(row):
    suggested = normalize_label(row["suggested_label"])
    final = normalize_label(row["final_label"])

    if final == "X":
        return "X"
    if final == "?":
        return flip_label(suggested)
    if final == "":
        return suggested
    return final


def make_record(row, label, split):
    return {
        "doc_id": row["doc_id"],
        "url": row["url"],
        "title": row["title"],
        # HF text-classification examples conventionally expose the model
        # input as "text". Here it is title + the opening document context.
        "text": row["text_for_model"],
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "split": split,
    }


def write_csv(path, rows):
    fieldnames = [
        "doc_id",
        "url",
        "title",
        "text",
        "label",
        "label_id",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open(newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f, delimiter=";"))

    excluded_howto_body_medium = 0
    deleted_x = 0
    usable = []

    for row in source_rows:
        suggested = normalize_label(row["suggested_label"])
        if suggested == "how-to" and row["label_source"] == "body_medium":
            excluded_howto_body_medium += 1
            continue

        label = resolved_label(row)
        if label == "X":
            deleted_x += 1
            continue

        usable.append((row, label))

    by_label = {
        label: [row for row, resolved in usable if resolved == label]
        for label in LABEL_TO_ID
    }

    for label, rows in by_label.items():
        if len(rows) < SAMPLES_PER_LABEL:
            raise ValueError(
                f"Need {SAMPLES_PER_LABEL} rows for {label}, but only found {len(rows)}"
            )

    selected = {}
    for label, rows in by_label.items():
        rows = rows[:]
        random.shuffle(rows)
        selected[label] = rows[:SAMPLES_PER_LABEL]

    split_rows = {split: [] for split in SPLIT_COUNTS}
    for label, rows in selected.items():
        offset = 0
        for split, count in SPLIT_COUNTS.items():
            chunk = rows[offset : offset + count]
            offset += count
            split_rows[split].extend(make_record(row, label, split) for row in chunk)

    for split in split_rows:
        random.shuffle(split_rows[split])
        write_csv(OUTPUT_DIR / f"{split}.csv", split_rows[split])

    info = {
        "dataset_name": "watsonx-docs-document-type",
        "source_dataset": "ibm-research/watsonxDocsQA",
        "task": "binary document-level technical documentation type classification",
        "labels": LABEL_TO_ID,
        "text_field": "text",
        "label_field": "label",
        "label_id_field": "label_id",
        "random_seed": RANDOM_SEED,
        "source_rows": len(source_rows),
        "excluded": {
            "suggested_how_to_body_medium": excluded_howto_body_medium,
            "final_label_X": deleted_x,
        },
        "usable_after_exclusion": dict(Counter(label for _, label in usable)),
        "selected": {label: len(rows) for label, rows in selected.items()},
        "splits": {
            split: dict(Counter(row["label"] for row in rows))
            for split, rows in split_rows.items()
        },
    }
    (OUTPUT_DIR / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
