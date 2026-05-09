"""
Train and evaluate embeddings-based text classifiers.

Experiments:
  A: all-MiniLM-L6-v2 + LogisticRegression  (baseline)
  B: all-MiniLM-L6-v2 + LinearSVC            (ablation: classifier)
  C: BAAI/bge-small-en-v1.5 + LogisticRegression (ablation: embedding)
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

LABELS = ["conceptual", "how-to"]

EXPERIMENTS = [
    {
        "id": "A",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "classifier": "LogisticRegression",
    },
    {
        "id": "B",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "classifier": "LinearSVC",
    },
    {
        "id": "C",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "classifier": "LogisticRegression",
    },
]


def load_splits():
    ds = load_dataset("itsjhuang/watsonx-docs-document-type")
    splits = {}
    for name in ("train", "validation", "test"):
        split = ds[name]
        splits[name] = {
            "texts": split["text"],
            "labels": split["label_id"],
            "doc_ids": split["doc_id"],
        }
    return splits


def build_embeddings(splits, model_name):
    model = SentenceTransformer(model_name)
    cache = {}
    for split_name, data in splits.items():
        print(f"  Encoding {split_name} ({len(data['texts'])} examples)...")
        cache[split_name] = model.encode(
            data["texts"], batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
    return cache


def build_classifier(name):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    if name == "LinearSVC":
        return LinearSVC(max_iter=2000, C=1.0, random_state=42)
    raise ValueError(f"Unknown classifier: {name}")


def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "predictions": preds.tolist(),
    }


def save_confusion_matrix(cm, condition_id, out_dir):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Condition {condition_id} (test)")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{condition_id}.png", dpi=150)
    plt.close(fig)


def main():
    results_dir = Path("results")
    models_dir = Path("models")
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    print("Loading dataset splits...")
    splits = load_splits()

    # Pre-compute embeddings, one pass per unique model
    embedding_cache = {}
    unique_models = list(dict.fromkeys(e["embedding_model"] for e in EXPERIMENTS))
    for model_name in unique_models:
        print(f"\nEmbedding with {model_name}...")
        embedding_cache[model_name] = build_embeddings(splits, model_name)

    all_metrics = []
    error_rows = []
    best_f1 = -1.0
    best_condition = None
    best_model_obj = None
    best_model_name = None

    for exp in EXPERIMENTS:
        cid = exp["id"]
        emb_name = exp["embedding_model"]
        clf_name = exp["classifier"]
        emb = embedding_cache[emb_name]

        print(f"\n--- Condition {cid}: {emb_name} + {clf_name} ---")
        clf = build_classifier(clf_name)
        clf.fit(emb["train"], splits["train"]["labels"])

        train_eval = evaluate(clf, emb["train"], splits["train"]["labels"])
        test_eval = evaluate(clf, emb["test"], splits["test"]["labels"])

        print(f"  Train  accuracy={train_eval['accuracy']:.4f}  macro_f1={train_eval['macro_f1']:.4f}")
        print(f"  Test   accuracy={test_eval['accuracy']:.4f}  macro_f1={test_eval['macro_f1']:.4f}")

        save_confusion_matrix(
            np.array(test_eval["confusion_matrix"]), cid, results_dir
        )

        # Collect errors on test split
        for i, (pred, true, doc_id, text) in enumerate(
            zip(
                test_eval["predictions"],
                splits["test"]["labels"],
                splits["test"]["doc_ids"],
                splits["test"]["texts"],
            )
        ):
            if pred != true:
                error_rows.append(
                    {
                        "condition_id": cid,
                        "doc_id": doc_id,
                        "text": text[:200],
                        "true_label": LABELS[true],
                        "predicted_label": LABELS[pred],
                    }
                )

        record = {
            "condition_id": cid,
            "embedding_model": emb_name,
            "classifier": clf_name,
            "train_accuracy": train_eval["accuracy"],
            "train_macro_f1": train_eval["macro_f1"],
            "test_accuracy": test_eval["accuracy"],
            "test_macro_f1": test_eval["macro_f1"],
        }
        all_metrics.append(record)

        if test_eval["macro_f1"] > best_f1:
            best_f1 = test_eval["macro_f1"]
            best_condition = cid
            best_model_obj = clf
            best_model_name = emb_name

    # Save metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save error analysis
    pd.DataFrame(error_rows).to_csv(results_dir / "error_analysis.csv", index=False)

    # Save best model
    joblib.dump(best_model_obj, models_dir / "best_model.joblib")
    (models_dir / "best_model_name.txt").write_text(best_model_name + "\n")

    # Print summary table
    print("\n=== Results Summary ===")
    header = f"{'Cond':<6} {'Embedding':<40} {'Classifier':<20} {'TrainAcc':>9} {'TrainF1':>8} {'TestAcc':>8} {'TestF1':>8}"
    print(header)
    print("-" * len(header))
    for r in all_metrics:
        print(
            f"{r['condition_id']:<6} {r['embedding_model']:<40} {r['classifier']:<20} "
            f"{r['train_accuracy']:>9.4f} {r['train_macro_f1']:>8.4f} "
            f"{r['test_accuracy']:>8.4f} {r['test_macro_f1']:>8.4f}"
        )
    print(f"\nBest condition: {best_condition} (test macro F1 = {best_f1:.4f})")
    print(f"Model saved to models/best_model.joblib")
    print(f"Results saved to results/")


if __name__ == "__main__":
    main()
