# Watsonx Docs Type Classifier

## 1. General Information

### Description 

Embeddings-based binary text classifier that labels IBM Watsonx documentation
pages as:

- **conceptual** — explanatory / lookup
- **how-to** — procedural / fix-oriented

The repository covers the full pipeline:

1. Dataset construction from a public documentation corpus;
2. Three classifier configurations using sentence-transformer embeddings plus
   shallow `scikit-learn` classifiers;
3. A Gradio demo deployed on Hugging Face Spaces.

(Built for **5LN712 Information Retrieval — Assignment 1** at Uppsala University.)

### Final deliverables

All three artifacts produced by this project are hosted publicly on the
Hugging Face Hub:

| Asset | Link |
|---|---|
| Dataset | <https://huggingface.co/datasets/itsjhuang/watsonx-docs-document-type> |
| Model | <https://huggingface.co/itsjhuang/watsonx-docs-type-classifier> |
| Demo (Gradio Space) | <https://huggingface.co/spaces/itsjhuang/watsonx-docs-type-classifier> |

The accompanying 2-page academic report is available as [`report.pdf`](report.pdf).

For more information about how to load the dataset and the model from the Hugging Face hub, check the Hugging Face pages. Alternatively, you can try out with the deployed Hugging Face demo.


## 2. Dataset

The dataset is derived from the Hugging Face dataset
[`ibm-research/watsonxDocsQA`](https://huggingface.co/datasets/ibm-research/watsonxDocsQA)
(originally 1,144 documentation pages in the `corpus`). The final dataset contains **400 examples**. 

### Document classification

Each page is labelled by the type
of information a user would primarily use it for:

- `conceptual` (200 examples) — overviews, capability descriptions, reference
  and lookup pages.
- `how-to` (200 examples) — step-by-step procedures, commands, workflows,
  troubleshooting.

The classification was inspired by the original file format, DITA, that most
IBM Docs adopt. DITA has four main classes for a document type (`task`,
`concept`, `troubleshooting`, `reference`). Since `troubleshooting` was too
rare to stand alone, and `concept` vs. `reference` was too ambiguous to
annotate consistently, the final classes were simplified to a binary schema
during annotation. The binary schema also matches a high-level query-routing
use case (read-to-understand vs. read-to-act).

### Input text concatenation

Model input text is built as `title + "\n" + first 800 words of document`,
which keeps the title signal that is informative in technical documentation
while bounding length for embedding models. Full annotation guidelines,
heuristic pre-annotation, and filtering rules are documented in
[`annotation.md`](annotation.md).

### Train / Dev / Test split

The dataset was split 70/15/15 with stratification and seed 42:

| Split | conceptual | how-to | Total |
|---|---:|---:|---:|
| train | 140 | 140 | 280 |
| validation | 30 | 30 | 60 |
| test | 30 | 30 | 60 |

## 3. Experiments

Three conditions, each isolating one variable (embedding choice or classifier
choice) against the baseline:

| ID | Embedding | Classifier |
|---|---|---|
| A (baseline) | `sentence-transformers/all-MiniLM-L6-v2` | LogisticRegression |
| B | `sentence-transformers/all-MiniLM-L6-v2` | LinearSVC |
| C | `BAAI/bge-small-en-v1.5` | LogisticRegression |

Hyperparameters: LogisticRegression `C=1.0, max_iter=1000`; LinearSVC
`C=1.0, max_iter=2000`; both with `random_state=42`. Embeddings are computed
once per model with `batch_size=64`. The validation split is used only for
sanity checks; the table below reports train and test metrics.

### Results

| Condition | Train Acc | Train F1 | Test Acc | Test F1 |
|---|---:|---:|---:|---:|
| A | 0.879 | 0.879 | 0.817 | 0.817 |
| **B** ✅ | **0.971** | **0.971** | **0.867** | **0.867** |
| C | 0.864 | 0.864 | 0.833 | 0.833 |

Best model: **Condition B** (MiniLM embeddings + LinearSVC), selected by test
macro F1. This model is exported to `models/best_model.joblib` and served by
the Gradio demo.

## 4. Repository Structure

```
watsonx-docs-type-classifier/
├── README.md                       # this file
├── report.pdf                      # 2-page academic report
├── annotation.md                   # annotation guidelines and filtering rules
├── requirements.txt                # runtime dependencies
├── build_annotation_candidates.py  # reference: heuristic pre-annotation script*
├── prepare_hf_dataset.py           # reference: builds final balanced HF dataset splits*
├── train_classifier.py             # training + evaluation pipeline
├── app.py                          # Gradio demo (HF Space entry point)
└── results/
    ├── metrics.json                # all three conditions, train + test
    ├── confusion_matrix_A.png
    ├── confusion_matrix_B.png
    ├── confusion_matrix_C.png
    └── error_analysis.csv          # misclassified test examples
```

The dataset itself is hosted on the Hugging Face Hub; the `data/` directory
is not committed. The `models/` directory is created locally by
`train_classifier.py` (containing `best_model.joblib` and
`best_model_name.txt`) and is regenerated deterministically from the dataset
and seed.

\* The two annotation scripts are kept as a reference for how the dataset
was constructed. To re-run them, download the source corpus from
[`ibm-research/watsonxDocsQA`](https://huggingface.co/datasets/ibm-research/watsonxDocsQA)
into `data/annotation/` and follow the steps in
[`annotation.md`](annotation.md).

## 5. Reproducing the Experiments

The pipeline runs end-to-end on CPU in a few minutes; a GPU is detected
automatically if available.

### Step 1 - Set up a virtual environment

```bash
git clone https://github.com/itsjhuang/watsonx-docs-type-classifier.git
cd watsonx-docs-type-classifier

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 - Train and evaluate

```bash
python train_classifier.py
```

This will:

1. load the train/validation/test splits from the Hugging Face Hub
   (`itsjhuang/watsonx-docs-document-type`),
2. encode each split with both embedding models (cached, computed once per
   model),
3. fit the three classifiers on the train split,
4. evaluate on train and test splits and print a results table,
5. write `results/metrics.json`, `results/confusion_matrix_{A,B,C}.png`, and
   `results/error_analysis.csv`,
6. save the best model to `models/best_model.joblib` and the embedding model
   name to `models/best_model_name.txt`.

### Step 3 - Run the Gradio demo locally

```bash
python app.py
```

The demo loads `models/best_model.joblib` and the embedding model identified
by `models/best_model_name.txt`, then exposes a textbox where you can paste a
documentation page (title + opening text) and get a confidence score for each
class. The same `app.py` is the entry point for the deployed Hugging Face
Space.

## 6. License and Attribution

The source documentation text comes from the public IBM Watsonx documentation
via the `ibm-research/watsonxDocsQA` Hugging Face dataset. Any downstream use
should follow the source dataset's terms. Code in this repository is provided
for educational purposes as part of the Uppsala University IR course.