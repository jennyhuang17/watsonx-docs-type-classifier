"""
Gradio demo for the Watsonx Docs Type Classifier.
Loads the best trained model from models/ and serves predictions.
"""

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

LABELS = ["conceptual", "how-to"]

model_name = (Path("models") / "best_model_name.txt").read_text().strip()
embedder = SentenceTransformer(model_name)
clf = joblib.load(Path("models") / "best_model.joblib")


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict(text: str) -> dict:
    if not text.strip():
        return {label: 0.0 for label in LABELS}
    embedding = embedder.encode([text], convert_to_numpy=True)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(embedding)[0]
    else:
        scores = clf.decision_function(embedding)[0]
        # LinearSVC returns a scalar for binary; wrap in array
        if np.ndim(scores) == 0:
            scores = np.array([-scores, scores])
        probs = softmax(scores)
    return {label: float(p) for label, p in zip(LABELS, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Document text (title + body)",
        lines=8,
        placeholder="Paste the title and opening text of a Watsonx documentation page here.",
    ),
    outputs=gr.Label(num_top_classes=2, label="Predicted document type"),
    title="Watsonx Docs Type Classifier",
    description=(
        "Predicts whether a Watsonx documentation page is **conceptual** or **how-to**. "
        "Paste the page title and opening text below."
    ),
)

if __name__ == "__main__":
    demo.launch(share=False)
