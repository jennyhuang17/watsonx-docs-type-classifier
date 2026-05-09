"""
Build annotation candidate CSV from ibm-research/watsonxDocsQA.

Labels: conceptual | how-to  (2-class)
  conceptual = original concept + reference  (user intent: understand/look up)
  how-to     = original task + troubleshooting  (user intent: do/fix something)

Output: data/annotation/watsonx_docs_annotation_candidates.csv
Columns: doc_id, url, title, text, text_for_model, suggested_label, label_source,
         final_label, notes
"""

import re
import csv
from pathlib import Path
from collections import Counter
from datasets import load_dataset

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_PATH = Path("data/annotation/watsonx_docs_annotation_candidates.csv")
TEXT_FOR_MODEL_WORDS = 800
BODY_MIN_SCORE = 3        # min matches to trust a body signal (else → concept)

# ── Title patterns  (priority: troubleshooting > task > concept) ─────────────

TROUBLESHOOTING_TITLE = re.compile(
    r"\b(troubleshoot(ing)?|known issues?|workaround|"
    r"debug(ging)?|cannot|can\'t|failure[s]?|"
    r"resolv(ing)? (the |this |an? )?(error|issue|problem)|"
    r"handling (error|exception|failure))\b"
    r"|errors?\s*$",          # title ends with "error(s)"
    re.IGNORECASE,
)

# Action verbs at start OR strong task signals anywhere in title
TASK_TITLE = re.compile(
    r"^(creating|configuring|setting up|deploying|installing|connecting|"
    r"adding|managing|running|using|building|enabling|disabling|uploading|"
    r"importing|exporting|integrating|migrating|training|fine[-\s]?tun|"
    r"evaluating|testing|monitoring|accessing|viewing|editing|deleting|"
    r"generating|defining|specifying|assigning|granting|registering|"
    r"authenticating|getting started|quick[- ]?start|how to|"
    r"working with|preparing|scheduling|automating|publishing)\b"
    r"|"
    r"\b(tutorial|step-by-step|walkthrough|procedure|workflow)\b",
    re.IGNORECASE,
)

# "About / Overview / Introduction / How X works / Key concepts / What is"
CONCEPT_TITLE = re.compile(
    r"^(about |overview|introduction|understanding|key concepts?|"
    r"what (is|are) |architecture|background|getting to know)\b"
    r"|"
    r"\b(overview|introduction to|understanding|concept[s]?|architecture|"
    r"how (it|this|the|\w+) work[s]?)\b",
    re.IGNORECASE,
)

# ── Body patterns ─────────────────────────────────────────────────────────────

TROUBLESHOOT_BODY = re.compile(
    r"\b(troubleshoot(ing)?|known (bug|issue|limitation)|workaround|"
    r"resolv(ing)? (this|the) (error|issue|problem)|error message[s]?|"
    r"if (you|the) (encounter|see|receive|get) (an? )?(error|issue|fail)|"
    r"cause[s]? and resolution|symptom|diagnos)\b",
    re.IGNORECASE,
)

TASK_BODY = re.compile(
    r"\b(step \d|follow(ing)? (the )?steps?|before you begin|prerequisites?|"
    r"procedure[: ]|"
    r"to (create|configure|deploy|install|connect|add|manage|run|use|"
    r"set up|enable|disable|upload|import|export|train|fine-tune|"
    r"evaluate|test|monitor|access|edit|delete|generate|specify|assign)|"
    r"in (the )?following (steps?|procedure)|"
    r"click (the |on )?|select (the )?|enter (the |your )?|open (the )?|"
    r"as follows[: ]|basic workflow|workflow is|"
    r"is as follows)\b"
    r"|"
    r"\b[1-9]\.\s+[A-Z]",    # numbered list: "1. Create …"
    re.IGNORECASE,
)

CONCEPT_BODY = re.compile(
    r"\b(overview|introduction|describe[s]?|explain[s]?|"
    r"understand(ing)?|is (a |an |the )|are (a |an |the )|"
    r"consist[s]? of|provide[s]? (a |an )|enable[s]? (you|users?)|"
    r"allow[s]? (you|users?)|architecture|feature[s]? of|"
    r"concept[s]?|definition[s]?|background|purpose of|"
    r"how (it|this|the) work[s]?)\b",
    re.IGNORECASE,
)


def count_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def suggest_label(title: str, text_for_model: str):
    """Return (label, label_source).

    label        : 'how-to' | 'conceptual'
    label_source : 'title_high' | 'body_medium' | 'default_low'
    """
    t = (title or "").strip()

    # 1. Title rules  (how-to signals take priority)
    if TROUBLESHOOTING_TITLE.search(t):
        return "how-to", "title_high"
    if TASK_TITLE.search(t):
        return "how-to", "title_high"
    if CONCEPT_TITLE.search(t):
        return "conceptual", "title_high"

    # 2. Body signals: merge troubleshooting+task scores vs concept score
    body = text_for_model or ""
    howto_score     = count_matches(TASK_BODY, body) + count_matches(TROUBLESHOOT_BODY, body)
    conceptual_score = count_matches(CONCEPT_BODY, body)

    best_score = max(howto_score, conceptual_score)
    if best_score < BODY_MIN_SCORE:
        return "conceptual", "default_low"
    if howto_score >= conceptual_score:
        return "how-to", "body_medium"
    return "conceptual", "body_medium"


def make_text_for_model(title: str, text: str, max_words: int = TEXT_FOR_MODEL_WORDS) -> str:
    words = (text or "").split()
    truncated = " ".join(words[:max_words])
    return f"{title}\n{truncated}"


def main():
    print("Loading ibm-research/watsonxDocsQA (corpus config) …")
    ds = load_dataset("ibm-research/watsonxDocsQA", "corpus", trust_remote_code=True)

    split_name = max(ds.keys(), key=lambda s: len(ds[s]))
    data = ds[split_name]
    print(f"Split '{split_name}': {len(data)} rows | columns: {data.column_names}")

    cols = data.column_names
    id_col    = next((c for c in cols if c == "doc_id"), cols[0])
    url_col   = next((c for c in cols if "url" in c.lower()), None)
    title_col = next((c for c in cols if "title" in c.lower()), None)
    text_col  = next((c for c in cols if c.lower() == "document"), None)
    if text_col is None:
        text_col = next((c for c in cols if c.lower() in ("text", "content", "passage")), None)
    if text_col is None:
        text_col = max(cols, key=lambda c: len(str(data[0].get(c, ""))))

    print(f"Mapped: id={id_col}  url={url_col}  title={title_col}  text={text_col}\n")

    rows = []
    for item in data:
        doc_id = str(item.get(id_col, ""))
        url    = str(item.get(url_col, ""))   if url_col   else ""
        title  = str(item.get(title_col, "")) if title_col else ""
        text   = str(item.get(text_col, ""))  if text_col  else ""

        text_for_model         = make_text_for_model(title, text)
        suggested, label_source = suggest_label(title, text_for_model)

        rows.append({
            "doc_id":          doc_id,
            "url":             url,
            "title":           title,
            "text":            text,
            "text_for_model":  text_for_model,
            "suggested_label": suggested,
            "label_source":    label_source,
            "final_label":     "",
            "notes":           "",
        })

    # ── Stats ─────────────────────────────────────────────────────────────────
    dist     = Counter(r["suggested_label"] for r in rows)
    src_dist = Counter(r["label_source"]    for r in rows)

    LABELS = ["how-to", "conceptual"]

    print("=== Suggested label distribution ===")
    for label in LABELS:
        count = dist.get(label, 0)
        bar   = "█" * (count // 10)
        print(f"  {label:20s}: {count:4d}  {bar}")

    print("\n=== Label source (confidence) ===")
    for src, count in sorted(src_dist.items()):
        print(f"  {src:20s}: {count}")

    # Cross-tab: label × source
    print("\n=== Label × confidence cross-tab ===")
    print(f"  {'':20s}  {'title_high':>10} {'body_medium':>11} {'default_low':>11}  {'total':>6}")
    for label in LABELS:
        th = sum(1 for r in rows if r["suggested_label"]==label and r["label_source"]=="title_high")
        bm = sum(1 for r in rows if r["suggested_label"]==label and r["label_source"]=="body_medium")
        dl = sum(1 for r in rows if r["suggested_label"]==label and r["label_source"]=="default_low")
        print(f"  {label:20s}  {th:>10} {bm:>11} {dl:>11}  {th+bm+dl:>6}")

    # Workload estimate after filtering documents shorter than 50 words
    kept  = [r for r in rows if len(r["text"].split()) >= 50]
    filt  = len(rows) - len(kept)
    k_th  = sum(1 for r in kept if r["label_source"] == "title_high")
    k_bm  = sum(1 for r in kept if r["label_source"] == "body_medium")
    k_dl  = sum(1 for r in kept if r["label_source"] == "default_low")
    print(f"\n=== After filtering <50-word documents ({filt} landing pages removed) ===")
    print(f"  Kept: {len(kept)} rows")
    print(f"  title_high  (batch accept, spot-check 10%): {k_th}  -> about {k_th//10} manual checks")
    print(f"  body_medium (review individually):           {k_bm}")
    print(f"  default_low (priority review):               {k_dl}")
    print(f"\n=== Manual review workload estimate ===")
    for target in [200, 300, 400]:
        per_cls   = target // 2
        # title_high contribution, capped separately for each class
        th_howto  = sum(1 for r in kept if r["suggested_label"]=="how-to"    and r["label_source"]=="title_high")
        th_conc   = sum(1 for r in kept if r["suggested_label"]=="conceptual" and r["label_source"]=="title_high")
        auto      = min(th_howto, per_cls) + min(th_conc, per_cls)
        need_review = max(0, target - auto)
        print(f"  Target {target} rows ({per_cls} per class): title_high auto-accepted {auto}, needs ~{need_review} manual reviews")

    print()
    for label in LABELS:
        samples = [r["title"] for r in rows if r["suggested_label"] == label][:6]
        print(f"--- {label} ({dist[label]}) ---")
        for t in samples:
            print(f"  {t}")
        print()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["doc_id", "url", "title", "text", "text_for_model",
                  "suggested_label", "label_source", "final_label", "notes"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {OUTPUT_PATH}")
    print("Fill in 'final_label' column to complete annotation.")


if __name__ == "__main__":
    main()
