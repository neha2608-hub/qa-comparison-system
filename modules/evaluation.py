"""Evaluate LLM and BERT on the 20-question dataset with Exact Match and token F1."""
import json
import pandas as pd
import config
from modules.hallucination import compare_answers
from utils.text_processing import exact_match, f1_score

COLUMNS = ["Question ID", "Passage", "Question", "Reference Answer", "LLM Answer", "BERT Answer",
           "LLM EM", "BERT EM", "LLM F1", "BERT F1", "Similarity", "Status",
           "Possible Hallucination", "LLM Error", "BERT Error"]


def load_dataset(path=config.DATASET_PATH) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(llm, bert, records, technique="Context-grounded",
                   threshold=config.SIMILARITY_THRESHOLD, progress_callback=None) -> pd.DataFrame:
    """Run both models on every record. Rows where a model fails get NaN scores (not fake ones)."""
    rows = []
    for i, rec in enumerate(records):
        llm_pred = bert_pred = llm_err = bert_err = None
        try:
            llm_pred = llm.answer(rec["passage"], rec["question"], technique)["answer"]
        except Exception as exc:
            llm_err = str(exc)
        try:
            bert_pred = bert.answer(rec["passage"], rec["question"])["answer"]
        except Exception as exc:
            bert_err = str(exc)

        ref = rec["reference_answer"]
        row = {"Question ID": rec["id"], "Passage": rec["passage"], "Question": rec["question"],
               "Reference Answer": ref, "LLM Answer": llm_pred, "BERT Answer": bert_pred,
               "LLM EM": exact_match(llm_pred, ref) if llm_pred is not None else None,
               "BERT EM": exact_match(bert_pred, ref) if bert_pred is not None else None,
               "LLM F1": round(f1_score(llm_pred, ref), 4) if llm_pred is not None else None,
               "BERT F1": round(f1_score(bert_pred, ref), 4) if bert_pred is not None else None,
               "Similarity": None, "Status": "N/A", "Possible Hallucination": None,
               "LLM Error": llm_err, "BERT Error": bert_err}
        if llm_pred is not None and bert_pred is not None:
            cmp_ = compare_answers(llm_pred, bert_pred, rec["passage"], threshold)
            row.update({"Similarity": cmp_["similarity"], "Status": cmp_["status"],
                        "Possible Hallucination": cmp_["possible_hallucination"]})
        rows.append(row)
        if progress_callback:
            progress_callback(i + 1, len(records))
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate metrics from the REAL per-question results."""
    n = len(df)
    n_llm, n_bert = int(df["LLM Answer"].notna().sum()), int(df["BERT Answer"].notna().sum())

    def pct(col):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.mean() * 100) if len(s) else float("nan")

    return {"total_questions": n, "llm_evaluated": n_llm, "bert_evaluated": n_bert,
            "llm_em_pct": pct("LLM EM"), "bert_em_pct": pct("BERT EM"),
            "llm_f1_pct": pct("LLM F1"), "bert_f1_pct": pct("BERT F1"),
            "disagreements": int((df["Status"] == "DISAGREEMENT").sum()),
            "hallucination_flags": int((df["Possible Hallucination"] == True).sum()),  # noqa: E712
            "complete": n_llm == n and n_bert == n}
