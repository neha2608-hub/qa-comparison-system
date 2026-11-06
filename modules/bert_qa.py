"""Extractive QA with deepset/bert-base-cased-squad2 (answers are spans of the passage)."""
import config
from utils.model_loader import safe_load
from utils.text_processing import NOT_FOUND, clean_text


class BertError(Exception):
    """Raised when BERT cannot load or run."""


class BertQA:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or config.BERT_MODEL_NAME
        self.device = device or config.get_device_name()

        def _load():
            from transformers import pipeline
            return pipeline("question-answering", model=self.model_name,
                            device=0 if self.device == "cuda" else -1)

        # Downloads on first use, then loads from the local cache. Retries on network errors.
        self.pipe = safe_load(_load, self.model_name, BertError)

    def answer(self, context: str, question: str) -> dict:
        """Return dict(answer, score, start, end, found).
        Long passages are automatically split into overlapping chunks by the pipeline
        (max_seq_len / doc_stride) and the best span over all chunks is returned."""
        context, question = clean_text(context), clean_text(question)
        if not context or not question:
            raise BertError("Passage and question must not be empty.")
        try:
            out = self.pipe(question=question, context=context, handle_impossible_answer=True,
                            max_seq_len=config.BERT_MAX_SEQ_LEN, doc_stride=config.BERT_DOC_STRIDE)
        except Exception as exc:
            raise BertError(f"BERT inference failed: {exc}") from exc
        if isinstance(out, list):
            out = out[0]
        text = (out.get("answer") or "").strip()
        found = bool(text)
        return {"answer": text if found else NOT_FOUND, "score": float(out.get("score", 0.0)),
                "start": out.get("start") if found else None,
                "end": out.get("end") if found else None, "found": found}
