"""Compare LLM vs BERT answers and flag possible hallucinations with passage evidence."""
from difflib import SequenceMatcher
import config
from utils.text_processing import (normalize_answer, f1_score, get_tokens, is_not_found,
                                   split_sentences)


def similarity(a: str, b: str) -> float:
    """Similarity in [0,1]: best of token-F1, character ratio, and containment (0.85)."""
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return 0.0
    score = max(f1_score(a, b), SequenceMatcher(None, na, nb).ratio())
    if na in nb or nb in na:
        score = max(score, 0.85)
    return round(score, 3)


def is_supported(answer: str, passage: str, min_coverage: float = 0.8) -> bool:
    """Is the answer grounded in the passage? (substring match or >=80% token coverage)"""
    if is_not_found(answer):
        return False
    na, np_ = normalize_answer(answer), normalize_answer(passage)
    if f" {na} " in f" {np_} ":
        return True
    toks = get_tokens(answer)
    if not toks:
        return False
    pass_tokens = set(get_tokens(passage))
    return sum(t in pass_tokens for t in toks) / len(toks) >= min_coverage


def find_evidence(answer: str, passage: str):
    """Return {'sentence', 'exact'} for the passage sentence that best supports the answer."""
    if is_not_found(answer):
        return None
    na, toks = normalize_answer(answer), set(get_tokens(answer))
    best, best_score = None, 0.0
    for sent in split_sentences(passage):
        if na and f" {na} " in f" {normalize_answer(sent)} ":
            return {"sentence": sent, "exact": True}
        score = len(toks & set(get_tokens(sent))) / len(toks) if toks else 0.0
        if score > best_score:
            best, best_score = sent, score
    return {"sentence": best, "exact": False} if best else None


def compare_answers(llm_answer: str, bert_answer: str, passage: str,
                    threshold: float = config.SIMILARITY_THRESHOLD) -> dict:
    """Full comparison. A disagreement is NOT proof of hallucination: it is a warning, and the
    LLM answer is only called a POSSIBLE HALLUCINATION if it is not supported by the passage."""
    llm_nf, bert_nf = is_not_found(llm_answer), is_not_found(bert_answer)
    llm_supported = is_supported(llm_answer, passage)
    bert_supported = is_supported(bert_answer, passage)
    res = {"llm_answer": llm_answer, "bert_answer": bert_answer,
           "llm_supported": llm_supported, "bert_supported": bert_supported,
           "llm_evidence": find_evidence(llm_answer, passage),
           "bert_evidence": find_evidence(bert_answer, passage)}

    if llm_nf and bert_nf:
        res.update(similarity=1.0, status="AGREEMENT", possible_hallucination=False, level="ok",
                   message="Both models report that the passage does not contain the answer.")
        return res

    sim = 0.0 if (llm_nf or bert_nf) else similarity(llm_answer, bert_answer)
    res["similarity"] = sim
    if sim >= threshold:
        res.update(status="AGREEMENT", possible_hallucination=False, level="ok",
                   message="Both models give matching or semantically similar answers.")
    elif llm_nf:
        res.update(status="DISAGREEMENT", possible_hallucination=False, level="review",
                   message="The LLM said 'Answer not found' but BERT extracted a span. "
                           "Check the evidence: the LLM may have been too cautious.")
    elif bert_nf and not llm_supported:
        res.update(status="DISAGREEMENT", possible_hallucination=True, level="hallucination",
                   message="POSSIBLE HALLUCINATION: BERT found no answer and the LLM answer is "
                           "not supported by the passage.")
    elif not llm_supported:
        res.update(status="DISAGREEMENT", possible_hallucination=True, level="hallucination",
                   message="POSSIBLE HALLUCINATION: the answers differ and the LLM answer is not "
                           "found in the passage.")
    else:
        res.update(status="DISAGREEMENT", possible_hallucination=False, level="review",
                   message="The answers differ, but the LLM answer does appear in the passage. "
                           "Manual verification needed (may be a different valid span).")
    return res
