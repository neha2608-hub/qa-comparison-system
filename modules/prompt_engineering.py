"""Four prompt techniques. Context-grounded uses the exact template from the spec."""
import re

GROUNDED_TEMPLATE = (
    "You are a question-answering assistant. Answer the question using ONLY the information "
    "provided in the passage. Do not use external knowledge. Do not make up facts. If the answer "
    "is not available in the passage, respond with 'Answer not found'.\n\n"
    "Passage: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

TECHNIQUES = ["Context-grounded", "Zero-shot", "Few-shot", "Chain-of-thought (evidence-based)"]

DESCRIPTIONS = {
    "Context-grounded": "Strict instructions: use ONLY the passage, never invent, else 'Answer not found'.",
    "Zero-shot": "Plain passage + question, no rules and no examples.",
    "Few-shot": "Grounded rules plus three worked examples (one is unanswerable).",
    "Chain-of-thought (evidence-based)": "Model quotes supporting evidence first, then a short final answer.",
}

_FEW_SHOT_EXAMPLES = (
    "Example 1\nPassage: The Nile is a river in Africa. It is about 6,650 km long.\n"
    "Question: How long is the Nile?\nAnswer: 6,650 km\n\n"
    "Example 2\nPassage: Marie Curie won the Nobel Prize in Physics in 1903.\n"
    "Question: When did Marie Curie win the Nobel Prize in Physics?\nAnswer: 1903\n\n"
    "Example 3\nPassage: Mount Fuji is a volcano in Japan.\n"
    "Question: Who first climbed Mount Fuji?\nAnswer: Answer not found\n\n"
)


def build_prompt(technique: str, context: str, question: str) -> str:
    """Return the full prompt for the chosen technique."""
    if technique == "Zero-shot":
        return f"Passage: {context}\n\nQuestion: {question}\n\nAnswer:"
    if technique == "Few-shot":
        return (
            "Answer the question using ONLY the passage. If the answer is not in the passage, "
            "reply 'Answer not found'. Give a short answer.\n\n"
            + _FEW_SHOT_EXAMPLES
            + f"Now your turn.\nPassage: {context}\nQuestion: {question}\nAnswer:"
        )
    if technique == "Chain-of-thought (evidence-based)":
        return (
            "Answer the question using ONLY the passage. First quote the sentence from the passage "
            "that supports your answer, then give a short final answer. If the passage does not "
            "contain the answer, write 'Evidence: none' and 'Answer: Answer not found'.\n\n"
            f"Passage: {context}\n\nQuestion: {question}\n\n"
            "Format:\nEvidence: <supporting sentence>\nAnswer: <short answer>\n\nEvidence:"
        )
    # Default / "Context-grounded"
    return GROUNDED_TEMPLATE.format(context=context, question=question)


def extract_final_answer(technique: str, raw_text: str) -> str:
    """For the evidence-based technique, keep only the text after the last 'Answer:'."""
    text = (raw_text or "").strip()
    if technique == "Chain-of-thought (evidence-based)":
        matches = re.findall(r"Answer:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            text = matches[-1].strip()
        elif text.lower().startswith("evidence: none") or text.lower().startswith("none"):
            text = "Answer not found"
    return text


def get_evidence_text(raw_text: str) -> str:
    """Return the 'Evidence:' part of an evidence-based output (empty if absent)."""
    m = re.search(r"^(?:Evidence:)?\s*(.*?)\s*Answer:", raw_text or "", flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""
