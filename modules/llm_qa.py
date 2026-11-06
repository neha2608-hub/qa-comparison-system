"""Generative QA with a configurable Hugging Face instruction-tuned LLM."""
import re
import config
from utils.model_loader import safe_load, classify_error, friendly_message
from modules.prompt_engineering import build_prompt, extract_final_answer, get_evidence_text
from utils.text_processing import NOT_FOUND, split_sentences, word_count, clean_text


class LLMError(Exception):
    """Raised when the LLM cannot load or generate."""


def prepare_context(context: str, question: str, max_words: int = config.LLM_MAX_CONTEXT_WORDS):
    """Trim long passages to the sentences most similar to the question (TF-IDF).
    Returns (context_text, was_trimmed)."""
    context = clean_text(context)
    if word_count(context) <= max_words:
        return context, False
    sentences = split_sentences(context)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english").fit(sentences + [question])
        scores = cosine_similarity(vec.transform([question]), vec.transform(sentences))[0]
    except Exception:  # fall back to simply keeping the start of the passage
        return " ".join(context.split()[:max_words]), True
    chosen, used = [], 0
    for idx in sorted(range(len(sentences)), key=lambda i: -scores[i]):
        n = word_count(sentences[idx])
        if used + n > max_words:
            continue
        chosen.append(idx)
        used += n
    if not chosen:
        return " ".join(context.split()[:max_words]), True
    return " ".join(sentences[i] for i in sorted(chosen)), True


class LLMQA:
    def __init__(self, model_name=None, arch=None, device=None):
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.arch = arch or config.LLM_ARCH
        self.device = device or config.get_device_name()

        def _load():
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_cls = AutoModelForSeq2SeqLM if self.arch == "seq2seq" else AutoModelForCausalLM
            model = model_cls.from_pretrained(self.model_name).to(self.device).eval()
            return torch, tokenizer, model

        # Downloads on first use, then loads from the local cache. Retries on network errors.
        self._torch, self.tokenizer, self.model = safe_load(_load, self.model_name, LLMError)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate(self, prompt: str) -> str:
        torch = self._torch
        if self.arch == "seq2seq":
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=config.LLM_MAX_INPUT_TOKENS).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                                          num_beams=4, do_sample=False)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        text = prompt
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                                      do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def answer(self, context: str, question: str, technique: str = "Context-grounded") -> dict:
        """Return dict(answer, raw_output, prompt, evidence, trimmed)."""
        if not context.strip() or not question.strip():
            raise LLMError("Passage and question must not be empty.")
        ctx, trimmed = prepare_context(context, question)
        prompt = build_prompt(technique, ctx, question.strip())
        try:
            raw = self._generate(prompt)
        except Exception as exc:
            kind = classify_error(exc)
            msg = (friendly_message(kind, self.model_name, exc) if kind == "memory"
                   else f"LLM generation failed: {exc}")
            raise LLMError(msg) from exc
        answer = extract_final_answer(technique, raw).strip().strip('"')
        if not answer or re.search(r"answer not found", answer, flags=re.IGNORECASE):
            answer = NOT_FOUND
        return {"answer": answer, "raw_output": raw, "prompt": prompt,
                "evidence": get_evidence_text(raw) if "Chain" in technique else "",
                "trimmed": trimmed}
