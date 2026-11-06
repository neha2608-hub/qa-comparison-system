---
title: Extractive vs Generative QA
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

# Extractive vs. Generative Question Answering System

*Compare BERT and LLM answers, detect possible hallucinations, and explore visual question answering.*

## 1. Introduction
A Streamlit web app that answers questions about a passage in two ways, compares them, and warns when the generative answer may be unreliable. It also includes image question answering (VQA), text-to-speech, and a 20-question evaluation with Exact Match and F1.

## 2. Problem Statement
Generative LLMs write fluent answers but can invent facts (hallucinate). Extractive models such as BERT can only copy a span from the passage, so they are grounded but less flexible. Comparing both on the same passage makes unreliable answers easier to spot.

## 3. Objectives
- Generative QA with a grounded prompt and four prompt techniques.
- Extractive QA with `deepset/bert-base-cased-squad2` (with no-answer support).
- Compare answers, score similarity, flag possible hallucinations and show passage evidence.
- VQA (ViLT), TTS (gTTS), and EM/F1 evaluation on 20 Q&A pairs.

## 4. System Architecture
```
            +-------------------- Streamlit UI (app.py) --------------------+
 Passage ──►| Tab1 Text QA | Tab2 Hallucination | Tab3 VQA | Tab4 TTS | Tab5 Eval |
 Question   +------+--------------+---------------+-----------+-----------+--+
                   |              |               |           |           |
        prompt_engineering.py     |            vqa.py      tts.py   evaluation.py
                   |              |         (ViLT)       (gTTS)   (EM, F1, charts)
   llm_qa.py (flan-t5-base)   hallucination.py
   bert_qa.py (BERT SQuAD2) ─► normalise → similarity → agree/disagree →
                               passage-support check → evidence sentence
```

## 5. Technologies
Python, Streamlit, Hugging Face Transformers, PyTorch, pandas, numpy, Pillow, gTTS, scikit-learn.
Models: `google/flan-t5-base` (LLM), `deepset/bert-base-cased-squad2` (extractive), `dandelin/vilt-b32-finetuned-vqa` (VQA). All free, no API key.

## 6. Features and Modules
| Module | File | What it does |
|---|---|---|
| 1 Prompts | `modules/prompt_engineering.py` | Grounded template + zero-shot, few-shot, evidence-based CoT, context-grounded |
| 2 Generative QA | `modules/llm_qa.py` | LLM answers; long passages trimmed to the most relevant sentences (TF-IDF); errors handled |
| 3 Extractive QA | `modules/bert_qa.py` | BERT span + confidence; no-answer support; long passages chunked (384 tokens, stride 128) |
| 4 Comparison | `modules/hallucination.py` | Normalise, similarity, agree/disagree, support check, evidence |
| 5 VQA | `modules/vqa.py` | ViLT; JPG/JPEG/PNG validation |
| 6 TTS | `modules/tts.py` | gTTS MP3 for both answers |
| 7 Evaluation | `modules/evaluation.py` | EM, F1, disagreements, hallucination flags |
| 8 UI | `app.py` | 5-tab dashboard, cached models |

**Hallucination logic.** Disagreement alone is *not* proof of hallucination. When answers disagree the LLM answer is labelled **POSSIBLE HALLUCINATION** only if it is not supported by the passage (not found as text, and < 80% of its words appear in the passage). If the LLM answer *is* in the passage, the app shows an orange "manual verification" warning instead. Passage evidence sentences are shown for both answers.

## 7. Folder Structure
```
qa-comparison-system/
├── app.py  config.py  requirements.txt  README.md  SETUP_WINDOWS.md  TROUBLESHOOTING.md
├── modules/  llm_qa.py bert_qa.py prompt_engineering.py hallucination.py vqa.py tts.py evaluation.py
├── utils/    text_processing.py  model_loader.py (download/cache/error handling)
├── scripts/  download_models.py  test_models.py
├── models_cache/  (created on first download)
├── data/     qa_dataset.json   (20 Q&A pairs; evaluation_results.csv is created on first run)
├── tests/    test_core.py      (offline unit tests)
└── assets/
```

## 8. Installation (Windows PowerShell, VS Code terminal)
Python 3.10, 3.11 or 3.12 (64-bit, not 3.13), ~4 GB disk, 8 GB RAM recommended. Full guide: **SETUP_WINDOWS.md**.
```powershell
cd C:\projects\qa-comparison-system
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_models.py     # one-time model download (~2 GB), cached in models_cache\
```

## 9. Run
```powershell
streamlit run app.py
```
Open http://localhost:8501. Models are downloaded automatically on first use if you skipped the download script, then loaded from the local cache. Test each model separately with `python scripts\test_models.py bert|llm|vqa|tts|all`. Problems? See **TROUBLESHOOTING.md**.

## 10. Evaluate the models
Open **Model Evaluation** -> choose a prompt technique -> **Run Evaluation**. Both real models run on the same 20 questions; tables and charts show actual EM/F1 and are saved to `data/evaluation_results.csv`. If a model fails the tab reports the evaluation as incomplete. Offline logic tests: `python -m unittest discover -s tests -v`.

## 11. Sample Input / Output
Passage: *The Eiffel Tower ... Construction began in 1887 and was completed in 1889 ...*  Question: *In which year was construction completed?*
- LLM: `1889` · BERT: `1889` (high confidence) · Similarity 1.00 → **AGREEMENT** (green)

Unanswerable: *Who painted the Eiffel Tower?* → BERT and a well-behaved LLM return `Answer not found` → AGREEMENT.
(Actual outputs depend on the models; run the app to see yours.)

## 12. EM and F1
Normalisation (SQuAD style): lowercase → remove punctuation → remove articles (a, an, the) → collapse spaces.
- **Exact Match** = 1 if normalised prediction == normalised reference, else 0.
- **Precision** = common tokens / predicted tokens; **Recall** = common tokens / reference tokens.
- **F1** = 2·P·R / (P + R). Example: pred "Gustave Eiffel", ref "Gustave Eiffel Tower" → P=1, R=0.67, F1=0.8.
For unanswerable questions the reference is "Answer not found"; BERT's empty (no-answer) output is mapped to that same text.

## 13. Limitations and Future Scope
- flan-t5-base is small; use `google/flan-t5-large` or a causal model (`LLM_ARCH = "causal"`) in `config.py` for better quality.
- Similarity is lexical (token F1 + string ratio), not semantic; sentence embeddings would improve it.
- Support check is word-overlap based, so a wrong answer built from passage words may pass as "supported" (it is then shown as "manual verification").
- 20 questions is a small test set; use SQuAD v2 dev for a real benchmark.
- gTTS needs internet. Future: multilingual TTS, PDF upload, retrieval for very long documents.

## Testing checklist (demo)
1. Same answer: default sample question → green AGREEMENT.
2. Different answers: ask a vague question (e.g. "What is special about the tower?") or switch to Zero-shot and compare.
3. Not in passage: "Who painted the Eiffel Tower?".
4. Empty passage → error. 5. Empty question → error.
6. VQA: upload a dog photo, ask "What animal is shown in the image?".
7. Unsupported file: rename a `.txt` to `.png` or upload a `.gif` → error message.
8. TTS tab → "Listen to LLM answer".
9–10. Evaluation tab → Run Evaluation → check metrics, table and charts.
