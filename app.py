"""Extractive vs. Generative Question Answering System - Streamlit dashboard."""
import html
import time

import pandas as pd
import streamlit as st

import config  # must stay first: sets the model cache folder before transformers loads
from utils.model_loader import is_model_cached, internet_available
from modules.bert_qa import BertQA, BertError
from modules.evaluation import load_dataset, run_evaluation, summarize
from modules.hallucination import compare_answers
from modules.llm_qa import LLMQA, LLMError
from modules.prompt_engineering import TECHNIQUES, DESCRIPTIONS
from modules.tts import text_to_speech, TTSError
from modules.vqa import VQAModel, VQAError, load_image

st.set_page_config(page_title="Extractive vs. Generative QA", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
.card{border-radius:14px;padding:18px 20px;margin-bottom:12px;border:1px solid #e5e7eb;background:#fff;
      box-shadow:0 1px 4px rgba(0,0,0,.06);color:#111827}
.card h4{margin:0 0 8px 0;font-size:1rem;color:#374151}
.card .ans{font-size:1.35rem;font-weight:600;word-wrap:break-word}
.card .meta{font-size:.85rem;color:#6b7280;margin-top:6px}
.llm{border-left:6px solid #6366f1}.bert{border-left:6px solid #0ea5e9}
.ok{background:#ecfdf5;border:1px solid #10b981;color:#065f46}
.review{background:#fffbeb;border:1px solid #f59e0b;color:#92400e}
.halluc{background:#fef2f2;border:1px solid #ef4444;color:#991b1b}
.banner{border-radius:12px;padding:14px 18px;font-weight:600;margin:8px 0}
.evidence{background:#f3f4f6;border-radius:10px;padding:10px 14px;color:#111827;margin-bottom:8px}
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------- cached model loaders
# st.cache_resource keeps each model in memory, so it is loaded only once per app run.
# The files themselves are cached on disk by Hugging Face (see config.HF_CACHE_DIR).
@st.cache_resource(show_spinner=False)
def get_llm(model_name, arch):
    return LLMQA(model_name, arch)

@st.cache_resource(show_spinner=False)
def get_bert():
    return BertQA()

@st.cache_resource(show_spinner=False)
def get_vqa():
    return VQAModel()


def load_with_status(kind, label, repo_id, size, getter, *args):
    """Load a model and show a clear loading / downloading indicator the first time."""
    loaded = st.session_state.setdefault("loaded", set())
    key = (kind, repo_id)
    if key in loaded:
        return getter(*args)
    if is_model_cached(repo_id):
        msg = f"Loading {label} from the local cache..."
    else:
        msg = (f"Downloading {label} ({size}) - first run only. This can take several minutes; "
               "the progress bar is shown in the VS Code terminal.")
    with st.status(msg, expanded=True) as status:
        model = getter(*args)
        status.update(label=f"{label} ready", state="complete", expanded=False)
    loaded.add(key)
    return model


def llm_getter():
    p = LLM_PRESET
    return load_with_status("llm", f"LLM ({p['name']})", p["name"], p["size"], get_llm, p["name"], p["arch"])

def bert_getter():
    return load_with_status("bert", "BERT", config.BERT_MODEL_NAME, config.BERT_SIZE, get_bert)

def vqa_getter():
    return load_with_status("vqa", "VQA model", config.VQA_MODEL_NAME, config.VQA_SIZE, get_vqa)

# ----------------------------------------------------------------- helpers
DATASET = load_dataset()
SAMPLE_PASSAGE, SAMPLE_QUESTION = DATASET[0]["passage"], DATASET[2]["question"]
LEVEL_CLASS = {"ok": "ok", "review": "review", "hallucination": "halluc"}


def card(title, answer, meta, css):
    st.markdown(f'<div class="card {css}"><h4>{title}</h4><div class="ans">{html.escape(str(answer))}'
                f'</div><div class="meta">{meta}</div></div>', unsafe_allow_html=True)


def banner(cmp_):
    icon = {"ok": "✅", "review": "⚠️", "hallucination": "🚨"}[cmp_["level"]]
    st.markdown(f'<div class="banner {LEVEL_CLASS[cmp_["level"]]}">{icon} {cmp_["status"]} — '
                f'{html.escape(cmp_["message"])}</div>', unsafe_allow_html=True)


def validate_inputs(passage, question):
    if not passage.strip():
        return "The passage is empty. Please paste a passage."
    if not question.strip():
        return "The question is empty. Please type a question."
    if len(passage) > config.MAX_PASSAGE_CHARS:
        return f"Passage is too long (max {config.MAX_PASSAGE_CHARS} characters)."
    return None


def show_evidence(label, ev):
    if ev is None:
        st.markdown(f"**{label}:** no supporting sentence found.")
    else:
        tag = "exact match in sentence" if ev["exact"] else "closest sentence (partial match)"
        st.markdown(f'**{label}** — {tag}')
        st.markdown(f'<div class="evidence">{html.escape(ev["sentence"])}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- header + sidebar
st.title("Extractive vs. Generative Question Answering System")
st.caption("Compare BERT and LLM answers, detect possible hallucinations, and explore visual question answering.")

with st.sidebar:
    st.header("⚙️ Settings")
    names = list(config.LLM_PRESETS)
    llm_choice = st.selectbox("Generative LLM", names, index=names.index(config.DEFAULT_LLM_PRESET),
                              help="Lighter models need less RAM and download size; larger ones answer better.")
    LLM_PRESET = config.LLM_PRESETS[llm_choice]
    st.caption(f"`{LLM_PRESET['name']}` · {LLM_PRESET['size']}. {LLM_PRESET['note']}")
    threshold = st.slider("Agreement threshold (similarity)", 0.1, 1.0, config.SIMILARITY_THRESHOLD, 0.05)
    st.markdown(f"**Device:** `{config.get_device_name()}`")

    st.subheader("Models")
    for label, repo in [("Extractive (BERT)", config.BERT_MODEL_NAME), ("LLM", LLM_PRESET["name"]),
                        ("VQA (ViLT)", config.VQA_MODEL_NAME)]:
        st.markdown(f"{'✅' if is_model_cached(repo) else '⬇️'} **{label}** — `{repo}`")
    st.caption("✅ = downloaded and cached · ⬇️ = will be downloaded on first use (internet needed once). "
               f"Cache folder: `{config.HF_CACHE_DIR}`")
    if st.button("Load / download all models now"):
        try:
            bert_getter(); llm_getter(); vqa_getter()
            st.success("All models are loaded.")
        except (BertError, LLMError, VQAError) as exc:
            st.error(str(exc))
    if st.button("Free memory (unload models)"):
        import gc
        st.cache_resource.clear()
        st.session_state["loaded"] = set()
        gc.collect()
        st.info("Models unloaded from RAM (files stay cached on disk).")
    if st.button("Check internet connection"):
        st.success("huggingface.co is reachable.") if internet_available() else st.error(
            "Cannot reach huggingface.co. Models that are already downloaded still work offline.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Text QA", "🔍 Hallucination Detection",
                                        "🖼️ Visual QA", "🔊 Text-to-Speech", "📊 Model Evaluation"])

# ================================================================= TAB 1
with tab1:
    passage = st.text_area("Passage", value=SAMPLE_PASSAGE, height=200, key="passage")
    question = st.text_input("Question", value=SAMPLE_QUESTION, key="question")
    c1, c2 = st.columns([2, 1])
    technique = c1.selectbox("Prompt technique", TECHNIQUES)
    c1.caption(DESCRIPTIONS[technique])
    compare_all = c2.checkbox("Compare all 4 prompt techniques")

    if st.button("Generate Answer", type="primary"):
        error = validate_inputs(passage, question)
        if error:
            st.error(error)
        else:
            try:
                bert, llm = bert_getter(), llm_getter()
                with st.spinner("Running BERT and the LLM..."):
                    t0 = time.perf_counter()
                    bert_res = bert.answer(passage, question)
                    t1 = time.perf_counter()
                    llm_res = llm.answer(passage, question, technique)
                    t2 = time.perf_counter()
                    comparison = compare_answers(llm_res["answer"], bert_res["answer"], passage, threshold)
                    technique_rows = []
                    if compare_all:
                        for tech in TECHNIQUES:
                            r = llm.answer(passage, question, tech)
                            c = compare_answers(r["answer"], bert_res["answer"], passage, threshold)
                            technique_rows.append({"Technique": tech, "LLM Answer": r["answer"],
                                                   "Similarity to BERT": c["similarity"],
                                                   "Status": c["status"]})
                st.session_state["result"] = {
                    "passage": passage, "question": question, "technique": technique,
                    "llm": llm_res, "bert": bert_res, "llm_model": LLM_PRESET["name"], "comparison": comparison,
                    "bert_time": t1 - t0, "llm_time": t2 - t1, "techniques": technique_rows}
                st.session_state.pop("audio", None)
            except (LLMError, BertError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    res = st.session_state.get("result")
    if res:
        st.divider()
        banner(res["comparison"])
        a, b = st.columns(2)
        with a:
            note = f"{res['llm_model']} · {res['technique']} · {res['llm_time']:.1f}s"
            if res["llm"]["trimmed"]:
                note += " · long passage trimmed to the most relevant sentences"
            card("🤖 Generative LLM answer", res["llm"]["answer"], note, "llm")
            if res["llm"]["evidence"]:
                st.caption(f"Model's stated evidence: {res['llm']['evidence']}")
        with b:
            conf = res["bert"]["score"]
            card("📌 BERT extractive answer", res["bert"]["answer"],
                 f"Model confidence: {conf:.1%} · {res['bert_time']:.1f}s", "bert")
            st.progress(min(max(conf, 0.0), 1.0))
        st.markdown(f"**Similarity score:** `{res['comparison']['similarity']:.2f}` "
                    f"(threshold {threshold:.2f})")
        if res["techniques"]:
            st.subheader("Prompt technique comparison")
            st.dataframe(pd.DataFrame(res["techniques"]), hide_index=True)
        with st.expander("Show the exact prompt sent to the LLM"):
            st.code(res["llm"]["prompt"])

# ================================================================= TAB 2
with tab2:
    res = st.session_state.get("result")
    if not res:
        st.info("Run a question in the **Text QA** tab first.")
    else:
        cmp_ = res["comparison"]
        banner(cmp_)
        m1, m2, m3 = st.columns(3)
        m1.metric("Similarity score", f"{cmp_['similarity']:.2f}")
        m2.metric("Status", cmp_["status"])
        m3.metric("Possible hallucination", "YES" if cmp_["possible_hallucination"] else "No")
        a, b = st.columns(2)
        a.markdown(f"**LLM answer:** {cmp_['llm_answer']}")
        a.markdown(f"Supported by passage: {'✅ yes' if cmp_['llm_supported'] else '❌ no'}")
        b.markdown(f"**BERT answer:** {cmp_['bert_answer']}")
        b.markdown(f"Supported by passage: {'✅ yes' if cmp_['bert_supported'] else '❌ no'}")
        st.subheader("Supporting evidence from the passage")
        show_evidence("Evidence for the LLM answer", cmp_["llm_evidence"])
        show_evidence("Evidence for the BERT answer", cmp_["bert_evidence"])
        st.caption("A disagreement does not prove a hallucination. Compare each answer with the "
                   "evidence above and verify manually.")

# ================================================================= TAB 3
with tab3:
    st.markdown("Upload an image and ask a question about it. (VQA is separate from the text QA comparison.)")
    up = st.file_uploader("Image (JPG, JPEG, PNG)", type=None, key="vqa_upload")
    img_q = st.text_input("Question about the image", value="What animal is shown in the image?")
    image = None
    if up is not None:
        try:
            image = load_image(up)
            st.image(image, caption=up.name, width=420)
        except VQAError as exc:
            st.error(str(exc))
    if st.button("Predict Answer", type="primary", key="vqa_btn"):
        if image is None:
            st.error("Please upload a valid JPG/JPEG/PNG image first.")
        elif not img_q.strip():
            st.error("Please type a question about the image.")
        else:
            try:
                vqa = vqa_getter()
                with st.spinner("Predicting..."):
                    out = vqa.predict(image, img_q)
                card("🖼️ VQA answer", out["answer"], f"Model score: {out['confidence']:.1%}", "bert")
                st.write("Top predictions:", ", ".join(f"{a} ({s:.1%})" for a, s in out["top"]))
            except VQAError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

# ================================================================= TAB 4
with tab4:
    res = st.session_state.get("result")
    if not res:
        st.info("Run a question in the **Text QA** tab first, then come back to listen.")
    else:
        st.session_state.setdefault("audio", {})
        for key, label, text in [("llm", "LLM answer", res["llm"]["answer"]),
                                 ("bert", "BERT answer", res["bert"]["answer"])]:
            st.subheader(label)
            st.write(text)
            if st.button(f"🔊 Listen to {label}", key=f"tts_{key}"):
                try:
                    with st.spinner("Generating audio..."):
                        st.session_state["audio"][key] = text_to_speech(text)
                except TTSError as exc:
                    st.error(str(exc))
            if key in st.session_state["audio"]:
                st.audio(st.session_state["audio"][key], format="audio/mp3")

# ================================================================= TAB 5
with tab5:
    st.markdown("Runs **both models on the same 20 questions** and compares them with the reference "
                "answers. Scores are computed from real predictions. 2 of the 20 questions are "
                "unanswerable (reference = 'Answer not found').")
    with st.expander("View the 20-question test dataset"):
        st.dataframe(pd.DataFrame(DATASET).rename(columns={"id": "Question ID", "passage": "Passage",
                     "question": "Question", "reference_answer": "Reference Answer"}), hide_index=True)
    eval_tech = st.selectbox("Prompt technique for the LLM", TECHNIQUES, key="eval_tech")
    if st.button("Run Evaluation", type="primary"):
        try:
            bert, llm = bert_getter(), llm_getter()
            bar = st.progress(0.0, text="Evaluating (real model inference, may take a few minutes on CPU)...")
            df = run_evaluation(llm, bert, DATASET, eval_tech, threshold,
                                lambda i, n: bar.progress(i / n, text=f"Evaluated {i}/{n} questions"))
            bar.empty()
            df.to_csv(config.RESULTS_PATH, index=False)
            st.session_state["eval_df"] = df
            st.session_state["eval_meta"] = f"LLM: {LLM_PRESET['name']} · prompt: {eval_tech} · threshold {threshold:.2f}"
        except (LLMError, BertError) as exc:
            st.error(f"Evaluation incomplete: {exc}")
        except Exception as exc:
            st.error(f"Evaluation incomplete: {exc}")

    df = st.session_state.get("eval_df")
    if df is not None:
        s = summarize(df)
        st.caption("Results from real model predictions — " + st.session_state.get("eval_meta", ""))
        if not s["complete"]:
            st.warning(f"Evaluation INCOMPLETE: LLM answered {s['llm_evaluated']}/{s['total_questions']}, "
                       f"BERT answered {s['bert_evaluated']}/{s['total_questions']}. "
                       "Scores only cover the questions that ran. See the error columns.")
        else:
            st.success(f"All {s['total_questions']} questions evaluated with both models.")
        fmt = lambda v: "n/a" if v != v else f"{v:.1f}%"
        k = st.columns(4)
        k[0].metric("LLM Exact Match", fmt(s["llm_em_pct"]))
        k[1].metric("BERT Exact Match", fmt(s["bert_em_pct"]))
        k[2].metric("LLM avg F1", fmt(s["llm_f1_pct"]))
        k[3].metric("BERT avg F1", fmt(s["bert_f1_pct"]))
        k2 = st.columns(2)
        k2[0].metric("Disagreements", s["disagreements"])
        k2[1].metric("Possible hallucination flags", s["hallucination_flags"])

        st.subheader("Per-question results")
        st.dataframe(df.drop(columns=["Passage"]), hide_index=True)
        st.download_button("⬇️ Download results CSV", df.to_csv(index=False), "evaluation_results.csv")

        st.subheader("BERT vs. LLM")
        ch1, ch2 = st.columns(2)
        ch1.bar_chart(pd.DataFrame({"LLM": [s["llm_em_pct"], s["llm_f1_pct"]],
                                    "BERT": [s["bert_em_pct"], s["bert_f1_pct"]]},
                                   index=["Exact Match (%)", "Average F1 (%)"]))
        ch2.bar_chart(df.set_index("Question ID")[["LLM F1", "BERT F1"]].astype(float))
        st.subheader("Disagreement statistics")
        st.bar_chart(pd.DataFrame({"Count": [int((df["Status"] == "AGREEMENT").sum()),
                                             s["disagreements"], s["hallucination_flags"]]},
                                  index=["Agreements", "Disagreements", "Possible hallucinations"]))
