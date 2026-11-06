"""Offline unit tests (no model downloads needed).  Run: python -m unittest discover -s tests -v"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processing import normalize_answer, exact_match, f1_score, is_not_found
from modules.hallucination import compare_answers, similarity
from modules.prompt_engineering import build_prompt, extract_final_answer, TECHNIQUES
from modules.evaluation import load_dataset, run_evaluation, summarize
from modules.llm_qa import prepare_context
from modules.vqa import load_image, VQAError
from modules.tts import text_to_speech, TTSError

PASSAGE = ("The Eiffel Tower is located in Paris, France. It was completed in 1889. "
           "It is named after Gustave Eiffel.")


class Stub:
    """Fake model that returns a fixed answer (only used to test the pipeline plumbing)."""
    def __init__(self, ans): self.ans = ans
    def answer(self, context, question, technique=None): return {"answer": self.ans}


class FakeUpload:
    def __init__(self, name, data): self.name, self._d = name, data
    def getvalue(self): return self._d


class Tests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_answer("The  Eiffel, Tower!"), "eiffel tower")

    def test_em_f1(self):
        self.assertEqual(exact_match("The Tower.", "tower"), 1)
        self.assertEqual(exact_match("tower", "bridge"), 0)
        # pred 2 tokens, ref 3 tokens, 2 common: P=1, R=2/3, F1=0.8
        self.assertAlmostEqual(f1_score("Gustave Eiffel", "Gustave Eiffel Tower"), 0.8)
        self.assertEqual(f1_score("a", "b"), 0.0)

    def test_agreement(self):
        r = compare_answers("1889", "1889", PASSAGE)
        self.assertEqual(r["status"], "AGREEMENT")

    def test_disagreement_hallucination(self):
        r = compare_answers("Berlin", "Paris, France", PASSAGE)
        self.assertEqual(r["status"], "DISAGREEMENT")
        self.assertTrue(r["possible_hallucination"])

    def test_disagreement_but_supported(self):
        r = compare_answers("1889", "Gustave Eiffel", PASSAGE)
        self.assertEqual(r["status"], "DISAGREEMENT")
        self.assertFalse(r["possible_hallucination"])   # LLM answer IS in passage -> manual review

    def test_not_found(self):
        self.assertEqual(compare_answers("Answer not found", "Answer not found", PASSAGE)["status"], "AGREEMENT")
        self.assertTrue(is_not_found(""))

    def test_prompts(self):
        for t in TECHNIQUES:
            p = build_prompt(t, "CTX", "Q?")
            self.assertIn("CTX", p); self.assertIn("Q?", p)
        self.assertIn("ONLY the information", build_prompt("Context-grounded", "c", "q"))
        self.assertEqual(extract_final_answer("Chain-of-thought (evidence-based)", "x. Answer: 1889"), "1889")

    def test_long_passage_trim(self):
        long_ctx = " ".join(f"Sentence number {i} is about topic{i}." for i in range(300))
        ctx, trimmed = prepare_context(long_ctx + " The capital is Paris.", "capital Paris?", 50)
        self.assertTrue(trimmed); self.assertIn("Paris", ctx)

    def test_dataset_and_evaluation(self):
        data = load_dataset()
        self.assertEqual(len(data), 20)
        df = run_evaluation(Stub("Answer not found"), Stub("Answer not found"), data)
        s = summarize(df)
        self.assertEqual(s["total_questions"], 20); self.assertTrue(s["complete"])
        self.assertAlmostEqual(s["llm_em_pct"], 10.0)    # 2 unanswerable of 20 match exactly
        class Broken:
            def answer(self, *a, **k): raise RuntimeError("boom")
        s2 = summarize(run_evaluation(Broken(), Stub("x"), data))
        self.assertFalse(s2["complete"])

    def test_image_validation(self):
        with self.assertRaises(VQAError): load_image(FakeUpload("a.gif", b"x"))
        with self.assertRaises(VQAError): load_image(FakeUpload("fake.png", b"not an image"))
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (8, 8), "red").save(buf, "PNG")
        self.assertEqual(load_image(FakeUpload("ok.png", buf.getvalue())).size, (8, 8))

    def test_tts_empty(self):
        with self.assertRaises(TTSError): text_to_speech("   ")


if __name__ == "__main__":
    unittest.main()


class ErrorHandlingTests(unittest.TestCase):
    """Offline tests for download / memory error handling (no internet needed)."""
    def test_classify(self):
        from utils.model_loader import classify_error
        self.assertEqual(classify_error(MemoryError()), "memory")
        self.assertEqual(classify_error(RuntimeError("[enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: not enough memory")), "memory")
        self.assertEqual(classify_error(OSError("[Errno 28] No space left on device")), "disk")
        self.assertEqual(classify_error(ConnectionError("Max retries exceeded with url")), "network")
        self.assertEqual(classify_error(OSError("DLL load failed while importing _C")), "dll")
        self.assertEqual(classify_error(ImportError("No module named 'sentencepiece'")), "dependency")
        self.assertEqual(classify_error(ValueError("something else")), "other")

    def test_safe_load_retries_then_friendly_error(self):
        import config
        from utils.model_loader import safe_load
        calls = {"n": 0}
        def flaky():
            calls["n"] += 1
            raise ConnectionError("getaddrinfo failed")
        class MyErr(Exception): pass
        import time
        orig, time.sleep = time.sleep, lambda s: None     # do not actually wait
        try:
            with self.assertRaises(MyErr) as cm:
                safe_load(flaky, "some/model", MyErr, retries=3)
        finally:
            time.sleep = orig
        self.assertEqual(calls["n"], 3)                   # retried on network errors
        self.assertIn("internet", str(cm.exception).lower())

    def test_memory_error_not_retried(self):
        from utils.model_loader import safe_load
        calls = {"n": 0}
        def oom():
            calls["n"] += 1
            raise MemoryError()
        class MyErr(Exception): pass
        with self.assertRaises(MyErr): safe_load(oom, "m", MyErr, retries=3)
        self.assertEqual(calls["n"], 1)

    def test_cache_check_false_for_unknown_model(self):
        from utils.model_loader import is_model_cached
        self.assertFalse(is_model_cached("nonexistent-org/nonexistent-model"))
