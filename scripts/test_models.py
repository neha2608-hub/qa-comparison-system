"""Test each model on its own with REAL inference (nothing is simulated).

    python scripts/test_models.py bert
    python scripts/test_models.py llm [--llm flan-t5-small]
    python scripts/test_models.py vqa [--image path\\to\\dog.jpg --question "What animal is shown?"]
    python scripts/test_models.py tts
    python scripts/test_models.py all

Every printed answer and score comes straight from the model. "matches expectation" only compares
the real output with the answer a human would expect; a "differs" line is not an error in the script.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PASSAGE = ("The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, "
           "France. It is named after the engineer Gustave Eiffel. Construction began in 1887 and "
           "was completed in 1889. The tower is 330 metres tall.")
FILLER = " ".join(f"Sentence {i} describes an unrelated detail about a quiet town and its weather."
                  for i in range(60))
LONG_PASSAGE = FILLER + " The annual lantern festival is held in Lyon every June. " + FILLER

results = []


def check(label, answer, expected):
    ok = expected.lower() in answer.lower()
    results.append(ok)
    print(f"   expected ~ '{expected}'  ->  {'matches expectation' if ok else 'DIFFERS (inspect)'}")


def test_bert():
    from modules.bert_qa import BertQA
    print("\n##### BERT (extractive) #####")
    t0 = time.perf_counter(); bert = BertQA(); print(f"loaded in {time.perf_counter()-t0:.1f}s on {bert.device}")
    cases = [("normal question", PASSAGE, "In which year was construction completed?", "1889"),
             ("unanswerable question", PASSAGE, "Who painted the Eiffel Tower?", "Answer not found"),
             (f"long passage ({len(LONG_PASSAGE.split())} words, chunked)", LONG_PASSAGE,
              "Where is the annual lantern festival held?", "Lyon")]
    for name, ctx, q, exp in cases:
        r = bert.answer(ctx, q)
        print(f"-- {name}\n   Q: {q}\n   A: {r['answer']}   (model confidence {r['score']:.3f}, span {r['start']}-{r['end']})")
        check(name, r["answer"], exp)
        if r["found"]:
            assert ctx[r["start"]:r["end"]].strip() == r["answer"], "BERT answer must be a span of the passage"


def test_llm(preset_key):
    from modules.llm_qa import LLMQA
    from modules.prompt_engineering import TECHNIQUES
    preset = config.LLM_PRESETS[preset_key]
    print(f"\n##### LLM ({preset['name']}) #####")
    t0 = time.perf_counter(); llm = LLMQA(preset["name"], preset["arch"])
    print(f"loaded in {time.perf_counter()-t0:.1f}s on {llm.device}")
    for q, exp in [("In which year was construction completed?", "1889"),
                   ("Who painted the Eiffel Tower?", "Answer not found")]:
        for tech in TECHNIQUES:
            t0 = time.perf_counter(); r = llm.answer(PASSAGE, q, tech)
            print(f"-- [{tech}] Q: {q}\n   A: {r['answer']}   ({time.perf_counter()-t0:.1f}s)")
            check(tech, r["answer"], exp)


def test_vqa(image_path, question):
    from modules.vqa import VQAModel
    from PIL import Image, ImageDraw
    print("\n##### VQA (ViLT) #####")
    if image_path:
        img, qs = Image.open(image_path).convert("RGB"), [question or "What is in the image?"]
        exps = [None]
    else:  # generate a simple test image: red circle on white background
        img = Image.new("RGB", (384, 384), "white")
        ImageDraw.Draw(img).ellipse((92, 92, 292, 292), fill="red")
        qs, exps = ["What color is the circle?", "What shape is shown?"], ["red", "circle"]
        print("(no --image given: using a generated image of a red circle; use --image for a real photo)")
    t0 = time.perf_counter(); vqa = VQAModel(); print(f"loaded in {time.perf_counter()-t0:.1f}s on {vqa.device}")
    for q, exp in zip(qs, exps):
        r = vqa.predict(img, q)
        print(f"-- Q: {q}\n   A: {r['answer']}   top-3: " + ", ".join(f"{a} ({s:.2f})" for a, s in r["top"]))
        if exp:
            check(q, r["answer"], exp)


def test_tts():
    from modules.tts import text_to_speech
    print("\n##### gTTS #####")
    data = text_to_speech("The Eiffel Tower was completed in eighteen eighty nine.")
    out = config.BASE_DIR / "assets" / "tts_test.mp3"
    out.write_bytes(data)
    print(f"OK - {len(data)} bytes written to {out}. Open it to listen.")
    assert len(data) > 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["bert", "llm", "vqa", "tts", "all"])
    ap.add_argument("--llm", default=config.DEFAULT_LLM_PRESET, choices=list(config.LLM_PRESETS))
    ap.add_argument("--image"); ap.add_argument("--question")
    a = ap.parse_args()
    try:
        if a.target in ("bert", "all"): test_bert()
        if a.target in ("llm", "all"): test_llm(a.llm)
        if a.target in ("vqa", "all"): test_vqa(a.image, a.question)
        if a.target in ("tts", "all"): test_tts()
    except Exception as exc:  # noqa: BLE001
        print(f"\nTEST FAILED: {exc}")
        sys.exit(1)
    if results:
        print(f"\nSummary: {sum(results)}/{len(results)} real outputs matched the expected answers.")
    print("Done - all components ran real inference without errors.")


if __name__ == "__main__":
    main()
