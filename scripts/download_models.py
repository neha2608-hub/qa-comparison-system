"""Download and cache every model once (so the app starts instantly later).

    python scripts/download_models.py                      # BERT + default LLM + ViLT + gTTS check
    python scripts/download_models.py --llm flan-t5-small  # use a lighter LLM
    python scripts/download_models.py --skip-vqa           # skip a model

Download progress bars appear in this terminal. Models are saved in the folder printed below
and are NOT downloaded again on later runs.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402  (must be imported first: sets the cache folder)
from utils.model_loader import internet_available, is_model_cached  # noqa: E402


def step(label, repo_id, size, loader):
    cached = is_model_cached(repo_id)
    print(f"\n=== {label}: {repo_id} ({size}) - "
          f"{'found in cache, loading' if cached else 'downloading (first time only)'} ===")
    t0 = time.perf_counter()
    loader()
    print(f"OK - ready in {time.perf_counter() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=config.DEFAULT_LLM_PRESET, choices=list(config.LLM_PRESETS))
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-vqa", action="store_true")
    args = ap.parse_args()

    print(f"Model cache folder: {config.HF_CACHE_DIR}")
    online = internet_available()
    print(f"Internet / huggingface.co reachable: {'yes' if online else 'NO'}")
    preset = config.LLM_PRESETS[args.llm]
    todo = []
    if not args.skip_bert:
        todo.append(("Extractive QA", config.BERT_MODEL_NAME, config.BERT_SIZE))
    if not args.skip_llm:
        todo.append(("Generative LLM", preset["name"], preset["size"]))
    if not args.skip_vqa:
        todo.append(("VQA", config.VQA_MODEL_NAME, config.VQA_SIZE))
    missing = [t for t in todo if not is_model_cached(t[1])]
    if missing and not online:
        print("\nERROR: these models are not downloaded yet and there is no internet connection:")
        for _, repo, _ in missing:
            print("  -", repo)
        print("Connect to the internet (check VPN/proxy/firewall) and run this script again.")
        sys.exit(1)

    from modules.bert_qa import BertQA, BertError
    from modules.llm_qa import LLMQA, LLMError
    from modules.vqa import VQAModel, VQAError
    loaders = {
        config.BERT_MODEL_NAME: lambda: BertQA(),
        preset["name"]: lambda: LLMQA(preset["name"], preset["arch"]),
        config.VQA_MODEL_NAME: lambda: VQAModel(),
    }
    failed = False
    for label, repo, size in todo:
        try:
            step(label, repo, size, loaders[repo])
        except (BertError, LLMError, VQAError) as exc:
            failed = True
            print(f"\nFAILED: {exc}")

    print("\n=== Text-to-Speech (gTTS) ===")
    if online:
        try:
            from modules.tts import text_to_speech
            n = len(text_to_speech("Text to speech is working."))
            print(f"OK - gTTS generated {n} bytes of audio")
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"FAILED: {exc}")
    else:
        print("Skipped (no internet). gTTS needs internet every time it generates audio.")

    print("\n" + ("Some steps FAILED - see messages above and TROUBLESHOOTING.md." if failed
                  else "All models are downloaded and cached. Start the app: streamlit run app.py"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
