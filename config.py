"""Central configuration. Edit model names here (or choose the LLM in the app sidebar)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "qa_dataset.json"
RESULTS_PATH = BASE_DIR / "data" / "evaluation_results.csv"

# ---- Model cache -----------------------------------------------------------
# Models are downloaded ONCE from Hugging Face and stored here (inside the project folder).
# Override with:  $env:QA_MODEL_CACHE="D:\hf_cache"   (PowerShell, before starting the app)
# NOTE: these variables must be set before transformers / huggingface_hub are imported,
# so every module imports `config` first.
if "QA_MODEL_CACHE" in os.environ:
    os.environ["HF_HOME"] = os.environ["QA_MODEL_CACHE"]
else:
    os.environ.setdefault("HF_HOME", str(BASE_DIR / "models_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows without Developer Mode
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")           # slower connections
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")           # avoids the Windows "libiomp5md.dll" crash
HF_CACHE_DIR = os.environ["HF_HOME"]

# ---- Generative LLM (instruction-tuned, free, runs on CPU) ------------------
# Choose in the app sidebar, or start with:  $env:QA_LLM_PRESET="flan-t5-small"
LLM_PRESETS = {
    "flan-t5-base": {"name": "google/flan-t5-base", "arch": "seq2seq",
                     "size": "~1.0 GB download, ~2 GB RAM",
                     "note": "Default. Good balance of quality and speed on a laptop CPU."},
    "flan-t5-small": {"name": "google/flan-t5-small", "arch": "seq2seq",
                      "size": "~0.3 GB download, ~1 GB RAM",
                      "note": "Lightest option for low-RAM laptops. Weaker answers."},
    "flan-t5-large": {"name": "google/flan-t5-large", "arch": "seq2seq",
                      "size": "~3.1 GB download, ~4-5 GB RAM",
                      "note": "Best quality here, but slow on CPU. Needs 8 GB+ RAM."},
    "qwen2.5-0.5b-instruct": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "arch": "causal",
                              "size": "~1.0 GB download, ~2.5 GB RAM",
                              "note": "Small chat-style instruction model."},
}
DEFAULT_LLM_PRESET = os.environ.get("QA_LLM_PRESET", "flan-t5-base")
if DEFAULT_LLM_PRESET not in LLM_PRESETS:
    DEFAULT_LLM_PRESET = "flan-t5-base"
LLM_MODEL_NAME = LLM_PRESETS[DEFAULT_LLM_PRESET]["name"]
LLM_ARCH = LLM_PRESETS[DEFAULT_LLM_PRESET]["arch"]
LLM_MAX_NEW_TOKENS = 64
LLM_MAX_INPUT_TOKENS = 1024
LLM_MAX_CONTEXT_WORDS = 350     # longer passages are trimmed to the most relevant sentences

# ---- Extractive model -------------------------------------------------------
BERT_MODEL_NAME = "deepset/bert-base-cased-squad2"
BERT_SIZE = "~0.45 GB download, ~1 GB RAM"
BERT_MAX_SEQ_LEN = 384          # the pipeline splits long passages into chunks of this size
BERT_DOC_STRIDE = 128           # overlap between chunks

# ---- VQA --------------------------------------------------------------------
VQA_MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
VQA_SIZE = "~0.5 GB download, ~1 GB RAM"
VQA_MAX_QUESTION_TOKENS = 40    # ViLT's text position limit
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
MAX_IMAGE_MB = 10

# ---- Comparison / misc ------------------------------------------------------
SIMILARITY_THRESHOLD = 0.5      # similarity >= threshold => AGREEMENT
TTS_LANGUAGE = "en"
MAX_PASSAGE_CHARS = 20000
MODEL_LOAD_RETRIES = 3          # retries for network failures while downloading


def get_device_name() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
