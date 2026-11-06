"""Shared helpers for downloading/loading Hugging Face models safely.

* classify_error()  - recognise network / memory / disk / DLL / dependency problems
* friendly_message() - turn the technical exception into a clear instruction
* safe_load()       - run a loader with retries on network failures
* is_model_cached() - is the model already in the local cache?
* internet_available() - quick check that huggingface.co can be reached
"""
import socket
import time

import config  # sets HF_HOME etc. before transformers is imported


def _chain_text(exc) -> str:
    """Text of an exception and all exceptions that caused it (lower-case)."""
    parts, seen, e = [], set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        parts.append(f"{type(e).__name__}: {e}")
        e = e.__cause__ or e.__context__
    return " | ".join(parts).lower()


_MEMORY = ("memoryerror", "out of memory", "paging file is too small", "not enough memory",
           "defaultcpuallocator", "alloc_cpu", "winerror 1455", "bad allocation")
_DISK = ("no space left", "errno 28", "not enough space on the disk", "disk full", "winerror 112")
_NOTFOUND = ("repositorynotfounderror", "404 client error", "401 client error", "403 client error",
             "gated repo", "is not a valid model identifier", "revisionnotfounderror")
_NETWORK = ("connectionerror", "max retries exceeded", "getaddrinfo", "name resolution",
            "couldn't connect", "could not connect", "timed out", "timeout", "proxyerror",
            "sslerror", "connection reset", "connection aborted", "localentrynotfounderror",
            "offlinemodeisenabled", "remote end closed", "network is unreachable",
            "incompleteread", "chunkedencodingerror", "failed to resolve", "gtts")
_DLL = ("dll load failed", "winerror 126", "winerror 1114", "the specified module could not be found")
_DEPENDENCY = ("no module named", "cannot import name", "requires the protobuf", "requires the sentencepiece")


def classify_error(exc) -> str:
    """Return one of: memory, disk, notfound, network, dll, dependency, other."""
    text = _chain_text(exc)
    for kind, hints in (("memory", _MEMORY), ("disk", _DISK), ("dll", _DLL),
                        ("notfound", _NOTFOUND), ("dependency", _DEPENDENCY),
                        ("network", _NETWORK)):
        if any(h in text for h in hints):
            return kind
    return "other"


def friendly_message(kind: str, model_name: str, exc=None) -> str:
    detail = f"\n\nTechnical detail: {type(exc).__name__}: {str(exc)[:300]}" if exc is not None else ""
    msgs = {
        "network": (f"Could not download '{model_name}' because huggingface.co could not be reached. "
                    "Check your internet connection, VPN/proxy or firewall and try again. Partial "
                    "downloads are resumed automatically. If the model was never downloaded before, "
                    "an internet connection is required once."),
        "memory": (f"Not enough memory (RAM) to load or run '{model_name}'. Close other programs, "
                   "choose a smaller model (e.g. flan-t5-small in the sidebar), or increase the "
                   "Windows virtual memory (paging file)."),
        "disk": (f"Not enough free disk space to store '{model_name}' in {config.HF_CACHE_DIR}. "
                 "Free some space or set QA_MODEL_CACHE to another drive."),
        "notfound": (f"Model '{model_name}' was not found or is not accessible on Hugging Face. "
                     "Check the model name in config.py and your internet connection."),
        "dll": ("A Windows DLL needed by PyTorch could not be loaded. Install the 'Microsoft Visual "
                "C++ Redistributable 2015-2022 (x64)' and make sure you use Python 3.10-3.12 (64-bit)."),
        "dependency": ("A required Python package is missing or has the wrong version. Activate the "
                       "virtual environment and run: pip install -r requirements.txt"),
        "other": f"Could not load or run '{model_name}'.",
    }
    return msgs.get(kind, msgs["other"]) + detail


def safe_load(loader, model_name: str, error_cls, retries: int = config.MODEL_LOAD_RETRIES):
    """Call loader(); retry on network errors; raise error_cls with a friendly message on failure."""
    last_exc, kind = None, "other"
    for attempt in range(1, retries + 1):
        try:
            return loader()
        except Exception as exc:  # noqa: BLE001 - we re-raise a friendlier error below
            last_exc, kind = exc, classify_error(exc)
            if kind == "network" and attempt < retries:
                time.sleep(2 * attempt)
                continue
            break
    raise error_cls(friendly_message(kind, model_name, last_exc)) from last_exc


def is_model_cached(repo_id: str) -> bool:
    """True if the weights of a model are already in the local cache (no download needed)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        for fname in ("model.safetensors", "pytorch_model.bin",
                      "model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if isinstance(try_to_load_from_cache(repo_id, fname), str):
                return True
    except Exception:
        pass
    return False


def internet_available(host: str = "huggingface.co", port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
