# Troubleshooting

| Symptom / message | Cause | Fix |
|---|---|---|
| `Could not download '...' because huggingface.co could not be reached` / `We couldn't connect to 'https://huggingface.co'` / `getaddrinfo failed` / `Max retries exceeded` | No internet, VPN, proxy or firewall | Check the connection and open https://huggingface.co in a browser. On a company/college network set a proxy: `$env:HTTPS_PROXY="http://host:port"`. Re-run `python scripts\download_models.py` (partial downloads resume). Sidebar → "Check internet connection". |
| Download is very slow or `ReadTimeout` | Slow network | `$env:HF_HUB_DOWNLOAD_TIMEOUT="300"` then rerun the download script. Or start with `--llm flan-t5-small`. |
| `SSLError` / `CERTIFICATE_VERIFY_FAILED` | Antivirus or corporate proxy intercepting HTTPS | Try another network (mobile hotspot) for the one-time download; afterwards the app works offline. |
| `403`/`404 Client Error`, `Repository Not Found` | Wrong model name or blocked access | Check names in `config.py`; ensure you are online. |
| Model was downloaded but the app downloads again | Cache folder changed (`QA_MODEL_CACHE` set in one terminal only) | Use the same environment variable every time, or remove it. Check the cache path in the sidebar. |
| Works offline? | Yes after the first download (gTTS excepted) | `$env:HF_HUB_OFFLINE="1"` forces offline mode. |
| `OSError: [Errno 28] No space left on device` | Disk full | Free ~4 GB or `$env:QA_MODEL_CACHE="D:\hf_cache"`. |
| `[WinError 1455] The paging file is too small` / `DefaultCPUAllocator: not enough memory` / `MemoryError` | Not enough RAM to load the model | Close browsers/apps; choose **flan-t5-small** in the sidebar; use "Free memory (unload models)" before switching LLMs; increase virtual memory (System → Advanced → Performance → Virtual memory). |
| App very slow / laptop freezes while generating | Large LLM on CPU | Use flan-t5-base or flan-t5-small. flan-t5-large needs 8 GB+ RAM and is slow. |
| `DLL load failed while importing _C` / `WinError 126` / `WinError 1114` | Missing Visual C++ runtime or wrong Python | Install "Microsoft Visual C++ Redistributable 2015-2022 (x64)"; use 64-bit Python 3.10–3.12. |
| `No matching distribution found for torch==2.4.1` | Python 3.13 or 32-bit Python | Install Python 3.11 (64-bit) and create the venv with `py -3.11 -m venv venv`. |
| `ModuleNotFoundError` / `cannot import name ...` | venv not active or packages mismatched | `venv\Scripts\Activate.ps1`, then `pip install -r requirements.txt`. Do not upgrade packages individually; the versions are pinned to work together. |
| `pyarrow requires NumPy 2.0` | Newer pyarrow than pinned | `pip install pyarrow==17.0.0 numpy==1.26.4` |
| `OMP: Error #15: libiomp5md.dll already initialized` | Duplicate OpenMP runtime | Already handled in `config.py` (`KMP_DUPLICATE_LIB_OK`). If it persists, restart the terminal. |
| `streamlit : The term 'streamlit' is not recognized` | venv not active | Activate it, or run `python -m streamlit run app.py`. |
| Activation: `running scripts is disabled on this system` | PowerShell policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| gTTS: `Could not reach the Google Text-to-Speech service` | gTTS needs internet each time | Check the connection; the answer text and all other features still work offline. |
| Port in use | Another Streamlit running | `streamlit run app.py --server.port 8502` |
| Warning `symlinks ... cache-system uses degraded mode` | Windows without Developer Mode | Harmless (suppressed by config). Enable Developer Mode to save disk space. |
| VQA error `not a valid image` | File renamed to .png or corrupted | Use a real JPG/JPEG/PNG. |
| LLM says "Answer not found" too often / differs from BERT | Small model limits | Try the Few-shot or Context-grounded technique, or flan-t5-large. This is real model behaviour; the app flags it instead of hiding it. |

Still stuck? Run `python scripts\test_models.py bert` (or `llm`, `vqa`, `tts`) to isolate which component fails, and read the message printed in the terminal.
