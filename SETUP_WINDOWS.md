# Windows 10/11 Setup (VS Code + PowerShell)

## 0. Requirements
- **Python 3.10, 3.11 or 3.12, 64-bit** (3.13 is NOT supported by the pinned PyTorch/NumPy). Check: `python --version`. Several versions installed? `py -0` lists them; use `py -3.11` below.
- **8 GB RAM recommended** (4 GB works with `flan-t5-small`), **~4 GB free disk** (models ~2 GB + packages).
- Internet for the first run, and for gTTS every time.
- Put the project in a **short path without special characters**, e.g. `C:\projects\qa-comparison-system` (avoids Windows path-length problems).

## 1. Install
Open the folder in VS Code (File → Open Folder), then Terminal → New Terminal (PowerShell):
```powershell
cd C:\projects\qa-comparison-system
python -m venv venv            # or: py -3.11 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.
In VS Code press **Ctrl+Shift+P → Python: Select Interpreter → `.\venv\Scripts\python.exe`**.
(Optional NVIDIA GPU: install the CUDA build from https://pytorch.org after step 1; the app then uses the GPU automatically.)

## 2. Download and cache the models (one time, ~2 GB)
```powershell
python scripts\download_models.py
```
Progress bars appear in the terminal. Files go to `models_cache\` inside the project and are **never downloaded again**. The app also downloads missing models automatically on first use, so this step is optional but recommended (you see progress and errors immediately).

| Model | Purpose | Size |
|---|---|---|
| `deepset/bert-base-cased-squad2` | extractive QA | ~0.45 GB |
| `google/flan-t5-base` (default) | generative QA, CPU-friendly | ~1.0 GB |
| `dandelin/vilt-b32-finetuned-vqa` | visual QA | ~0.5 GB |
| gTTS | text-to-speech | online service, no download |

**Lighter / heavier LLM (you choose, nothing is swapped silently):**
```powershell
python scripts\download_models.py --llm flan-t5-small     # ~0.3 GB, for low-RAM laptops
python scripts\download_models.py --llm flan-t5-large     # ~3.1 GB, better, slow on CPU
python scripts\download_models.py --llm qwen2.5-0.5b-instruct
$env:QA_LLM_PRESET="flan-t5-small"                         # make it the default for this terminal
```
The app sidebar also has an LLM dropdown. Skip a model with `--skip-vqa`, `--skip-llm`, `--skip-bert`.

Other cache options:
```powershell
$env:QA_MODEL_CACHE="D:\hf_cache"     # store models on another drive (set before running anything)
$env:HF_HUB_OFFLINE="1"               # force offline use of already-downloaded models
```
To free space, delete the `models_cache` folder (models will download again when needed).

## 3. Test each model independently (real inference)
```powershell
python scripts\test_models.py bert
python scripts\test_models.py llm
python scripts\test_models.py llm --llm flan-t5-small
python scripts\test_models.py vqa
python scripts\test_models.py vqa --image C:\path\to\dog.jpg --question "What animal is shown in the image?"
python scripts\test_models.py tts
python scripts\test_models.py all
```
Each script prints the real answer and the model's own score. Lines marked `matches expectation` / `DIFFERS` compare the real output with the answer a human expects; a `DIFFERS` line is a genuine model result, not a script bug. `tts` writes `assets\tts_test.mp3` so you can listen.
Offline unit tests (logic only, no downloads): `python -m unittest discover -s tests -v`

## 4. Run the full application
```powershell
streamlit run app.py
```
Open http://localhost:8501. Sidebar: pick the LLM, see which models are cached (✅) or will download (⬇️), load all models, free memory, check internet.
Stop with **Ctrl+C**. After editing code, stop and start again (file watching is disabled in `.streamlit\config.toml`).

## 5. Real evaluation on the 20 Q&A pairs
Tab **Model Evaluation** → choose prompt technique → **Run Evaluation**. Both models run on all 20 questions; EM and F1 come from their actual predictions (CPU: roughly 1–3 minutes). Results are saved to `data\evaluation_results.csv`. If a model fails, the tab shows "Evaluation INCOMPLETE" instead of inventing scores.
