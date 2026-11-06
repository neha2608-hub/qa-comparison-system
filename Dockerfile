# Dockerfile for Hugging Face Spaces (Docker SDK) - also works on any Docker host.
FROM python:3.11-slim

# Hugging Face Spaces runs containers as a non-root user with id 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    QA_MODEL_CACHE=/home/user/models_cache \
    PYTHONUNBUFFERED=1
RUN mkdir -p /home/user/app
WORKDIR /home/user/app

# CPU-only PyTorch (about 200 MB instead of ~4 GB with CUDA packages)
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Download the models at build time so visitors do not wait ~2 GB on the first request.
# (If this step fails, the app still downloads the models on first use.)
RUN python scripts/download_models.py || echo "Model pre-download skipped"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
