# Climate Change Awareness API: container image
# Build (from the repo root, after filling model_store/):
#     python3 src/api/download_models.py
#     docker build -t climate-api .
# Run:
#     docker run --rm -p 8000:8000 climate-api
# Then open http://127.0.0.1:8000/docs

# Same Python version as training (3.12), small "slim" variant
FROM python:3.12-slim

# No .pyc files, logs appear immediately, src/ is importable, models come only from the baked-in folder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/model_store \
    MODEL_REGISTRY_FALLBACK=0

WORKDIR /app

# Libraries first: this layer is cached and only rebuilt when requirements-api.txt changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only the code the API needs (no notebooks, no training scripts, no data)
COPY src/api src/api
COPY src/climate_models src/climate_models

# The champion models, downloaded from the registry before the build
COPY model_store model_store

# Fail the build if the models do not load and predict with these library versions
RUN python -m api.smoke_test

# Do not run as root inside the container
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Docker marks the container healthy only while /health answers
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
