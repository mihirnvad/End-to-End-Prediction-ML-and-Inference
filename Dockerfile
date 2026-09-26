# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   base       -> slim OS layer + non-root user
#   deps-api   -> lean virtualenv for serving (no SHAP / numba / streamlit)
#   deps-full  -> + training, plotting and dashboard dependencies
#   trainer    -> generates data, trains + evaluates the model with the *same* library
#                 versions the runtime uses (pickled pipelines are version-sensitive)
#   api        -> FastAPI runtime        (docker build --target api .)
#   dashboard  -> Streamlit runtime      (docker build --target dashboard .)

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# libgomp1: OpenMP runtime required by XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app
WORKDIR /app

# ---------------------------------------------------------------------------------------
FROM base AS deps-api
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# ---------------------------------------------------------------------------------------
FROM deps-api AS deps-full
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------------------------------
FROM deps-full AS trainer
ARG N_SAMPLES=25000
# e.g. --build-arg TRAIN_ARGS=--fast for a quick CI build
ARG TRAIN_ARGS=""
COPY src/ src/
COPY data/generate_dataset.py data/
RUN python data/generate_dataset.py --n-samples ${N_SAMPLES} \
    && python src/train.py ${TRAIN_ARGS} \
    && python src/evaluate.py

# ---------------------------------------------------------------------------------------
FROM base AS api
ENV PATH=/opt/venv/bin:$PATH \
    CHURN_ARTIFACT_DIR=/app/artifacts
COPY --from=deps-api /opt/venv /opt/venv
COPY --chown=app:app src/ src/
COPY --chown=app:app app/ app/
COPY --from=trainer --chown=app:app \
    /app/artifacts/model_pipeline.joblib \
    /app/artifacts/model_metadata.json \
    /app/artifacts/metrics.json \
    artifacts/
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]
# One worker per container; scale horizontally with replicas.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

# ---------------------------------------------------------------------------------------
FROM base AS dashboard
ENV PATH=/opt/venv/bin:$PATH \
    CHURN_ARTIFACT_DIR=/app/artifacts \
    API_URL=http://api:8000
COPY --from=deps-full /opt/venv /opt/venv
COPY --chown=app:app src/ src/
COPY --chown=app:app app/ app/
COPY --chown=app:app frontend/ frontend/
COPY --from=trainer --chown=app:app /app/artifacts/ artifacts/
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).status != 200)"]
CMD ["streamlit", "run", "frontend/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
