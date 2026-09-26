"""FastAPI real-time churn inference service.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
Interactive OpenAPI docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse

from app.middleware import LatencyMiddleware, LatencyTracker
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnPredictionRequest,
    ExplanationResponse,
    FeatureContribution,
    HealthResponse,
    LatencyMetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from app.service import ModelService, load_model_service
from src.config import ArtifactPaths
from src.pipeline import SchemaError

API_VERSION = "1.0.0"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("churn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = str(ArtifactPaths.from_env().root)
    app.state.started_at = time.monotonic()
    app.state.model_service = None
    try:
        service = load_model_service(artifact_dir)
        service.warmup()
        app.state.model_service = service
    except Exception:  # keep serving /health so orchestrators can see *why* it is down
        logger.exception("Model failed to load from %s; serving 503s", artifact_dir)
    yield


async def get_service(request: Request) -> ModelService:
    # async: a sync dependency would cost a threadpool hop on every request
    service = request.app.state.model_service
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return service


router = APIRouter()


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
    responses={503: {"model": HealthResponse, "description": "Model not loaded"}},
)
async def health(request: Request, response: Response) -> HealthResponse:
    """Liveness + readiness: model status, version, and process uptime."""
    service: ModelService | None = request.app.state.model_service
    if service is None:
        response.status_code = 503
    return HealthResponse(
        status="ok" if service else "unavailable",
        model_loaded=service is not None,
        model_version=service.version if service else None,
        model_type=service.model_type if service else None,
        decision_threshold=service.threshold if service else None,
        uptime_seconds=round(time.monotonic() - request.app.state.started_at, 3),
        api_version=API_VERSION,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    payload: ChurnPredictionRequest, service: ModelService = Depends(get_service)
) -> PredictionResponse:
    """Score one customer: calibrated churn probability, decision, and risk tier."""
    # Single-row scoring on the compiled path is ~3 ms of CPU, cheaper than the threadpool
    # hand-off it would otherwise pay. The slower sklearn fallback, batch, and explain run
    # in the threadpool so they never block the event loop (health checks) for long.
    records = [payload.feature_record()]
    if service.compiled is not None:
        result = service.predict(records)[0]
    else:
        result = (await run_in_threadpool(service.predict, records))[0]
    return PredictionResponse(
        customer_id=payload.customer_id,
        decision_threshold=service.threshold,
        model_version=service.version,
        **result,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(
    payload: BatchPredictionRequest, service: ModelService = Depends(get_service)
) -> BatchPredictionResponse:
    """Vectorised scoring: the whole batch goes through the pipeline in a single call."""
    results = service.predict([c.feature_record() for c in payload.customers])
    predictions = [
        PredictionResponse(
            customer_id=customer.customer_id,
            decision_threshold=service.threshold,
            model_version=service.version,
            **result,
        )
        for customer, result in zip(payload.customers, results)
    ]
    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        n_predicted_churners=sum(p.churn_prediction for p in predictions),
        model_version=service.version,
    )


@router.post("/explain", response_model=ExplanationResponse, tags=["inference"])
def explain(
    payload: ChurnPredictionRequest, service: ModelService = Depends(get_service)
) -> ExplanationResponse:
    """Exact TreeSHAP attribution of one prediction, grouped by input feature."""
    result = service.explain(payload.feature_record())
    return ExplanationResponse(
        customer_id=payload.customer_id,
        churn_probability=result["churn_probability"],
        base_value=result["base_value"],
        base_probability=result["base_probability"],
        contributions=[FeatureContribution(**c) for c in result["contributions"]],
        model_version=service.version,
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["operations"])
async def model_info(service: ModelService = Depends(get_service)) -> ModelInfoResponse:
    meta = service.metadata
    return ModelInfoResponse(
        model_version=meta["model_version"],
        model_type=meta["model_type"],
        trained_at=meta["trained_at"],
        decision_threshold=meta["decision_threshold"],
        thresholds=meta["thresholds"],
        calibration=meta["calibration"],
        hyperparameters=meta["hyperparameters"],
        features=meta["features"],
        holdout_metrics=service.holdout_metrics,
        inference_path="compiled" if service.compiled is not None else "sklearn",
    )


@router.get("/metrics", response_model=LatencyMetricsResponse, tags=["operations"])
async def latency_metrics(request: Request) -> LatencyMetricsResponse:
    """Rolling server-side latency percentiles per route."""
    tracker: LatencyTracker = request.app.state.latency_tracker
    return LatencyMetricsResponse(window_size=tracker.window_size, routes=tracker.snapshot())


def create_app() -> FastAPI:
    app = FastAPI(
        title="Churn Prediction Inference API",
        description=(
            "Real-time customer churn scoring backed by a calibrated XGBoost pipeline. "
            "Probabilities are calibrated; decisions use a threshold tuned for campaign ROI."
        ),
        version=API_VERSION,
        lifespan=lifespan,
    )
    tracker = LatencyTracker()
    app.state.latency_tracker = tracker
    app.add_middleware(LatencyMiddleware, tracker=tracker)
    app.include_router(router)

    @app.exception_handler(SchemaError)
    async def schema_error_handler(_: Request, exc: SchemaError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app


app = create_app()
