"""Shared fixtures.

Tests are hermetic: a small dataset is generated and a fast model is trained into a
temporary directory once per session, so the suite never depends on (or overwrites)
local artifacts.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.schemas import EXAMPLE_CUSTOMER
from src.config import PROJECT_ROOT, ArtifactPaths
from src.data import DatasetSplits, load_splits
from src.train import train

TEST_DATASET_SIZE = 6000
TEST_SEED = 7


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_dataset", PROJECT_ROOT / "data" / "generate_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def generator():
    return _load_generator()


@pytest.fixture(scope="session")
def dataset_path(generator, tmp_path_factory) -> Path:
    df = generator.generate_customers(n_samples=TEST_DATASET_SIZE, seed=TEST_SEED)
    path = tmp_path_factory.mktemp("data") / "churn_dataset.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def artifact_dir(dataset_path, tmp_path_factory) -> Path:
    logging.getLogger("churn").setLevel(logging.WARNING)
    out = tmp_path_factory.mktemp("artifacts")
    train(data_path=dataset_path, artifact_dir=out, n_iter=2, n_boot=50, seed=TEST_SEED)
    return out


@pytest.fixture(scope="session")
def artifacts(artifact_dir) -> ArtifactPaths:
    return ArtifactPaths(artifact_dir)


@pytest.fixture(scope="session")
def pipeline(artifacts):
    return joblib.load(artifacts.model)


@pytest.fixture(scope="session")
def splits(dataset_path) -> DatasetSplits:
    return load_splits(dataset_path, seed=TEST_SEED)


@pytest.fixture
def customer() -> dict:
    """A valid API payload (fresh copy per test)."""
    return dict(EXAMPLE_CUSTOMER)


@pytest.fixture
def raw_frame(customer) -> pd.DataFrame:
    record = {k: v for k, v in customer.items() if k != "customer_id"}
    return pd.DataFrame([record])


@pytest.fixture(scope="session")
def client(artifact_dir):
    from app.main import create_app

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CHURN_ARTIFACT_DIR", str(artifact_dir))
        with TestClient(create_app()) as test_client:
            yield test_client


@pytest.fixture
def unavailable_client(tmp_path):
    from app.main import create_app

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CHURN_ARTIFACT_DIR", str(tmp_path / "does-not-exist"))
        with TestClient(create_app()) as test_client:
            yield test_client
