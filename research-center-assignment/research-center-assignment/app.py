"""
Research Center Quality Classifier — FastAPI Endpoint
======================================================
Loads the trained K-Means model artefacts and exposes a /predict endpoint
that classifies a research center into Premium, Standard, or Basic tier.

Run:
    uvicorn app:app --reload
    # or in Docker: uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Load model artefacts at startup ─────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "cluster_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Run EDA_and_Model.ipynb first to generate cluster_model.pkl."
    )

kmeans, scaler, selected_features, cluster_to_tier = joblib.load(MODEL_PATH)
logger.info("Model artefacts loaded from %s", MODEL_PATH)
logger.info("Selected features: %s", selected_features)
logger.info("Tier mapping: %s", cluster_to_tier)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Research Center Quality Classifier",
    description=(
        "Classifies UK research centers into **Premium**, **Standard**, or **Basic** "
        "quality tiers using a pre-trained K-Means clustering model."
    ),
    version="1.0.0",
    contact={"name": "Srinidhi Radhakrishna", "email": "srinidhiriyengar22@gmail.com"},
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class ResearchCenterInput(BaseModel):
    """Input features for quality tier prediction."""

    internalFacilitiesCount: float = Field(
        ..., ge=0, description="Number of internal facilities (labs, workstations, etc.)"
    )
    hospitals_10km: float = Field(
        ..., ge=0, description="Number of hospitals within 10 km"
    )
    pharmacies_10km: float = Field(
        ..., ge=0, description="Number of pharmacies within 10 km"
    )
    facilityDiversity_10km: float = Field(
        ..., ge=0.0, le=1.0, description="Facility diversity index (0–1)"
    )
    facilityDensity_10km: float = Field(
        ..., ge=0.0, description="Approximate facility density per area unit"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "internalFacilitiesCount": 9,
                "hospitals_10km": 3,
                "pharmacies_10km": 2,
                "facilityDiversity_10km": 0.82,
                "facilityDensity_10km": 0.45,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction result returned by /predict."""

    predictedCluster: int
    predictedCategory: str
    confidence: str  # qualitative note — K-Means is hard-assignment


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root() -> dict:
    """Health check / root endpoint."""
    return {
        "status": "ok",
        "model": "K-Means Research Center Quality Classifier",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Liveness probe for container orchestration."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_quality(data: ResearchCenterInput) -> JSONResponse:
    """
    Predict the quality tier for a research center.

    Returns one of:
    - **Premium** — highest facility count and healthcare access
    - **Standard** — moderate infrastructure
    - **Basic** — limited internal facilities and nearby healthcare
    """
    try:
        # Build DataFrame in the exact feature order the scaler expects
        input_df = pd.DataFrame([data.model_dump()])[selected_features]

        # Standardise
        X_scaled = scaler.transform(input_df)

        # Predict cluster label
        cluster_label: int = int(kmeans.predict(X_scaled)[0])

        # Map to human-readable tier
        tier: str = cluster_to_tier.get(cluster_label, "Unknown")

        logger.info(
            "Prediction: cluster=%d → tier=%s | input=%s",
            cluster_label, tier, data.model_dump(),
        )

        return JSONResponse(
            content={
                "predictedCluster": cluster_label,
                "predictedCategory": tier,
                "confidence": "hard-assignment (K-Means)",
            }
        )

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
