from typing import List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mcdm_core import merec, topsis, mairca


app = FastAPI(
    title="MCDM Backend",
    description="MEREC + TOPSIS + MAIRCA backend",
    version="1.0.0",
)

# CORS so frontend can call backend from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # if you want, later restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MCDMRequest(BaseModel):
    decision_matrix: List[List[float]]
    criteria_types: List[str]
    method: Literal["topsis", "mairca", "all"] = "all"
    use_merec_weights: bool = True
    weights: Optional[List[float]] = None  # only used if use_merec_weights = False


def sanitize_for_json(obj):
    """Convert numpy types + NaN/Inf into JSON-safe Python values."""
    if isinstance(obj, np.ndarray):
        obj = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            return 0.0
        return v
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


@app.get("/")
def root():
    return {"message": "MCDM backend running. Use /docs for API docs."}


@app.post("/mcdm/run")
def run_mcdm(req: MCDMRequest):
    # Convert decision matrix
    X = np.asarray(req.decision_matrix, dtype=float)
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="decision_matrix must be 2D (m x n)")

    m, n = X.shape
    if len(req.criteria_types) != n:
        raise HTTPException(
            status_code=400,
            detail="criteria_types length must match number of columns in decision_matrix",
        )

    # Decide weights
    if req.use_merec_weights:
        w = merec(X, req.criteria_types)
    else:
        if req.weights is None:
            raise HTTPException(
                status_code=400,
                detail="weights must be provided if use_merec_weights is False",
            )
        w = np.asarray(req.weights, dtype=float)
        if w.shape != (n,):
            raise HTTPException(
                status_code=400,
                detail="weights length must match number of criteria",
            )
        w_sum = w.sum()
        if w_sum == 0:
            raise HTTPException(
                status_code=400,
                detail="Sum of weights cannot be zero",
            )
        w = w / w_sum

    response = {
        "weights_used": w,
    }

    # TOPSIS
    if req.method in ("topsis", "all"):
        scores, ranking = topsis(X, w, req.criteria_types)
        response["topsis_scores"] = scores
        response["topsis_ranking"] = ranking

    # MAIRCA
    if req.method in ("mairca", "all"):
        Q, ranking_m = mairca(X, w, req.criteria_types)
        response["mairca_Q"] = Q
        response["mairca_ranking"] = ranking_m

    # Make sure everything is JSON-safe
    return sanitize_for_json(response)
