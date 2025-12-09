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

# CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MCDMRequest(BaseModel):
    decision_matrix: List[List[float]]
    criteria_types: List[str]
    method: Literal["topsis", "mairca", "all"] = "all"
    use_merec_weights: bool = True
    weights: Optional[List[float]] = None  # if use_merec_weights = False


@app.get("/")
def root():
    return {"message": "MCDM backend running. Go to /docs to test the API."}


@app.post("/mcdm/run")
def run_mcdm(req: MCDMRequest):
    # ----- 1) Basic validation -----
    X = np.asarray(req.decision_matrix, dtype=float)
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="decision_matrix must be 2D (m x n)")

    m, n = X.shape
    if len(req.criteria_types) != n:
        raise HTTPException(
            status_code=400,
            detail="criteria_types length must match number of columns in decision_matrix",
        )

    # ----- 2) Weights (MEREC or custom) -----
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

    # JSON-safe weights
    w_safe = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    response = {
        "weights_used": w_safe.tolist(),
    }

    # ----- 3) TOPSIS -----
    if req.method in ("topsis", "all"):
        scores, ranking = topsis(X, w, req.criteria_types)

        scores_safe = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        ranking_safe = np.asarray(ranking, dtype=int)

        response["topsis_scores"] = scores_safe.tolist()
        response["topsis_ranking"] = ranking_safe.tolist()

    # ----- 4) MAIRCA -----
    if req.method in ("mairca", "all"):
        Q, ranking_m = mairca(X, w, req.criteria_types)

        Q_safe = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        ranking_m_safe = np.asarray(ranking_m, dtype=int)

        response["mairca_Q"] = Q_safe.tolist()
        response["mairca_ranking"] = ranking_m_safe.tolist()

    # ----- 5) Return pure Python types (no numpy, no NaN/Inf) -----
    return response
