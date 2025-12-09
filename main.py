from enum import Enum
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from mcdm_core import merec, topsis, mairca


# ---------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------

class CriterionType(str, Enum):
    max = "max"
    min = "min"
    benefit = "benefit"
    cost = "cost"
    higher = "higher"
    lower = "lower"


class MCDMBaseRequest(BaseModel):
    decision_matrix: List[List[float]] = Field(
        ..., description="Matrix of alternatives x criteria (rows = alternatives)."
    )
    criteria_types: List[CriterionType] = Field(
        ..., description="Type of each criterion: 'max', 'min', 'benefit', 'cost', etc."
    )

    @validator("decision_matrix")
    def matrix_not_empty(cls, v):
        if not v:
            raise ValueError("decision_matrix must have at least one alternative")
        row_len = len(v[0])
        if row_len == 0:
            raise ValueError("decision_matrix must have at least one criterion")
        for i, row in enumerate(v):
            if len(row) != row_len:
                raise ValueError(
                    f"All rows in decision_matrix must have same length; "
                    f"row 0 length = {row_len}, row {i} length = {len(row)}"
                )
        return v

    @validator("criteria_types")
    def criteria_len_match(cls, v, values):
        dm = values.get("decision_matrix")
        if dm is not None:
            n = len(dm[0])
            if len(v) != n:
                raise ValueError(
                    f"criteria_types length ({len(v)}) must match number of "
                    f"criteria in decision_matrix ({n})"
                )
        return v


class MerecRequest(MCDMBaseRequest):
    """Request body for /mcdm/merec"""


class MerecResponse(BaseModel):
    weights: List[float]


class TopsisRequest(MCDMBaseRequest):
    """Request body for /mcdm/topsis"""

    weights: Optional[List[float]] = Field(
        None,
        description="Optional explicit weights. If omitted and use_merec_weights=True, "
                    "weights are computed via MEREC.",
    )
    use_merec_weights: bool = Field(
        True,
        description="If true and weights is None, compute weights using MEREC.",
    )


class TopsisResponse(BaseModel):
    weights_used: List[float]
    scores: List[float]
    ranking: List[int]  # indices of alternatives from best to worst


class MaircaRequest(MCDMBaseRequest):
    """Request body for /mcdm/mairca"""

    weights: Optional[List[float]] = Field(
        None,
        description="Optional explicit weights. If omitted and use_merec_weights=True, "
                    "weights are computed via MEREC.",
    )
    use_merec_weights: bool = Field(
        True,
        description="If true and weights is None, compute weights using MEREC.",
    )


class MaircaResponse(BaseModel):
    weights_used: List[float]
    Q: List[float]       # Total gap for each alternative (lower = better)
    ranking: List[int]   # indices of alternatives from best to worst


class RunMethod(str, Enum):
    topsis = "topsis"
    mairca = "mairca"
    all = "all"  # run both


class MCDMRunRequest(MCDMBaseRequest):
    """
    Unified endpoint body for /mcdm/run
    """

    method: RunMethod = Field(
        RunMethod.topsis,
        description="Which method to run: 'topsis', 'mairca', or 'all'.",
    )
    weights: Optional[List[float]] = Field(
        None,
        description="Optional explicit weights. If omitted and use_merec_weights=True, "
                    "weights are computed via MEREC.",
    )
    use_merec_weights: bool = Field(
        True,
        description="If true and weights is None, compute weights using MEREC.",
    )


class MCDMRunResponse(BaseModel):
    """
    Flexible response for /mcdm/run.

    Fields that are not applicable to the chosen method are omitted.
    """

    # Common
    weights_used: List[float]

    # TOPSIS outputs
    topsis_scores: Optional[List[float]] = None
    topsis_ranking: Optional[List[int]] = None

    # MAIRCA outputs
    mairca_Q: Optional[List[float]] = None
    mairca_ranking: Optional[List[int]] = None


# ---------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------

app = FastAPI(
    title="MCDM API (MEREC, TOPSIS, MAIRCA)",
    version="1.0.0",
    description="Backend maths API for your MCDM web app.",
)

# CORS – allow everything for dev; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, put your frontend URL(s) here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def _resolve_weights(
    matrix: List[List[float]],
    criteria_types: List[CriterionType],
    weights: Optional[List[float]],
    use_merec_weights: bool,
) -> List[float]:
    """
    Decide which weights to use:
    - if weights is provided and use_merec_weights is False -> use as-is
    - else -> compute via MEREC
    """
    n = len(matrix[0])

    if weights is not None and not use_merec_weights:
        if len(weights) != n:
            raise HTTPException(
                status_code=400,
                detail=f"weights length ({len(weights)}) must match number of "
                       f"criteria in decision_matrix ({n})",
            )
        if sum(weights) == 0:
            raise HTTPException(
                status_code=400,
                detail="Sum of weights cannot be zero.",
            )
        return weights

    # Compute weights via MEREC
    try:
        w = merec(matrix, [ct.value for ct in criteria_types])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"MEREC error: {str(e)}")
    return w.tolist()


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "MCDM API is alive. Use /docs for Swagger UI.",
        "tease_the_life": True,
    }


@app.post("/mcdm/merec", response_model=MerecResponse)
def compute_merec(req: MerecRequest):
    try:
        weights = merec(req.decision_matrix, [ct.value for ct in req.criteria_types])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return MerecResponse(weights=weights.tolist())


@app.post("/mcdm/topsis", response_model=TopsisResponse)
def compute_topsis(req: TopsisRequest):
    # Resolve weights
    weights_used = _resolve_weights(
        req.decision_matrix,
        req.criteria_types,
        req.weights,
        req.use_merec_weights,
    )

    try:
        scores, ranking = topsis(
            req.decision_matrix,
            weights_used,
            [ct.value for ct in req.criteria_types],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return TopsisResponse(
        weights_used=weights_used,
        scores=scores.tolist(),
        ranking=ranking.tolist(),
    )


@app.post("/mcdm/mairca", response_model=MaircaResponse)
def compute_mairca(req: MaircaRequest):
    # Resolve weights
    weights_used = _resolve_weights(
        req.decision_matrix,
        req.criteria_types,
        req.weights,
        req.use_merec_weights,
    )

    try:
        Q, ranking = mairca(
            req.decision_matrix,
            weights_used,
            [ct.value for ct in req.criteria_types],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return MaircaResponse(
        weights_used=weights_used,
        Q=Q.tolist(),
        ranking=ranking.tolist(),
    )


@app.post("/mcdm/run", response_model=MCDMRunResponse)
def run_mcdm(req: MCDMRunRequest):
    """
    Unified endpoint for the frontend.

    - method = 'topsis'  -> returns TOPSIS scores & ranking
    - method = 'mairca'  -> returns MAIRCA Q & ranking
    - method = 'all'     -> returns both
    """
    # Resolve weights once
    weights_used = _resolve_weights(
        req.decision_matrix,
        req.criteria_types,
        req.weights,
        req.use_merec_weights,
    )

    topsis_scores = topsis_ranking = None
    mairca_Q = mairca_ranking = None

    try:
        if req.method in (RunMethod.topsis, RunMethod.all):
            s, r = topsis(
                req.decision_matrix,
                weights_used,
                [ct.value for ct in req.criteria_types],
            )
            topsis_scores = s.tolist()
            topsis_ranking = r.tolist()

        if req.method in (RunMethod.mairca, RunMethod.all):
            Q, r = mairca(
                req.decision_matrix,
                weights_used,
                [ct.value for ct in req.criteria_types],
            )
            mairca_Q = Q.tolist()
            mairca_ranking = r.tolist()

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return MCDMRunResponse(
        weights_used=weights_used,
        topsis_scores=topsis_scores,
        topsis_ranking=topsis_ranking,
        mairca_Q=mairca_Q,
        mairca_ranking=mairca_ranking,
    )
