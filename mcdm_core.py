import numpy as np


def _parse_criteria_types(criteria_types):
    """
    Convert user-friendly criteria types into internal 'benefit'/'cost' labels.

    Accepts values like:
    - 'max', 'benefit', 'higher', 1, True  -> 'benefit' (larger is better)
    - 'min', 'cost', 'lower', -1, False   -> 'cost' (smaller is better)
    """
    parsed = []
    for ct in criteria_types:
        if isinstance(ct, str):
            s = ct.strip().lower()
            if s in ("max", "benefit", "higher", "larger", "bigger", "gain"):
                parsed.append("benefit")
            elif s in ("min", "cost", "lower", "smaller", "loss"):
                parsed.append("cost")
            else:
                raise ValueError(f"Unknown criteria type string: {ct!r}")
        elif isinstance(ct, (int, float, bool)):
            if ct in (1, True):
                parsed.append("benefit")
            elif ct in (-1, 0, False):
                parsed.append("cost")
            else:
                raise ValueError(f"Unknown criteria type numeric: {ct!r}")
        else:
            raise TypeError(f"Unsupported criteria type: {type(ct)}")

    return np.array(parsed, dtype=object)


# ---------------------------------------------------------------------
# MEREC – criteria weights
# ---------------------------------------------------------------------
def merec(decision_matrix, criteria_types):
    X = np.asarray(decision_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be 2D (m x n)")
    m, n = X.shape

    c_types = _parse_criteria_types(criteria_types)
    if len(c_types) != n:
        raise ValueError(
            "criteria_types length must match number of columns in decision_matrix"
        )

    eps = 1e-15  # small value to avoid division by zero

    # Step 2: normalized matrix h_ij
    hij = np.zeros_like(X, dtype=float)
    for j in range(n):
        col = X[:, j]
        col_min = col.min()
        col_max = col.max()

        if col_max == col_min:
            # All values equal => no information; set hij = 1 for all alts
            hij[:, j] = 1.0
            continue

        if c_types[j] == "benefit":
            # bigger is better: h_ij = min(x_ij) / x_ij, but avoid /0
            denom = np.where(col == 0, eps, col)
            hij[:, j] = col_min / denom
        else:  # cost (smaller is better)
            # smaller is better: h_ij = x_ij / max(x_ij)
            denom = col_max if col_max != 0 else eps
            hij[:, j] = col / denom

    # Clean weird values
    hij = np.nan_to_num(hij, nan=eps, posinf=1.0, neginf=eps)
    hij_safe = np.clip(hij, eps, None)
    log_hij = np.log(hij_safe)

    # Step 3: S_i
    base = np.abs(log_hij)
    Si = np.log(1.0 + (base.sum(axis=1) / n))

    # Step 4: S'_ij
    S_prime = np.zeros((m, n), dtype=float)
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        if mask.sum() == 0:
            S_prime[:, j] = 0.0
            continue

        sub = base[:, mask]
        S_prime[:, j] = np.log(1.0 + (sub.sum(axis=1) / mask.sum()))

    # Step 5: E_j
    E = np.abs(S_prime - Si[:, None]).sum(axis=0)
    E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 6: normalize weights
    E_sum = E.sum()
    if E_sum == 0:
        return np.full(n, 1.0 / n)

    weights = E / E_sum
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weights


# ---------------------------------------------------------------------
# TOPSIS – ranking
# ---------------------------------------------------------------------
def topsis(decision_matrix, weights, criteria_types):
    X = np.asarray(decision_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be 2D (m x n)")
    m, n = X.shape

    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError("weights must be 1D array of length n")
    w = w / w.sum()

    c_types = _parse_criteria_types(criteria_types)
    if len(c_types) != n:
        raise ValueError(
            "criteria_types length must match number of columns in decision_matrix"
        )

    # Step 2: vector normalization
    denom = np.linalg.norm(X, axis=0)
    denom[denom == 0] = 1.0
    R = X / denom

    # Step 3: weighted matrix
    V = R * w

    # Step 4: ideal best & worst
    ideal_best = np.zeros(n, dtype=float)
    ideal_worst = np.zeros(n, dtype=float)
    for j in range(n):
        col = V[:, j]
        if c_types[j] == "benefit":
            ideal_best[j] = col.max()
            ideal_worst[j] = col.min()
        else:
            ideal_best[j] = col.min()
            ideal_worst[j] = col.max()

    # Step 5: distances
    D_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # Step 6: closeness coefficient
    denom_dm = D_plus + D_minus
    denom_dm[denom_dm == 0] = 1.0
    scores = D_minus / denom_dm
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    ranking = np.argsort(scores)[::-1]
    return scores, ranking


# ---------------------------------------------------------------------
# MAIRCA – ranking
# ---------------------------------------------------------------------
def mairca(decision_matrix, weights, criteria_types):
    X = np.asarray(decision_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be 2D (m x n)")
    m, n = X.shape

    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError("weights must be 1D array of length n")
    w = w / w.sum()

    c_types = _parse_criteria_types(criteria_types)
    if len(c_types) != n:
        raise ValueError(
            "criteria_types length must match number of columns in decision_matrix"
        )

    # Step 2: equal preference for alternatives
    PA = 1.0 / m

    # Step 3: theoretical rating matrix t_pij = PA * w_j
    tp = PA * w
    tp_mat = np.tile(tp, (m, 1))

    # Step 4: real rating matrix t_rij
    tr = np.zeros_like(X, dtype=float)
    for j in range(n):
        col = X[:, j]
        col_min = col.min()
        col_max = col.max()

        if col_max == col_min:
            factor = np.ones(m, dtype=float)
        else:
            if c_types[j] == "benefit":
                factor = (col - col_min) / (col_max - col_min)
            else:
                factor = (col - col_max) / (col_min - col_max)

        tr[:, j] = tp[j] * factor

    # Step 5: gap matrix
    g = tp_mat - tr

    # Step 6: total gap per alternative
    Q = g.sum(axis=1)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)

    ranking = np.argsort(Q)
    return Q, ranking
