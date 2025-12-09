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
# MEREC – criteria weights (Section 3 in the paper) 
# ---------------------------------------------------------------------
def merec(decision_matrix, criteria_types):
    """
    Compute objective criteria weights using the MEREC method.

    Parameters
    ----------
    decision_matrix : array-like, shape (m, n)
        Matrix of alternatives (rows) vs criteria (columns), raw values x_ij.
    criteria_types : list/array-like of length n
        Type of each criterion ('max'/'min', 'benefit'/'cost', 1/-1, etc.).

    Returns
    -------
    weights : np.ndarray, shape (n,)
        Normalized weights that sum to 1.
    """
    X = np.asarray(decision_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be 2D (m x n)")
    m, n = X.shape

    c_types = _parse_criteria_types(criteria_types)
    if len(c_types) != n:
        raise ValueError("criteria_types length must match number of columns in decision_matrix")

    # Step 2: normalized matrix h_ij using Eqs. (25) and (26) 
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
            # bigger is better: h_ij = min(x_ij) / x_ij
            hij[:, j] = col_min / col
        else:  # cost (smaller is better)
            # smaller is better: h_ij = x_ij / max(x_ij)
            hij[:, j] = col / col_max

    # Guard against log(0)
    eps = 1e-15
    hij_safe = np.clip(hij, eps, None)
    log_hij = np.log(hij_safe)

    # Step 3: performance of alternatives S_i – Eq. (27) 
    # S_i = ln( 1 + (1/n) * Σ_j |ln(h_ij)| )
    base = np.abs(log_hij)
    Si = np.log(1.0 + (base.sum(axis=1) / n))

    # Step 4: performance with criterion j removed, S'_ij – Eq. (28) 
    S_prime = np.zeros((m, n), dtype=float)
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        if mask.sum() == 0:
            # Only one criterion; if removed, no info
            S_prime[:, j] = 0.0
            continue
        sub = base[:, mask]
        S_prime[:, j] = np.log(1.0 + (sub.sum(axis=1) / mask.sum()))

    # Step 5: removal effect of j-th criterion E_j – Eq. (29) 
    E = np.abs(S_prime - Si[:, None]).sum(axis=0)

    # Step 6: normalize to get weights – Eq. (30) 
    E_sum = E.sum()
    if E_sum == 0:
        # No discriminatory power; assign equal weights
        return np.full(n, 1.0 / n)
    weights = E / E_sum
    return weights


# ---------------------------------------------------------------------
# TOPSIS – ranking (Section 2.2 in the paper) 
# ---------------------------------------------------------------------
def topsis(decision_matrix, weights, criteria_types):
    """
    Run TOPSIS and return preference scores and ranking.

    Parameters
    ----------
    decision_matrix : array-like, shape (m, n)
        Alternatives x criteria matrix x_ij.
    weights : array-like, shape (n,)
        Criteria weights w_j (will be normalized internally).
    criteria_types : list/array-like of length n
        Types for each criterion ('max'/'min', 'benefit'/'cost', etc.).

    Returns
    -------
    scores : np.ndarray, shape (m,)
        Closeness coefficient R_i for each alternative (higher is better). (Eq. 18)
    ranking : np.ndarray, shape (m,)
        Indices of alternatives sorted from best (0) to worst (m-1).
    """
    X = np.asarray(decision_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be 2D (m x n)")
    m, n = X.shape

    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError("weights must be 1D array of length n")
    w = w / w.sum()  # normalize

    c_types = _parse_criteria_types(criteria_types)
    if len(c_types) != n:
        raise ValueError("criteria_types length must match number of columns in decision_matrix")

    # Step 2: vector normalization – Eq. (12) 
    denom = np.linalg.norm(X, axis=0)
    denom[denom == 0] = 1.0  # avoid division by zero
    R = X / denom

    # Step 3: weighted normalized matrix – Eq. (13) 
    V = R * w  # broadcasting

    # Step 4: ideal best & worst – Eqs. (14), (15) 
    ideal_best = np.zeros(n, dtype=float)
    ideal_worst = np.zeros(n, dtype=float)
    for j in range(n):
        col = V[:, j]
        if c_types[j] == "benefit":
            ideal_best[j] = col.max()
            ideal_worst[j] = col.min()
        else:  # cost
            ideal_best[j] = col.min()
            ideal_worst[j] = col.max()

    # Step 5: distances to ideal best & worst – Eqs. (16), (17) 
    D_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # Step 6: closeness coefficient – Eq. (18) 
    denom_dm = D_plus + D_minus
    denom_dm[denom_dm == 0] = 1.0
    scores = D_minus / denom_dm  # higher is better

    ranking = np.argsort(scores)[::-1]
    return scores, ranking


# ---------------------------------------------------------------------
# MAIRCA – ranking (Section 2.3 in the paper) 
# ---------------------------------------------------------------------
def mairca(decision_matrix, weights, criteria_types):
    """
    Run MAIRCA and return Q values and ranking.

    Parameters
    ----------
    decision_matrix : array-like, shape (m, n)
        Alternatives x criteria matrix x_ij.
    weights : array-like, shape (n,)
        Criteria weights w_j (will be normalized internally).
    criteria_types : list/array-like of length n
        Types for each criterion ('max'/'min', 'benefit'/'cost', etc.).

    Returns
    -------
    Q : np.ndarray, shape (m,)
        Total gap Q_i for each alternative (lower is better). (Eq. 24)
    ranking : np.ndarray, shape (m,)
        Indices of alternatives sorted from best (0) to worst (m-1).
    """
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
        raise ValueError("criteria_types length must match number of columns in decision_matrix")

    # Step 2: equal preference for alternatives – Eq. (19) 
    PA = 1.0 / m

    # Step 3: theoretical rating matrix t_pij = PA * w_j – Eq. (20) 
    # tp has shape (n,), then broadcast to (m, n)
    tp = PA * w
    tp_mat = np.tile(tp, (m, 1))

    # Step 4: real rating matrix t_rij – Eqs. (21), (22) 
    tr = np.zeros_like(X, dtype=float)
    for j in range(n):
        col = X[:, j]
        col_min = col.min()
        col_max = col.max()
        if col_max == col_min:
            # No discrimination: factor = 1 (all identical)
            factor = np.ones(m, dtype=float)
        else:
            if c_types[j] == "benefit":
                # (x_ij - x_j^-)/(x_j^+ - x_j^-)
                factor = (col - col_min) / (col_max - col_min)
            else:
                # cost: (x_ij - x_j^+)/(x_j^- - x_j^+)
                factor = (col - col_max) / (col_min - col_max)

        tr[:, j] = tp[j] * factor

    # Step 5: gap matrix g_ij = t_pij - t_rij – Eq. (23) 
    g = tp_mat - tr

    # Step 6: total gap per alternative Q_i – Eq. (24) 
    Q = g.sum(axis=1)

    # Lower Q is better
    ranking = np.argsort(Q)
    return Q, ranking
