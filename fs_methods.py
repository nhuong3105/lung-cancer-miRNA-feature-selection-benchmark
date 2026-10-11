"""
fs_methods.py — feature-selection algorithms benchmarked in PaperLu3.
Every method returns a full ranking (best first) of column indices so that
top-k subsets of any size can be taken. Methods that natively return a subset
(Boruta, stability selection, LASSO, RFECV) are converted to rankings using
their own internal score (hits, selection frequency, |coef| path entry, RFE rank).
"""
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False


def _rank(score):
    s = np.nan_to_num(np.asarray(score, float), nan=-np.inf)
    return np.argsort(-s, kind="mergesort")


# 1. ANOVA F-test (filter, univariate, linear/Gaussian)
def fs_anova(X, y, rs=0):
    F, _ = f_classif(X, y)
    return _rank(F)


# 2. Mutual information (filter, univariate, non-parametric)
def fs_mutual_info(X, y, rs=0):
    mi = mutual_info_classif(X, y, n_neighbors=5, random_state=rs)
    return _rank(mi)


# 3. mRMR (filter, multivariate; F-statistic relevance / mean |Pearson r| redundancy, quotient form)
def fs_mrmr(X, y, rs=0, k_max=60):
    F, _ = f_classif(X, y)
    F = np.nan_to_num(F)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    n, p = X.shape
    selected = [int(np.argmax(F))]
    red_sum = np.zeros(p)
    remaining = np.ones(p, bool); remaining[selected[0]] = False
    k_max = min(k_max, p)
    while len(selected) < k_max:
        last = selected[-1]
        r = np.abs(Z.T @ Z[:, last]) / n
        red_sum += r
        score = F / (red_sum / len(selected) + 1e-6)
        score[~remaining] = -np.inf
        j = int(np.argmax(score))
        selected.append(j); remaining[j] = False
    rest = [i for i in _rank(F) if remaining[i]]
    return np.array(selected + rest)


# 4. LASSO (embedded; L1 logistic regression; ranking = order of entry along a regularisation path)
def fs_lasso(X, y, rs=0):
    Xs = StandardScaler().fit_transform(X)
    Cs = np.logspace(-3.3, 0, 18)
    entry = np.full(X.shape[1], np.inf)
    last_coef = np.zeros(X.shape[1])
    for step, C in enumerate(Cs):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000,
                               class_weight="balanced", random_state=rs).fit(Xs, y)
        nz = np.abs(m.coef_[0]) > 0
        newly = nz & np.isinf(entry)
        entry[newly] = step
        last_coef = np.abs(m.coef_[0])
    # earlier entry is better; ties broken by final |coef|
    score = -entry + 1e-3 * last_coef / (last_coef.max() + 1e-12)
    score[np.isinf(entry)] = -1e9 + last_coef[np.isinf(entry)]
    return _rank(score)


# 5. Random-forest importance (embedded, impurity decrease)
def fs_rf(X, y, rs=0):
    m = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                               class_weight="balanced", n_jobs=-1, random_state=rs).fit(X, y)
    return _rank(m.feature_importances_)


# 6. XGBoost importance (embedded, total gain)
def fs_xgb(X, y, rs=0):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.5, min_child_weight=2, reg_lambda=1.0,
                      scale_pos_weight=spw, eval_metric="logloss", n_jobs=2,
                      random_state=rs, verbosity=0).fit(X, y)
    booster = m.get_booster()
    gain = booster.get_score(importance_type="total_gain")
    score = np.zeros(X.shape[1])
    for kf, v in gain.items():
        score[int(kf[1:])] = v
    return _rank(score)


# 7. RFE with linear SVM (wrapper); CV-optimal size reported separately
def fs_rfe(X, y, rs=0, step_frac=0.1, min_keep=5):
    Xs = StandardScaler().fit_transform(X)
    p = X.shape[1]
    alive = np.arange(p)
    elim_order = []
    while len(alive) > min_keep:
        m = LinearSVC(C=0.05, class_weight="balanced", max_iter=5000, dual=False).fit(Xs[:, alive], y)
        w = np.abs(m.coef_[0])
        n_drop = max(1, int(len(alive) * step_frac))
        n_drop = min(n_drop, len(alive) - min_keep)
        drop = np.argsort(w)[:n_drop]
        elim_order.extend(alive[drop].tolist())
        alive = np.delete(alive, drop)
    m = LinearSVC(C=0.05, class_weight="balanced", max_iter=5000, dual=False).fit(Xs[:, alive], y)
    last = alive[np.argsort(-np.abs(m.coef_[0]))].tolist()
    return np.array(last + elim_order[::-1])


# 8. Boruta (wrapper, all-relevant; shadow features + binomial test); ranking by hit rate then RF importance
def fs_boruta(X, y, rs=0, n_iter=30, n_estimators=150, alpha=0.05):
    rng = np.random.default_rng(rs)
    n, p = X.shape
    hits = np.zeros(p)
    imp_sum = np.zeros(p)
    for it in range(n_iter):
        sh = rng.permuted(X, axis=0)
        XX = np.hstack([X, sh])
        m = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=7, max_features="sqrt",
                                 class_weight="balanced", n_jobs=-1,
                                 random_state=int(rng.integers(1e9))).fit(XX, y)
        imp = m.feature_importances_
        thr = imp[p:].max()
        hits += imp[:p] > thr
        imp_sum += imp[:p]
    # two-sided binomial decision (as in Kursa & Rudnicki 2010), Bonferroni-corrected
    p_acc = stats.binom.sf(hits - 1, n_iter, 0.5)
    confirmed = p_acc < alpha / p
    score = hits / n_iter + 1e-3 * imp_sum / (imp_sum.max() + 1e-12) + confirmed * 10
    return _rank(score), int(confirmed.sum())


# 9. Stability selection (randomised L1 logistic regression on half-subsamples)
def _c_for_size(Xs, y, q):
    """Smallest C on a log grid whose L1 fit selects >= q features."""
    for C in np.logspace(-3.3, 0.5, 20):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000,
                               class_weight="balanced").fit(Xs, y)
        if (np.abs(m.coef_[0]) > 0).sum() >= q:
            return C
    return C


def fs_stability(X, y, rs=0, n_rep=50, C=None, weakness=0.5, q=25):
    rng = np.random.default_rng(rs)
    Xs = StandardScaler().fit_transform(X)
    n, p = Xs.shape
    if C is None:
        C = _c_for_size(Xs, y, q)
    freq = np.zeros(p)
    coef_sum = np.zeros(p)
    for b in range(n_rep):
        idx = rng.choice(n, n // 2, replace=False)
        scale = rng.uniform(weakness, 1.0, size=p)
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000,
                               class_weight="balanced").fit(Xs[idx] * scale, y[idx])
        nz = np.abs(m.coef_[0]) > 0
        freq += nz
        coef_sum += np.abs(m.coef_[0])
    freq /= n_rep
    return _rank(freq + 1e-3 * coef_sum / (coef_sum.max() + 1e-12)), freq


# 10. ReliefF (filter, multivariate/interaction-aware), on a stratified subsample of instances
def fs_relieff(X, y, rs=0, n_neighbors=10, m_max=800):
    rng = np.random.default_rng(rs)
    Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    n, p = Xn.shape
    idx = rng.choice(n, min(n, m_max), replace=False)
    W = np.zeros(p)
    sq = (Xn ** 2).sum(1)
    for chunk in np.array_split(idx, max(1, len(idx) // 100)):
        D = sq[chunk][:, None] + sq[None, :] - 2 * Xn[chunk] @ Xn.T
        D[np.arange(len(chunk)), chunk] = np.inf
        for r, i in enumerate(chunk):
            same = np.where(y == y[i])[0]; diff = np.where(y != y[i])[0]
            h = same[np.argsort(D[r, same])[:n_neighbors]]
            m = diff[np.argsort(D[r, diff])[:n_neighbors]]
            W += -np.abs(Xn[h] - Xn[i]).mean(0) + np.abs(Xn[m] - Xn[i]).mean(0)
    return _rank(W / len(idx))


METHODS = {
    "ANOVA": fs_anova,
    "MutualInfo": fs_mutual_info,
    "mRMR": fs_mrmr,
    "ReliefF": fs_relieff,
    "LASSO": fs_lasso,
    "StabilitySel": lambda X, y, rs=0: fs_stability(X, y, rs)[0],
    "RF-Imp": fs_rf,
    "XGB-Gain": fs_xgb,
    "SVM-RFE": fs_rfe,
    "Boruta": lambda X, y, rs=0: fs_boruta(X, y, rs)[0],
}
METHOD_FAMILY = {
    "ANOVA": "Filter (univariate)", "MutualInfo": "Filter (univariate)",
    "mRMR": "Filter (multivariate)", "ReliefF": "Filter (multivariate)",
    "LASSO": "Embedded (sparse linear)", "StabilitySel": "Embedded (resampled sparse)",
    "RF-Imp": "Embedded (tree ensemble)", "XGB-Gain": "Embedded (boosting)",
    "SVM-RFE": "Wrapper (linear SVM)", "Boruta": "Wrapper (all-relevant)",
}


def borda(rankings, p, top=None):
    """Borda-count aggregation of several full rankings -> consensus ranking."""
    score = np.zeros(p)
    for r in rankings:
        pos = np.empty(p); pos[r] = np.arange(p)
        if top is not None:
            pos = np.minimum(pos, top)
        score -= pos
    return _rank(score)


def timed(fn, X, y, rs=0):
    t = time.time()
    r = fn(X, y, rs)
    return r, time.time() - t
