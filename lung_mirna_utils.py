"""
lung_mirna_utils.py
Shared utilities for the PaperLu1 / PaperLu2 / PaperLu3 notebooks.

- Loads the seven STRICT lung-cancer-vs-healthy blood cohorts and the TCGA tissue cohort
- Harmonises legacy miRBase v13 names (GPL9040 / febit cohorts) to MIMAT accessions
  using a miRBase all-version alias table (derived from miRBaseConverter, Xu et al. 2018)
- Provides cohort metadata (specimen, platform, chemistry) used throughout the papers
- Provides metrics (AUC with DeLong CI, calibration, net benefit) and ANOVA helpers
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------------
# Cohort metadata (verified against GEO / ArrayExpress records and source publications)
# ----------------------------------------------------------------------------------
COHORTS = {
    "E-MTAB-8026": dict(file="E_MTAB_8026_STRICT_lung_vs_healthy_preprocessed.csv",
                        specimen="Whole blood (PAXgene)", platform="Agilent SurePrint miRNA microarray",
                        chemistry="Agilent", country="Germany (multicentre)", family="WB-Agilent"),
    "GSE137140": dict(file="GSE137140_STRICT_lung_vs_healthy_preprocessed.csv",
                      specimen="Serum", platform="Toray 3D-Gene Human miRNA V21 (GPL21263)",
                      chemistry="3D-Gene", country="Japan (NCC Biobank)", family="Serum-3DGene"),
    "GSE106817": dict(file="GSE106817_STRICT_lung_vs_healthy_preprocessed.csv",
                      specimen="Serum", platform="Toray 3D-Gene Human miRNA V21 (GPL21263)",
                      chemistry="3D-Gene", country="Japan (NCC Biobank)", family="Serum-3DGene"),
    "GSE112264": dict(file="GSE112264_STRICT_lung_vs_healthy_preprocessed.csv",
                      specimen="Serum", platform="Toray 3D-Gene Human miRNA V21 (GPL21263)",
                      chemistry="3D-Gene", country="Japan (NCC Biobank)", family="Serum-3DGene"),
    "GSE113486": dict(file="GSE113486_STRICT_lung_vs_healthy_preprocessed.csv",
                      specimen="Serum", platform="Toray 3D-Gene Human miRNA V21 (GPL21263)",
                      chemistry="3D-Gene", country="Japan (NCC Biobank)", family="Serum-3DGene"),
    "GSE31568": dict(file="GSE31568_STRICT_lung_vs_healthy_preprocessed.csv",
                     specimen="Whole blood (PAXgene)", platform="febit Geniom Biochip miRBase v13 (GPL9040)",
                     chemistry="Geniom", country="Germany", family="WB-Geniom"),
    "GSE61741": dict(file="GSE61741_STRICT_lung_vs_healthy_preprocessed.csv",
                     specimen="Whole blood (PAXgene)", platform="febit Geniom Biochip miRBase v13 (GPL9040)",
                     chemistry="Geniom", country="Germany", family="WB-Geniom"),
}
TCGA_FILE = "TCGA_Lung_miRNA_preprocessed_with_labels.csv"
COHORT_ORDER = list(COHORTS.keys())
SHORT = {"E-MTAB-8026": "EMTAB", "GSE137140": "137140", "GSE106817": "106817",
         "GSE112264": "112264", "GSE113486": "113486", "GSE31568": "31568", "GSE61741": "61741"}
META_COLS = ["sample_id", "dataset", "raw_group", "class_type", "class_label"]


def find_data_dir(start=None):
    """Locate Lung_miRNA_Preprocessed_Datasets by walking up from the notebook folder."""
    p = os.path.abspath(start or os.getcwd())
    for _ in range(5):
        cand = os.path.join(p, "Lung_miRNA_Preprocessed_Datasets")
        if os.path.isdir(cand):
            return cand
        p = os.path.dirname(p)
    raise FileNotFoundError("Lung_miRNA_Preprocessed_Datasets not found")


def load_alias_table(path=None):
    path = path or os.path.join(HERE, "mirbase_hsa_alias_all_versions.csv")
    return pd.read_csv(path)


def legacy_to_mimat(names, alias):
    """Map miRBase v13 symbols (e.g. 'hsa-miR-21', 'hsa-miR-21*') to MIMAT accessions.
    Symbols that map to >1 accession across miRBase history are resolved using the
    star/non-star partner; unresolved ambiguities are dropped (reported)."""
    g = alias.groupby("sym")["acc"].apply(lambda s: sorted(set(s)))
    mapping, report = {}, []
    for n in names:
        accs = g.get(n)
        if accs is None:
            report.append((n, "unmapped", ""))
            continue
        if len(accs) == 1:
            mapping[n] = accs[0]
            continue
        # try resolving with star partner
        partner = n[:-1] if n.endswith("*") else n + "*"
        p_accs = g.get(partner)
        resolved = None
        if p_accs is not None:
            rest = [a for a in accs if a not in p_accs]
            if len(rest) == 1:
                resolved = rest[0]
        if resolved:
            mapping[n] = resolved
            report.append((n, "resolved_by_partner", resolved))
        else:
            report.append((n, "ambiguous_dropped", ";".join(accs)))
    # drop many-to-one collisions
    inv = pd.Series(mapping).reset_index()
    inv.columns = ["sym", "acc"]
    dup = inv[inv.acc.duplicated(keep=False)]
    for _, r in dup.iterrows():
        report.append((r.sym, "collision_dropped", r.acc))
        mapping.pop(r.sym, None)
    return mapping, pd.DataFrame(report, columns=["legacy_name", "status", "accession"])


def load_cohort(name, data_dir=None, alias=None, return_report=False):
    data_dir = data_dir or find_data_dir()
    info = COHORTS[name]
    df = pd.read_csv(os.path.join(data_dir, info["file"]), low_memory=False)
    feats = [c for c in df.columns if c not in META_COLS]
    rep = None
    if not feats[0].startswith("MIMAT"):
        alias = alias if alias is not None else load_alias_table()
        mapping, rep = legacy_to_mimat(feats, alias)
        X = df[list(mapping.keys())].rename(columns=mapping)
    else:
        X = df[feats]
    X = X.astype(float)
    y = df["class_type"].astype(int).values
    X.index = df["sample_id"].astype(str).values
    if return_report:
        return X, y, rep
    return X, y


def load_all(data_dir=None):
    alias = load_alias_table()
    data = {}
    for c in COHORT_ORDER:
        X, y = load_cohort(c, data_dir, alias)
        data[c] = (X, y)
    return data


def load_tcga(data_dir=None):
    data_dir = data_dir or find_data_dir()
    t = pd.read_csv(os.path.join(data_dir, TCGA_FILE))
    X = t[[c for c in t.columns if c.startswith("MIMAT")]].astype(float)
    return X, t["class_type"].astype(int).values, t["cancer_type"].values


def restandardize(X, ref_mask=None):
    """Re-z-score features. Because the deposited matrices are already per-cohort z-scores,
    an affine re-standardisation on a subset is mathematically identical to z-scoring the
    original log-scale data on that subset (e.g. control-referenced standardisation)."""
    ref = X if ref_mask is None else X[ref_mask]
    mu = ref.mean(0)
    sd = ref.std(0, ddof=1).replace(0, 1.0)
    return (X - mu) / sd


# ----------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------
def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def anova_table(X, y):
    """One-way ANOVA (2 groups => F = t^2) with effect sizes and single-marker AUC."""
    Xv = X.values
    g1, g0 = Xv[y == 1], Xv[y == 0]
    F, p = stats.f_oneway(g1, g0, axis=0)
    n1, n0 = len(g1), len(g0)
    m1, m0 = g1.mean(0), g0.mean(0)
    s_p = np.sqrt(((n1 - 1) * g1.var(0, ddof=1) + (n0 - 1) * g0.var(0, ddof=1)) / (n1 + n0 - 2))
    d = (m1 - m0) / np.where(s_p == 0, np.nan, s_p)
    J = 1 - 3 / (4 * (n1 + n0) - 9)  # Hedges correction
    # single-marker AUC via Mann-Whitney U
    r = stats.rankdata(Xv, axis=0)
    auc = (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    ss_b = n1 * (m1 - Xv.mean(0)) ** 2 + n0 * (m0 - Xv.mean(0)) ** 2
    ss_t = ((Xv - Xv.mean(0)) ** 2).sum(0)
    eta2 = ss_b / np.where(ss_t == 0, np.nan, ss_t)
    with np.errstate(divide="ignore"):
        nlp = -stats.f.logsf(F, 1, n1 + n0 - 2) / np.log(10)
    out = pd.DataFrame({"feature": X.columns, "F": F, "p": p, "neglog10p": nlp, "FDR": bh_fdr(np.nan_to_num(p, nan=1.0)),
                        "mean_diff": m1 - m0, "hedges_g": d * J, "eta2": eta2, "AUC": auc,
                        "direction": np.where(m1 > m0, "up", "down")})
    return out.sort_values("F", ascending=False).reset_index(drop=True)


def auc_delong(y, s):
    """AUC with DeLong variance -> (auc, lo, hi)."""
    y = np.asarray(y); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan
    allv = np.concatenate([pos, neg])
    r = stats.rankdata(allv)
    rp = stats.rankdata(pos); rn = stats.rankdata(neg)
    auc = (r[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (r[:m] - rp) / n
    v01 = 1 - (r[m:] - rn) / m
    var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    se = np.sqrt(var)
    return auc, max(0, auc - 1.96 * se), min(1, auc + 1.96 * se)


def delong_paired(y, s1, s2):
    """Two-sided DeLong test for two correlated AUCs."""
    y = np.asarray(y)
    def comps(s):
        pos, neg = s[y == 1], s[y == 0]
        m, n = len(pos), len(neg)
        allv = np.concatenate([pos, neg]); r = stats.rankdata(allv)
        rp = stats.rankdata(pos); rn = stats.rankdata(neg)
        auc = (r[:m].sum() - m * (m + 1) / 2) / (m * n)
        return auc, (r[:m] - rp) / n, 1 - (r[m:] - rn) / m
    a1, v10a, v01a = comps(np.asarray(s1, float))
    a2, v10b, v01b = comps(np.asarray(s2, float))
    m, n = len(v10a), len(v01a)
    S10 = np.cov(np.vstack([v10a, v10b])); S01 = np.cov(np.vstack([v01a, v01b]))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return a1, a2, 2 * stats.norm.sf(abs(z))


def calibration_stats(y, p):
    """Calibration-in-the-large (intercept) and slope via logistic recalibration."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    slope = lr.coef_[0, 0]
    # intercept with slope fixed at 1 (offset) -> Newton iterations
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp)))
        g = np.sum(y - q); h = np.sum(q * (1 - q))
        a += g / max(h, 1e-9)
    return a, slope


def net_benefit(y, p, thresholds):
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def threshold_metrics(y, p, thr=0.5):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0)); fn = np.sum((pred == 0) & (y == 1))
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    acc = (tp + tn) / len(y)
    bacc = (sens + spec) / 2
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0
    return dict(sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc, bacc=bacc, f1=f1, mcc=mcc)


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return thr[np.argmax(j)]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def kuncheva(sets, p):
    """Kuncheva consistency index averaged over all pairs of equal-size subsets."""
    sets = [list(s) for s in sets]
    k = len(sets[0])
    if k == 0 or k >= p:
        return np.nan
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            r = len(set(sets[i]) & set(sets[j]))
            vals.append((r * p - k * k) / (k * (p - k)))
    return float(np.mean(vals))


def set_plot_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


# colour-blind-safe palette (Okabe-Ito) for specimen/platform families
FAMILY_COLORS = {"WB-Agilent": "#0072B2", "Serum-3DGene": "#E69F00", "WB-Geniom": "#009E73"}


# ----------------------------------------------------------------------------------
# Cross-series sample-overlap audit
# ----------------------------------------------------------------------------------
def _rowz(A):
    A = A - A.mean(1, keepdims=True)
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def overlap_audit(data, r_thr=0.80, pairs=None):
    """Detect the same hybridisation/participant deposited in more than one series.
    Profiles are compared by Pearson correlation across shared miRNAs (after per-cohort
    z-scoring); r > r_thr is used as the duplicate call (the empirical distribution is
    bimodal with an empty 0.80-0.85 gap). Returns (pair_table, cluster_df)."""
    names = list(data.keys())
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rows, edges = [], []
    for a, b in pairs:
        Xa, ya = data[a]; Xb, yb = data[b]
        f = sorted(set(Xa.columns) & set(Xb.columns))
        if len(f) < 50:
            continue
        C = _rowz(Xa[f].values) @ _rowz(Xb[f].values).T
        mx = C.max(1); j = C.argmax(1)
        d = mx > r_thr
        for i in np.where(d)[0]:
            edges.append(((a, Xa.index[i]), (b, Xb.index[j[i]]), float(mx[i])))
        rows.append(dict(cohort_a=a, cohort_b=b, n_a=len(ya), n_b=len(yb), shared_features=len(f),
                         dup_a=int(d.sum()), dup_cases=int(ya[d].sum()), dup_controls=int((ya[d] == 0).sum()),
                         label_concordance=float((ya[d] == yb[j[d]]).mean()) if d.sum() else np.nan,
                         median_max_r_nondup=float(np.median(mx[~d])) if (~d).sum() else np.nan,
                         min_r_dup=float(mx[d].min()) if d.sum() else np.nan))
    # union-find clusters
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for u, v, _ in edges:
        parent[find(u)] = find(v)
    recs = []
    for c in names:
        X, y = data[c]
        for sid, lab in zip(X.index, y):
            recs.append(dict(cohort=c, sample_id=sid, label=int(lab), cluster=str(find((c, sid)))))
    cl = pd.DataFrame(recs)
    sizes = cl.groupby("cluster").cohort.transform("nunique")
    cl["n_series"] = sizes
    return pd.DataFrame(rows), cl


def overlap_ids(clusters, test_cohort, train_cohorts):
    """Sample IDs of test_cohort whose participant/hybridisation also occurs in train_cohorts."""
    train_cl = set(clusters.loc[clusters.cohort.isin(train_cohorts), "cluster"])
    t = clusters[clusters.cohort == test_cohort]
    return set(t.loc[t.cluster.isin(train_cl), "sample_id"])


def mimat_names(alias=None):
    """MIMAT -> current miRBase name (last alias recorded in the all-version table)."""
    alias = alias if alias is not None else load_alias_table()
    def pick(names):
        names = list(names)
        arm = [n for n in names if re.search(r"-[35]p$", n)]
        if arm:
            return arm[-1]
        nostar = [n for n in names if not n.endswith("*")]
        return (nostar or names)[-1]
    return alias.groupby("acc")["sym"].apply(pick).to_dict()


def hedges_g_var(g, n1, n0):
    return (n1 + n0) / (n1 * n0) + g ** 2 / (2 * (n1 + n0))


def dl_meta(G, V):
    """DerSimonian-Laird random-effects meta-analysis, vectorised over features.
    G, V: arrays (n_features, n_studies) with NaN for missing."""
    W = 1 / V
    W = np.where(np.isfinite(W), W, 0)
    Gz = np.where(np.isfinite(G), G, 0)
    k = (W > 0).sum(1)
    fe = (W * Gz).sum(1) / W.sum(1)
    Q = (W * (Gz - fe[:, None]) ** 2).sum(1)
    c = W.sum(1) - (W ** 2).sum(1) / W.sum(1)
    tau2 = np.maximum(0, (Q - (k - 1)) / np.where(c > 0, c, np.nan))
    Ws = np.where(W > 0, 1 / (V + tau2[:, None]), 0)
    re = (Ws * Gz).sum(1) / Ws.sum(1)
    se = np.sqrt(1 / Ws.sum(1))
    z = re / se
    p = 2 * stats.norm.sf(np.abs(z))
    I2 = np.where(Q > 0, np.maximum(0, (Q - (k - 1)) / Q), 0)
    return pd.DataFrame({"g_meta": re, "se": se, "z": z, "p": p, "tau2": tau2, "I2": I2, "k": k})


def make_models(rs=42, spw=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)),
        "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                     n_jobs=-1, random_state=rs),
        "XGB": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                             n_jobs=2, random_state=rs, verbosity=0),
    }


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    return model.decision_function(X)


def draw_flow(ax, rows, title=None):
    """Simple vertical flow diagram. rows = list of lists of (text, colour) boxes; arrows connect successive rows."""
    import matplotlib.patches as mpatches
    ax.axis("off")
    n = len(rows)
    ys = np.linspace(0.92, 0.08, n)
    centers = []
    for r, (row, yv) in enumerate(zip(rows, ys)):
        m = len(row); xs = np.linspace(0.5 / m, 1 - 0.5 / m, m); w = 0.9 / m
        cs = []
        for (txt, col), xv in zip(row, xs):
            box = mpatches.FancyBboxPatch((xv - w / 2, yv - 0.055), w, 0.11, boxstyle="round,pad=0.008",
                                          fc=col, ec="#333333", lw=0.6, transform=ax.transAxes)
            ax.add_patch(box)
            ax.text(xv, yv, txt, ha="center", va="center", fontsize=7, transform=ax.transAxes, wrap=True)
            cs.append((xv, yv))
        centers.append(cs)
    for r in range(n - 1):
        for (x1, y1) in centers[r]:
            for (x2, y2) in centers[r + 1]:
                if len(centers[r]) > 1 and len(centers[r + 1]) > 1 and abs(x1 - x2) > 0.3:
                    continue
                ax.annotate("", xy=(x2, y2 + 0.058), xytext=(x1, y1 - 0.058), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", lw=0.7, color="#555555"))
    if title:
        ax.set_title(title, loc="left", fontsize=10)
