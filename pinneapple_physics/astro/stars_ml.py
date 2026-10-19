"""Machine learning on the bright-star catalogue, checked against the catalogue's own spectral classification.

* Unsupervised: a Gaussian mixture (EM, NumPy) on the HR diagram (B-V, M_V) finds the main sequence, the giant branch
  and the luminous stars without labels; its clusters are scored against the luminosity classes of the spectral types.
* Supervised: luminosity class (dwarf / subgiant / giant / bright giant-supergiant) from (B-V, M_V), and spectral
  class (O B A F G K M) from B-V alone, with k-nearest neighbours; 5-fold cross-validated accuracy against the
  catalogue labels, next to the majority-class baseline, and per-class recall.

Labels come from the Yale BSC5 MK spectral types (Morgan-Keenan luminosity classes parsed from strings such as
"K0IIIbCN-0.5" -> III). Only stars with a trigonometric parallax >= 5 mas enter the HR-diagram tasks.
"""
from __future__ import annotations

import re

import numpy as np

from .stars import StarCatalog, load_bright_stars

LUM_GROUPS = ["dwarf (V)", "subgiant (IV)", "giant (III)", "bright giant / supergiant (I-II)"]
SPEC = "OBAFGKM"
_LUM_RE = re.compile(r"(Iab|Ia|Ib|III|II|IV|VI|V|I)")


def luminosity_group(sp: str):
    s = sp.split("+")[0]
    m = re.match(r"^[a-z]*([OBAFGKM])\d*(?:\.\d+)?(?:-\d+)?\s*(.*)$", s)
    if not m:
        return None
    rest = m.group(2)
    k = _LUM_RE.match(rest)
    if not k:
        return None
    c = k.group(1)
    return {"V": 0, "VI": 0, "IV": 1, "III": 2}.get(c, 3)


def spectral_class(sp: str):
    m = re.match(r"^[a-z]*([OBAFGKM])", sp.split("+")[0])
    return SPEC.index(m.group(1)) if m else None


def dataset(cat: StarCatalog | None = None, min_parallax_mas: float = 5.0):
    cat = cat or load_bright_stars()
    ok = ~np.isnan(cat.bv) & ~np.isnan(cat.plx) & (cat.plx >= min_parallax_mas) & ~cat.plx_dyn
    idx = np.nonzero(ok)[0]
    Mv = cat.v[idx] - 5 * np.log10(1000.0 / cat.plx[idx] / 10.0)
    lum = np.array([luminosity_group(cat.sp[i]) if cat.sp[i] else None for i in idx], dtype=object)
    spc = np.array([spectral_class(cat.sp[i]) if cat.sp[i] else None for i in idx], dtype=object)
    return {"index": idx, "hr": cat.hr[idx], "bv": cat.bv[idx], "Mv": Mv, "lum": lum, "spec": spc}


def gmm_fit(X, k=3, iters=200, seed=0):
    """Gaussian mixture by expectation-maximisation (full covariances), k-means++-style initialisation."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mu = X[rng.choice(n, 1)]
    for _ in range(1, k):
        dist = np.min(((X[:, None] - mu[None]) ** 2).sum(-1), 1)
        mu = np.vstack([mu, X[rng.choice(n, p=dist / dist.sum())]])
    cov = np.array([np.cov(X.T) for _ in range(k)])
    w = np.full(k, 1.0 / k)
    ll_old = -np.inf
    for _ in range(iters):
        logp = np.stack([np.log(w[j]) - 0.5 * (np.linalg.slogdet(cov[j])[1]
                         + np.einsum("ni,ij,nj->n", X - mu[j], np.linalg.inv(cov[j]), X - mu[j])
                         + d * np.log(2 * np.pi)) for j in range(k)], 1)
        m = logp.max(1, keepdims=True)
        ll = float((m[:, 0] + np.log(np.exp(logp - m).sum(1))).sum())
        R = np.exp(logp - m)
        R /= R.sum(1, keepdims=True)
        Nk = R.sum(0) + 1e-12
        w = Nk / n
        mu = (R.T @ X) / Nk[:, None]
        cov = np.array([((R[:, j, None] * (X - mu[j])).T @ (X - mu[j])) / Nk[j] + 1e-6 * np.eye(d) for j in range(k)])
        if abs(ll - ll_old) < 1e-6 * abs(ll):
            break
        ll_old = ll
    return {"weights": w, "means": mu, "covs": cov, "resp": R, "labels": R.argmax(1), "loglik": ll}


def _knn_predict(Xtr, ytr, Xte, k=15):
    d = ((Xte[:, None] - Xtr[None]) ** 2).sum(-1)
    nn = np.argsort(d, 1)[:, :k]
    votes = ytr[nn]
    n_cls = int(ytr.max()) + 1
    counts = np.stack([(votes == c).sum(1) for c in range(n_cls)], 1)
    return counts.argmax(1)


def cross_validate(X, y, k=15, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    pred = np.empty(len(y), int)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        pred[te] = _knn_predict(Z[tr], y[tr], Z[te], k)
    n_cls = int(y.max()) + 1
    conf = np.zeros((n_cls, n_cls), int)
    np.add.at(conf, (y, pred), 1)
    recall = [float(conf[c, c] / conf[c].sum()) if conf[c].sum() else None for c in range(n_cls)]
    baseline = float(np.bincount(y).max() / len(y))
    return {"accuracy": float((pred == y).mean()), "baseline": baseline, "confusion": conf.tolist(),
            "recall": recall, "pred": pred}


def analyse(seed: int = 0) -> dict:
    ds = dataset()
    X = np.stack([ds["bv"], ds["Mv"]], 1)
    g = gmm_fit(X, 3, seed=seed)
    labelled = np.array([v is not None for v in ds["lum"]])
    lum = np.array([v if v is not None else -1 for v in ds["lum"]])
    # name each cluster by its majority luminosity group; purity = fraction agreeing
    names, purity_hits = [], 0
    for j in range(3):
        m = (g["labels"] == j) & labelled
        maj = int(np.bincount(lum[m], minlength=4).argmax()) if m.any() else 0
        names.append(LUM_GROUPS[maj])
        purity_hits += int((lum[m] == maj).sum())
    purity = purity_hits / max(int(labelled.sum()), 1)
    lum_cv = cross_validate(X[labelled], lum[labelled].astype(int), seed=seed)
    # spectral class from colour alone uses every star with B-V (no parallax needed)
    cat = load_bright_stars()
    allok = ~np.isnan(cat.bv)
    sp_all = np.array([spectral_class(s) if s else None for s in cat.sp[allok]], dtype=object)
    m2 = np.array([v is not None for v in sp_all])
    spec_cv = cross_validate(cat.bv[allok][m2][:, None], sp_all[m2].astype(int), seed=seed)
    checks = [
        {"id": "ml_lum_class", "name": "Luminosity class from (B-V, M_V): CV accuracy vs majority baseline",
         "value": lum_cv["accuracy"], "threshold": lum_cv["baseline"] + 0.1,
         "status": "pass" if lum_cv["accuracy"] >= lum_cv["baseline"] + 0.1 else "fail", "higher_is_better": True},
        {"id": "ml_spec_class", "name": "Spectral class from B-V: CV accuracy vs majority baseline",
         "value": spec_cv["accuracy"], "threshold": spec_cv["baseline"] + 0.2,
         "status": "pass" if spec_cv["accuracy"] >= spec_cv["baseline"] + 0.2 else "fail", "higher_is_better": True},
        {"id": "ml_cluster_purity", "name": "Unsupervised clusters: purity against MK luminosity classes", "value": purity, "threshold": 0.6,
         "status": "pass" if purity >= 0.6 else "warn", "higher_is_better": True},
    ]
    return {
        "n_stars_hr": int(len(X)), "n_labelled": int(labelled.sum()),
        "hr": ds["hr"].tolist(), "bv": np.round(ds["bv"], 3).tolist(), "Mv": np.round(ds["Mv"], 3).tolist(),
        "cluster": g["labels"].tolist(), "cluster_names": names, "cluster_means": g["means"].tolist(),
        "cluster_purity": purity,
        "lum_true": lum.tolist(), "lum_groups": LUM_GROUPS,
        "lum_cv": {k: v for k, v in lum_cv.items() if k != "pred"},
        "spec_cv": {k: v for k, v in spec_cv.items() if k != "pred"}, "spec_classes": list(SPEC),
        "checks": checks,
        "verdict": {"status": "trustworthy" if all(c["status"] == "pass" for c in checks) else "use_with_caution"},
        "method": "Gaussian mixture (EM) on (B-V, M_V); k-nearest neighbours (k = 15, standardised features), "
                  "5-fold cross-validation against the BSC5 MK spectral types",
    }
