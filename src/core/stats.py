"""The paper's detection statistics, implemented from its Methods (Fousek et al. 2024, Eqs. 6-8 and text).
- dFC_w: sliding-window FC correlation matrix (window 60 s, step 2 s).
- switching index: variance of dFC_w(i,j) over pairs with i <= j - W (W = window length in steps).
- KS histogram statistic: KS distance between centred distributions of dFC_w upper-triangle values.
- VAF2: variance accounted for by the first two PCs of a (T, N) series (a function of its covariance only).
- edge time series, RSS, events (95th percentile), dFC_e.
"""
import numpy as np
from scipy.stats import ks_2samp


def dfc_w(bold, tr_s=0.72, win_s=60.0, step_s=2.0):
    T, N = bold.shape
    w = int(round(win_s / tr_s))
    st = max(1, int(round(step_s / tr_s)))
    iu = np.triu_indices(N, 1)
    fcs = []
    for a in range(0, T - w + 1, st):
        fcs.append(
            np.nan_to_num(np.corrcoef(bold[a : a + w].T)[iu])
        )  # constant nodes (two-state model) give NaN correlations; treat as 0
    if len(fcs) < 3:
        return np.full((max(len(fcs), 1),) * 2, np.nan), w // st
    F = np.array(fcs)
    F = F - F.mean(1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    return F @ F.T, w // st


def switching_index(D, Wsteps):
    n = D.shape[0]
    i, j = np.triu_indices(n, 1)
    m = i <= j - Wsteps
    return float(np.var(D[i[m], j[m]])) if m.any() else np.nan


def upper(D):
    i, j = np.triu_indices(D.shape[0], 1)
    return D[i, j]


def ks_centred(Da, Db):
    a, b = upper(Da), upper(Db)
    return float(ks_2samp(a - a.mean(), b - b.mean()).statistic)


def vaf2(X):
    X = np.asarray(X, np.float64)
    if not np.isfinite(X).all():
        return np.nan
    C = np.cov(X.T)
    C = 0.5 * (C + C.T)
    try:
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    except np.linalg.LinAlgError:
        ev = np.sort(np.linalg.eigvals(C).real)[::-1]
    return float(ev[:2].sum() / ev.sum()) if ev.sum() > 0 else np.nan


def edge_series(bold):
    z = (bold - bold.mean(0)) / (bold.std(0) + 1e-12)
    N = z.shape[1]
    i, j = np.triu_indices(N, 1)
    return z[:, i] * z[:, j]


def events(E, pct=95):
    rss = np.sqrt((E**2).sum(1))
    thr = np.percentile(rss, pct)
    return rss, rss > thr


def event_stats(bold):
    """Similarity of edge vectors within events vs event-nonevent, as in Fig. 3D (bottom)."""
    E = edge_series(bold)
    rss, ev = events(E)
    C = np.corrcoef(E)
    n = len(ev)
    i, j = np.triu_indices(n, 1)
    within = C[i, j][ev[i] & ev[j]]
    cross = C[i, j][ev[i] ^ ev[j]]
    none = C[i, j][~ev[i] & ~ev[j]]
    n_exc = int(((ev[1:] & ~ev[:-1])).sum() + ev[0])
    return dict(
        within=float(within.mean()) if within.size else np.nan,
        cross=float(cross.mean()),
        nonevents=float(none.mean()),
        n_excursions=n_exc,
        frac_event=float(ev.mean()),
    )


def phase_randomise(bold, rng):
    """Multivariate phase randomisation: same random phases added to every channel (preserves cross-spectrum,
    hence FC and all autocovariances) -> a stationary Gaussian-like surrogate."""
    T, N = bold.shape
    X = np.fft.rfft(bold - bold.mean(0), axis=0)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(X.shape[0], 1)))
    ph[0] = 1
    if T % 2 == 0:
        ph[-1] = 1
    return np.fft.irfft(X * ph, n=T, axis=0) + bold.mean(0)


def all_stats(bold, tr_s=0.72):
    D, Ws = dfc_w(bold, tr_s)
    st = event_stats(bold)
    return dict(switch=switching_index(D, Ws), vaf2=vaf2(bold), **st), D
