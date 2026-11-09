"""Observables del paper: h, G(q) = q² |h_q|², RDF y G0(q) armónico."""

from __future__ import annotations

import numpy as np

from .lattice import KB_EV, KAPPA_300K


def height_fluctuations(pos: np.ndarray) -> tuple[np.ndarray, float]:
    """h_i = z_i - <z> (plano del centro de masa, Fig. 2)."""
    h = pos[:, 2] - pos[:, 2].mean()
    h_rms = float(np.sqrt(np.mean(h * h)))
    return h, h_rms


def area_per_atom(lx: float, ly: float, n: int) -> float:
    return lx * ly / n


def G0_harmonic(q: np.ndarray, n: int, s0: float, T: float = 300.0, kappa: float = KAPPA_300K):
    """G0(q)/N = T / (κ S0 q²)  (paper, eqs. 5–6, T en unidades de energía)."""
    t_ev = T * KB_EV
    q = np.asarray(q, dtype=np.float64)
    out = np.full_like(q, np.nan, dtype=np.float64)
    mask = q > 0
    out[mask] = t_ev / (kappa * s0 * q[mask] ** 2)
    return out


def normal_normal_Gq(pos: np.ndarray, lx: float, ly: float, nmax: int = 12):
    """G(q) = q² |h_q|² con h_q = Σ_j h_j exp(-i q · r_j).

    Convención del paper: |h_q|² = T N / (κ S0 q^4) en el límite armónico.
    """
    n = pos.shape[0]
    h, _ = height_fluctuations(pos)
    xy = pos[:, :2]
    records = []
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            if nx == 0 and ny == 0:
                continue
            qx = 2.0 * np.pi * nx / lx
            qy = 2.0 * np.pi * ny / ly
            q = math_hypot(qx, qy)
            phase = qx * xy[:, 0] + qy * xy[:, 1]
            hq = np.sum(h * np.exp(-1j * phase))
            g = (q * q) * float(hq.real * hq.real + hq.imag * hq.imag)
            records.append((q, g / n, nx, ny))
    rec = np.array(records, dtype=np.float64)
    order = np.argsort(rec[:, 0])
    return rec[order]


def math_hypot(qx: float, qy: float) -> float:
    return float(np.hypot(qx, qy))


def radial_distribution(
    pos: np.ndarray,
    lx: float,
    ly: float,
    rmax: float = 3.5,
    nbins: int = 350,
) -> tuple[np.ndarray, np.ndarray]:
    """g(r) 2D con imagen mínima en x,y (Fig. 5 del paper)."""
    n = pos.shape[0]
    density = n / (lx * ly)
    edges = np.linspace(0.0, rmax, nbins + 1)
    hist = np.zeros(nbins, dtype=np.float64)
    for i in range(n):
        dx = pos[:, 0] - pos[i, 0]
        dy = pos[:, 1] - pos[i, 1]
        dz = pos[:, 2] - pos[i, 2]
        dx -= lx * np.round(dx / lx)
        dy -= ly * np.round(dy / ly)
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        r[i] = 1e9
        hist += np.histogram(r, bins=edges)[0]
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    shell = 2.0 * np.pi * r_mid * dr * density
    g = hist / (n * np.maximum(shell, 1e-30))
    return r_mid, g


def bin_Gq(q_g: np.ndarray, n_bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Mediana angular de G(q)/N en bins logarítmicos de |q|.

    En la Fig. 3 del paper la curva es ese promedio sobre la dirección de q.
    La mediana sigue a los modos térmicos; unos pocos modos de Bragg no
    convierten el pico en una subida suave que tape la ley 1/q².
    """
    q = q_g[:, 0]
    g = q_g[:, 1]
    qb, gb, _ = angular_shells(q, g, n_bins=n_bins)
    return qb, gb


def cluster_shells(
    q: np.ndarray,
    g: np.ndarray,
    tol: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Un cascarón por estrella de |q| (modos a menos de `tol` relativo).

    Cada estrella entra una sola vez, con la mediana de G. Así la Fig. 3 no
    dibuja escalones de la malla discreta de N=240 ni deja que un modo de
    Bragg aisle el promedio.
    """
    q = np.asarray(q, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    mask = (q > 0) & np.isfinite(q) & np.isfinite(g) & (g > 0)
    q, g = q[mask], g[mask]
    if q.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    order = np.argsort(q)
    q, g = q[order], g[order]
    q_out, g_out, e_out = [], [], []
    i = 0
    n = q.size
    while i < n:
        j = i + 1
        limite = q[i] * (1.0 + tol)
        while j < n and q[j] <= limite:
            j += 1
        gg = g[i:j]
        q_out.append(float(np.exp(np.mean(np.log(q[i:j])))))
        g_out.append(float(np.median(gg)))
        if gg.size >= 4:
            lo, hi = np.percentile(gg, [16.0, 84.0])
            e_out.append(float(0.5 * (hi - lo)))
        else:
            e_out.append(0.0)
        i = j
    return np.array(q_out), np.array(g_out), np.array(e_out)


def angular_shells(
    q: np.ndarray,
    g: np.ndarray,
    n_bins: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(q_geom, mediana de G, semiancho del intervalo 16–84 %)."""
    q = np.asarray(q, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    mask = (q > 0) & np.isfinite(q) & np.isfinite(g) & (g > 0)
    q, g = q[mask], g[mask]
    if q.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    edges = np.logspace(np.log10(q.min()), np.log10(q.max() * 1.0000001), n_bins + 1)
    q_out, g_out, e_out = [], [], []
    for i in range(n_bins):
        sel = (q >= edges[i]) & (q < edges[i + 1])
        if np.count_nonzero(sel) < 1:
            continue
        gg = g[sel]
        q_out.append(np.exp(np.mean(np.log(q[sel]))))
        g_out.append(float(np.median(gg)))
        if gg.size >= 4:
            lo, hi = np.percentile(gg, [16, 84])
            e_out.append(0.5 * (hi - lo))
        else:
            e_out.append(0.0)
    return np.array(q_out), np.array(g_out), np.array(e_out)
