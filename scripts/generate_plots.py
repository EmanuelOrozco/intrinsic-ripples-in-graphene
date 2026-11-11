#!/usr/bin/env python3
"""Figuras PNG de la reimplementación (nunca PDF)."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from ripples.lattice import KAPPA_300K  # noqa: E402
from ripples.observables import G0_harmonic, spectrum_curve  # noqa: E402

DATOS = ROOT / "results" / "datos"
GRAF = ROOT / "results" / "graficas"
T = 300.0


def sistema() -> tuple[int, float]:
    files = sorted(DATOS.glob("replica_*_summary.csv"))
    if not files:
        return 240, 2.62
    ns, s0s = [], []
    for path in files:
        s = np.genfromtxt(path, delimiter=",", names=True)
        ns.append(int(float(np.atleast_1d(s["N"])[0])))
        s0s.append(float(np.atleast_1d(s["S0"])[0]))
    return int(np.median(ns)), float(np.mean(s0s))


def _ejes_log(ax) -> None:
    """Décadas limpias. El formateo por defecto llega a rotular 3×10^4×10^-6."""
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0))
        axis.set_major_formatter(LogFormatterMathtext(base=10.0))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)


def guardar(fig, nombre: str) -> None:
    GRAF.mkdir(parents=True, exist_ok=True)
    out = GRAF / nombre
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {out}")


def load_csv(path: Path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def plot_snapshot():
    files = sorted(DATOS.glob("replica_*_snapshot.csv"))
    if not files:
        return
    data = np.genfromtxt(files[0], delimiter=",", names=True)
    x, y, z = data["x"], data["y"], data["z"]
    h = z - z.mean()
    lx, ly = float(data["lx"][0]), float(data["ly"][0])

    n_atoms, _ = sistema()
    # mplot3d iguala los tres ejes al rango de los datos: 2 Å de altura se ven
    # como 70 Å de caja. Se amplía z un factor fijo y se dice en el título.
    z_factor = 6.0
    dx = max(float(np.ptp(x)), 1.0)
    dy = max(float(np.ptp(y)), 1.0)
    dz = max(float(np.ptp(z)), 1e-3)
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=h, cmap="coolwarm", s=18 if n_atoms <= 2160 else 6, depthshade=False)
    ax.set_box_aspect((dx, dy, z_factor * dz))
    pad = 0.05 * dz
    ax.set_zlim(float(z.min()) - pad, float(z.max()) + pad)
    ax.set_zticks([-1.0, 0.0, 1.0])
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(
        f"Instantánea N={n_atoms}, T=300 K  (caja {lx:.2f}×{ly:.2f} Å, eje z ×{z_factor:.0f})"
    )
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12, label="h (Å)")
    guardar(fig, "snapshot_3d.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sc = ax.scatter(x, y, c=h, cmap="coolwarm", s=28, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title("Altura h sobre el plano del centro de masa")
    fig.colorbar(sc, ax=ax, label="h (Å)")
    guardar(fig, "snapshot_h_top.png")


def _promedio_modos(files) -> tuple[np.ndarray, np.ndarray]:
    """Promedia G(q)/N de las réplicas en cada (nx, ny)."""
    acc: dict[tuple[int, int], list[float]] = {}
    qacc: dict[tuple[int, int], list[float]] = {}
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        for nx, ny, q, g in zip(d["nx"], d["ny"], d["q"], d["G_over_N"]):
            key = (int(nx), int(ny))
            acc.setdefault(key, []).append(float(g))
            qacc.setdefault(key, []).append(float(q))
    q = np.array([np.mean(qacc[k]) for k in acc])
    g = np.array([np.mean(acc[k]) for k in acc])
    return q, g


def _modos_instantaneas(nmax: int = 22) -> tuple[np.ndarray, np.ndarray]:
    """G(q)=q²|h_q|² de las instantáneas, hasta el segundo pico de Bragg.

    El CSV de producción guarda |n|≤10 (q≲3.5 Å⁻¹). El pico q=4π/(√3 a)=5.11
    sale de las mismas configuraciones finales, con la convención del paper.
    """
    files = sorted(DATOS.glob("replica_*_snapshot.csv"))
    if not files:
        return np.array([]), np.array([])
    espec = []
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        pos = np.column_stack([d["x"], d["y"], d["z"]])
        lx, ly = float(d["lx"][0]), float(d["ly"][0])
        h = pos[:, 2] - pos[:, 2].mean()
        n = pos.shape[0]
        qs, gs = [], []
        for nx in range(-nmax, nmax + 1):
            for ny in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0:
                    continue
                qx = 2.0 * np.pi * nx / lx
                qy = 2.0 * np.pi * ny / ly
                phase = qx * pos[:, 0] + qy * pos[:, 1]
                hq = np.sum(h * np.exp(-1j * phase))
                q = float(np.hypot(qx, qy))
                gs.append((q * q) * float(hq.real * hq.real + hq.imag * hq.imag) / n)
                qs.append(q)
        espec.append((qs, gs))
    q = np.mean([np.array(s[0]) for s in espec], axis=0)
    g = np.mean([np.array(s[1]) for s in espec], axis=0)
    return q, g


def plot_gq():
    files = sorted(DATOS.glob("replica_*_gq.csv"))
    if not files:
        return
    q, g = _promedio_modos(files)
    n_atoms, s0 = sistema()
    if q.size and float(np.max(q)) < 4.5 and n_atoms <= 400:
        q_hi, g_hi = _modos_instantaneas()
        if q_hi.size:
            extra = q_hi > q.max()
            q = np.concatenate([q, q_hi[extra]])
            g = np.concatenate([g, g_hi[extra]])
    # Mismos ejes que la Fig. 3 del paper: q de 0.05 a 5, G/N de 1e-3 a 1.
    # La curva es la mediana en ventana logarítmica, no cada estrella de |q|.
    q_line, g_line = spectrum_curve(q, g)
    q_th = np.logspace(np.log10(0.05), np.log10(5.0), 400)
    g0 = G0_harmonic(q_th, n_atoms, s0, T=T, kappa=KAPPA_300K)

    fig, ax = plt.subplots(figsize=(4.9, 6.15))
    ax.set_box_aspect(202.0 / 156.0)
    ax.loglog(q_th, g0, color="#d62728", lw=1.3, label=r"$G_0(q)$", zorder=2)
    ax.loglog(q_line, g_line, color="#2ca02c", lw=1.8, label=str(n_atoms), zorder=3)
    ax.axvline(2.0 * np.pi / 70.0, color="0.35", ls=":", lw=0.9, zorder=1)
    ax.axvline(4.0 * np.pi / (3.0 * 1.42), color="0.35", ls="--", lw=0.8, zorder=1)
    ax.set_xlim(0.05, 5.0)
    ax.set_ylim(1.0e-3, 1.0)
    _ejes_log(ax)
    ax.set_xlabel(r"$q$ (Å$^{-1}$)")
    ax.set_ylabel(r"$G(q)/N$")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    guardar(fig, "Gq_normal_normal.png")


def plot_rdf():
    files = sorted(DATOS.glob("replica_*_rdf.csv"))
    if not files:
        return
    r = None
    acc = None
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        if r is None:
            r = d["r"]
            acc = d["g"].astype(np.float64)
        else:
            acc += d["g"]
    g = acc / len(files)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(r, g, color="#1f77b4", lw=1.5)
    ax.axvline(1.31, color="#d62728", ls="--", lw=1, label="doble 1.31 Å")
    ax.axvline(1.42, color="#2ca02c", ls="--", lw=1.2, label="conjugado 1.42 Å")
    ax.axvline(1.54, color="#ff7f0e", ls="--", lw=1, label="simple 1.54 Å")
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0, 3.5)
    ax.set_title(f"Función de distribución radial, N={sistema()[0]}, T=300 K")
    ax.legend(fontsize=8)
    guardar(fig, "rdf.png")
    np.savetxt(
        DATOS / "rdf_mean.csv",
        np.column_stack([r, g]),
        delimiter=",",
        header="r,g",
        comments="",
    )


def plot_trace():
    files = sorted(DATOS.glob("replica_*_trace.csv"))
    if not files:
        return
    n_atoms = sistema()[0]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        axes[0].plot(d["step"], d["h_rms"], lw=0.8, alpha=0.75)
        axes[1].plot(d["step"], d["energy"] / n_atoms, lw=0.8, alpha=0.75)
    axes[0].set_ylabel(r"$h_{\mathrm{rms}}$ (Å)")
    axes[0].set_title("Equilibrio de altura y energía (réplicas independientes)")
    axes[1].set_ylabel("E/N (eV/átomo)")
    axes[1].set_xlabel("paso MC")
    guardar(fig, "traza_equilibrio.png")


def plot_height_hist():
    files = sorted(DATOS.glob("replica_*_snapshot.csv"))
    if not files:
        return
    hs = []
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        z = d["z"]
        hs.append(z - z.mean())
    h = np.concatenate(hs)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.hist(h, bins=40, density=True, color="#1f77b4", alpha=0.85, edgecolor="none")
    ax.set_xlabel("h (Å)")
    ax.set_ylabel("densidad")
    ax.set_title(r"Distribución de alturas, $\langle h\rangle=0$")
    guardar(fig, "histograma_h.png")


def plot_summary_table_png():
    files = sorted(DATOS.glob("replica_*_summary.csv"))
    if not files:
        return
    rows = [np.genfromtxt(f, delimiter=",", names=True) for f in files]
    hrms = np.array([float(np.atleast_1d(r["h_rms"])[0]) for r in rows])
    en = np.array([float(np.atleast_1d(r["energy"])[0]) for r in rows])
    acc = np.array([float(np.atleast_1d(r["acc_move"])[0]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(np.arange(len(hrms)), hrms, color="#1f77b4")
    ax.axhline(hrms.mean(), color="k", ls="--", label=fr"media {hrms.mean():.3f} Å")
    ax.set_xlabel("réplica")
    ax.set_ylabel(r"$h_{\mathrm{rms}}$ (Å)")
    ax.set_title("Altura RMS por réplica (producción)")
    ax.legend()
    guardar(fig, "hrms_replicas.png")
    n_atoms, _ = sistema()
    with (DATOS / "resumen_global.csv").open("w", encoding="utf-8") as fh:
        fh.write("n_replicas,h_rms_mean,h_rms_std,E_over_N_mean,acc_move_mean,T,N\n")
        fh.write(
            f"{len(rows)},{hrms.mean():.6f},{hrms.std():.6f},"
            f"{(en / n_atoms).mean():.6f},{acc.mean():.4f},300,{n_atoms}\n"
        )


def main():
    global DATOS, GRAF
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", type=Path, default=None)
    parser.add_argument("--graf", type=Path, default=None)
    args = parser.parse_args()
    if args.datos is not None:
        DATOS = args.datos
    if args.graf is not None:
        GRAF = args.graf
    plot_snapshot()
    plot_gq()
    plot_rdf()
    plot_trace()
    plot_height_hist()
    plot_summary_table_png()
    print("Figuras escritas en", GRAF)


if __name__ == "__main__":
    main()
