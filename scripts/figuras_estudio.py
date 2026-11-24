#!/usr/bin/env python3
"""Figuras del estudio LCBOPII (N<=4860) y de la comparación con Tersoff.

G(q) dibuja todos los modos como puntos, sin mediana ni espaciado.
Los ejes son los de la Fig. 3 del paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from contrastar_paper import informe  # noqa: E402
from ripples.lattice import KAPPA_300K  # noqa: E402
from ripples.observables import G0_harmonic  # noqa: E402

SIZES = [240, 960, 2160, 4860]
COLORES = {240: "#1f77b4", 960: "#ff7f0e", 2160: "#2ca02c", 4860: "#d62728"}
MARCAS = {240: "o", 960: "s", 2160: "^", 4860: "D"}
Q_RIPPLE = 2.0 * np.pi / 70.0
A_CC = 1.42
BRAGG_1 = 4.0 * np.pi / (3.0 * A_CC)
FIG3_Q = (0.05, 5.0)
FIG3_G = (1.0e-3, 1.0)
FIG5_R = (0.0, 3.5)

L_BASE = ROOT / "results" / "lcbopii"
T_BASE = ROOT / "results"
L_GRAF = L_BASE / "tabla" / "graficas"
C_GRAF = ROOT / "results" / "comparacion" / "graficas"


def _ejes(ax) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def cargar(base: Path, n: int) -> dict:
    return informe(base / f"n{n}" / "datos")


def rdf(base: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    path = base / f"n{n}" / "datos" / "rdf_mean.csv"
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(d["r"], dtype=np.float64), np.asarray(d["g"], dtype=np.float64)


def serie_gq(ax, info: dict, marca: str, relleno: bool) -> None:
    del marca, relleno
    color = COLORES[info["N"]]
    q, g = info["q"], info["g"]
    sel = (q >= FIG3_Q[0]) & (q <= FIG3_Q[1]) & (g > 0)
    ax.plot(
        q[sel], g[sel],
        color=color, marker="o", ms=1.2, ls="none",
        markeredgewidth=0, alpha=0.55, label=str(info["N"]),
        zorder=3, rasterized=True,
    )


def marco_gq(ax, s0: float) -> None:
    q_th = np.logspace(np.log10(FIG3_Q[0]), np.log10(FIG3_Q[1]), 400)
    g0 = G0_harmonic(q_th, 1, s0, T=300.0, kappa=KAPPA_300K)
    ax.loglog(q_th, g0, color="#d62728", lw=1.3, label=r"$G_0$", zorder=2)
    ax.axvline(Q_RIPPLE, color="0.35", ls=":", lw=0.9, zorder=1)
    ax.axvline(BRAGG_1, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.set_xlim(*FIG3_Q)
    ax.set_ylim(*FIG3_G)
    ax.set_xlabel(r"$q$ (Å$^{-1}$)")
    ax.set_ylabel(r"$G(q)/N$")
    ax.legend(frameon=False, fontsize=8, loc="lower left", markerscale=3)
    _ejes(ax)


def figura_gq(filas: list[dict], destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 6.15))
    ax.set_box_aspect(202.0 / 156.0)
    for info in filas:
        serie_gq(ax, info, "L", True)
    marco_gq(ax, filas[-1]["S0"])
    fig.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(fig)


def figura_rdf(filas: list[dict], base: Path, destino: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    ax, zoom = axes
    for info in filas:
        r, g = rdf(base, info["N"])
        color = COLORES[info["N"]]
        marca = MARCAS[info["N"]]
        ax.plot(
            r, g, color=color, lw=0.7, marker=marca, ms=3.2, markevery=4,
            markerfacecolor=color, markeredgewidth=0, label=str(info["N"]),
        )
        m = (r >= 1.15) & (r <= 1.75)
        zoom.plot(
            r[m], g[m], color=color, lw=0.7, marker=marca, ms=4.0, markevery=1,
            markerfacecolor=color, markeredgewidth=0,
        )
    for eje in (ax, zoom):
        eje.axvline(1.31, color="0.25", ls=":", lw=0.8)
        eje.axvline(1.42, color="0.15", ls="--", lw=0.9)
        eje.axvline(1.54, color="0.25", ls=":", lw=0.8)
        eje.set_ylim(0.0, 16.0)
        eje.set_xlabel(r"$r$ (Å)")
        _ejes(eje)
    ax.axhspan(0.0, 3.0, color="#d62728", alpha=0.08, zorder=0)
    ax.set_xlim(*FIG5_R)
    ax.set_ylabel(r"$g(r)$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    zoom.set_xlim(1.15, 1.75)
    zoom.set_title("primer pico", fontsize=9)
    fig.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(fig)


def figura_h(filas: list[dict], destino: Path, ancla: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    largo = np.array([max(f["lx"], f["ly"]) for f in filas])
    h = np.array([f["h_rms"] for f in filas])
    e = np.array([f["h_rms_std"] for f in filas])
    ax.errorbar(
        largo, h, yerr=e, fmt="o", color="#1f77b4", ms=7, lw=0.8, capsize=3,
        label=r"$h_{\mathrm{rms}}$",
    )
    guia = h[0] * largo / largo[0]
    ax.plot(largo, guia, "--", color="#d62728", label=r"armónico $\propto L$")
    for f, x, y in zip(filas, largo, h):
        ax.annotate(str(f["N"]), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(r"$L$ (Å)")
    ax.set_ylabel(r"$h_{\mathrm{rms}}$ (Å)")
    ax.legend(frameon=False)
    _ejes(ax)
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)
    del ancla


def figura_foto(destino: Path) -> None:
    snap = L_BASE / "n4860" / "datos" / "replica_00_snapshot.csv"
    d = np.genfromtxt(snap, delimiter=",", names=True)
    h = d["z"] - np.mean(d["z"])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sc = ax.scatter(d["x"], d["y"], c=h, cmap="coolwarm", s=3.0, linewidths=0)
    x0 = float(np.min(d["x"])) + 8.0
    y0 = float(np.min(d["y"])) + 10.0
    ax.plot([x0, x0 + 70.0], [y0, y0], color="black", lw=2.4, solid_capstyle="butt")
    ax.text(x0 + 35.0, y0 + 4.5, "70 Å", ha="center", va="bottom", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (Å)")
    ax.set_ylabel(r"$y$ (Å)")
    fig.colorbar(sc, ax=ax, shrink=0.78, label=r"$h$ (Å)")
    _ejes(ax)
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)


def figura_comparacion(tersoff: list[dict], lcbop: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.4), sharey=True)
    for ax, filas, titulo, relleno in (
        (axes[0], tersoff, "Tersoff", False),
        (axes[1], lcbop, "LCBOPII", True),
    ):
        ax.set_box_aspect(202.0 / 156.0)
        for info in filas:
            serie_gq(ax, info, "T" if not relleno else "L", relleno)
        marco_gq(ax, filas[-1]["S0"])
        ax.set_title(titulo, fontsize=11)
    fig.savefig(C_GRAF / "Gq_comparacion.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    for ax, filas, base, titulo, relleno in (
        (axes[0], tersoff, T_BASE, "Tersoff", False),
        (axes[1], lcbop, L_BASE, "LCBOPII", True),
    ):
        for info in filas:
            r, g = rdf(base, info["N"])
            m = (r >= 1.15) & (r <= 1.75)
            ax.plot(
                r[m], g[m], color=COLORES[info["N"]], lw=0.6,
                marker=MARCAS[info["N"]], ms=3.6, markevery=1,
                markerfacecolor=COLORES[info["N"]] if relleno else "white",
                markeredgecolor=COLORES[info["N"]], markeredgewidth=0.6,
                label=str(info["N"]),
            )
        for x, estilo in ((1.31, ":"), (1.42, "--"), (1.54, ":")):
            ax.axvline(x, color="0.25", ls=estilo, lw=0.8)
        ax.set_xlim(1.15, 1.75)
        ax.set_ylim(0.0, 16.0)
        ax.set_xlabel(r"$r$ (Å)")
        ax.set_title(titulo, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        _ejes(ax)
    axes[0].set_ylabel(r"$g(r)$")
    fig.savefig(C_GRAF / "rdf_comparacion.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for filas, color, marca, nombre in (
        (tersoff, "#1f77b4", "o", "Tersoff"),
        (lcbop, "#d62728", "D", "LCBOPII"),
    ):
        largo = np.array([max(f["lx"], f["ly"]) for f in filas])
        h = np.array([f["h_rms"] for f in filas])
        e = np.array([f["h_rms_std"] for f in filas])
        ax.errorbar(
            largo, h, yerr=e, fmt=marca, color=color, ms=7, lw=0.8, capsize=3, label=nombre,
        )
    ax.set_xlabel(r"$L$ (Å)")
    ax.set_ylabel(r"$h_{\mathrm{rms}}$ (Å)")
    ax.legend(frameon=False)
    _ejes(ax)
    fig.savefig(C_GRAF / "hrms_comparacion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    L_GRAF.mkdir(parents=True, exist_ok=True)
    C_GRAF.mkdir(parents=True, exist_ok=True)
    lcbop = [cargar(L_BASE, n) for n in SIZES]
    tersoff = [cargar(T_BASE, n) for n in SIZES]
    figura_gq(lcbop, L_GRAF / "Gq_fig3.png")
    figura_rdf(lcbop, L_BASE, L_GRAF / "rdf_fig5.png")
    figura_h(lcbop, L_GRAF / "hrms_vs_L.png", "L")
    figura_foto(L_GRAF / "snapshot_fig1.png")
    figura_comparacion(tersoff, lcbop)
    print("PNG:", L_GRAF)
    print("PNG:", C_GRAF)


if __name__ == "__main__":
    main()
