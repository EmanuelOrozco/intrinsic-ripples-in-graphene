#!/usr/bin/env python3
"""Escribe docs/resultados_macros.tex a partir de los CSV de la simulación."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "results" / "datos"
OUT = ROOT / "docs" / "resultados_macros.tex"


def f(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def main() -> None:
    summaries = sorted(DATOS.glob("replica_*_summary.csv"))
    if not summaries:
        sys.exit("no hay summary CSV")
    rows = [np.genfromtxt(p, delimiter=",", names=True) for p in summaries]
    hrms = np.array([float(np.atleast_1d(r["h_rms"])[0]) for r in rows])
    heq = np.array([float(np.atleast_1d(r["h_rms_eq"])[0]) for r in rows])
    en = np.array([float(np.atleast_1d(r["energy"])[0]) for r in rows]) / 240.0
    acc = np.array([float(np.atleast_1d(r["acc_move"])[0]) for r in rows])
    acca = np.array([float(np.atleast_1d(r["acc_area"])[0]) for r in rows])
    lx = np.array([float(np.atleast_1d(r["lx"])[0]) for r in rows])
    ly = np.array([float(np.atleast_1d(r["ly"])[0]) for r in rows])
    s0 = np.array([float(np.atleast_1d(r["S0"])[0]) for r in rows])
    ns = np.array([int(np.atleast_1d(r["nsamples"])[0]) for r in rows])
    nrep = len(rows)

    rdf_peak = 1.42
    rdf_path = DATOS / "rdf_mean.csv"
    if rdf_path.exists():
        d = np.genfromtxt(rdf_path, delimiter=",", names=True)
        w = (d["r"] > 1.2) & (d["r"] < 1.65)
        rdf_peak = float(d["r"][w][np.argmax(d["g"][w])])

    qmin = 2.0 * np.pi / max(lx.mean(), ly.mean())
    qrip = 2.0 * np.pi / 70.0

    gq_ratio = float("nan")
    gq_path = DATOS / "gq_binned.csv"
    if gq_path.exists():
        g = np.genfromtxt(gq_path, delimiter=",", names=True)
        # comparar G/N con G0 alrededor de q=0.5–1.0 Å⁻¹
        t = 300.0 * 8.617333262145e-5
        kappa, s0m = 1.1, float(s0.mean())
        sel = (g["q"] > 0.45) & (g["q"] < 1.2)
        if np.any(sel):
            g0 = t / (kappa * s0m * g["q"][sel] ** 2)
            gq_ratio = float(np.median(g["G_over_N"][sel] / g0))

    lines = [
        r"% macros generadas; no editar a mano",
        rf"\newcommand{{\Nrep}}{{{nrep}}}",
        rf"\newcommand{{\HrmsMean}}{{{f(hrms.mean())}}}",
        rf"\newcommand{{\HrmsStd}}{{{f(hrms.std())}}}",
        rf"\newcommand{{\HeqMean}}{{{f(heq.mean())}}}",
        rf"\newcommand{{\EoverN}}{{{f(en.mean(), 3)}}}",
        rf"\newcommand{{\AccMove}}{{{f(100*acc.mean(), 1)}}}",
        rf"\newcommand{{\AccArea}}{{{f(100*acca.mean(), 1)}}}",
        rf"\newcommand{{\LxMean}}{{{f(lx.mean(), 2)}}}",
        rf"\newcommand{{\LyMean}}{{{f(ly.mean(), 2)}}}",
        rf"\newcommand{{\Szero}}{{{f(s0.mean(), 3)}}}",
        rf"\newcommand{{\NsampTot}}{{{int(ns.sum())}}}",
        rf"\newcommand{{\RdfPeak}}{{{f(rdf_peak, 3)}}}",
        rf"\newcommand{{\Qmin}}{{{f(qmin, 3)}}}",
        rf"\newcommand{{\Qripple}}{{{f(qrip, 3)}}}",
        rf"\newcommand{{\GqRatio}}{{{f(gq_ratio, 2) if np.isfinite(gq_ratio) else '---'}}}",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("escrito", OUT)


if __name__ == "__main__":
    main()
