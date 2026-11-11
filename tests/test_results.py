"""Corrobora la corrida N=240, T=300 K contra las predicciones del paper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATOS = ROOT / "results" / "n240" / "datos"
GRAF = ROOT / "results" / "n240" / "graficas"

REQUIRED_PNG = [
    "snapshot_3d.png",
    "snapshot_h_top.png",
    "Gq_normal_normal.png",
    "rdf.png",
    "traza_equilibrio.png",
    "histograma_h.png",
    "hrms_replicas.png",
]


def _summaries():
    files = sorted(DATOS.glob("replica_*_summary.csv"))
    if not files:
        pytest.skip("no hay resultados de simulación")
    return [np.genfromtxt(f, delimiter=",", names=True) for f in files]


def test_only_n240_and_t300():
    for row in _summaries():
        assert int(np.atleast_1d(row["N"])[0]) == 240
        assert float(np.atleast_1d(row["T"])[0]) == pytest.approx(300.0)
        assert float(np.atleast_1d(row["T"])[0]) <= 300.0


def test_ripples_are_present():
    """El paper: hay fluctuaciones de altura en equilibrio (Fig. 1)."""
    h = np.array([float(np.atleast_1d(r["h_rms"])[0]) for r in _summaries()])
    assert h.mean() > 0.02
    assert h.mean() < 2.0


def test_energy_still_bound():
    e = np.array([float(np.atleast_1d(r["energy"])[0]) for r in _summaries()])
    assert (e / 240).mean() < -4.0


def test_acceptance_reasonable():
    acc = np.array([float(np.atleast_1d(r["acc_move"])[0]) for r in _summaries()])
    assert acc.mean() > 0.15
    assert acc.mean() < 0.85


def test_rdf_peak_conjugated_bond():
    path = DATOS / "rdf_mean.csv"
    if not path.exists():
        pytest.skip("falta rdf_mean.csv")
    d = np.genfromtxt(path, delimiter=",", names=True)
    window = (d["r"] > 1.2) & (d["r"] < 1.65)
    peak = d["r"][window][np.argmax(d["g"][window])]
    assert peak == pytest.approx(1.42, abs=0.08)


def test_gq_shape_matches_paper_fig3():
    """Fig. 3: 1/q² hasta un mínimo y subida en los Bragg 2.94 y 5.11."""
    path = DATOS / "gq_binned.csv"
    if not path.exists():
        pytest.skip("falta gq_binned.csv")
    d = np.genfromtxt(path, delimiter=",", names=True)
    q, g = d["q"], d["G_over_N"]
    assert q.min() > 0.15
    assert q.max() > 5.0
    low = g[q < 0.4]
    mid = g[(q > 1.2) & (q < 1.9)]
    bragg = g[(q > 4.7) & (q < 5.8)]
    assert low.size and mid.size and bragg.size
    assert mid.min() < low.mean() * 0.4
    assert bragg.max() > mid.min() * 4.0


def test_no_70A_peak_accessible():
    """El paper: las muestras más pequeñas que 70 Å no muestran el máximo a q≈0.1."""
    files = sorted(DATOS.glob("replica_*_gq.csv"))
    if not files:
        pytest.skip("falta G(q)")
    qs = []
    gs = []
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        qs.append(d["q"])
        gs.append(d["G_over_N"])
    q = np.concatenate(qs)
    g = np.concatenate(gs)
    assert q.min() > 0.15  # 2π/L ≳ 0.25 Å⁻¹
    # no hay puntos en q≈0.1 Å⁻¹
    assert np.count_nonzero(q < 0.15) == 0
    assert np.all(np.isfinite(g))


def test_png_not_pdf_figures():
    if not GRAF.exists():
        pytest.skip("no hay gráficas")
    pngs = list(GRAF.glob("*.png")) + list(GRAF.glob("*.jpg"))
    pdfs = list(GRAF.glob("*.pdf"))
    assert len(pngs) >= 5
    assert pdfs == []
    for name in REQUIRED_PNG:
        assert (GRAF / name).exists(), name


def test_box_stays_near_table_i():
    for row in _summaries():
        lx = float(np.atleast_1d(row["lx"])[0])
        ly = float(np.atleast_1d(row["ly"])[0])
        assert 20.0 < lx < 30.0
        assert 20.0 < ly < 30.0
