"""G(q) y RDF: convención del paper y picos de la red ideal."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ripples.lattice import A_CC, graphene_sheet
from ripples.observables import (
    G0_harmonic,
    cluster_shells,
    height_fluctuations,
    normal_normal_Gq,
    radial_distribution,
)


def test_flat_sheet_zero_Gq():
    pos, lx, ly = graphene_sheet(10, 6)
    rec = normal_normal_Gq(pos, lx, ly, nmax=6)
    assert np.max(np.abs(rec[:, 1])) < 1e-18


def test_height_com_removed():
    pos, _, _ = graphene_sheet(10, 6)
    pos = pos.copy()
    pos[:, 2] = 0.3
    h, hrms = height_fluctuations(pos)
    assert h.mean() == pytest.approx(0.0, abs=1e-12)
    assert hrms == pytest.approx(0.0, abs=1e-12)


def test_harmonic_G0_matches_paper_formula():
    s0 = 2.62
    g = G0_harmonic(np.array([1.0]), 240, s0, T=300.0, kappa=1.1)
    t = 300.0 * 8.617333262145e-5
    assert g[0] == pytest.approx(t / (1.1 * s0), rel=1e-9)


def test_single_mode_Gq_convention():
    """Un modo h = A cos(qx x) debe dar |h_q|² = (N A / 2)² en q y -q."""
    pos, lx, ly = graphene_sheet(10, 6)
    pos = pos.copy()
    qx = 2.0 * np.pi / lx
    A = 0.2
    pos[:, 2] = A * np.cos(qx * pos[:, 0])
    rec = normal_normal_Gq(pos, lx, ly, nmax=3)
    # modos nx=±1, ny=0
    sel = (np.abs(rec[:, 2]) == 1) & (rec[:, 3] == 0)
    g_over_n = rec[sel, 1]
    n = pos.shape[0]
    hq = 0.5 * A * n  # Σ (A/2) e^{±iqx} 
    q = qx
    expected = (q * q) * (hq * hq) / n
    assert np.mean(g_over_n) == pytest.approx(expected, rel=0.08)


def test_rdf_first_peak_ideal():
    pos, lx, ly = graphene_sheet(10, 6)
    r, g = radial_distribution(pos, lx, ly, rmax=2.0, nbins=200)
    peak_r = r[np.argmax(g)]
    assert peak_r == pytest.approx(A_CC, abs=0.05)
    assert g.max() > 1.0


def test_cluster_shells_median_keeps_bragg_outlier_out():
    """Un modo enorme en la estrella no debe fijar la mediana angular."""
    q = np.array([1.0, 1.01, 1.02, 1.03, 2.0])
    g = np.array([0.10, 0.11, 0.09, 40.0, 0.02])
    qs, gs, _ = cluster_shells(q, g, tol=0.05)
    assert qs[0] == pytest.approx(1.015, abs=0.02)
    assert gs[0] == pytest.approx(0.105, abs=0.02)
    assert qs[-1] == pytest.approx(2.0)
    assert gs[-1] == pytest.approx(0.02)


def test_qmin_larger_than_paper_ripple_peak():
    """N=240 tiene L≈25 Å < 70 Å: no puede hospedar el máximo de Fig. 3."""
    pos, lx, ly = graphene_sheet(10, 6)
    qmin = 2.0 * np.pi / max(lx, ly)
    q_ripple = 2.0 * np.pi / 70.0
    assert qmin > q_ripple
    assert lx < 70.0 and ly < 70.0
