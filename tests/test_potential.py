"""El Tersoff C debe pegar grafeno plano y coincidir con el núcleo C."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ripples.lattice import graphene_sheet
from ripples.potential import TersoffCarbon

BIN = ROOT / "src" / "c" / "graphene_mc"


def _run_c(args):
    return subprocess.check_output([str(BIN), *args], text=True)


@pytest.mark.skipif(not BIN.exists(), reason="compilar graphene_mc primero")
def test_c_self_test():
    out = _run_c(["--self-test"])
    assert "PASS" in out
    assert "N=240" in out


@pytest.mark.skipif(not BIN.exists(), reason="compilar graphene_mc primero")
def test_c_energy_matches_python_small_cell():
    """p=q=2 → N=16, lo bastante pequeño para el Tersoff en Python."""
    pos, lx, ly = graphene_sheet(2, 2)
    e_py = TersoffCarbon().energy(pos, lx, ly)
    line = _run_c(["--energy-only", "--p", "2", "--q", "2"]).strip().split()
    e_c = float(line[0])
    assert e_c == pytest.approx(e_py, rel=1e-6, abs=1e-5)


@pytest.mark.skipif(not BIN.exists(), reason="compilar graphene_mc primero")
def test_c_n8640_matches_brute_force():
    out = _run_c(["--self-test-large"])
    assert "PASS" in out
    assert "N=8640" in out
    assert "nn_malos=0" in out


@pytest.mark.skipif(not BIN.exists(), reason="compilar graphene_mc primero")
def test_flat_graphene_is_bound():
    line = _run_c(["--energy-only", "--p", "10", "--q", "6"]).strip().split()
    e, lx, ly, n = map(float, line)
    assert int(n) == 240
    assert e / n < -4.0
    assert e / n > -10.0


def test_python_energy_negative_n16():
    pos, lx, ly = graphene_sheet(2, 2)
    e = TersoffCarbon().energy(pos, lx, ly)
    assert e / 16.0 < -4.0


def test_cutoff_range():
    pot = TersoffCarbon()
    assert pot.cutoff(1.42) == 1.0
    assert pot.cutoff(2.10) == 0.0
    assert 0.0 < pot.cutoff(1.95) < 1.0
