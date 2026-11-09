"""La red N=240 debe reproducir la Tabla I del paper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ripples.lattice import (
    A_CC,
    coordination_numbers,
    graphene_sheet,
    nearest_neighbor_distances,
    paper_box,
)


def test_table_i_n240():
    lx, ly, n = paper_box(10, 6)
    assert n == 240
    assert lx == pytest.approx(24.59, abs=0.02)
    assert ly == pytest.approx(25.56, abs=0.02)


def test_sheet_matches_box():
    pos, lx, ly = graphene_sheet(10, 6)
    assert pos.shape == (240, 3)
    assert np.allclose(pos[:, 2], 0.0)
    assert pos[:, 0].min() >= -1e-9
    assert pos[:, 0].max() < lx + 1e-9


def test_three_fold_coordination():
    pos, lx, ly = graphene_sheet(10, 6)
    coord = coordination_numbers(pos, lx, ly, cutoff=1.60)
    assert np.all(coord == 3)


def test_cc_bond_length():
    pos, lx, ly = graphene_sheet(10, 6)
    d = nearest_neighbor_distances(pos, lx, ly, cutoff=1.60)
    assert d.size == 240 * 3 // 2
    assert np.allclose(d, A_CC, atol=1e-9)


def test_area_per_atom_graphene():
    lx, ly, n = paper_box(10, 6)
    s0 = lx * ly / n
    # celda primitiva de 2 átomos: (√3 a)* (3a)/2 / 2 = (3√3/4) a² ≈ 2.62 Å²
    assert s0 == pytest.approx(2.62, abs=0.03)
