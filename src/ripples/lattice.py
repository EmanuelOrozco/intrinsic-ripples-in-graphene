"""Red hexagonal de grafeno de la Tabla I del paper (N = 4 p q)."""

from __future__ import annotations

import math

import numpy as np

A_CC = 1.42  # Å, distancia conjugada C–C del paper
KB_EV = 8.617333262145e-5  # eV / K
KAPPA_300K = 1.1  # eV, rigidez de flexión usada en G0(q) a 300 K


def paper_box(p: int = 10, q: int = 6) -> tuple[float, float, int]:
    """Caja rectangular de la Tabla I: Lx = p |a1|, Ly = q |a1+2a2|, N = 4pq.

    El texto escribe (Lx, Ly) = (2p|a1|, q|a1+2a2|), pero los números de la
    tabla (Lx = 24.59 Å para p = 10) coinciden con Lx = p |a1| y una celda
    rectangular de 4 átomos. Se siguen los valores tabulados.
    """
    a1 = A_CC * math.sqrt(3.0)
    ly_cell = 3.0 * A_CC
    lx = p * a1
    ly = q * ly_cell
    n_atoms = 4 * p * q
    return lx, ly, n_atoms


def graphene_sheet(p: int = 10, q: int = 6) -> tuple[np.ndarray, float, float]:
    """Hoja plana, periódica en x,y. Devuelve (posiciones (N,3), Lx, Ly)."""
    lx, ly, n_atoms = paper_box(p, q)
    a1 = A_CC * math.sqrt(3.0)
    basis = np.array(
        [
            [0.0, 0.0],
            [0.0, A_CC],
            [0.5 * a1, 1.5 * A_CC],
            [0.5 * a1, 2.5 * A_CC],
        ],
        dtype=np.float64,
    )
    pos = np.zeros((n_atoms, 3), dtype=np.float64)
    idx = 0
    for iy in range(q):
        for ix in range(p):
            shift = np.array([ix * a1, iy * 3.0 * A_CC])
            for b in basis:
                pos[idx, 0] = b[0] + shift[0]
                pos[idx, 1] = b[1] + shift[1]
                pos[idx, 2] = 0.0
                idx += 1
    return pos, lx, ly


def nearest_neighbor_distances(pos: np.ndarray, lx: float, ly: float, cutoff: float = 1.6):
    """Distancias mínimas-imagen menores que cutoff (para tests de coordinación)."""
    n = pos.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dx -= lx * np.round(dx / lx)
            dy -= ly * np.round(dy / ly)
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < cutoff:
                dists.append(r)
    return np.array(dists, dtype=np.float64)


def coordination_numbers(pos: np.ndarray, lx: float, ly: float, cutoff: float = 1.6) -> np.ndarray:
    n = pos.shape[0]
    coord = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dx -= lx * np.round(dx / lx)
            dy -= ly * np.round(dy / ly)
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < cutoff:
                coord[i] += 1
    return coord
