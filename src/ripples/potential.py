"""Tersoff de carbono (Lindsay–Broido 2010), misma clase que LCBOP: bond-order.

El paper usa LCBOPII (Los et al., PRB 72, 214102). Esa parametrización no está
disponible como biblioteca aquí; se reimplementa un potencial de orden de enlace
ajustado a grafeno (Lindsay y Broido, PRB 81, 205441), que conserva coordinación
dependiente de los ángulos y la distancia C–C de 1.42 Å. El protocolo MC, la
red N = 240 y los observables son los del paper.
"""

from __future__ import annotations

import math

import numpy as np


class TersoffCarbon:
    """Parámetros Lindsay–Broido sobre Tersoff 1988/89 para C."""

    A = 1393.6
    B = 430.0
    lambda1 = 3.4879
    lambda2 = 2.2119
    beta = 1.5724e-7
    n = 0.72751
    c = 38049.0
    d = 4.3484
    h = -0.93
    R = 1.80
    S = 2.10

    def __init__(self) -> None:
        self._c2 = self.c * self.c
        self._d2 = self.d * self.d
        self._inv_2n = -0.5 / self.n
        self._beta_n = self.beta**self.n

    def cutoff(self, r: float) -> float:
        if r < self.R:
            return 1.0
        if r >= self.S:
            return 0.0
        x = math.pi * (r - self.R) / (self.S - self.R)
        return 0.5 + 0.5 * math.cos(x)

    def g_theta(self, cos_theta: float) -> float:
        dh = self.h - cos_theta
        return 1.0 + self._c2 / self._d2 - self._c2 / (self._d2 + dh * dh)

    def _pair_vecs(self, pos, i, j, lx, ly):
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        dx -= lx * round(dx / lx)
        dy -= ly * round(dy / ly)
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dx, dy, dz, r

    def energy(self, pos: np.ndarray, lx: float, ly: float) -> float:
        n = pos.shape[0]
        e = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx, dy, dz, rij = self._pair_vecs(pos, i, j, lx, ly)
                fc = self.cutoff(rij)
                if fc == 0.0 or rij < 1e-12:
                    continue
                zeta = 0.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    dxk, dyk, dzk, rik = self._pair_vecs(pos, i, k, lx, ly)
                    fck = self.cutoff(rik)
                    if fck == 0.0 or rik < 1e-12:
                        continue
                    cos_t = (dx * dxk + dy * dyk + dz * dzk) / (rij * rik)
                    cos_t = max(-1.0, min(1.0, cos_t))
                    zeta += fck * self.g_theta(cos_t)
                b = (1.0 + self._beta_n * (zeta**self.n if zeta > 0 else 0.0)) ** self._inv_2n
                fR = self.A * math.exp(-self.lambda1 * rij)
                fA = -self.B * math.exp(-self.lambda2 * rij)
                e += 0.5 * fc * (fR + b * fA)
        return e


def cohesive_energy(pos: np.ndarray, lx: float, ly: float) -> float:
    return TersoffCarbon().energy(pos, lx, ly) / pos.shape[0]
