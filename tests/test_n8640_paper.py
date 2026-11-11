"""La corrida larga se contrasta con el paper solo cuando el pipeline la cerró."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from contrastar_paper import informe  # noqa: E402

RUN = ROOT / "results" / "n8640"
DATOS = RUN / "datos"


@pytest.fixture(scope="module")
def info():
    if not (RUN / "COMPLETO").exists():
        pytest.skip("la corrida N=8640 todavía no está cerrada")
    return informe(DATOS)


def test_caja_grande_y_temperatura(info):
    assert info["N"] == 8640
    assert info["T"] == pytest.approx(300.0)
    assert info["caja_mayor_70A"]
    assert info["ventana_q_accesible"]
    assert info["qmin"] < info["q_ripple"]


def test_ondulaciones_y_energia(info):
    assert 0.02 < info["h_rms"] < 8.0
    assert -9.5 < info["E_over_N"] < -4.0
    assert 0.15 < info["acc"] < 0.85


def test_rdf_enlace_conjugado(info):
    assert info["rdf_peak"] == pytest.approx(1.42, abs=0.08)


def test_gq_en_el_orden_de_la_solucion_armonica(info):
    """G0 es la referencia analítica del paper. G puede apartarse, no dispararse."""
    for clave in ("G_sobre_G0_baja_q", "G_sobre_G0_media_q"):
        valor = info[clave]
        assert np.isfinite(valor)
        assert 1e-3 < valor < 50.0


def test_contraste_escrito():
    if not (RUN / "COMPLETO").exists():
        pytest.skip("la corrida N=8640 todavía no está cerrada")
    texto = (RUN / "contraste_paper.txt").read_text(encoding="utf-8")
    assert "G0" in texto
    assert "70" in texto
    assert "1.42" in texto
