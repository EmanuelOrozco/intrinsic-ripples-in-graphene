"""El vigilante distingue una réplica sana, una caída y una CPU quieta."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ripples.guard import decidir


def test_replica_sana():
    assert decidir(True, None, 2.0, False, 0, 6, 600.0) == "seguir"


def test_cpu_congelada():
    assert decidir(True, None, 900.0, False, 0, 6, 600.0) == "congelada"


def test_congelada_sin_reintentos_restantes():
    assert decidir(True, None, 900.0, False, 6, 6, 600.0) == "fallida"


def test_caida_se_reanuda():
    assert decidir(False, 2, None, False, 1, 6, 600.0) == "reiniciar"


def test_caida_agotada():
    assert decidir(False, 2, None, False, 6, 6, 600.0) == "fallida"


def test_salida_limpia():
    assert decidir(False, 0, None, True, 0, 6, 600.0) == "lista"


def test_salida_cero_sin_sentinel_no_esta_lista():
    assert decidir(False, 0, None, False, 0, 6, 600.0) == "reiniciar"


def test_primer_arranque():
    assert decidir(False, None, None, False, 0, 6, 600.0) == "reiniciar"
