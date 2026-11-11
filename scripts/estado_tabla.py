#!/usr/bin/env python3
"""Estado de la Tabla I (N<=8640) sin estar en la sesión de la corrida."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESTADO = ROOT / "results" / "tabla" / "estado.txt"
LOG = ROOT / "results" / "tabla" / "pipeline.log"


def main() -> None:
    if not ESTADO.exists():
        sys.exit(f"todavía no hay corrida de la tabla en {ESTADO.parent}")
    print(ESTADO.read_text(encoding="utf-8"), end="")
    if LOG.exists():
        lineas = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        print("--- últimas líneas ---")
        print("\n".join(lineas[-20:]))


if __name__ == "__main__":
    main()
