#!/usr/bin/env python3
"""Resume el estado de la corrida N=8640 sin tener que estar en esa sesión."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "n8640"
ESTADO = RUN / "estado.txt"
PIDFILE = RUN / "pipeline.pid"
LOG = RUN / "logs" / "pipeline.log"


def pid_vivo(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    if not ESTADO.exists():
        sys.exit(f"todavía no hay corrida en {RUN}")
    if PIDFILE.exists():
        try:
            pid = int(PIDFILE.read_text().strip())
        except ValueError:
            pid = -1
        print(f"pipeline pid {pid}: {'vivo' if pid_vivo(pid) else 'no está corriendo'}")
    print(ESTADO.read_text(encoding="utf-8"), end="")
    if LOG.exists():
        lineas = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        print("--- últimas líneas del log ---")
        print("\n".join(lineas[-15:]))


if __name__ == "__main__":
    main()
