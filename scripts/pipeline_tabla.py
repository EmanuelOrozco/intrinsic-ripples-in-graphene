#!/usr/bin/env python3
"""Tabla I del paper hasta N=8640, un tamaño tras otro.

N=8640 se reutiliza si results/n8640/COMPLETO ya existe. El resto usa el mismo
protocolo: T=300 K, 4e5 de equilibrado y 5e5 de producción. Al cerrar todos,
escribe el contraste y las figuras.

    setsid ../.venv/bin/python -u scripts/pipeline_tabla.py \
        >> results/tabla/pipeline.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT.parent / ".venv"
RUN = ROOT / "results" / "tabla"
ESTADO = RUN / "estado.txt"
# 19940 queda fuera: el tope pedido es N=8640.
SIZES = [240, 960, 2160, 4860, 8640]


def exigir_venv() -> None:
    esperado = (VENV / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != esperado:
        sys.exit(f"hay que usar el entorno {esperado}")


def completo(n: int) -> bool:
    return (ROOT / "results" / f"n{n}" / "COMPLETO").exists()


def escribir(fase: str) -> None:
    lineas = [
        f"fase: {fase}",
        f"actualizado: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"pid: {os.getpid()}",
        "Tamaños de la Tabla I con N<=8640: " + ", ".join(str(n) for n in SIZES),
        "",
    ]
    for n in SIZES:
        estado = ROOT / "results" / f"n{n}" / "estado.txt"
        marca = "cerrado" if completo(n) else "pendiente"
        if estado.exists() and not completo(n):
            for linea in estado.read_text(encoding="utf-8").splitlines():
                if linea.startswith("fase:"):
                    marca = linea.split(":", 1)[1].strip()
                if linea.startswith("réplica "):
                    lineas.append(f"  N={n}  {linea}")
        lineas.append(f"N={n}  {marca}")
    lineas.append("")
    lineas.append(f"mirar: {VENV / 'bin' / 'python'} {ROOT / 'scripts' / 'estado_tabla.py'}")
    ESTADO.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    exigir_venv()
    RUN.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for i, n in enumerate(SIZES):
        if completo(n):
            print(f"N={n} ya está cerrado", flush=True)
            escribir(f"N={n} ya cerrado")
            continue
        escribir(f"corriendo N={n}")
        cmd = [py, "-u", str(ROOT / "scripts" / "pipeline_n8640.py"), "--n", str(n)]
        if i:
            cmd.append("--sin-preflight")
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0 or not completo(n):
            escribir(f"falló N={n} código {r.returncode}")
            raise SystemExit(r.returncode or 1)
        escribir(f"cerró N={n}")
    escribir("analizando")
    r = subprocess.run([py, "-u", str(ROOT / "scripts" / "analizar_tabla.py")], cwd=str(ROOT))
    escribir("documentos listos" if r.returncode == 0 else "análisis con diferencias")
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
