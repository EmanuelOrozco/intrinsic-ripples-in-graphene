"""Corrobora el binario Mojo: geometría Tabla I y consistencia ΔE local."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "src" / "mojo" / "graphene_mc"


def test_mojo_binary_exists():
    assert BIN.is_file(), "falta src/mojo/graphene_mc; compilar con pixi run mojo build -O3"


def test_mojo_self_test():
    p = subprocess.run(
        [str(BIN), "--self-test"],
        cwd=str(BIN.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    out = p.stdout
    assert "self-test PASS" in out
    assert "N= 240" in out
    assert "E/N=" in out
    # hoja plana Lindsay–Broido
    energy_line = [ln for ln in out.splitlines() if "E/N=" in ln][0]
    en = float(energy_line.split("E/N=")[1].strip().split()[0])
    assert en < -4.0
    assert en > -10.0
    diff_line = [ln for ln in out.splitlines() if "|diff|=" in ln][0]
    err = float(diff_line.split("|diff|=")[1].strip().split()[0])
    assert err < 1e-6
