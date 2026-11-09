#!/usr/bin/env python3
"""Lanza el Monte Carlo. Por defecto usa el núcleo Mojo (nproc-2 procesos)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "src" / "c"
DATOS = ROOT / "results" / "datos"
BIN_C = C_DIR / "graphene_mc"
RUN_MOJO = ROOT / "scripts" / "run_mojo.py"


def n_workers(leave: int = 2) -> int:
    n = os.cpu_count() or 2
    return max(1, n - leave)


def compile_c() -> None:
    subprocess.run(["make", "-C", str(C_DIR), "all"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=300.0)
    parser.add_argument("--n-eq", type=int, default=400000)
    parser.add_argument("--n-prod", type=int, default=500000)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="intervalo de muestreo en producción (Mojo). En C, 0 = según RAM",
    )
    parser.add_argument("--leave-cores", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("mojo", "c"),
        default="mojo",
        help="núcleo: mojo (defecto) o c/OpenMP",
    )
    args, rest = parser.parse_known_args()
    if args.T > 300.0:
        sys.exit("T máxima = 300 K")

    if args.backend == "mojo":
        cmd = [
            sys.executable,
            str(RUN_MOJO),
            "--T",
            str(args.T),
            "--n-eq",
            str(args.n_eq),
            "--n-prod",
            str(args.n_prod),
            "--sample-every",
            str(args.sample_every if args.sample_every > 0 else 10),
            "--leave-cores",
            str(args.leave_cores),
            "--seed",
            str(args.seed),
        ]
        if args.skip_compile:
            cmd.append("--skip-compile")
        cmd.extend(rest)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        return

    if not args.skip_compile:
        compile_c()
    DATOS.mkdir(parents=True, exist_ok=True)
    workers = n_workers(args.leave_cores)
    cmd = [
        str(BIN_C),
        "--T",
        str(args.T),
        "--n-eq",
        str(args.n_eq),
        "--n-prod",
        str(args.n_prod),
        "--sample-every",
        str(args.sample_every),
        "--leave-cores",
        str(args.leave_cores),
        "--replicas",
        str(workers),
        "--seed",
        str(args.seed),
        "--out",
        str(DATOS),
        "--p",
        "10",
        "--q",
        "6",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
