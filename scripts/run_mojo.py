#!/usr/bin/env python3
"""Compila el Monte Carlo en Mojo (-O3) y lanza nproc-2 réplicas, una por proceso."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOJO_DIR = ROOT / "src" / "mojo"
DATOS = ROOT / "results" / "datos"
LOGS = ROOT / "results" / "logs"
BIN = MOJO_DIR / "graphene_mc"
PIXI = Path.home() / ".pixi" / "bin" / "pixi"


def n_workers(leave: int = 2) -> int:
    n = os.cpu_count() or 2
    return max(1, n - leave)


def compile_mojo() -> None:
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".pixi" / "bin") + os.pathsep + env.get("PATH", "")
    cmd = [str(PIXI), "run", "mojo", "build", "-O3", "-o", "graphene_mc", "graphene_mc.mojo"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(MOJO_DIR), env=env)


def self_test() -> None:
    subprocess.run([str(BIN), "--self-test"], check=True, cwd=str(MOJO_DIR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=300.0)
    parser.add_argument("--n-eq", type=int, default=400000)
    parser.add_argument("--n-prod", type=int, default=500000)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--leave-cores", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if args.T > 300.0:
        sys.exit("T máxima = 300 K")
    if not args.skip_compile:
        compile_mojo()
    if not args.skip_test:
        self_test()

    DATOS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    workers = n_workers(args.leave_cores)
    nproc = os.cpu_count() or 2
    print(
        f"Mojo MC  N=240  T={args.T}  réplicas={workers}  "
        f"(se dejan {args.leave_cores} de {nproc} núcleos)  "
        f"eq={args.n_eq} prod={args.n_prod} sample_every={args.sample_every}",
        flush=True,
    )

    procs: list[subprocess.Popen] = []
    logs = []
    try:
        for r in range(workers):
            logf = (LOGS / f"replica_{r:02d}.log").open("w", encoding="utf-8")
            cmd = [
                "taskset",
                "-c",
                str(r),
                str(BIN),
                "--T",
                str(args.T),
                "--n-eq",
                str(args.n_eq),
                "--n-prod",
                str(args.n_prod),
                "--sample-every",
                str(args.sample_every),
                "--replica",
                str(r),
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
            p = subprocess.Popen(
                cmd,
                cwd=str(MOJO_DIR),
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append(p)
            logs.append(logf)

        t0 = time.time()
        remaining = set(range(workers))
        while remaining:
            done = []
            for r in remaining:
                rc = procs[r].poll()
                if rc is None:
                    continue
                done.append(r)
                print(
                    f"réplica {r:02d} terminó rc={rc}  t={time.time()-t0:.0f}s",
                    flush=True,
                )
                if rc != 0:
                    print((LOGS / f"replica_{r:02d}.log").read_text(encoding="utf-8")[-1500:])
            remaining -= set(done)
            if remaining:
                time.sleep(5)
        elapsed = time.time() - t0
        print(f"todas las réplicas listo en {elapsed:.1f} s", flush=True)
        if any(p.returncode != 0 for p in procs):
            sys.exit(1)
    except KeyboardInterrupt:
        print("interrompo réplicas", flush=True)
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        raise
    finally:
        for f in logs:
            f.close()


if __name__ == "__main__":
    main()
