# Intrinsic ripples in graphene

Atomistic reimplementation of Fasolino, Los and Katsnelson, arXiv:0704.1793, restricted to \(N = 240\) and \(T \le 300\,\mathrm{K}\). The Monte Carlo core is Mojo; a C/OpenMP backend implements the same protocol. Python holds the lattice, the Tersoff potential used by the tests, the observables, and the plotters.

## What you need on the remote machine

- Python 3.11 or newer
- A C compiler with OpenMP (`gcc`) if you use the C backend
- [Pixi](https://pixi.sh) and a Mojo 1.2 toolchain if you use the default backend
- `tectonic` only if you want to rebuild the notes in `docs/`

The paper PDF is not in this repository. It is arXiv:0704.1793.

## Setup

```bash
git clone https://github.com/EmanuelOrozco/intrinsic-ripples-in-graphene.git
cd intrinsic-ripples-in-graphene
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Checks that do not need a compiled kernel:

```bash
pytest tests/test_lattice.py tests/test_potential.py tests/test_observables.py
```

## Mojo run

From the repository root, with the virtualenv active and `pixi` on `PATH`:

```bash
cd src/mojo
pixi install
cd ../..
python scripts/run_simulation.py
python scripts/generate_plots.py
python scripts/write_result_macros.py
```

`run_simulation.py` compiles `src/mojo/graphene_mc.mojo` at `-O3`, runs the self-test, then starts one process per core minus two (`--leave-cores`). Output lands in `results/`, which is not versioned. A production run uses \(4\times 10^5\) equilibration steps and \(5\times 10^5\) production steps; shorten it with `--n-eq` and `--n-prod` while checking the cluster.

`tests/test_mojo.py` expects the binary `src/mojo/graphene_mc`. `tests/test_results.py` expects the CSV files written by a finished run.

## C/OpenMP run

```bash
python scripts/run_simulation.py --backend c
```

That builds `src/c/graphene_mc` with the Makefile and launches the same replica count.

## Notes

`docs/reimplementacion.tex` describes the protocol. `docs/resultados.tex` reads `docs/resultados_macros.tex`, which `scripts/write_result_macros.py` rewrites from `results/datos/` after a run. Rebuild both PDFs with `docs/compilar.sh` if `tectonic` is installed.
