# Intrinsic ripples in graphene

Atomistic reimplementation of Fasolino, Los and Katsnelson, arXiv:0704.1793, for every Table I sheet up to \(N = 8640\) at \(300\,\mathrm{K}\). The Monte Carlo kernel is C with OpenMP (`src/c/graphene_mc.c`). Python holds the lattice, the Tersoff potential used by the tests, the observables, and the figures.

## What you need

- Python 3.11 or newer
- A C compiler with OpenMP (`gcc`)
- `tectonic` only if you want to rebuild the notes in `docs/`

The preprint this reimplementation follows is [`references/Intrinsic_ripples_in_graphene.pdf`](references/Intrinsic_ripples_in_graphene.pdf). Copyright in that file stays with its authors; see [Copyright](#copyright).

## Setup

```bash
git clone https://github.com/EmanuelOrozco/intrinsic-ripples-in-graphene.git
cd intrinsic-ripples-in-graphene
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
pytest
```

## Layout

- `src/c/` is the Monte Carlo kernel. `src/ripples/` is the lattice, the potential, and the observables.
- `scripts/pipeline_tabla.py` runs the five sizes one after another. `scripts/pipeline_n8640.py` runs a single size. `scripts/analizar_tabla.py` draws the comparison.
- `results/n240`, `results/n960`, `results/n2160`, `results/n4860`, and `results/n8640` hold one finished size each. `results/tabla/graficas/` holds the figures shared by the notes.
- `docs/` holds the three notes below. `references/` holds the preprint.

## Reports

- `docs/informe_reimplementacion.pdf` is the report: the five sizes against Figs. 1, 3, and 5, and why each choice was made.
- `docs/informe_codigo.pdf` walks through the simulation and names the function for each step.

Rebuild them with `docs/compilar.sh`.

## Run

From the repository, with the virtualenv that lives next to it:

```bash
../.venv/bin/python scripts/pipeline_tabla.py
```

That uses all 64 hardware threads, \(4\times 10^5\) equilibration steps and \(5\times 10^5\) production steps at \(300\,\mathrm{K}\), and skips a size that already has `results/nN/COMPLETO`. Check it later with `../.venv/bin/python scripts/estado_tabla.py`.

## Copyright

The file [`references/Intrinsic_ripples_in_graphene.pdf`](references/Intrinsic_ripples_in_graphene.pdf) is the arXiv preprint of the paper this repository reimplements. It is included so the simulation can be compared with the original figures. Copyright in that work remains with the authors:

A. Fasolino, J. H. Los and M. I. Katsnelson, “Intrinsic ripples in graphene,” arXiv:0704.1793v1 [cond-mat.mtrl-sci] (2007). Published as Nature Materials **6**, 858–861 (2007), https://doi.org/10.1038/nmat2011.

Nothing in this repository transfers that copyright. The Monte Carlo code, the notes in `docs/`, and the figures produced by the runs are separate from the preprint.
