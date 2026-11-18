# Intrinsic ripples in graphene

Atomistic reimplementation of Fasolino, Los and Katsnelson, arXiv:0704.1793, for every Table I sheet up to \(N = 8640\) at \(300\,\mathrm{K}\). Two Monte Carlo kernels share the lattice, the observables, and the run protocol:

- Tersoff (Lindsay–Broido), the finished campaign: `src/c/graphene_mc.c`
- LCBOPII (Los et al., PRB 72, 214102): `src/c/graphene_lcbopii`, built from `lcbopii_mc.c` and `lcbopii_energy.c`

Python holds the lattice, the Tersoff potential used by the tests, the observables, and the figures.

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

- `src/c/graphene_mc.c` is the Tersoff kernel. `src/c/lcbopii_energy.c` is the LCBOPII energy, and `src/c/lcbopii_mc.c` is its Monte Carlo driver. `src/ripples/` is the lattice, the Tersoff potential, and the observables.
- `scripts/pipeline_tabla.py` runs the five Tersoff sizes. `scripts/pipeline_lcbopii.py` runs the same sizes with LCBOPII and writes `results/lcbopii/`. `scripts/pipeline_n8640.py` runs one size; `--potencial` chooses the kernel and defaults to Tersoff.
- `results/n240` … `results/n8640` are the Tersoff runs. `results/lcbopii/n240` … `n4860` are the finished LCBOPII runs. `results/tabla/graficas/` holds the Tersoff figures. `results/lcbopii/tabla/graficas/` and `results/comparacion/graficas/` hold the LCBOPII figures and the comparison.
- `docs/` holds the notes below. `references/` holds the preprint.

## Reports

- `docs/informe_tersoff.pdf` is the Tersoff campaign: the five sizes against Figs. 1, 3, and 5.
- `docs/informe_lcbopii.pdf` is the LCBOPII campaign for \(N = 240, 960, 2160, 4860\).
- `docs/informe_comparacion.pdf` compares the two potentials, the error of each, and which one to keep implementing.
- `docs/informe_codigo.pdf` walks through the simulation and names the function for each step.

Rebuild them with `docs/compilar.sh`.

## Run

From the repository, with the virtualenv that lives next to it:

```bash
../.venv/bin/python scripts/pipeline_tabla.py
```

That uses all 64 hardware threads, \(4\times 10^5\) equilibration steps and \(5\times 10^5\) production steps at \(300\,\mathrm{K}\), and skips a size that already has `results/nN/COMPLETO`. Check it later with `../.venv/bin/python scripts/estado_tabla.py`.

The LCBOPII campaign is the same protocol and does not overwrite the Tersoff results:

```bash
../.venv/bin/python scripts/pipeline_lcbopii.py
```

Check it with `../.venv/bin/python scripts/estado_lcbopii.py`.

## Copyright

The file [`references/Intrinsic_ripples_in_graphene.pdf`](references/Intrinsic_ripples_in_graphene.pdf) is the arXiv preprint of the paper this repository reimplements. It is included so the simulation can be compared with the original figures. Copyright in that work remains with the authors:

A. Fasolino, J. H. Los and M. I. Katsnelson, “Intrinsic ripples in graphene,” arXiv:0704.1793v1 [cond-mat.mtrl-sci] (2007). Published as Nature Materials **6**, 858–861 (2007), https://doi.org/10.1038/nmat2011.

Nothing in this repository transfers that copyright. The Monte Carlo code, the notes in `docs/`, and the figures produced by the runs are separate from the preprint.
