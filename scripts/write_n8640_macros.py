#!/usr/bin/env python3
"""Escribe docs/resultados_n8640_macros.tex desde results/n8640/datos."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from contrastar_paper import informe  # noqa: E402

DATOS = ROOT / "results" / "n8640" / "datos"
OUT = ROOT / "docs" / "resultados_n8640_macros.tex"
H_TIPICO_PAPER = 0.6


def f(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def frase_maximo(hay: bool) -> str:
    if hay:
        return (
            r"En la ventana $q\in[0.06,0.25]$~\AA$^{-1}$ el promedio de $G(q)$ tiene un m\'aximo "
            r"ancho cerca de $q\approx 0.06$--$0.09$~\AA$^{-1}$, el lugar donde el paper lo "
            r"espera para muestras m\'as largas que $70$~\AA. La dispersi\'on entre las dos "
            r"r\'eplicas es grande a ese $q$, as\'i que no se identifica con el pico agudo "
            r"que el paper atribuye al canal de conjugaci\'on de LCBOPII."
        )
    return (
        r"En la ventana $q\in[0.06,0.25]$~\AA$^{-1}$ no hay un m\'aximo interior de $G(q)$. "
        r"El paper lo ve en muestras m\'as largas que $\sim 70$~\AA\ y lo atribuye al canal "
        r"de conjugaci\'on de LCBOPII. Este potencial es Tersoff de Lindsay--Broido, sin ese "
        r"canal, as\'i que la ausencia no invalida el protocolo: la caja s\'i entra en el "
        r"r\'egimen de la figura grande ($q_{\min}<2\pi/70$)."
    )


def frase_h(h_rms: float) -> str:
    razon = h_rms / H_TIPICO_PAPER
    if 0.3 <= razon <= 3.0:
        cierre = "Queda en el mismo orden de magnitud."
    elif razon < 0.3:
        cierre = "Queda por debajo de ese orden de magnitud."
    else:
        cierre = "Queda por encima de ese orden de magnitud."
    return (
        rf"El paper describe, para $N=8640$, una altura t\'ipica $\sim {H_TIPICO_PAPER:.1f}$~\AA\ "
        rf"(Fig.~1). Aqu\'i $h_{{\mathrm{{rms}}}}={f(h_rms)}$~\AA, que es $\sqrt{{\langle h^2\rangle}}$ "
        rf"y no el mismo estimador visual. {cierre}"
    )


def main() -> None:
    if not (DATOS.parent / "COMPLETO").exists():
        sys.exit("la corrida N=8640 todavía no tiene results/n8640/COMPLETO")
    info = informe(DATOS)
    rows = [np.genfromtxt(p, delimiter=",", names=True) for p in sorted(DATOS.glob("replica_*_summary.csv"))]
    heq = np.array([float(np.atleast_1d(r["h_rms_eq"])[0]) for r in rows])
    acca = np.array([float(np.atleast_1d(r["acc_area"])[0]) for r in rows])
    lines = [
        "% macros generadas; no editar a mano",
        rf"\newcommand{{\Nrep}}{{{info['n_replicas']}}}",
        rf"\newcommand{{\HrmsMean}}{{{f(info['h_rms'])}}}",
        rf"\newcommand{{\HrmsStd}}{{{f(info['h_rms_std'])}}}",
        rf"\newcommand{{\HeqMean}}{{{f(float(heq.mean()))}}}",
        rf"\newcommand{{\EoverN}}{{{f(info['E_over_N'], 3)}}}",
        rf"\newcommand{{\AccMove}}{{{f(100 * info['acc'], 1)}}}",
        rf"\newcommand{{\AccArea}}{{{f(100 * float(acca.mean()), 1)}}}",
        rf"\newcommand{{\LxMean}}{{{f(info['lx'], 2)}}}",
        rf"\newcommand{{\LyMean}}{{{f(info['ly'], 2)}}}",
        rf"\newcommand{{\Szero}}{{{f(info['S0'], 3)}}}",
        rf"\newcommand{{\NsampTot}}{{{info['nsamples']}}}",
        rf"\newcommand{{\RdfPeak}}{{{f(info['rdf_peak'], 3)}}}",
        rf"\newcommand{{\Qmin}}{{{f(info['qmin'], 3)}}}",
        rf"\newcommand{{\Qripple}}{{{f(info['q_ripple'], 3)}}}",
        rf"\newcommand{{\GqBaja}}{{{f(info['G_sobre_G0_baja_q'], 2)}}}",
        rf"\newcommand{{\GqMedia}}{{{f(info['G_sobre_G0_media_q'], 2)}}}",
        r"\newcommand{\MaximoFrase}{" + frase_maximo(bool(info["maximo_cerca_0.1"])) + "}",
        r"\newcommand{\HrmsFrase}{" + frase_h(float(info["h_rms"])) + "}",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("escrito", OUT)


if __name__ == "__main__":
    main()
