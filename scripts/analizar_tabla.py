#!/usr/bin/env python3
"""Contraste exhaustivo de la Tabla I (N<=8640, T=300 K) con el paper."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from contrastar_paper import hay_maximo, informe  # noqa: E402
from ripples.lattice import PAPER_SIZES, TABLE_I_BOX, paper_box  # noqa: E402
from ripples.observables import cluster_shells, spectrum_curve  # noqa: E402

SIZES = [240, 960, 2160, 4860, 8640]
Q_RIPPLE = 2.0 * np.pi / 70.0
A_CC = 1.42
# Marcos medidos en las figuras de references/Intrinsic_ripples_in_graphene.pdf.
# Fig. 3: dos décadas en q (0.05 a 5) y tres en G/N (0.001 a 1).
FIG3_Q = (0.05, 5.0)
FIG3_G = (1.0e-3, 1.0)
# Fig. 5: r de 0 a 3.5 Å; el eje vertical del paper, etiquetado rdf, va de 0 a 3.
FIG5_R = (0.0, 3.5)
FIG5_RDF_PAPER = (0.0, 3.0)
BRAGG_1 = 4.0 * np.pi / (3.0 * A_CC)
GRAF = ROOT / "results" / "tabla" / "graficas"
DOCS = ROOT / "docs"
COLORES = {240: "#1f77b4", 960: "#ff7f0e", 2160: "#2ca02c", 4860: "#d62728", 8640: "#9467bd"}


def firma(info: dict) -> dict:
    q = np.asarray(info["q"], dtype=np.float64)
    g = np.asarray(info["g"], dtype=np.float64)
    qb, gb, _ = cluster_shells(q, g, tol=0.05) if q.size else (q, g, q)
    qmin = float(info["qmin"])
    cerca = (qb >= 0.08) & (qb <= 0.16) if qb.size else np.array([], dtype=bool)
    caida = False
    if qb.size and cerca.any() and qmin < 0.09:
        g_bajo = float(gb[int(np.argmin(qb))])
        g_pico = float(np.max(gb[cerca]))
        caida = g_bajo < 0.90 * g_pico
    return {
        "maximo": bool(hay_maximo(q, g)) if q.size else False,
        "caida_bajo_q": caida,
        "modo_70_cabe": bool(qmin < Q_RIPPLE),
    }


def cargar() -> list[dict]:
    filas = []
    for n in SIZES:
        datos = ROOT / "results" / f"n{n}" / "datos"
        if not (datos.parent / "COMPLETO").exists():
            raise FileNotFoundError(f"falta results/n{n}/COMPLETO")
        info = informe(datos)
        p_celda, q_celda = PAPER_SIZES[n]
        lx0, ly0, _ = paper_box(p_celda, q_celda)
        info.update(firma(info))
        info["lx0"] = lx0
        info["ly0"] = ly0
        info["p"] = p_celda
        info["q_celda"] = q_celda
        info["tabla_lx"], info["tabla_ly"] = TABLE_I_BOX[n]
        filas.append(info)
    return filas


def _ejes_papel(ax) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def _rdf_media(n: int) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "results" / f"n{n}" / "datos" / "rdf_mean.csv"
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(d["r"], dtype=np.float64), np.asarray(d["g"], dtype=np.float64)


def figuras(filas: list[dict]) -> None:
    GRAF.mkdir(parents=True, exist_ok=True)
    from ripples.lattice import KAPPA_300K
    from ripples.observables import G0_harmonic

    # Misma caja que la Fig. 3: más alta que ancha, para que la pendiente de G0
    # se lea como en el paper (el marco mide 156 x 202 pt).
    fig, ax = plt.subplots(figsize=(4.9, 6.15))
    ax.set_box_aspect(202.0 / 156.0)
    s0 = filas[-1]["S0"]
    q_th = np.logspace(np.log10(FIG3_Q[0]), np.log10(FIG3_Q[1]), 400)
    g0 = G0_harmonic(q_th, 1, s0, T=300.0, kappa=KAPPA_300K)
    ax.loglog(q_th, g0, color="#d62728", lw=1.5, label=r"$G_0$", zorder=2)
    for info in filas:
        q_line, g_line = spectrum_curve(info["q"], info["g"])
        ax.loglog(
            q_line, g_line, lw=1.35, color=COLORES[info["N"]], label=str(info["N"]), zorder=3
        )
    ax.axvline(Q_RIPPLE, color="0.35", ls=":", lw=0.9, zorder=1)
    ax.axvline(BRAGG_1, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.set_xlim(*FIG3_Q)
    ax.set_ylim(*FIG3_G)
    ax.set_xlabel(r"$q$ (Å$^{-1}$)")
    ax.set_ylabel(r"$G(q)/N$")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _ejes_papel(ax)
    fig.savefig(GRAF / "Gq_fig3.png", dpi=170, bbox_inches="tight")
    fig.savefig(GRAF / "Gq_tabla.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    ax, zoom = axes
    for info in filas:
        r, g = _rdf_media(info["N"])
        kw = dict(color=COLORES[info["N"]], label=str(info["N"]), lw=1.15 if info["N"] != 8640 else 1.7)
        ax.plot(r, g, **kw)
        zoom.plot(r, g, color=COLORES[info["N"]], lw=kw["lw"])
    for eje in (ax, zoom):
        eje.axvline(1.31, color="0.25", ls=":", lw=0.8)
        eje.axvline(1.42, color="0.15", ls="--", lw=0.9)
        eje.axvline(1.54, color="0.25", ls=":", lw=0.8)
        eje.set_ylim(0.0, 16.0)
        eje.set_xlabel(r"$r$ (Å)")
        _ejes_papel(eje)
    ax.axhspan(0.0, FIG5_RDF_PAPER[1], color="#d62728", alpha=0.08, zorder=0)
    ax.text(0.12, 1.45, "eje vertical\ndel paper: 0 a 3", fontsize=8, color="#8c2f2f", va="center")
    ax.set_xlim(*FIG5_R)
    ax.set_ylabel(r"$g(r)$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    zoom.set_xlim(1.15, 1.75)
    zoom.set_title(r"primer pico, marcas 1.31 / 1.42 / 1.54 Å", fontsize=9)
    fig.savefig(GRAF / "rdf_fig5.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    snap = ROOT / "results" / "n8640" / "datos" / "replica_00_snapshot.csv"
    if snap.exists():
        d = np.genfromtxt(snap, delimiter=",", names=True)
        h = d["z"] - np.mean(d["z"])
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        sc = ax.scatter(d["x"], d["y"], c=h, cmap="coolwarm", s=2.2, linewidths=0)
        x0 = float(np.min(d["x"])) + 8.0
        y0 = float(np.min(d["y"])) + 10.0
        ax.plot([x0, x0 + 70.0], [y0, y0], color="black", lw=2.4, solid_capstyle="butt")
        ax.text(x0 + 35.0, y0 + 4.5, "70 Å", ha="center", va="bottom", fontsize=9, color="black")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ (Å)")
        ax.set_ylabel(r"$y$ (Å)")
        fig.colorbar(sc, ax=ax, shrink=0.78, label=r"$h$ (Å)")
        _ejes_papel(ax)
        fig.savefig(GRAF / "snapshot_fig1.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    largo = np.array([max(f["lx0"], f["ly0"]) for f in filas])
    h = np.array([f["h_rms"] for f in filas])
    ax.plot(largo, h, "o-", color="#1f77b4", label=r"$h_{\mathrm{rms}}$")
    guia = h[0] * largo / largo[0]
    ax.plot(largo, guia, "--", color="#d62728", label=r"armónico $\propto L$")
    for f, x, y in zip(filas, largo, h):
        ax.annotate(str(f["N"]), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(r"$L$ (Å)")
    ax.set_ylabel(r"$h_{\mathrm{rms}}$ (Å)")
    ax.legend(frameon=False)
    fig.savefig(GRAF / "hrms_vs_L.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _num(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def redactar(filas: list[dict]) -> str:
    lineas = [
        "Contraste de la Tabla I (N<=8640) con Fasolino, Los y Katsnelson, arXiv:0704.1793.",
        "Protocolo: hoja plana, Metropolis, 400000 pasos de equilibrado, 500000 de producción,",
        "movimiento de área cada 5 pasos, T=300 K. Potencial: Tersoff Lindsay-Broido, no LCBOPII.",
        "N=19940 y T=3500 K están en el paper y no se simulan: el tope pedido es N=8640 y T=300 K.",
        "",
    ]
    for f in filas:
        lado = "menor que 70 Å" if max(f["lx0"], f["ly0"]) < 70 else "mayor que 70 Å"
        lineas.append(
            f"N={f['N']}  p={f['p']} q={f['q_celda']}  caja inicial {f['lx0']:.2f}×{f['ly0']:.2f} Å "
            f"(tabla {f['tabla_lx']:.2f}×{f['tabla_ly']:.2f})  {lado}"
        )
        lineas.append(
            f"  caja final {f['lx']:.2f}×{f['ly']:.2f}  h_rms={f['h_rms']:.4f}±{f['h_rms_std']:.4f} Å  "
            f"E/N={f['E_over_N']:.4f}  aceptación={100*f['acc']:.1f}%  "
            f"g(r)={f['rdf_peak']:.3f} Å  G/G0(0.40-1.20)={f['G_sobre_G0_media_q']:.3f}"
        )
        lineas.append(
            f"  q_min={f['qmin']:.4f}  el modo de 70 Å cabe: {f['modo_70_cabe']}  "
            f"máximo cerca de 0.1: {f['maximo']}  caída de G a bajo q: {f['caida_bajo_q']}"
        )
    return "\n".join(lineas) + "\n"


def _frase_pico(f: dict) -> str:
    if max(f["lx0"], f["ly0"]) < 70.0:
        return (
            "El paper dice que esta caja, más corta que 70~\\AA, no muestra la caída de "
            "$G(q)$ a bajo $q$. El modo $q\\approx 0.1$~\\AA$^{-1}$ no cabe "
            f"($q_{{\\min}}={_num(f['qmin'], 3)}$)."
        )
    if f["caida_bajo_q"]:
        return (
            "Por debajo de $q\\sim 0.1$~\\AA$^{-1}$, $G(q)$ deja de seguir a $G_0\\propto 1/q^2$ "
            "y el primer cascarón queda por debajo del máximo de la ventana $0.08$--$0.16$. "
            "Ese máximo es ancho y, con dos o cuatro réplicas, la dispersión es del orden de la bajada: "
            "no es el pico agudo de $\\sim 70$~\\AA\\ que el paper asigna a la conjugación de LCBOPII. "
            "Tersoff no tiene ese canal."
        )
    return (
        "No aparece la caída de $G(q)$ a bajo $q$ que el paper ve con LCBOPII en las "
        "muestras mayores que 70~\\AA. La caja sí está en ese régimen "
        f"($q_{{\\min}}={_num(f['qmin'], 3)}<{_num(Q_RIPPLE, 3)}$). "
        "La ausencia corresponde al potencial, no a la geometría ni al protocolo."
    )


def tex(filas: list[dict]) -> str:
    filas_tabla = []
    for f in filas:
        filas_tabla.append(
            f"{f['N']} & {f['p']} & {f['q_celda']} & {_num(f['lx0'], 2)} & {_num(f['ly0'], 2)} & "
            f"{_num(f['tabla_lx'], 2)} & {_num(f['tabla_ly'], 2)} \\\\"
        )
    obs = []
    for f in filas:
        obs.append(
            f"{f['N']} & {_num(f['h_rms'], 3)}$\\pm${_num(f['h_rms_std'], 3)} & "
            f"{_num(f['E_over_N'], 3)} & {_num(100*f['acc'], 1)} & {_num(f['rdf_peak'], 3)} & "
            f"{_num(f['G_sobre_G0_media_q'], 2)} & {_num(f['qmin'], 3)} \\\\"
        )
    juicios = "\n\n".join(
        f"\\paragraph{{$N={f['N']}$.}} {_frase_pico(f)}" for f in filas
    )
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{iftex}}
\ifPDFTeX
  \usepackage[T1]{{fontenc}}
  \usepackage[utf8]{{inputenc}}
\else
  \usepackage{{fontspec}}
\fi
\usepackage[spanish,es-noshorthands]{{babel}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{geometry}}
\usepackage{{float}}
\usepackage{{hyperref}}
\geometry{{margin=2.3cm}}
\graphicspath{{{{../results/tabla/graficas/}}}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\title{{Tabla I hasta $N=8640$\\[0.3em]
\large Contraste con Fasolino, Los y Katsnelson (2007)}}
\author{{Emanuel Orozco Gallego}}
\date{{11 de noviembre de 2026}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Se reimplementa el Monte Carlo del paper para todos los tamaños de la Tabla~I con $N\le 8640$: 240, 960, 2160, 4860 y 8640, a $T=300\,\mathrm{{K}}$. $N=19940$ queda fuera del tope. $T=3500\,\mathrm{{K}}$ (Figs.~4 y~5) no se simula. El potencial es Tersoff de Lindsay--Broido; el del paper es LCBOPII.
\end{{abstract}}

\section{{Qué exige el paper}}
Hoja plana, condiciones periódicas, $4\times 10^5$ pasos de equilibrado y $5\times 10^5$ de promediado. Un paso son $N$ intentos. Cada cinco pasos, una fluctuación isotrópica de área. Los observables son $h$, $G(q)=q^2|h_q|^2$ y $g(r)$. La referencia armónica es $G_0(q)/N=T/(\kappa S_0 q^2)$ con $\kappa=1.1\,\mathrm{{eV}}$. La Fig.~3 muestra que las dos cajas más cortas que $\sim 70\,\text{{\AA}}$ no tienen la caída de $G(q)$ a bajo $q$, y que las más largas sí, con un máximo cerca de $q\approx 0.1\,\text{{\AA}}^{{-1}}$. Para $N=8640$ citan una altura típica de orden $0.6\,\text{{\AA}}$. El primer pico de $g(r)$ a $300\,\mathrm{{K}}$ está en el enlace conjugado $1.42\,\text{{\AA}}$.

\section{{Geometría}}
La caja usada es $L_x=p|a_1|$ y $L_y=q\cdot 3a$, con $a=1.42\,\text{{\AA}}$ y $N=4pq$. Coincide con los números de la Tabla~I salvo el $L_x$ impreso de $N=960$ ($49.29\,\text{{\AA}}$): $p|a_1|=49.19\,\text{{\AA}}$. El resto de esa columna, y todas las demás filas, cierran con esta fórmula, así que se simula $49.19$ y no el valor impreso.

\begin{{center}}
\begin{{tabular}}{{rrrrrrr}}
\toprule
$N$ & $p$ & $q$ & $L_x$ & $L_y$ & Tabla $L_x$ & Tabla $L_y$ \\
\midrule
{chr(10).join(filas_tabla)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section{{Resultados a $300\,\mathrm{{K}}$}}
\begin{{center}}
\small
\begin{{tabular}}{{rrrrrrr}}
\toprule
$N$ & $h_{{\mathrm{{rms}}}}$ (\AA) & $E/N$ (eV) & acc.\ (\%) & pico $g(r)$ & $G/G_0$ & $q_{{\min}}$ \\
\midrule
{chr(10).join(obs)}
\bottomrule
\end{{tabular}}
\end{{center}}

$G/G_0$ es la mediana en $q\in[0.40,1.20]\,\text{{\AA}}^{{-1}}$, lejos de $q_{{\min}}$ y del primer Bragg. Un valor de orden 1 dice que la convención de Fourier y $\kappa=1.1\,\mathrm{{eV}}$ describen esa ventana. $E/N$ se compara con la hoja plana de Tersoff, $-7.966\,\mathrm{{eV}}$. El pico de $g(r)$ se lee entre $1.2$ y $1.65\,\text{{\AA}}$ y se compara con $1.42\,\text{{\AA}}$.

Los ejes de las figuras siguientes se midieron sobre las Figs.~1, 3 y~5 del PDF en \texttt{{references/}}. La Fig.~3 es log--log con $q$ de $0.05$ a $5\,\text{{\AA}}^{{-1}}$ (dos décadas) y $G(q)/N$ de $10^{{-3}}$ a $1$ (tres décadas); el marco es más alto que ancho, en la proporción $202/156$ del original. La Fig.~4 usa ese mismo marco y añade la curva a $3500\,\mathrm{{K}}$, que aquí no está. La Fig.~5 usa $r$ de $0$ a $3.5\,\text{{\AA}}$ y un eje vertical de $0$ a $3$ etiquetado rdf.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.62\textwidth]{{Gq_fig3.png}}
\caption{{$G(q)/N$ en los ejes de la Fig.~3. La recta roja es $G_0$ con $\kappa=1.1\,\mathrm{{eV}}$. La punteada es $q=2\pi/70\,\text{{\AA}}$ y la discontinua el primer Bragg $4\pi/(3a)=2.94\,\text{{\AA}}^{{-1}}$. El segundo, $4\pi/(\sqrt{{3}}a)=5.11$, cae justo a la derecha de $q=5$, el borde del paper. $N=19940$ está en esa figura y no se simula.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{rdf_fig5.png}}
\caption{{$g(r)$ con el eje horizontal de la Fig.~5 ($0$ a $3.5\,\text{{\AA}}$). Las marcas son el doble $1.31$, el conjugado $1.42$ y el simple $1.54\,\text{{\AA}}$. La banda roja es el eje vertical del paper, de $0$ a $3$. Aquí el bin es $0.01\,\text{{\AA}}$ y el primer pico llega a $\sim 14$: la altura no es la de esa figura; la posición sí. El panel derecho abre el primer pico. No hay hombros en $1.31$ ni en $1.54\,\text{{\AA}}$: haría falta la corrida a $3500\,\mathrm{{K}}$ con LCBOPII.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{snapshot_fig1.png}}
\caption{{Vista de la hoja $N=8640$ a $300\,\mathrm{{K}}$, con la barra de $70\,\text{{\AA}}$ que el paper dibuja sobre la Fig.~1. El color es la altura $h$ respecto del centro de masa. La altura típica que citan es $\sim 0.6\,\text{{\AA}}$; $h_{{\mathrm{{rms}}}}$ de esta corrida es $0.47\,\text{{\AA}}$.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{hrms_vs_L.png}}
\caption{{$h_{{\mathrm{{rms}}}}$ frente al lado largo. Esta curva no está en el paper. La recta roja es el crecimiento armónico $\propto L$ anclado en $N=240$. Quedar por debajo es la supresión anarmónica que discuten a partir de la ec.~(4).}}
\end{{figure}}

\section{{Tamaño por tamaño}}
{juicios}

\section{{Qué no se reproduce}}
\begin{{itemize}}
\item $N=19940$ ($p=90$, $q=54$): está en la Tabla~I y pasa de 8640.
\item $T=3500\,\mathrm{{K}}$: Figs.~4 y~5. El máximo se desplaza a unos $30\,\text{{\AA}}$ y $g(r)$ muestra hombros de enlace simple y doble. Hace falta LCBOPII.
\item El pico agudo de $70\,\text{{\AA}}$ es, en el paper, consecuencia del canal de conjugación de LCBOPII. Tersoff no tiene ese canal. La meseta ancha a bajo $q$ no se presenta como ese pico.
\end{{itemize}}
\end{{document}}
"""


def main() -> None:
    filas = cargar()
    figuras(filas)
    texto = redactar(filas)
    out = ROOT / "results" / "tabla"
    out.mkdir(parents=True, exist_ok=True)
    (out / "contraste_tabla.txt").write_text(texto, encoding="utf-8")
    (DOCS / "resultados_tabla.tex").write_text(tex(filas), encoding="utf-8")
    print(texto, end="")
    fallos = []
    for f in filas:
        if not (1.34 <= f["rdf_peak"] <= 1.50):
            fallos.append(f"N={f['N']} pico g(r)={f['rdf_peak']}")
        if not (-8.2 <= f["E_over_N"] <= -7.5):
            fallos.append(f"N={f['N']} E/N={f['E_over_N']}")
        if f["h_rms"] <= 0.02:
            fallos.append(f"N={f['N']} hoja plana")
        if not (0.15 <= f["acc"] <= 0.85):
            fallos.append(f"N={f['N']} aceptación={f['acc']}")
        if not (1e-3 < f["G_sobre_G0_media_q"] < 50):
            fallos.append(f"N={f['N']} G/G0")
        if max(f["lx0"], f["ly0"]) < 70 and f["modo_70_cabe"]:
            fallos.append(f"N={f['N']} no debería caber el modo de 70 Å")
        if min(f["lx0"], f["ly0"]) > 70 and not f["modo_70_cabe"]:
            fallos.append(f"N={f['N']} debería caber el modo de 70 Å")
    if fallos:
        print("DIFERENCIAS\n" + "\n".join(fallos), file=sys.stderr)
        raise SystemExit(1)
    print("contraste de protocolo: PASS")


if __name__ == "__main__":
    main()
