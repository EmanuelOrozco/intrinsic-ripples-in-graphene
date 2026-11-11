#!/usr/bin/env python3
"""Contrasta los CSV de N=8640 con G0(q) y con las afirmaciones del paper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ripples.lattice import KAPPA_300K  # noqa: E402
from ripples.observables import G0_harmonic, cluster_shells  # noqa: E402

Q_RIPPLE = 2.0 * np.pi / 70.0


def _col(row, name: str) -> float:
    return float(np.atleast_1d(row[name])[0])


def cargar(datos: Path) -> dict:
    summaries = sorted(datos.glob("replica_*_summary.csv"))
    if not summaries:
        raise FileNotFoundError(f"no hay summaries en {datos}")
    rows = [np.genfromtxt(p, delimiter=",", names=True) for p in summaries]
    gq_files = sorted(datos.glob("replica_*_gq.csv"))
    acc: dict[tuple[int, int], list[float]] = {}
    qacc: dict[tuple[int, int], list[float]] = {}
    for f in gq_files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        for nx, ny, q, g in zip(d["nx"], d["ny"], d["q"], d["G_over_N"]):
            key = (int(nx), int(ny))
            acc.setdefault(key, []).append(float(g))
            qacc.setdefault(key, []).append(float(q))
    if acc:
        q = np.array([float(np.mean(qacc[k])) for k in acc])
        g = np.array([float(np.mean(acc[k])) for k in acc])
    else:
        q = np.array([])
        g = np.array([])
    rdf_peak = float("nan")
    rdf = datos / "rdf_mean.csv"
    if not rdf.exists():
        partes = sorted(datos.glob("replica_*_rdf.csv"))
        if partes:
            r = None
            suma = None
            for f in partes:
                d = np.genfromtxt(f, delimiter=",", names=True)
                if r is None:
                    r = d["r"]
                    suma = d["g"].astype(np.float64)
                else:
                    suma = suma + d["g"]
            g_rdf = suma / len(partes)
            ventana = (r > 1.2) & (r < 1.65)
            rdf_peak = float(r[ventana][np.argmax(g_rdf[ventana])])
    else:
        d = np.genfromtxt(rdf, delimiter=",", names=True)
        ventana = (d["r"] > 1.2) & (d["r"] < 1.65)
        rdf_peak = float(d["r"][ventana][np.argmax(d["g"][ventana])])
    return {"rows": rows, "q": q, "g": g, "rdf_peak": rdf_peak}


def hay_maximo(q: np.ndarray, g: np.ndarray, lo: float = 0.06, hi: float = 0.25) -> bool:
    """Máximo interior de G(q) en la ventana donde el paper lo espera."""
    if q.size < 5:
        return False
    qb, gb, _ = cluster_shells(q, g, tol=0.08)
    sel = (qb >= lo) & (qb <= hi)
    qq, gg = qb[sel], gb[sel]
    if qq.size < 3:
        return False
    for i in range(1, qq.size - 1):
        if gg[i] > gg[i - 1] * 1.05 and gg[i] > gg[i + 1] * 1.05:
            return True
    return False


def informe(datos: Path) -> dict:
    paquete = cargar(datos)
    rows = paquete["rows"]
    n = np.array([_col(r, "N") for r in rows])
    t = np.array([_col(r, "T") for r in rows])
    h = np.array([_col(r, "h_rms") for r in rows])
    e = np.array([_col(r, "energy") for r in rows])
    acc = np.array([_col(r, "acc_move") for r in rows])
    lx = np.array([_col(r, "lx") for r in rows])
    ly = np.array([_col(r, "ly") for r in rows])
    s0 = np.array([_col(r, "S0") for r in rows])
    ns = np.array([_col(r, "nsamples") for r in rows])
    q, g = paquete["q"], paquete["g"]
    qmin = float(2.0 * np.pi / max(lx.mean(), ly.mean()))
    s0m = float(s0.mean())
    relacion_baja = float("nan")
    relacion_media = float("nan")
    if q.size:
        orden = np.argsort(q)
        q, g = q[orden], g[orden]
        def ratio(lo: float, hi: float) -> float:
            sel = (q >= lo) & (q <= hi) & (g > 0)
            if not np.any(sel):
                return float("nan")
            g0 = G0_harmonic(q[sel], 1, s0m, T=300.0, kappa=KAPPA_300K)
            return float(np.median(g[sel] / g0))

        relacion_baja = ratio(0.06, 0.20)
        relacion_media = ratio(0.40, 1.20)
    return {
        "n_replicas": len(rows),
        "N": int(n[0]),
        "T": float(t.mean()),
        "h_rms": float(h.mean()),
        "h_rms_std": float(h.std()),
        "E_over_N": float((e / n).mean()),
        "acc": float(acc.mean()),
        "lx": float(lx.mean()),
        "ly": float(ly.mean()),
        "S0": s0m,
        "nsamples": int(ns.sum()),
        "qmin": qmin,
        "q_ripple": float(Q_RIPPLE),
        "caja_mayor_70A": bool(lx.min() > 70.0 and ly.min() > 70.0),
        "ventana_q_accesible": bool(qmin < Q_RIPPLE),
        "rdf_peak": float(paquete["rdf_peak"]),
        "G_sobre_G0_baja_q": relacion_baja,
        "G_sobre_G0_media_q": relacion_media,
        "maximo_cerca_0.1": bool(hay_maximo(q, g)),
        "q": q,
        "g": g,
    }


def redactar(info: dict) -> str:
    pico = "sí" if info["maximo_cerca_0.1"] else "no"
    lineas = [
        "Contraste con Fasolino, Los y Katsnelson, arXiv:0704.1793",
        "",
        f"Réplicas {info['n_replicas']}   N={info['N']}   T={info['T']:.0f} K",
        f"Caja {info['lx']:.2f} × {info['ly']:.2f} Å   S0={info['S0']:.3f} Å²",
        f"q_min={info['qmin']:.4f} Å⁻¹   q(70 Å)={info['q_ripple']:.4f} Å⁻¹",
        f"La caja supera 70 Å: {info['caja_mayor_70A']}.",
        f"La ventana q≈0.1 Å⁻¹ es accesible: {info['ventana_q_accesible']}.",
        f"h_rms = {info['h_rms']:.4f} ± {info['h_rms_std']:.4f} Å",
        f"E/N = {info['E_over_N']:.4f} eV/átomo",
        f"Aceptación de movimientos = {100 * info['acc']:.1f} %",
        f"Pico de g(r) entre 1.2 y 1.65 Å = {info['rdf_peak']:.3f} Å (el paper marca 1.42 Å a 300 K).",
        f"Mediana G/G0 en q=0.06–0.20 = {_fmt(info['G_sobre_G0_baja_q'])}",
        f"Mediana G/G0 en q=0.40–1.20 = {_fmt(info['G_sobre_G0_media_q'])}",
        f"Máximo interior de G(q) cerca de 0.1 Å⁻¹: {pico}.",
        "",
        "G0(q)/N = T / (κ S0 q²), con κ = 1.1 eV, es la referencia armónica de las Figs. 3 y 4.",
        "El paper dice que las muestras más cortas que ~70 Å no muestran el máximo de G(q)",
        "en q≈0.1 Å⁻¹, y que las más largas sí. Ese máximo lo atribuyen al canal de",
        "conjugación de LCBOPII. Aquí el potencial es Tersoff de Lindsay–Broido, de la",
        "misma clase (orden de enlace) pero sin ese canal, así que el máximo puede no aparecer.",
        "La prueba de cordura es que G(q) quede en el orden de G0, con ondulaciones, g(r)",
        "centrado en el enlace conjugado y la caja en el régimen de la figura grande.",
    ]
    return "\n".join(lineas) + "\n"


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "no disponible"
    return f"{x:.3f}"


def main() -> None:
    datos = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "n8640" / "datos"
    info = informe(datos)
    texto = redactar(info)
    out = datos.parent / "contraste_paper.txt"
    out.write_text(texto, encoding="utf-8")
    print(texto, end="")
    print("escrito", out)


if __name__ == "__main__":
    main()
