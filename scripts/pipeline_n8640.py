#!/usr/bin/env python3
"""Pipeline N=8640: pruebas, réplicas en paralelo, vigilancia y contraste con el paper.

Se lanza con el Python del entorno y queda despegado de la terminal:

    setsid /home/eavelas/magnetic_graphene/.venv/bin/python -u scripts/pipeline_n8640.py \\
        >> results/n8640/logs/pipeline.log 2>&1 < /dev/null &

El estado se lee con scripts/estado_n8640.py. Si una réplica muere o su heartbeat
deja de avanzar, se reanuda desde el último checkpoint. No hace falta sudo.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ripples.guard import decidir  # noqa: E402
from ripples.lattice import paper_size  # noqa: E402

VENV = ROOT.parent / ".venv"
BIN = ROOT / "src" / "c" / "graphene_mc"
RUN = ROOT / "results" / "n8640"
DATOS = RUN / "datos"
LOGS = RUN / "logs"
CKPT = RUN / "checkpoints"
PROBE = RUN / "probe"
ESTADO = RUN / "estado.txt"
PLAN = RUN / "plan.json"
PIDFILE = RUN / "pipeline.pid"

N_ATOMS = 8640
P, Q = paper_size(N_ATOMS)
T = 300.0
N_EQ = 400_000
N_PROD = 500_000
SEED = 42
# 32 hilos por réplica saturan los FPU compartidos del Opteron; 64 hilos en un
# solo proceso no baja el tiempo de pared. Dos réplicas cubren los 64 núcleos.
THREADS_POR_REPLICA = 32
FREEZE_S = 600.0
MAX_REINTENTOS = 6
CKPT_EVERY = 4000
BEAT_EVERY = 200
POLL_S = 15.0
HILOS = 1
N_REPLICAS = 1


def aplicar_reparto() -> None:
    global HILOS, N_REPLICAS
    nproc = os.cpu_count() or 1
    HILOS = min(THREADS_POR_REPLICA, nproc)
    N_REPLICAS = max(1, nproc // HILOS)


def exigir_venv() -> None:
    esperado = (VENV / "bin" / "python").resolve()
    actual = Path(sys.executable).resolve()
    if actual != esperado:
        sys.exit(f"hay que usar el entorno {esperado}, no {actual}")


def pid_vivo(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def proc_estado(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat")
    try:
        texto = stat.read_text()
    except OSError:
        return "?"
    # pid (comm) state  — comm puede contener espacios y paréntesis
    fin = texto.rfind(")")
    if fin < 0 or fin + 2 >= len(texto):
        return "?"
    return texto[fin + 2]


def mem_disponible_gib() -> float:
    try:
        for linea in Path("/proc/meminfo").read_text().splitlines():
            if linea.startswith("MemAvailable:"):
                kb = float(linea.split()[1])
                return kb / 1024.0 / 1024.0
    except OSError:
        return -1.0
    return -1.0


@dataclass
class Replica:
    indice: int
    proceso: subprocess.Popen | None = None
    codigo: int | None = None
    reintentos: int = 0
    congeladas: int = 0
    iniciada: float = 0.0
    vida_corta: bool = False
    accion: str = "pendiente"
    log: object | None = field(default=None, repr=False)

    @property
    def done(self) -> Path:
        return DATOS / f"replica_{self.indice:02d}.done"

    @property
    def ckpt(self) -> Path:
        return CKPT / f"replica_{self.indice:02d}.ckpt"

    @property
    def ckpt_prev(self) -> Path:
        return CKPT / f"replica_{self.indice:02d}.ckpt.prev"

    @property
    def heartbeat(self) -> Path:
        return RUN / "heartbeats" / f"replica_{self.indice:02d}.hb"

    def cerrar_log(self) -> None:
        if self.log is not None:
            self.log.close()
            self.log = None


REPLICAS: list[Replica] = []
DETENER = False


def al_senal(signum, _frame) -> None:
    global DETENER
    DETENER = True
    print(f"señal {signum}: paro el vigilante y guardo las réplicas", flush=True)


def plan_texto() -> str:
    if not PLAN.exists():
        return ""
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    horas = float(plan["horas_mc_por_replica"]) + float(plan["horas_muestras_por_replica"])
    return (
        f"ritmo: {plan['sec_per_step']*1000:.1f} ms/paso, {plan['sec_per_sample']:.2f} s/muestra, "
        f"muestra cada {plan['sample_every']} pasos. Pared estimada por réplica: {horas:.1f} h."
    )


def escribir_estado(fase: str, extra: str = "") -> None:
    ahora = time.time()
    lineas = [
        f"fase: {fase}",
        f"actualizado: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"pid del pipeline: {os.getpid()}",
        f"N={N_ATOMS}  p={P}  q={Q}  T={T:.0f} K  eq={N_EQ}  prod={N_PROD}",
        f"réplicas={len(REPLICAS)}  hilos por réplica={HILOS}  "
        f"núcleos en uso={len(REPLICAS) * HILOS}  RAM libre={mem_disponible_gib():.1f} GiB",
        "",
        "Para mirar esto sin la sesión de trabajo:",
        f"  {VENV / 'bin' / 'python'} {ROOT / 'scripts' / 'estado_n8640.py'}",
        "",
    ]
    if extra:
        lineas.append(extra.rstrip())
        lineas.append("")
    previsto = plan_texto()
    if previsto:
        lineas.append(previsto)
        lineas.append("")
    vivas = listas = fallidas = 0
    for rep in REPLICAS:
        paso, total = leer_paso(rep)
        edad = edad_heartbeat(rep, ahora)
        if rep.accion == "lista":
            listas += 1
        elif rep.accion == "fallida":
            fallidas += 1
        elif rep.proceso is not None and rep.proceso.poll() is None:
            vivas += 1
        pct = 100.0 * paso / total if total else 0.0
        edad_txt = f"{edad:.0f}" if edad is not None else "-"
        lineas.append(
            f"réplica {rep.indice:02d}  {rep.accion:10}  paso {paso}/{total} ({pct:5.1f}%)  "
            f"reintentos={rep.reintentos}  congeladas={rep.congeladas}  "
            f"heartbeat={edad_txt} s"
        )
    lineas.append("")
    lineas.append(f"vivas={vivas}  listas={listas}  fallidas={fallidas}  total={len(REPLICAS)}")
    ESTADO.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def leer_paso(rep: Replica) -> tuple[int, int]:
    if not rep.heartbeat.exists():
        return 0, N_EQ + N_PROD
    paso = 0
    total = N_EQ + N_PROD
    try:
        for linea in rep.heartbeat.read_text(encoding="utf-8").splitlines():
            if linea.startswith("step="):
                paso = int(linea.split("=", 1)[1])
            elif linea.startswith("total="):
                total = int(linea.split("=", 1)[1])
    except (OSError, ValueError):
        return 0, N_EQ + N_PROD
    return paso, total


def edad_heartbeat(rep: Replica, ahora: float) -> float | None:
    if rep.proceso is not None and rep.proceso.poll() is None and rep.iniciada > 0:
        desde_arranque = max(0.0, ahora - rep.iniciada)
        if rep.heartbeat.exists():
            return min(desde_arranque, max(0.0, ahora - rep.heartbeat.stat().st_mtime))
        return desde_arranque
    if rep.heartbeat.exists():
        return max(0.0, ahora - rep.heartbeat.stat().st_mtime)
    return None


def correr(cmd: list[str], cwd: Path | None = None) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or ROOT))


def preflight() -> None:
    correr(["make", "-C", str(ROOT / "src" / "c"), "all"])
    correr([str(BIN), "--self-test"])
    correr([str(BIN), "--self-test-large"])
    pruebas = [
        "tests/test_lattice.py",
        "tests/test_potential.py",
        "tests/test_observables.py",
        "tests/test_pipeline_guard.py",
    ]
    correr([sys.executable, "-m", "pytest", "-q", *pruebas])


def medir() -> int:
    """Elige cada cuántos pasos de producción se acumula G(q)."""
    if PLAN.exists():
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        return int(plan["sample_every"])
    PROBE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    correr(
        [
            str(BIN),
            "--replica",
            "0",
            "--p",
            str(P),
            "--q",
            str(Q),
            "--T",
            str(T),
            "--n-eq",
            "30",
            "--n-prod",
            "0",
            "--sample-every",
            "10",
            "--seed",
            "1",
            "--out",
            str(PROBE),
            "--checkpoint-every",
            "0",
            "--heartbeat-every",
            "0",
            "--threads",
            str(HILOS),
        ]
    )
    t_paso = (time.perf_counter() - t0) / 30.0
    t1 = time.perf_counter()
    correr(
        [
            str(BIN),
            "--replica",
            "1",
            "--p",
            str(P),
            "--q",
            str(Q),
            "--T",
            str(T),
            "--n-eq",
            "1",
            "--n-prod",
            "4",
            "--sample-every",
            "2",
            "--seed",
            "1",
            "--out",
            str(PROBE),
            "--checkpoint-every",
            "0",
            "--heartbeat-every",
            "0",
            "--threads",
            str(HILOS),
        ]
    )
    # 5 pasos y 2 muestras. Se resta el costo de los pasos.
    t_muestra = max(0.05, (time.perf_counter() - t1 - 5.0 * t_paso) / 2.0)
    presupuesto = 4.0 * 3600.0
    n_por_tiempo = max(80, int(presupuesto / t_muestra))
    n_muestras = min(300, n_por_tiempo)
    sample_every = max(10, (N_PROD + n_muestras - 1) // n_muestras)
    mc_h = t_paso * (N_EQ + N_PROD) / 3600.0
    muestras_h = (N_PROD / sample_every) * t_muestra / 3600.0
    plan = {
        "sample_every": sample_every,
        "sec_per_step": t_paso,
        "sec_per_sample": t_muestra,
        "horas_mc_por_replica": mc_h,
        "horas_muestras_por_replica": muestras_h,
        "workers": N_REPLICAS,
        "threads": HILOS,
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"plan: {t_paso*1000:.1f} ms/paso, {t_muestra:.2f} s/muestra, "
        f"sample_every={sample_every}, pared ≈ {mc_h + muestras_h:.1f} h por réplica",
        flush=True,
    )
    return sample_every


def elegir_resume(rep: Replica) -> Path | None:
    if rep.vida_corta and rep.ckpt_prev.exists():
        if rep.ckpt.exists():
            rep.ckpt.replace(CKPT / f"replica_{rep.indice:02d}.ckpt.bad")
        rep.ckpt_prev.replace(rep.ckpt)
        rep.vida_corta = False
    if rep.ckpt.exists():
        return rep.ckpt
    if rep.ckpt_prev.exists():
        return rep.ckpt_prev
    return None


def lanzar(rep: Replica, sample_every: int) -> None:
    rep.cerrar_log()
    if rep.done.exists() and (rep.proceso is None or rep.proceso.poll() is not None):
        rep.accion = "lista"
        rep.codigo = 0
        return
    if rep.heartbeat.exists():
        rep.heartbeat.unlink()
    inicio = rep.indice * HILOS
    nucleo = f"{inicio}-{inicio + HILOS - 1}"
    cmd = [
        "taskset",
        "-c",
        nucleo,
        str(BIN),
        "--replica",
        str(rep.indice),
        "--p",
        str(P),
        "--q",
        str(Q),
        "--T",
        str(T),
        "--n-eq",
        str(N_EQ),
        "--n-prod",
        str(N_PROD),
        "--sample-every",
        str(sample_every),
        "--seed",
        str(SEED),
        "--out",
        str(DATOS),
        "--heartbeat",
        str(rep.heartbeat),
        "--checkpoint",
        str(rep.ckpt),
        "--checkpoint-every",
        str(CKPT_EVERY),
        "--heartbeat-every",
        str(BEAT_EVERY),
        "--threads",
        str(HILOS),
    ]
    resume = elegir_resume(rep)
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    logf = (LOGS / f"replica_{rep.indice:02d}.log").open("a", encoding="utf-8")
    logf.write(f"\n# arranque {time.strftime('%Y-%m-%d %H:%M:%S')} reintento={rep.reintentos}\n")
    logf.flush()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(HILOS)
    rep.proceso = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    rep.log = logf
    rep.codigo = None
    rep.iniciada = time.time()
    rep.accion = "corre"
    print(" ".join(cmd), flush=True)


def matar(rep: Replica) -> None:
    proc = rep.proceso
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=10)


def vigilar(sample_every: int) -> int:
    while not DETENER:
        ahora = time.time()
        for rep in REPLICAS:
            if rep.accion in ("lista", "fallida"):
                continue
            vivo = rep.proceso is not None and rep.proceso.poll() is None
            if rep.proceso is not None and not vivo and rep.codigo is None:
                rep.codigo = rep.proceso.returncode
                vida = ahora - rep.iniciada
                rep.vida_corta = rep.codigo not in (0, 75) and vida < 90.0
            accion = decidir(
                vivo,
                rep.codigo,
                edad_heartbeat(rep, ahora),
                rep.done.exists(),
                rep.reintentos,
                MAX_REINTENTOS,
                FREEZE_S,
            )
            rep.accion = accion
            if accion == "congelada":
                print(f"réplica {rep.indice:02d} congelada; la reinicio", flush=True)
                matar(rep)
                rep.codigo = rep.proceso.returncode if rep.proceso is not None else None
                rep.congeladas += 1
                rep.reintentos += 1
                if rep.reintentos >= MAX_REINTENTOS:
                    rep.accion = "fallida"
                else:
                    lanzar(rep, sample_every)
            elif accion == "reiniciar":
                ya_corrio = rep.proceso is not None
                if ya_corrio:
                    rep.reintentos += 1
                if ya_corrio and rep.reintentos >= MAX_REINTENTOS:
                    rep.accion = "fallida"
                    print(f"réplica {rep.indice:02d} agotó los reintentos", flush=True)
                else:
                    print(
                        f"réplica {rep.indice:02d} arranque {rep.reintentos + 1} código {rep.codigo}",
                        flush=True,
                    )
                    lanzar(rep, sample_every)
            elif accion == "fallida":
                matar(rep)
                print(f"réplica {rep.indice:02d} agotó los reintentos", flush=True)
        escribir_estado("corriendo")
        if all(rep.accion in ("lista", "fallida") for rep in REPLICAS):
            break
        time.sleep(POLL_S)
    if DETENER:
        for rep in REPLICAS:
            matar(rep)
        escribir_estado("interrumpido")
        return 75
    fallidas = sum(rep.accion == "fallida" for rep in REPLICAS)
    escribir_estado("réplicas cerradas" if fallidas == 0 else "réplicas con fallos")
    return 0 if fallidas == 0 else 1


def cerrar() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_plots import main as graficar
    import contrastar_paper

    sys.argv = ["generate_plots.py", "--datos", str(DATOS), "--graf", str(RUN / "graficas")]
    graficar()
    texto = contrastar_paper.redactar(contrastar_paper.informe(DATOS))
    (RUN / "contraste_paper.txt").write_text(texto, encoding="utf-8")
    print(texto, flush=True)
    (RUN / "COMPLETO").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_n8640_paper.py"],
        cwd=str(ROOT),
    )
    escribir_estado("contraste listo" if r.returncode == 0 else "contraste con diferencias")
    return r.returncode


def main() -> None:
    exigir_venv()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, al_senal)
    signal.signal(signal.SIGINT, al_senal)
    for ruta in (DATOS, LOGS, CKPT, RUN / "heartbeats", RUN / "graficas"):
        ruta.mkdir(parents=True, exist_ok=True)
    if PIDFILE.exists():
        try:
            viejo = int(PIDFILE.read_text().strip())
        except ValueError:
            viejo = -1
        if pid_vivo(viejo) and viejo != os.getpid():
            sys.exit(f"el pipeline ya corre con pid {viejo}")
    PIDFILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    aplicar_reparto()
    os.environ["OMP_NUM_THREADS"] = str(HILOS)
    print(
        f"pipeline N={N_ATOMS}  réplicas={N_REPLICAS}  hilos={HILOS}  "
        f"núcleos={N_REPLICAS * HILOS}  RAM libre {mem_disponible_gib():.1f} GiB",
        flush=True,
    )
    escribir_estado("preflight")
    preflight()
    sample_every = medir()
    global REPLICAS
    REPLICAS = [Replica(i) for i in range(N_REPLICAS)]
    for rep in REPLICAS:
        if rep.done.exists():
            rep.accion = "lista"
            rep.codigo = 0
    codigo = vigilar(sample_every)
    for rep in REPLICAS:
        rep.cerrar_log()
    if codigo == 0 and not DETENER:
        codigo = cerrar()
    escribir_estado("terminado" if codigo == 0 else f"salida {codigo}")
    raise SystemExit(codigo)


if __name__ == "__main__":
    main()
