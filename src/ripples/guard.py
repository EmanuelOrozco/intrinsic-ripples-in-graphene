"""Decisiones del vigilante: proceso muerto, checkpoint o CPU quieta."""

from __future__ import annotations


def decidir(
    proceso_vivo: bool,
    codigo: int | None,
    edad_s: float | None,
    hay_done: bool,
    reintentos: int,
    max_reintentos: int,
    freeze_s: float,
) -> str:
    """Qué hacer con una réplica.

    seguir: sigue viva y el heartbeat es reciente.
    congelada: sigue viva pero el heartbeat no avanza; hay que matarla y reanudar.
    lista: terminó bien.
    reiniciar: hay que relanzarla desde el checkpoint.
    fallida: se agotaron los reintentos.
    """
    if not proceso_vivo and codigo == 0 and hay_done:
        return "lista"
    congelada = proceso_vivo and edad_s is not None and edad_s > freeze_s
    if reintentos >= max_reintentos and (not proceso_vivo or congelada):
        return "fallida"
    if congelada:
        return "congelada"
    if not proceso_vivo:
        return "reiniciar"
    return "seguir"
