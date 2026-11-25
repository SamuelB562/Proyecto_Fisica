
"""
GESTOR DE CIRCUITO

Prepara una vuelta continua desde cualquier punto
del trazado, con margen antes y después del coche.

No modifica el circuito original.
"""

import numpy as np


# Localiza inicio/meta y produce la geometría necesaria para evaluar una vuelta.
def preparar_vuelta(
    centro,
    izquierdo,
    derecho,
    posicion,
    margen_atras_m=200.0,
    margen_despues_m=300.0,
):
    """Localiza inicio/meta y produce la geometría necesaria para evaluar una vuelta."""
    centro = np.asarray(centro, dtype=float)
    izquierdo = np.asarray(izquierdo, dtype=float)
    derecho = np.asarray(derecho, dtype=float)
    posicion = np.asarray(posicion, dtype=float)

    n = len(centro)

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if n < 3:
        raise ValueError("El circuito tiene pocos puntos")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if izquierdo.shape != centro.shape or derecho.shape != centro.shape:
        raise ValueError("Los tres trazados deben coincidir")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not all(np.all(np.isfinite(p)) for p in
               (centro, izquierdo, derecho)):
        raise ValueError("Coordenadas no válidas")

    # Longitud total de la pista cerrada.
    siguientes = np.roll(centro, -1, axis=0)
    segmentos = siguientes - centro

    longitudes = np.linalg.norm(segmentos, axis=1)

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(longitudes < 1e-8):
        raise ValueError("Hay segmentos de longitud nula")

    estaciones = np.r_[0.0, np.cumsum(longitudes)]
    longitud_vuelta = float(estaciones[-1])

    # Proyectar el Porsche sobre la línea central.
    longitud2 = longitudes ** 2

    t = np.clip(
        np.sum((posicion - centro) * segmentos, axis=1)
        / longitud2,
        0.0,
        1.0
    )

    proyecciones = centro + t[:, None] * segmentos

    distancias2 = np.sum(
        (proyecciones - posicion) ** 2,
        axis=1
    )

    indice_coche = int(np.argmin(distancias2))

    progreso_coche = (
        estaciones[indice_coche]
        + t[indice_coche] * longitudes[indice_coche]
    )

    # Seleccionar un inicio anterior al Porsche.
    progreso_corte = (
        progreso_coche - margen_atras_m
    ) % longitud_vuelta

    indice_corte = int(
        np.searchsorted(
            estaciones[:-1],
            progreso_corte,
            side="right"
        ) - 1
    )

    # Repetimos los datos temporalmente para poder
    # extraer una sección continua que cruce la meta.
    centro_repetido = np.tile(centro, (3, 1))
    izquierdo_repetido = np.tile(izquierdo, (3, 1))
    derecho_repetido = np.tile(derecho, (3, 1))

    # Una vuelta más márgenes anterior y posterior.
    distancia_objetivo = (
        longitud_vuelta
        + margen_atras_m
        + margen_despues_m
    )

    # Tomamos un tramo de hasta dos vueltas.
    tramo = centro_repetido[
        indice_corte:indice_corte + 2 * n
    ]

    pasos = np.linalg.norm(
        np.diff(tramo, axis=0),
        axis=1
    )

    distancias = np.r_[0.0, np.cumsum(pasos)]

    indice_final = int(
        np.searchsorted(distancias, distancia_objetivo)
    )

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if indice_final >= len(tramo):
        raise ValueError("Extensión insuficiente del circuito")

    fin = indice_corte + indice_final + 1

    centro_ext = centro_repetido[indice_corte:fin].copy()
    izquierdo_ext = izquierdo_repetido[indice_corte:fin].copy()
    derecho_ext = derecho_repetido[indice_corte:fin].copy()

    # Distancia de la posición inicial desde el nuevo
    # comienzo de la geometría extendida.
    progreso_inicial = (
        progreso_coche - estaciones[indice_corte]
    ) % longitud_vuelta

    return (
        centro_ext,
        izquierdo_ext,
        derecho_ext,
        longitud_vuelta,
        float(progreso_inicial),
    )


# Suaviza de manera sincronizada centro y bordes respetando continuidad de la vuelta.
def suavizar_geometria(
    centro,
    izquierdo,
    derecho,
    pasadas=3,
    desplazamiento_max_m=1.0,
):
    """
    Suavizado periódico y sincronizado de los tres trazados.

    No modifica los arrays originales.
    No cambia el modelo físico.
    Rechaza correcciones geométricas excesivas.
    """

    originales = [
        np.asarray(p, dtype=float).copy()
        for p in (centro, izquierdo, derecho)
    ]

    suavizados = [p.copy() for p in originales]

    # Filtro local: conserva principalmente el punto actual.
    for _ in range(pasadas):
        suavizados = [
            (
                np.roll(p, 1, axis=0)
                + 2.0 * p
                + np.roll(p, -1, axis=0)
            ) / 4.0
            for p in suavizados
        ]

    cambios = [
        float(np.max(np.linalg.norm(nuevo - original, axis=1)))
        for nuevo, original in zip(suavizados, originales)
    ]

    print("\n=== PREPROCESAMIENTO DE GEOMETRÍA ===")
    print(f"Desplazamiento del centro: {cambios[0]:.3f} m")
    print(f"Desplazamiento borde izquierdo: {cambios[1]:.3f} m")
    print(f"Desplazamiento borde derecho: {cambios[2]:.3f} m")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if max(cambios) > desplazamiento_max_m:
        raise ValueError(
            "El suavizado requiere cambiar demasiado "
            "la geometría. Revisar datos originales."
        )

    # Comprobación de anchura respecto a la geometría
    # original. No sustituye la validación física del agente.
    centro_s, izq_s, der_s = suavizados

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(np.linalg.norm(izq_s - der_s, axis=1) < 4.0):
        raise ValueError(
            "La geometría suavizada tiene un ancho sospechoso."
        )

    return tuple(suavizados)
