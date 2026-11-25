
"""Exporta la trayectoria optimizada en coordenadas 3D de Assetto Corsa."""

import csv
from pathlib import Path

import numpy as np

from detectar_circuito import detectar_circuito, buscar_geometria
from geometria_circuito import SIGNO_Z_CSV


DESTINO_CSP = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common"
    r"\assettocorsa\apps\lua\TrazadaFisica\trayectoria.csv"
)

NUM_MUESTRAS = 1500


# A. LEER BORDES ORIGINALES EN 3D
def leer_borde(ruta):
    """Lee muestras de un borde del circuito y valida sus coordenadas."""
    filas = []

    with Path(ruta).open(encoding="utf-8-sig", newline="") as archivo:
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for fila in csv.reader(archivo):
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if len(fila) < 4:
                continue

            # Protege recursos externos y permite limpieza ante errores.
            try:
                x = float(fila[0])
                altura = float(fila[1])
                z = float(fila[2])
                progreso = float(fila[3])
            except ValueError:
                continue

            # Convencion horizontal del modelo de bicicleta.
            filas.append((progreso, x, altura, z * SIGNO_Z_CSV))

    datos = np.asarray(filas, dtype=float)

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if datos.ndim != 2 or len(datos) < 3:
        raise ValueError(f"Geometria insuficiente: {ruta}")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not np.isfinite(datos).all():
        raise ValueError(f"Coordenadas invalidas: {ruta}")

    datos = datos[np.argsort(datos[:, 0])]

    # Quitar progresos repetidos.
    _, indices = np.unique(datos[:, 0], return_index=True)
    return datos[np.sort(indices)]


# B. INTERPOLAR BORDES EN UN PROGRESO COMUN
def interpolar_borde(datos, progreso):
    """Reconstruye muestras del borde para compartir una parametrización longitudinal."""
    s = datos[:, 0]

    # Extender el circuito para interpolar el cierre de vuelta.
    s_ext = np.concatenate((s - 1, s, s + 1))

    columnas = []
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for col in (1, 2, 3):
        valores = np.tile(datos[:, col], 3)
        columnas.append(np.interp(progreso, s_ext, valores))

    # Cada fila contiene X, altura Y, Z horizontal del modelo.
    return np.column_stack(columnas)


# C. ESTIMAR ALTURA DE LA RUTA OPTIMIZADA
def recuperar_alturas(ruta_2d, izquierdo, derecho):
    # Coordenadas horizontales de ambos bordes.
    """Recupera cotas de la geometría sin alterar las coordenadas planas de la pista."""
    li = izquierdo[:, [0, 2]]
    ld = derecho[:, [0, 2]]

    centro = (li + ld) / 2.0

    # Segmentos de la linea central cerrada.
    inicio = centro
    vector = np.roll(centro, -1, axis=0) - inicio

    longitud2 = np.sum(vector * vector, axis=1)
    longitud2 = np.maximum(longitud2, 1e-12)

    alturas = np.empty(len(ruta_2d), dtype=float)

    # Procesar por bloques para no crear matrices enormes.
    for comienzo in range(0, len(ruta_2d), 128):
        bloque = ruta_2d[comienzo:comienzo + 128]

        relativo = bloque[:, None, :] - inicio[None, :, :]

        t = np.sum(
            relativo * vector[None, :, :],
            axis=2
        ) / longitud2[None, :]

        t = np.clip(t, 0.0, 1.0)

        proyecciones = (
            inicio[None, :, :]
            + t[:, :, None] * vector[None, :, :]
        )

        distancia2 = np.sum(
            (bloque[:, None, :] - proyecciones) ** 2,
            axis=2
        )

        indices = np.argmin(distancia2, axis=1)
        filas = np.arange(len(bloque))
        factores = t[filas, indices]

        siguiente = (indices + 1) % len(centro)

        # Interpolar ambos bordes en la misma estacion.
        iz = (
            izquierdo[indices] * (1 - factores[:, None])
            + izquierdo[siguiente] * factores[:, None]
        )
        de = (
            derecho[indices] * (1 - factores[:, None])
            + derecho[siguiente] * factores[:, None]
        )

        # Posicion lateral entre el borde izquierdo y derecho.
        lateral = de[:, [0, 2]] - iz[:, [0, 2]]

        fraccion = np.sum(
            (bloque - iz[:, [0, 2]]) * lateral,
            axis=1
        ) / np.maximum(np.sum(lateral ** 2, axis=1), 1e-12)

        # Interpolar altura: permite aproximar el peralte.
        alturas[comienzo:comienzo + len(bloque)] = (
            iz[:, 1] * (1 - fraccion)
            + de[:, 1] * fraccion
        )

    return alturas


# D. EXPORTAR CSV PARA CSP
def exportar_trayectoria(ruta, destino=DESTINO_CSP):
    """Escribe la trayectoria preparada para análisis o visualización posterior."""
    ruta = np.asarray(ruta, dtype=float)

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if ruta.ndim != 2 or ruta.shape[1] != 2:
        raise ValueError("La ruta debe contener puntos 2D (N, 2)")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if len(ruta) < 2 or not np.isfinite(ruta).all():
        raise ValueError("Trayectoria vacia o invalida")

    circuito, variante = detectar_circuito()
    _, archivos = buscar_geometria(circuito, variante)

    izquierdo = leer_borde(archivos["side_l"])
    derecho = leer_borde(archivos["side_r"])

    progreso = np.linspace(
        0.0, 1.0, NUM_MUESTRAS, endpoint=False
    )

    izquierdo = interpolar_borde(izquierdo, progreso)
    derecho = interpolar_borde(derecho, progreso)

    altura = recuperar_alturas(ruta, izquierdo, derecho)

    # Convertir del plano del modelo a coordenadas Assetto X,Y,Z.
    # La ruta ya está expresada en las coordenadas horizontales
    # que coinciden con la telemetría real de Assetto.
    # No invertir Z nuevamente.
    puntos_xyz = np.column_stack((
        ruta[:, 0],
        altura,
        ruta[:, 1]
    ))

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    temporal = destino.with_suffix(".tmp")

    with temporal.open(
        "w", newline="", encoding="utf-8"
    ) as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["x", "y", "z"])
        escritor.writerows(puntos_xyz)

    temporal.replace(destino)

    print(f"Trayectoria exportada: {len(puntos_xyz)} puntos")
    print(f"Circuito: {circuito} / {variante}")
    print(
        f"Alturas aproximadas: "
        f"{altura.min():.3f} a {altura.max():.3f} m"
    )
    print(f"CSV: {destino}")

    return destino
