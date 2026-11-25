
"""
GEOMETRÍA AUTOMÁTICA DEL CIRCUITO

Lee los bordes izquierdo y derecho de Assetto Corsa.
Los sincroniza por progreso normalizado y obtiene:

- Borde izquierdo
- Borde derecho
- Línea central
- Ancho aproximado de la pista
- Distancia del vehículo a la línea central

Solo lectura. No controla el vehículo.
"""

import csv
import time
from dataclasses import dataclass

import numpy as np

from detectar_circuito import detectar_circuito, buscar_geometria
from telemetria_assetto import Graphics, Physics, conectar, leer


# Confirmado experimentalmente para Tocancipá proyfinal.
# En otras pistas debemos validar esta convención.
SIGNO_Z_CSV = -1

NUM_PUNTOS = 1500


# Agrupa geometría del circuito con bordes y eje central en una misma representación.
@dataclass
class GeometriaCircuito:
    izquierdo: np.ndarray
    derecho: np.ndarray
    centro: np.ndarray
    progreso: np.ndarray
    anchos_m: np.ndarray


# Interpreta las coordenadas de un borde a partir del archivo CSV.
def cargar_csv(ruta, signo_z):
    """Lee X, Z y progreso de un borde."""

    filas = []

    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for fila in csv.reader(archivo):
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if not fila or len(fila) < 4:
                continue

            x = float(fila[0])
            z = float(fila[2]) * signo_z
            progreso = float(fila[3])

            filas.append((progreso, x, z))

    datos = np.asarray(filas, dtype=float)

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if datos.ndim != 2 or datos.shape[0] < 3:
        raise ValueError(f"CSV insuficiente: {ruta}")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not np.all(np.isfinite(datos)):
        raise ValueError(f"Valores no finitos: {ruta}")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any((datos[:, 0] < 0) | (datos[:, 0] > 1)):
        raise ValueError("El progreso debe estar entre 0 y 1")

    # Ordenar según el progreso de vuelta.
    datos = datos[np.argsort(datos[:, 0])]

    # Eliminar progresos repetidos para interpolar.
    _, indices = np.unique(datos[:, 0], return_index=True)
    datos = datos[np.sort(indices)]

    return datos


# Reconstruye muestras del borde para compartir una parametrización longitudinal.
def interpolar_borde(datos, progreso_objetivo):
    """Interpola un borde cerrado sobre el mismo progreso."""

    s = datos[:, 0]
    x = datos[:, 1]
    z = datos[:, 2]

    # Extender una vuelta antes y después para evitar
    # discontinuidades cerca del inicio y el final.
    s_ext = np.concatenate((s - 1, s, s + 1))
    x_ext = np.tile(x, 3)
    z_ext = np.tile(z, 3)

    x_nuevo = np.interp(progreso_objetivo, s_ext, x_ext)
    z_nuevo = np.interp(progreso_objetivo, s_ext, z_ext)

    return np.column_stack((x_nuevo, z_nuevo))


# Reúne los bordes y centro del circuito activo en una geometría coherente.
def construir_geometria(signo_z=SIGNO_Z_CSV):
    """Detecta y reconstruye el circuito actual."""

    circuito, variante = detectar_circuito()
    _, archivos = buscar_geometria(circuito, variante)

    izquierda = cargar_csv(archivos["side_l"], signo_z)
    derecha = cargar_csv(archivos["side_r"], signo_z)

    # Mismo progreso para ambos bordes.
    progreso = np.linspace(
        0.0, 1.0, NUM_PUNTOS, endpoint=False
    )

    borde_izq = interpolar_borde(izquierda, progreso)
    borde_der = interpolar_borde(derecha, progreso)

    # Punto medio entre bordes correspondientes.
    centro = (borde_izq + borde_der) / 2.0

    # Separación entre bordes emparejados.
    anchos = np.linalg.norm(borde_der - borde_izq, axis=1)

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not np.all(np.isfinite(anchos)):
        raise ValueError("Geometría inválida")

    return circuito, variante, GeometriaCircuito(
        izquierdo=borde_izq,
        derecho=borde_der,
        centro=centro,
        progreso=progreso,
        anchos_m=anchos,
    )


# Calcula el punto del eje del circuito más cercano por proyección ortogonal sobre segmentos.
def proyectar_centro(punto, centro):
    """
    Encuentra el punto más cercano de la línea central,
    considerando segmentos y no solamente vértices.
    """

    a = centro
    b = np.roll(centro, -1, axis=0)

    segmentos = b - a
    relativos = punto - a

    longitud2 = np.sum(segmentos**2, axis=1)

    t = np.divide(
        np.sum(relativos * segmentos, axis=1),
        longitud2,
        out=np.zeros_like(longitud2),
        where=longitud2 > 1e-12,
    )

    t = np.clip(t, 0.0, 1.0)

    proyecciones = a + t[:, None] * segmentos
    distancias2 = np.sum((proyecciones - punto)**2, axis=1)

    indice = int(np.argmin(distancias2))

    return (
        float(np.sqrt(distancias2[indice])),
        indice,
        proyecciones[indice],
    )


# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main():
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    circuito, variante, geometria = construir_geometria()

    print("=== GEOMETRÍA DEL CIRCUITO ===")
    print(f"Circuito: {circuito}")
    print(f"Variante: {variante or 'Sin variante'}")
    print(f"Puntos de referencia: {len(geometria.centro)}")
    print(f"Ancho mínimo: {geometria.anchos_m.min():.2f} m")
    print(f"Ancho máximo: {geometria.anchos_m.max():.2f} m")

    segmentos = (
        np.roll(geometria.centro, -1, axis=0)
        - geometria.centro
    )

    longitud = np.linalg.norm(segmentos, axis=1).sum()
    print(f"Longitud central aproximada: {longitud:.1f} m")
    print("\nObservando posición real del Porsche...")
    print("Ctrl+C para detener.\n")

    grafica_mem = conectar("acpmf_graphics", Graphics)
    fisica_mem = conectar("acpmf_physics", Physics)

    # Protege recursos externos y permite limpieza ante errores.
    try:
        # Repite el proceso hasta alcanzar la condición de parada o el límite de seguridad.
        while True:
            grafica = leer(grafica_mem, Graphics)
            fisica = leer(fisica_mem, Physics)

            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if grafica.status != 2:
                print("Esperando sesión LIVE...")
                time.sleep(0.5)
                continue

            x, altura, z = grafica.carCoordinates
            punto = np.array([x, z], dtype=float)

            distancia, indice, proyeccion = proyectar_centro(
                punto, geometria.centro
            )

            ancho = geometria.anchos_m[indice]
            progreso = geometria.progreso[indice]

            print(
                f"Velocidad: {fisica.speedKmh:.1f} km/h | "
                f"Posición: ({x:.2f}, {z:.2f})"
            )

            print(
                f"Distancia al centro: {distancia:.2f} m | "
                f"Ancho local aproximado: {ancho:.2f} m"
            )

            print(
                f"Progreso geométrico: {progreso * 100:.1f}% | "
                f"Segmento: {indice}"
            )

            print("-" * 55)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nObservación finalizada.")

    finally:
        grafica_mem.close()
        fisica_mem.close()


if __name__ == "__main__":
    main()
