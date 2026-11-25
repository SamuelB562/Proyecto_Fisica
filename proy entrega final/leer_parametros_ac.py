"""DOCUMENTACIÓN DEL MÓDULO — PROYECTO FINAL.

Los comentarios explican entradas, transformaciones y supuestos. Se conserva
la lógica preexistente; los límites estimados no son mediciones directas de AC.
"""


from pathlib import Path
from configparser import ConfigParser
import re

from telemetria_assetto import (
    Physics, Graphics, Static, conectar, leer
)


# A. CONFIGURACIÓN
# Cambia únicamente esta ruta por la carpeta de Assetto Corsa.
AC_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")


# B. LEER ARCHIVOS INI
# B. LEER ARCHIVOS INI
def leer_ini(ruta):
    """Carga parámetros físicos del automóvil sin interpretar como valores los comentarios inline."""
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe: {ruta}")

    ini = ConfigParser(
        interpolation=None,
        inline_comment_prefixes=(";", "#")
    )

    ini.read(ruta, encoding="utf-8-sig")
    return ini


# C. IDENTIFICAR EL COMPUESTO
def buscar_compuesto(ini, nombre_activo):
    # Ejemplo: "Hypercar road (HR)"
    """Relaciona el compuesto activo informado por telemetría con las secciones de tyres.ini."""
    nombre_activo = nombre_activo.strip().lower()

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not nombre_activo:
        raise ValueError("No se recibió un compuesto activo.")

    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for sufijo in ("", "_1", "_2", "_3", "_4", "_5"):
        seccion = f"FRONT{sufijo}"

        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if not ini.has_section(seccion):
            continue

        nombre = ini.get(seccion, "NAME").lower()
        corto = ini.get(seccion, "SHORT_NAME").lower()

        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if (
            nombre_activo == nombre
            or nombre_activo == f"{nombre} ({corto})"
        ):
            return sufijo

    raise ValueError(
        f"No se encontró el compuesto: {nombre_activo}"
    )


# D. OBTENER LOS PARÁMETROS
def obtener_parametros():
    """Obtiene parámetros del automóvil activo y estima límites longitudinales simplificados."""
    fisica = conectar("acpmf_physics", Physics)
    graficos = conectar("acpmf_graphics", Graphics)
    estatica = conectar("acpmf_static", Static)

    # Protege recursos externos y permite limpieza ante errores.
    try:
        p = leer(fisica, Physics)
        g = leer(graficos, Graphics)
        s = leer(estatica, Static)

        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if g.status != 2:
            raise RuntimeError(
                "Assetto Corsa debe estar en una sesión LIVE."
            )

        auto = str(s.carModel).strip("\x00 ")
        compuesto = str(g.tyreCompound).strip("\x00 ")

        # E. CARPETA DEL AUTOMÓVIL ACTIVO
        carpeta = AC_DIR / "content" / "cars" / auto / "data"

        car = leer_ini(carpeta / "car.ini")
        suspension = leer_ini(carpeta / "suspensions.ini")
        neumaticos = leer_ini(carpeta / "tyres.ini")
        frenos = leer_ini(carpeta / "brakes.ini")

        # F. IDENTIFICAR LOS NEUMÁTICOS ACTIVOS
        sufijo = buscar_compuesto(neumaticos, compuesto)
        delantero = f"FRONT{sufijo}"
        trasero = f"REAR{sufijo}"
        # G1. ESTIMACIONES SIMPLIFICADAS DE ACELERACIÓN Y FRENADO
        g = 9.81

        masa = car.getfloat("BASIC", "TOTALMASS")
        peso_delantero = suspension.getfloat("BASIC", "CG_LOCATION")
        peso_trasero = 1.0 - peso_delantero

        mu_long = min(
            neumaticos.getfloat(delantero, "DX_REF"),
            neumaticos.getfloat(trasero, "DX_REF")
        )

        mu_lat = min(
            neumaticos.getfloat(delantero, "DY_REF"),
            neumaticos.getfloat(trasero, "DY_REF")
        )

        # Aceleración: límite aproximado por tracción trasera.
        aceleracion_max = mu_long * peso_trasero * g

        # Frenado: límite mecánico de los frenos.
        torque = frenos.getfloat("DATA", "MAX_TORQUE")
        reparto = frenos.getfloat("DATA", "FRONT_SHARE")

        radio_f = neumaticos.getfloat(delantero, "RADIUS")
        radio_r = neumaticos.getfloat(trasero, "RADIUS")

        # Aproximación: MAX_TORQUE por rueda, repartido entre ejes.
        fuerza_frenos = (
            2 * torque * reparto / radio_f
            + 2 * torque * (1 - reparto) / radio_r
        )

        frenado_mecanico = fuerza_frenos / masa

        # Límite simplificado de adherencia longitudinal.
        frenado_agarre = mu_long * g

        # Usamos el más restrictivo de ambos.
        frenado_max = min(frenado_mecanico, frenado_agarre)

        # G. PARÁMETROS EXTRAÍDOS
        return {
            "automovil": auto,
            "circuito": str(s.track).strip("\x00 "),
            "compuesto": compuesto,

            "masa_kg": car.getfloat("BASIC", "TOTALMASS"),
            "distancia_ejes_m": suspension.getfloat(
                "BASIC", "WHEELBASE"
            ),

            "fraccion_peso_delantero": suspension.getfloat(
                "BASIC", "CG_LOCATION"
            ),

            "radio_delantero_m": neumaticos.getfloat(
                delantero, "RADIUS"
            ),
            "radio_trasero_m": neumaticos.getfloat(
                trasero, "RADIUS"
            ),

            "mu_lateral_ref_delantero": neumaticos.getfloat(
                delantero, "DY_REF"
            ),
            "mu_lateral_ref_trasero": neumaticos.getfloat(
                trasero, "DY_REF"
            ),

            "mu_longitudinal_ref_delantero": neumaticos.getfloat(
                delantero, "DX_REF"
            ),
            "mu_longitudinal_ref_trasero": neumaticos.getfloat(
                trasero, "DX_REF"
            ),

            "potencia_maxima_telemetria": float(s.maxPower),
            "direccion_steer_lock": car.getfloat(
                "CONTROLS", "STEER_LOCK"
            ),
            "direccion_steer_ratio": car.getfloat(
                "CONTROLS", "STEER_RATIO"
            ),
            "aceleracion_max_m_s2": aceleracion_max,
            "frenado_max_m_s2": frenado_max,
            "coef_friccion_modelo": mu_lat,
            "frenado_mecanico_m_s2": frenado_mecanico,
        }

    finally:
        fisica.close()
        graficos.close()
        estatica.close()


# H. EJECUTAR
if __name__ == "__main__":
    parametros = obtener_parametros()

    print("\nPARÁMETROS DEL AUTOMÓVIL ACTIVO\n")

    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for nombre, valor in parametros.items():
        print(f"{nombre}: {valor}")
