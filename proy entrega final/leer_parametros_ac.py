"""DOCUMENTACIÓN DEL MÓDULO — PROYECTO FINAL.

Los comentarios explican entradas, transformaciones y supuestos. Se conserva
la lógica preexistente; los límites estimados no son mediciones directas de AC.
"""


from pathlib import Path
from configparser import ConfigParser
import csv  # Estadística ponderada de las muestras experimentales guardadas.
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


# C1. CALIBRACIÓN: UNA ÚNICA FUENTE DE LÍMITES EXPERIMENTALES.
# La carpeta identifica la medición concreta; NO se aplica a otros automóviles.
RUTA_CALIBRACION = (Path(__file__).resolve().parent / 'medicion_ac_vjoy'
                    / '20261125_133617' / 'resumen_por_velocidad.csv')
AUTO_CALIBRADO = 'jw_porsche_991.2_gt2_rs_us'
COMPUESTO_CALIBRADO = 'hypercar road (hr)'

def limites_experimentales(auto, compuesto):
    """Promedio ponderado por número de muestras de cada intervalo de velocidad.

    Una sola prueba 0–121 km/h no permite afirmar límites para 187 km/h.
    Se usa como aproximación CONSTANTE del modelo, no extrapolación validada.
    """
    if auto != AUTO_CALIBRADO or compuesto.lower() != COMPUESTO_CALIBRADO:
        return None
    if not RUTA_CALIBRACION.is_file():
        raise FileNotFoundError(f'Falta la calibración declarada: {RUTA_CALIBRACION}')
    acumulados = {'acelerando': [0.0, 0], 'frenando': [0.0, 0]}
    with RUTA_CALIBRACION.open(encoding='utf-8-sig', newline='') as archivo:
        for fila in csv.DictReader(archivo):
            tipo = fila.get('tipo')
            if tipo not in acumulados:
                continue
            n = int(fila['muestras'])
            a = float(fila['aceleracion_media_m_s2'])
            if n <= 0 or not 0 < a < 35:
                raise ValueError('Datos de calibración inválidos')
            acumulados[tipo][0] += a*n
            acumulados[tipo][1] += n
    if any(n == 0 for _,n in acumulados.values()):
        raise ValueError('Calibración sin muestras suficientes en alguna fase')
    return (acumulados['acelerando'][0] / acumulados['acelerando'][1],
            acumulados['frenando'][0] / acumulados['frenando'][1])


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

        # G0. Preferir límites experimentales cuando corresponden al coche
        # y compuesto activos. La fuente es el CSV, nunca números duplicados.
        experimento = limites_experimentales(auto, compuesto)
        fuente_limites = 'física idealizada (sin calibración del coche)'
        if experimento is not None:
            aceleracion_max, frenado_max = experimento
            fuente_limites = f'ensayo único ponderado: {RUTA_CALIBRACION.name}'

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
            "origen_limites": fuente_limites,
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
