
"""Adaptador del modo SOLO: Assetto Corsa <-> agentes.

Por ahora solo convierte datos. No controla el vehículo.
"""

import math

from modelado_fisico import Estado, Vehiculo
from agente_conductor import Controles


# Normaliza la orientación al intervalo angular [-π,π].
def normalizar_angulo(angulo):
    """Representa un ángulo entre -pi y pi."""
    return math.atan2(math.sin(angulo), math.cos(angulo))


# Convierte telemetría de Assetto Corsa al estado usado por nuestro modelo de bicicleta.
def convertir_estado(fisica, grafica):
    """Convierte la telemetría RAW a nuestro Estado."""

    # Assetto utiliza X, Y (altura), Z.
    x, altura, z = grafica.carCoordinates

    # Nuestro modelo utiliza un plano horizontal X, Y.
    # Inicialmente: X_modelo = X_assetto,
    #               Y_modelo = Z_assetto.
    x_modelo = float(x)
    y_modelo = float(z)

    # Conversión inicial de heading a orientación desde +X.
    psi = normalizar_angulo(
        float(fisica.heading) + math.pi / 2
    )

    # Conversión de km/h a m/s.
    velocidad = float(fisica.speedKmh) / 3.6

    return Estado(
        x=x_modelo,
        y=y_modelo,
        psi_rad=psi,
        velocidad_m_s=velocidad,
        delta_rad=0.0,
    )


# Traduce órdenes de seguimiento en señales para el controlador virtual.
def convertir_controles(controles: Controles, vehiculo: Vehiculo):
    """Prepara los controles del conductor para vJoy."""

    # Dirección física solicitada por el conductor.
    direccion_rad = controles.direccion_rad

    # Primera aproximación de conversión a [-1, 1].
    # Necesita calibración con el Porsche real.
    direccion_normalizada = (
        direccion_rad / vehiculo.giro_ruedas_max_rad
    )

    direccion_normalizada = float(max(
    -1.0, min(1.0, direccion_normalizada)
    ))

    return {
        "acelerador": controles.acelerador,
        "freno": controles.freno,
        "direccion": direccion_normalizada,
    }


if __name__ == "__main__":
    # Prueba de conversiones sin controlar Assetto Corsa.
    vehiculo = Vehiculo()

    controles = Controles(
        acelerador=0.35,
        freno=0.0,
        direccion_rad=math.radians(3),
    )

    resultado = convertir_controles(controles, vehiculo)

    print("Controles del conductor:", controles)
    print("Controles convertidos:", resultado)
    print("Ninguna orden enviada a Assetto Corsa.")
