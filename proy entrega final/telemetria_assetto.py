"""DOCUMENTACIÓN DEL MÓDULO — PROYECTO FINAL.

Los comentarios explican entradas, transformaciones y supuestos. Se conserva
la lógica preexistente; los límites estimados no son mediciones directas de AC.
"""


import ctypes
import mmap
import time

from ctypes import c_float, c_int32, c_wchar


# 1. Estructura de los datos físicos.
# Respetamos exactamente el orden inicial de Shared Memory.

class Physics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int32),
        ("gas", c_float),
        ("brake", c_float),
        ("fuel", c_float),
        ("gear", c_int32),
        ("rpms", c_int32),
        ("steerAngle", c_float),
        ("speedKmh", c_float),

        # Velocidad del vehículo en coordenadas globales.
        ("velocity", c_float * 3),

        # Aceleración en coordenadas locales, medida en g.
        ("accG", c_float * 3),

        # Campos intermedios necesarios para conservar
        # las posiciones correctas en la memoria.
        ("wheelSlip", c_float * 4),
        ("wheelLoad", c_float * 4),
        ("wheelsPressure", c_float * 4),
        ("wheelAngularSpeed", c_float * 4),
        ("tyreWear", c_float * 4),
        ("tyreDirtyLevel", c_float * 4),
        ("tyreCoreTemperature", c_float * 4),
        ("camberRAD", c_float * 4),
        ("suspensionTravel", c_float * 4),
        ("drs", c_float),
        ("tc", c_float),

        # Orientación del vehículo.
        ("heading", c_float),
        ("pitch", c_float),
        ("roll", c_float),
    ]


# 2. Estructura de los datos de sesión y posición.
class Graphics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int32),
        ("status", c_int32),
        ("session", c_int32),
        ("currentTime", c_wchar * 15),
        ("lastTime", c_wchar * 15),
        ("bestTime", c_wchar * 15),
        ("split", c_wchar * 15),
        ("completedLaps", c_int32),
        ("position", c_int32),
        ("iCurrentTime", c_int32),
        ("iLastTime", c_int32),
        ("iBestTime", c_int32),
        ("sessionTimeLeft", c_float),
        ("distanceTraveled", c_float),
        ("isInPit", c_int32),
        ("currentSectorIndex", c_int32),
        ("lastSectorTime", c_int32),
        ("numberOfLaps", c_int32),
        ("tyreCompound", c_wchar * 33),
        ("replayTimeMultiplier", c_float),
        ("normalizedCarPosition", c_float),
        ("carCoordinates", c_float * 3),
    ]


# 3. Estructura con información del coche y circuito.

class Static(ctypes.Structure):
    _pack_ = 4

    _fields_ = [
        ("smVersion", c_wchar * 15),
        ("acVersion", c_wchar * 15),
        ("numberOfSessions", c_int32),
        ("numCars", c_int32),
        ("carModel", c_wchar * 33),
        ("track", c_wchar * 33),

        ("playerName", c_wchar * 33),
        ("playerSurname", c_wchar * 33),
        ("playerNick", c_wchar * 33),
        ("sectorCount", c_int32),

        ("maxTorque", c_float),
        ("maxPower", c_float),
        ("maxRpm", c_int32),
        ("maxFuel", c_float),

        ("suspensionMaxTravel", c_float * 4),
        ("tyreRadius", c_float * 4),

        ("maxTurboBoost", c_float),
        ("airTemp", c_float),
        ("roadTemp", c_float),
        ("penaltiesEnabled", c_int32),

        ("aidFuelRate", c_float),
        ("aidTireRate", c_float),
        ("aidMechanicalDamage", c_float),
        ("aidAllowTyreBlankets", c_int32),
        ("aidStability", c_float),
        ("aidAutoClutch", c_int32),
        ("aidAutoBlip", c_int32),

        ("hasDRS", c_int32),
        ("hasERS", c_int32),
        ("hasKERS", c_int32),
        ("kersMaxJ", c_float),
        ("engineBrakeSettingsCount", c_int32),
        ("ersPowerControllerCount", c_int32),

        ("trackSPlineLength", c_float),
        ("trackConfiguration", c_wchar * 33),
    ]



# Abre el segmento de memoria compartida correspondiente a la telemetría del juego.
def conectar(nombre, estructura):
    """Abre una zona de memoria compartida de Assetto Corsa."""
    memoria = mmap.mmap(
        -1,
        ctypes.sizeof(estructura),
        tagname=nombre,
        access=mmap.ACCESS_READ,
    )
    return memoria


# Copia y decodifica la estructura de telemetría para evitar lecturas parcialmente cambiantes.
def leer(memoria, estructura):
    """Convierte una copia de los bytes en datos Python."""
    memoria.seek(0)
    datos = memoria.read(ctypes.sizeof(estructura))
    return estructura.from_buffer_copy(datos)


# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main():
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    print("Buscando telemetría de Assetto Corsa...")

    fisica_mem = conectar("acpmf_physics", Physics)
    grafica_mem = conectar("acpmf_graphics", Graphics)
    estatica_mem = conectar("acpmf_static", Static)

    print("Memoria compartida abierta.")
    print("Presiona Ctrl+C para detener.\n")

    # Protege recursos externos y permite limpieza ante errores.
    try:
        # Repite el proceso hasta alcanzar la condición de parada o el límite de seguridad.
        while True:
            fisica = leer(fisica_mem, Physics)
            grafica = leer(grafica_mem, Graphics)
            estatica = leer(estatica_mem, Static)

            # 2 significa sesión LIVE en la interfaz de AC.
            if grafica.status != 2:
                print(
                    f"Esperando sesión activa... "
                    f"Estado: {grafica.status}"
                )
            else:
                
                # Posición del vehículo
                x, y, z = grafica.carCoordinates
    
                # Velocidad global en cada eje (m/s).
                vx, vy, vz = fisica.velocity

                # Aceleración local en unidades de g.
                ax_g, ay_g, az_g = fisica.accG

                # Orientación del vehículo (radianes).
                heading = fisica.heading
                pitch = fisica.pitch
                roll = fisica.roll

                print(
                    f"Coche: {estatica.carModel} | "
                    f"Pista: {estatica.track}"
                )
                print(
                    f"Velocidad: {fisica.speedKmh:.1f} km/h | "
                    f"RPM: {fisica.rpms} | "
                    f"Marcha: {fisica.gear}"
                )
                print(
                    f"Gas: {fisica.gas * 100:.1f}% | "
                    f"Freno: {fisica.brake * 100:.1f}% | "
                    f"Dirección RAW: {fisica.steerAngle:.4f}"
                )
                print(
                    f"Posición: X={x:.2f}, "
                    f"Y={y:.2f}, Z={z:.2f} | "
                    f"Boxes: {bool(grafica.isInPit)}"
                )
                
                
                print(
                    f"Velocidad vectorial (m/s): "
                    f"X={vx:.2f}, Y={vy:.2f}, Z={vz:.2f}"
                )

                print(
                    f"Aceleraciones locales (g): "
                    f"X={ax_g:.3f}, Y={ay_g:.3f}, Z={az_g:.3f}"
                )

                print(
                    f"Orientación (rad): "
                    f"Heading={heading:.4f}, "
                    f"Pitch={pitch:.4f}, Roll={roll:.4f}"
                )

                print("-" * 65)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nLectura detenida.")

    finally:
        fisica_mem.close()
        grafica_mem.close()
        estatica_mem.close()


if __name__ == "__main__":
    main()
