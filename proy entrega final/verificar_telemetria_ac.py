"""DOCUMENTACIÓN DEL MÓDULO — PROYECTO FINAL.

Los comentarios explican entradas, transformaciones y supuestos. Se conserva
la lógica preexistente; los límites estimados no son mediciones directas de AC.
"""

# A. IMPORTACIONES
# Reutilizamos las funciones que ya existen en el proyecto.

from telemetria_assetto import (
    Physics,
    Graphics,
    Static,
    conectar,
    leer,
)


# B. CONEXIÓN CON ASSETTO CORSA

def main():

    # B1. Abrir los dos bloques de memoria compartida.
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    fisica = conectar("acpmf_physics", Physics)
    graficos = conectar("acpmf_graphics", Graphics)

    # Protege recursos externos y permite limpieza ante errores.
    try:

        # C. LEER LA TELEMETRÍA ACTUAL

        datos_fisicos = leer(fisica, Physics)
        datos_graficos = leer(graficos, Graphics)

        # C1. Estado del simulador.
        print("Estado:", datos_graficos.status)

        # C2. Velocidad real del automóvil.
        print("Velocidad:", datos_fisicos.speedKmh, "km/h")

        # C3. Acelerador y freno.
        print("Acelerador:", datos_fisicos.gas)
        print("Freno:", datos_fisicos.brake)

        # C4. Aceleraciones.
        print("Aceleraciones:", list(datos_fisicos.accG))

        # C5. Dirección.
        print("Volante:", datos_fisicos.steerAngle)
        
        
        # F. COMPUESTO ACTIVO
        print("Neumáticos:", datos_graficos.tyreCompound)
        
        # E. DATOS ESTÁTICOS DE ASSETTO CORSA
        estatica = conectar("acpmf_static", Static)

        # Protege recursos externos y permite limpieza ante errores.
        try:
            datos = leer(estatica, Static)

            print("Automóvil:", datos.carModel)
            print("Circuito:", datos.track)
            print("Variante:", datos.trackConfiguration)
            print("Potencia máxima:", datos.maxPower)


        finally:
            estatica.close()
        

    finally:

        # D. CERRAR LAS CONEXIONES
        fisica.close()
        graficos.close()


if __name__ == "__main__":
    main()