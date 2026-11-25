"""DOCUMENTACIÓN DEL MÓDULO — PROYECTO FINAL.

Los comentarios explican entradas, transformaciones y supuestos. Se conserva
la lógica preexistente; los límites estimados no son mediciones directas de AC.
"""


import pyvjoy


# Aísla la interfaz del dispositivo virtual y garantiza el envío y liberación de los mandos.
class ControladorVJoy:
    """Envía controles a Assetto Corsa mediante vJoy."""

    MINIMO = 0x0001
    MAXIMO = 0x8000
    CENTRO = (MINIMO + MAXIMO) // 2

    # Gestiona la etapa «__init__» del flujo del módulo, validando los datos que procesa.
    def __init__(self, dispositivo=1):
        # Conexión al volante virtual.
        """Gestiona la etapa «__init__» del flujo del módulo, validando los datos que procesa."""
        self.volante = pyvjoy.VJoyDevice(dispositivo)

        # Empezamos sin acelerar, frenar ni girar.
        self.reposo()

    # Gestiona la etapa «limitar» del flujo del módulo, validando los datos que procesa.
    @staticmethod
    def limitar(valor, minimo, maximo):
        """Mantiene un valor dentro de un intervalo."""
        return max(minimo, min(maximo, float(valor)))

    # Gestiona la etapa «enviar» del flujo del módulo, validando los datos que procesa.
    def enviar(self, acelerador=0.0, freno=0.0, direccion=0.0):
        """
        acelerador: 0.0 a 1.0
        freno:      0.0 a 1.0
        direccion: -1.0 a 1.0
        """

        acelerador = self.limitar(acelerador, 0.0, 1.0)
        freno = self.limitar(freno, 0.0, 1.0)
        direccion = self.limitar(direccion, -1.0, 1.0)

        # Conversión de los controles a valores vJoy.
        eje_x = round(
            self.MINIMO
            + (direccion + 1.0) / 2.0
            * (self.MAXIMO - self.MINIMO)
        )

        eje_y = round(
            self.MINIMO
            + acelerador * (self.MAXIMO - self.MINIMO)
        )

        eje_z = round(
            self.MINIMO
            + freno * (self.MAXIMO - self.MINIMO)
        )

        # Enviar las órdenes al dispositivo virtual.
        self.volante.set_axis(pyvjoy.HID_USAGE_X, eje_x)
        self.volante.set_axis(pyvjoy.HID_USAGE_Y, eje_y)
        self.volante.set_axis(pyvjoy.HID_USAGE_Z, eje_z)

    # Gestiona la etapa «reposo» del flujo del módulo, validando los datos que procesa.
    def reposo(self):
        """Devuelve los controles a su posición neutral."""
        self.enviar(0.0, 0.0, 0.0)

    # Gestiona la etapa «cerrar» del flujo del módulo, validando los datos que procesa.
    def cerrar(self):
        """Libera los controles dejando los ejes en reposo."""
        self.reposo()
