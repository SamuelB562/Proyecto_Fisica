"""Agente 1: conductor de un vehículo en una trayectoria de referencia.

Este módulo NO se conecta por sí mismo con Assetto Corsa Gym y NO planifica ápices.
El Agente 2 publica órdenes temporizadas y, opcionalmente, una trayectoria de
referencia. Un bucle externo lee telemetría, llama a ``actualizar`` y aplica
los tres controles devueltos a Assetto Corsa Gym.

Dependencias locales esperadas:
    modelado_fisico.py  -> clases Estado, Vehiculo (sección A)
    estimaciones.py     -> carro_dentro_de_pista (sección F; uso opcional)

Las órdenes son parciales: los campos no indicados mantienen su último valor.
Cuando termina la cola, la última orden permanece vigente (NO frena por falta
 de instrucciones). El agente conductor NO solicita replantear la trayectoria:
solo publica su estado y el porcentaje de pérdida para que el Agente 2 consulte.
"""

# ===========================================================================
# A. IMPORTACIONES Y DEPENDENCIAS
# ===========================================================================
# Este módulo organiza las órdenes y calcula controles; no abre Assetto Corsa.
# Estado y Vehiculo vienen del módulo físico; el detector de límites viene de F.
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Callable, Iterable, Optional
import math

import numpy as np

from modelado_fisico import Estado, Vehiculo
# Las verificaciones de F quedan disponibles si se pasa una pista al conductor.
from estimaciones import carro_dentro_de_pista


# ===========================================================================
# B. ESTRUCTURAS DE DATOS: ÓRDENES, CONTROLES E INFORMES
# ===========================================================================
# Las dataclasses almacenan información; por sí solas no ejecutan el vehículo.
# frozen=True impide modificar una instancia después de crearla.
@dataclass(frozen=True)
class Orden:
    """Cambios de referencias durante [inicio_s, fin_s).

    Los tiempos son segundos de simulación, no tiempo del reloj del computador.
    ``direccion_rad`` es el ángulo de las RUEDAS DELANTERAS, no del volante.
    ``acelerador`` y ``freno`` son porcentajes normalizados (0..1).
    Campos None: NO modificar ese canal respecto a la última referencia.
    Los canales pueden operar simultáneamente.
    """
    inicio_s: float  # Instante en que se aplicará la orden [s de simulación].
    fin_s: float  # Fin nominal de la orden; NO restaura referencias previas.
    acelerador: Optional[float] = None  # 0..1; None conserva el anterior.
    freno: Optional[float] = None  # 0..1; None conserva el anterior.
    direccion_rad: Optional[float] = None  # Ruedas delanteras; no volante.
    velocidad_objetivo_m_s: Optional[float] = None  # Referencia opcional para control de velocidad.

    # Gestiona la etapa «__post_init__» del flujo del módulo, validando los datos que procesa.
    def __post_init__(self) -> None:  # Valida que la orden tenga sentido antes de usarla.
        """Gestiona la etapa «__post_init__» del flujo del módulo, validando los datos que procesa."""
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if not (math.isfinite(self.inicio_s) and math.isfinite(self.fin_s)):
            raise ValueError("Los tiempos de las órdenes deben ser finitos")
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self.inicio_s < 0 or self.fin_s <= self.inicio_s:
            raise ValueError("Se requiere 0 <= inicio_s < fin_s")
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for campo in ("acelerador", "freno"):
            valor = getattr(self, campo)
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if valor is not None and (not math.isfinite(valor) or not 0 <= valor <= 1):
                raise ValueError(f"{campo} debe estar entre 0 y 1")
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self.direccion_rad is not None and not math.isfinite(self.direccion_rad):
            raise ValueError("direccion_rad debe ser finito")
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self.velocidad_objetivo_m_s is not None and (
            not math.isfinite(self.velocidad_objetivo_m_s) or self.velocidad_objetivo_m_s < 0
        ):
            raise ValueError("velocidad_objetivo_m_s debe ser no negativo")


# Una orden expresa referencias deseadas; Controles almacena lo que aplicaremos.
# Cada valor se conserva por separado si la orden siguiente no lo menciona.
@dataclass(frozen=True)
class Controles:
    acelerador: float  # Control final normalizado [0,1].
    freno: float  # Control final normalizado [0,1].
    direccion_rad: float  # Ángulo de ruedas aplicado [rad].


# El informe es la salida pública de cada actualización del conductor.
# Optional indica que puede no haber medición (None), por ejemplo sin ruta.
@dataclass(frozen=True)
class InformeConductor:
    tiempo_s: float  # Momento del informe.
    controles: Controles  # Acción calculada para el simulador.
    error_lateral_m: Optional[float]  # Desviación con signo de la ruta [m].
    error_orientacion_rad: Optional[float]  # Diferencia angular respecto a la ruta [rad].
    perdida_trayectoria_pct: Optional[float]  # Desviación lateral normalizada [0,100].
    dentro_pista: Optional[bool]  # None si no se configuró pista.
    ordenes_pendientes: int  # Número de órdenes futuras en cola.

# ===========================================================================
# C. AGENTE CONDUCTOR: CONFIGURACIÓN Y MEMORIA INTERNA
# ===========================================================================
class AgenteConductor:
    """Ejecutor temporal con seguimiento y límites de actuadores.

    Las órdenes se aplican al llegar su ``inicio_s``. ``fin_s`` indica hasta
    cuándo está activa la orden, pero NO borra sus valores al finalizar.
    Si el planificador no publica nada más, se conservan las últimas referencias.
    """

    # Gestiona la etapa «__init__» del flujo del módulo, validando los datos que procesa.
    def __init__(
        self,
        vehiculo: Vehiculo,
        *,
        periodo_control_s: float = 0.02,
        tasa_acelerador_por_s: float = 3.0,
        tasa_freno_por_s: float = 4.0,
        ganancia_velocidad: float = 0.25,
        ganancia_lateral: float = 0.06,
        ganancia_orientacion: float = 0.7,
        margen_perdida_m: float = 2.0,
    ) -> None:
        """Gestiona la etapa «__init__» del flujo del módulo, validando los datos que procesa."""
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if periodo_control_s <= 0 or margen_perdida_m <= 0:
            raise ValueError("Periodo y margen deben ser positivos")
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if tasa_acelerador_por_s <= 0 or tasa_freno_por_s <= 0:
            raise ValueError("Las tasas de actuadores deben ser positivas")
        self.veh = vehiculo  # Parámetros mecánicos del vehículo (parte A).
        self.periodo_control_s = periodo_control_s  # dt usado en la primera actualización.
        self.tasa_acelerador_por_s = tasa_acelerador_por_s  # Cambio máximo del pedal por segundo.
        self.tasa_freno_por_s = tasa_freno_por_s  # Cambio máximo del freno por segundo.
        self.k_v = ganancia_velocidad  # Respuesta proporcional al error de velocidad.
        self.k_e = ganancia_lateral  # Corrección de dirección por error lateral.
        self.k_psi = ganancia_orientacion  # Corrección de dirección por error angular.
        self.margen_perdida_m = margen_perdida_m  # Distancia equivalente a 100 % de pérdida.

        self._cola: deque[Orden] = deque()  # Instrucciones pendientes ordenadas temporalmente.
        self._ultima_orden_inicio = -math.inf  # Marca del último inicio aplicado.
        self._referencia = Controles(0.0, 0.0, 0.0)  # Últimos objetivos de control recibidos.
        self._velocidad_objetivo: Optional[float] = None  # Si None, usa pedales de las órdenes.
        self._aplicado = Controles(0.0, 0.0, 0.0)  # Controles realmente solicitados al juego.
        self._trayectoria: Optional[np.ndarray] = None  # Array (N,2) de referencia.
        self._centro_pista: Optional[np.ndarray] = None  # Centro geométrico para diagnóstico F.
        self._ancho_pista: Optional[float] = None  # Ancho constante del corredor simplificado.
        self._tiempo_anterior: Optional[float] = None  # Necesario para calcular dt real.
        self._informe: Optional[InformeConductor] = None  # Última lectura disponible para Agente 2.

    # =======================================================================
    # D. COLA TEMPORAL: AGREGAR, REEMPLAZAR Y VALIDAR ÓRDENES
    # =======================================================================
    def agregar_ordenes(self, ordenes: Iterable[Orden]) -> None:
        """Añade órdenes futuras no solapadas, ordenadas por inicio.

        Para cambiar una planificación futura usa ``reemplazar_futuras``.
        """
        nuevas = list(ordenes)  # Materializa un iterable para comprobarlo una sola vez.
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if any(not isinstance(o, Orden) for o in nuevas):
            raise TypeError("Todas las instrucciones deben ser Orden")
        conjunto = list(self._cola) + nuevas  # Une pendientes y nuevas órdenes.
        conjunto.sort(key=lambda o: o.inicio_s)  # Ordena por tiempo de inicio.
        self._validar_no_solapadas(conjunto)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self._tiempo_anterior is not None and any(
            o.inicio_s < self._tiempo_anterior for o in nuevas
        ):
            raise ValueError("No se pueden añadir instrucciones en el pasado")
        self._cola = deque(conjunto)  # Actualización de cola validada (no altera referencias activas).

    # Gestiona la etapa «reemplazar_futuras» del flujo del módulo, validando los datos que procesa.
    def reemplazar_futuras(self, desde_s: float, ordenes: Iterable[Orden]) -> None:
        """Reemplaza de forma atómica el segmento futuro de la cola.

        Nunca cambia referencias ya aplicadas. Las órdenes con inicio anterior
        a ``desde_s`` se conservan; las posteriores se sustituyen. El instante
        de corte no puede estar en el pasado.
        """
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self._tiempo_anterior is not None and desde_s < self._tiempo_anterior:
            raise ValueError("El corte no puede estar en el pasado")
        nuevas = list(ordenes)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if any(not isinstance(o, Orden) for o in nuevas):
            raise TypeError("Todas las instrucciones deben ser Orden")
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if any(o.inicio_s < desde_s for o in nuevas):
            raise ValueError("Toda nueva orden debe comenzar en o después del corte")
        conservadas = [o for o in self._cola if o.inicio_s < desde_s]  # Mantiene órdenes previas al corte.
        conjunto = sorted(conservadas + nuevas, key=lambda o: o.inicio_s)
        self._validar_no_solapadas(conjunto)
        self._cola = deque(conjunto)

    # Gestiona la etapa «_validar_no_solapadas» del flujo del módulo, validando los datos que procesa.
    @staticmethod
    def _validar_no_solapadas(ordenes: list[Orden]) -> None:
        """Gestiona la etapa «_validar_no_solapadas» del flujo del módulo, validando los datos que procesa."""
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for anterior, siguiente in zip(ordenes, ordenes[1:]):
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if anterior.fin_s > siguiente.inicio_s:
                raise ValueError("Las órdenes no deben solaparse; combine los canales en una orden")

    # =======================================================================
    # E. REFERENCIAS GEOMÉTRICAS: TRAYECTORIA Y PISTA
    # =======================================================================
    def establecer_trayectoria(self, puntos_xy: Iterable[Iterable[float]]) -> None:
        """Referencia geométrica opcional del Agente 2; no crea nueva trayectoria."""
        puntos = np.asarray(puntos_xy, dtype=float)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if (puntos.ndim != 2 or puntos.shape[1] != 2 or len(puntos) < 2
                or not np.all(np.isfinite(puntos))):
            raise ValueError("La trayectoria debe tener forma (N,2) con N>=2")
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if np.any(np.linalg.norm(np.diff(puntos, axis=0), axis=1) < 1e-9):
            raise ValueError("La trayectoria contiene puntos consecutivos repetidos")
        self._trayectoria = puntos.copy()  # Almacena copia para evitar modificaciones externas.

    # Gestiona la etapa «establecer_pista» del flujo del módulo, validando los datos que procesa.
    def establecer_pista(self, centro_xy: Iterable[Iterable[float]], ancho_m: float) -> None:
        """Permite diagnóstico F: salir de pista no detiene automáticamente el coche."""
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if ancho_m <= 0:
            raise ValueError("El ancho de pista debe ser positivo")
        centro = np.asarray(centro_xy, dtype=float)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if centro.ndim != 2 or centro.shape[1] != 2 or len(centro) < 2:
            raise ValueError("Centro de pista inválido")
        self._centro_pista = centro.copy()  # Datos para comprobar la huella del vehículo.
        self._ancho_pista = float(ancho_m)  # Se supone un ancho constante.

    # =======================================================================
    # F. CICLO PRINCIPAL DE CONDUCCIÓN: UNA ACTUALIZACIÓN POR TICK
    # =======================================================================
    def actualizar(self, tiempo_s: float, estado: Estado) -> InformeConductor:
        """Produce una acción a partir de la telemetría y la última planificación.

        Llamar una vez por ciclo de control, con tiempo de SIMULACIÓN creciente.
        NO mueve el vehículo: el adaptador externo ejecuta los controles.
        """
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if not math.isfinite(tiempo_s) or tiempo_s < 0:
            raise ValueError("Tiempo inválido")
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self._tiempo_anterior is not None and tiempo_s <= self._tiempo_anterior:
            raise ValueError("El tiempo de simulación debe aumentar")
        # dt se obtiene de los instantes reales de simulación, no del reloj externo.
        dt = (tiempo_s - self._tiempo_anterior if self._tiempo_anterior is not None
              else self.periodo_control_s)
        self._tiempo_anterior = tiempo_s  # Guarda marca temporal para próximo dt.

        # Aplicar todas las órdenes cuyo inicio ya llegó; conservar la última.
        # Una misma actualización puede consumir varias órdenes si el tiempo saltó.
        # fin_s no desactiva el control: la última referencia sigue vigente.
        while self._cola and self._cola[0].inicio_s <= tiempo_s:
            orden = self._cola.popleft()  # Quita la orden al llegar su inicio.
            self._ultima_orden_inicio = orden.inicio_s
            r = self._referencia  # Valores anteriores para conservar canales omitidos.
            self._referencia = Controles(
                r.acelerador if orden.acelerador is None else orden.acelerador,
                r.freno if orden.freno is None else orden.freno,
                r.direccion_rad if orden.direccion_rad is None else orden.direccion_rad,
            )
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if orden.velocidad_objetivo_m_s is not None:  # Solo actualiza si trae un objetivo.
                self._velocidad_objetivo = orden.velocidad_objetivo_m_s

        error_lat = None
        error_psi = None
        objetivo_dir = self._referencia.direccion_rad  # Dirección solicitada por Agente 2.
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self._trayectoria is not None:
            error_lat, error_psi = self._errores_trayectoria(estado, self._trayectoria)  # Desviaciones medidas.
            # Seguimiento local: cambia las acciones, NO la ruta del Agente 2.
            objetivo_dir += self.k_e * error_lat + self.k_psi * error_psi  # Feedback lateral y angular.

        max_dir = float(self.veh.giro_ruedas_max_rad)  # Ángulo máximo permitido por las ruedas.
        objetivo_dir = float(np.clip(objetivo_dir, -max_dir, max_dir))
        max_delta = float(self.veh.velocidad_giro_ruedas_max_rad_s) * dt  # Giro máximo durante este tick.
        # Limita la variación desde el ángulo previamente aplicado.
        # No asegura por sí solo que la ruta sea viable con la adherencia real.
        direccion = float(np.clip(objetivo_dir,
                                  self._aplicado.direccion_rad - max_delta,
                                  self._aplicado.direccion_rad + max_delta))

        acelerador = self._referencia.acelerador  # Pedal solicitado por la última orden.
        freno = self._referencia.freno  # Freno solicitado por la última orden.
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self._velocidad_objetivo is not None:
            # Control proporcional de velocidad (sin optimizar la trayectoria).
            error_v = self._velocidad_objetivo - float(estado.velocidad_m_s)  # Objetivo menos velocidad real.
            demanda = self.k_v * error_v  # Control proporcional (P), sin término integral.
            acelerador = float(np.clip(demanda, 0.0, 1.0))  # Demanda positiva -> acelerador.
            freno = float(np.clip(-demanda, 0.0, 1.0))  # Demanda negativa -> freno.
            # Si no alcanzamos el giro solicitado, evitar acelerar hacia una
            # referencia de velocidad demasiado exigente; no borra la orden.
            error_giro = abs(objetivo_dir - direccion)  # Giro pendiente de alcanzar.
            if error_giro > math.radians(2.0) and estado.velocidad_m_s > 1:
                acelerador = 0.0  # Evita seguir acelerando ante gran error de giro.
                freno = max(freno, min(0.5, error_giro / max_dir * 0.5))

        # Suaviza los PEDALES; primero se calcula la demanda y después se limita.
        acelerador = self._limitar_tasa(acelerador, self._aplicado.acelerador,
                                        self.tasa_acelerador_por_s * dt)
        freno = self._limitar_tasa(freno, self._aplicado.freno,
                                  self.tasa_freno_por_s * dt)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if freno > 0.01:
            acelerador = 0.0  # Evitar acelerar y frenar simultáneamente.
        self._aplicado = Controles(acelerador, freno, direccion)

        # La pérdida es una métrica de seguimiento, no probabilidad de choque.
        perdida = None
        if error_lat is not None:
            perdida = 100.0 * min(1.0, abs(error_lat) / self.margen_perdida_m)  # 2 m/2 m -> 100 %.
        dentro = None
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self._centro_pista is not None:
            dentro = bool(carro_dentro_de_pista(
                estado, self.veh, self._centro_pista, self._ancho_pista
            ))
        self._informe = InformeConductor(
            tiempo_s, self._aplicado, error_lat, error_psi, perdida,
            dentro, len(self._cola)
        )
        return self._informe  # Última lectura, sin volver a calcular ni mover el coche.

    # =======================================================================
    # G. OPERACIONES AUXILIARES: SUAVIZADO Y ERROR DE SEGUIMIENTO
    # =======================================================================
    @staticmethod
    def _limitar_tasa(objetivo: float, actual: float, max_cambio: float) -> float:
        """Gestiona la etapa «_limitar_tasa» del flujo del módulo, validando los datos que procesa."""
        return float(np.clip(objetivo, actual - max_cambio, actual + max_cambio))  # Limita variación por tick.

    # Gestiona la etapa «_errores_trayectoria» del flujo del módulo, validando los datos que procesa.
    @staticmethod
    def _errores_trayectoria(estado: Estado, trayectoria: np.ndarray) -> tuple[float, float]:
        """Error lateral con signo y error angular en el segmento más cercano."""
        pos = np.array([estado.x, estado.y], dtype=float)  # Posición de eje trasero en 2D.
        inicio = trayectoria[:-1]  # Todos los comienzos de segmentos.
        vectores = np.diff(trayectoria, axis=0)  # Diferencias P[i+1]-P[i].
        long2 = np.sum(vectores * vectores, axis=1)  # Longitud al cuadrado de cada segmento.
        # Proyecta el carro sobre CADA segmento; recorta t a [0,1] para no
        # prolongar segmentos indefinidamente. Es geometría, no predicción.
        t = np.clip(np.sum((pos - inicio) * vectores, axis=1) / long2, 0.0, 1.0)
        proyecciones = inicio + t[:, None] * vectores  # Pie perpendicular sobre cada segmento.
        i = int(np.argmin(np.sum((proyecciones - pos) ** 2, axis=1)))  # Índice del segmento cercano.
        tangente = vectores[i] / math.sqrt(float(long2[i]))  # Dirección unitaria de la ruta.
        normal_izq = np.array([-tangente[1], tangente[0]])  # Normal izquierda de la ruta.
        # +error_lat: coche a la izquierda -> corregir a la derecha.
        error_lat = float(np.dot(pos - proyecciones[i], normal_izq))  # Distancia lateral firmada.
        # +error_psi: coche apunta a la izquierda -> corregir a la derecha.
        psi_ref = math.atan2(float(tangente[1]), float(tangente[0]))
        error_psi = math.atan2(math.sin(estado.psi_rad - psi_ref),
                               math.cos(estado.psi_rad - psi_ref))
        return -error_lat, -error_psi  # Signo invertido para corregir hacia la ruta.

    # =======================================================================
    # H. SALIDAS DEL AGENTE: INFORME PARA EL PLANIFICADOR Y CONTROLES
    # =======================================================================
    def consultar_estado(self) -> Optional[InformeConductor]:
        """Lectura sin efectos secundarios para el polling del Agente 2."""
        return self._informe

    # Gestiona la etapa «controles_actuales» del flujo del módulo, validando los datos que procesa.
    @property
    def controles_actuales(self) -> Controles:
        """Gestiona la etapa «controles_actuales» del flujo del módulo, validando los datos que procesa."""
        return self._aplicado  # Último comando calculado para enviar a Gym.


# ===========================================================================
# I. EJEMPLO DE USO (SOLO COMENTARIOS, NO SE EJECUTA)
# ===========================================================================
# EJEMPLO DE INTEGRACIÓN (no se ejecuta automáticamente):
# conductor = AgenteConductor(Vehiculo())
# conductor.agregar_ordenes([
#     Orden(0.0, 2.0, acelerador=0.8),
#     Orden(2.0, 2.3, direccion_rad=math.radians(10)),
# ])
# conductor.establecer_trayectoria([[0, 0], [10, 0], [20, -2]])
# En cada tick del simulador:
#     informe = conductor.actualizar(tiempo_simulacion_s, estado_actual)
#     controles = informe.controles
#     # El adaptador Gym traduce controles a throttle/brake/steer propios de Gym.
#     # El Agente 2 consulta conductor.consultar_estado() cuando lo desee.
