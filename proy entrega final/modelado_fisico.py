"""MODELO FÍSICO DE CARRERAS — VERSIÓN NUMPY, EXPLICADA PASO A PASO.

No existe un agente de IA en este archivo: solo geometría, cinemática,
restricciones físicas, estimación de tiempos y una búsqueda matemática opcional (sin pruebas ni demostraciones).

Instalación: python -m pip install numpy
Uso: import modelo_fisico_carreras_numpy_limpio  # módulo sin ejecución automática

CONVENCIONES IMPORTANTES:
- Para el CARRO: x=Este, y=Norte y psi se mide desde el eje +x.
- Para las INTEGRALES DEL PAPER: psi_r se mide desde +Norte y crece hacia -Este.
  Por eso dE/ds=-sin(psi_r), dN/ds=cos(psi_r).
- La posición (x,y) de la bicicleta corresponde al centro del eje TRASERO.
- La carrocería se representa, por separado, con un RECTÁNGULO de colisión.
- Las fórmulas de agarre son aproximaciones; NO reproducen derrapes reales.
- La búsqueda no garantiza el óptimo global ni simula Assetto Corsa.
"""

from __future__ import annotations  # Permite usar anotaciones de tipos con comodidad.

from dataclasses import dataclass  # Crea clases sencillas de parámetros y resultados.
import math  # Constantes matemáticas y funciones escalares.
import numpy as np  # Arrays, vectores, diferencias y operaciones por lotes.


G = 9.81  # Aceleración gravitacional terrestre aproximada [m/s²].
EPS = 1e-9  # Pequeña tolerancia para evitar divisiones entre cero.


# ===========================================================================
# A. PARÁMETROS DEL VEHÍCULO Y SU ESTADO
# ===========================================================================
@dataclass(frozen=True)
class Vehiculo:
    """Constantes aproximadas del carro (cambiar tras medirlo en el simulador)."""

    masa_kg: float = 1200.0  # Masa total del automóvil [kg].
    potencia_util_w: float = 125_000.0  # Potencia efectiva disponible [W].
    distancia_ejes_m: float = 2.60  # Separación entre eje delantero y trasero [m].
    largo_m: float = 4.20  # Longitud de carrocería para colisiones [m].
    ancho_m: float = 1.80  # Ancho de carrocería para colisiones [m].
    relacion_direccion: float = 15.0  # Ángulo volante / ángulo ruedas.
    giro_ruedas_max_rad: float = np.deg2rad(30.0)  # Giro máximo de ruedas [rad].
    velocidad_giro_ruedas_max_rad_s: float = np.deg2rad(360.0)  # Rapidez máxima [rad/s].
    aceleracion_max_m_s2: float = 3.0  # Máxima aceleración longitudinal [m/s²].
    frenado_max_m_s2: float = 8.0  # Máxima desaceleración longitudinal [m/s²].
    coef_friccion: float = 1.1  # Coeficiente de adherencia global (supuesto).

    # Gestiona la etapa «__post_init__» del flujo del módulo, validando los datos que procesa.
    def __post_init__(self):
        # Valida parámetros que físicamente deben ser positivos.
        """Gestiona la etapa «__post_init__» del flujo del módulo, validando los datos que procesa."""
        for nombre in ('masa_kg', 'potencia_util_w', 'distancia_ejes_m',
                       'largo_m', 'ancho_m', 'relacion_direccion',
                       'aceleracion_max_m_s2', 'frenado_max_m_s2', 'coef_friccion',
                       'giro_ruedas_max_rad', 'velocidad_giro_ruedas_max_rad_s'):
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if getattr(self, nombre) <= 0:  # Lee el atributo por su nombre.
                raise ValueError(f'{nombre} debe ser positivo')


# Describe posición, orientación, velocidad y giro de las ruedas del modelo.
@dataclass(frozen=True)
class Estado:
    """Estado instantáneo de la bicicleta, ubicada en su eje trasero."""

    x: float  # Coordenada Este del eje trasero [m].
    y: float  # Coordenada Norte del eje trasero [m].
    psi_rad: float  # Orientación del auto desde +x [rad].
    velocidad_m_s: float  # Rapidez longitudinal [m/s].
    delta_rad: float = 0.0  # Giro de ruedas DELANTERAS [rad], no del volante.


# Agrupa tiempo, distancia, perfiles y porcentaje de adherencia consumida.
@dataclass
class Evaluacion:
    """Resultados de evaluar una trayectoria determinada."""

    tiempo_s: float  # Tiempo estimado para recorrerla [s].
    longitud_m: float  # Longitud total de la trayectoria [m].
    velocidades_m_s: np.ndarray  # Velocidad estimada en cada punto [m/s].
    curvaturas_menos1: np.ndarray  # Curvatura geométrica con signo [1/m].
    direcciones_rad: np.ndarray  # Ángulo de ruedas necesario [rad].
    max_uso_lateral: float  # Máxima fracción de adherencia lateral utilizada.


# ===========================================================================
# B. UTILIDADES DE ARRAYS Y GEOMETRÍA DE PISTA
# ===========================================================================
def validar_puntos(puntos, minimo=2):
    """Convierte cualquier colección de puntos en un array de forma (N,2)."""
    p = np.asarray(puntos, dtype=float)  # Convierte lista o array a NumPy.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < minimo:
        raise ValueError(f'Se requieren al menos {minimo} puntos (N, 2)')
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not np.all(np.isfinite(p)):  # Rechaza valores NaN o infinito.
        raise ValueError('Los puntos deben ser finitos')
    return p  # Cada fila representa [x, y].


# Calcula diferencia angular mínima y evita discontinuidades de 2π.
def diferencia_angular(a, b):
    """Diferencia de ángulos en [-pi,pi], incluso si cruzan 360°."""
    return np.arctan2(np.sin(np.asarray(a)-b), np.cos(np.asarray(a)-b))


# Integra la orientación y posición utilizando la curvatura de cada segmento.
def integrar_pista_desde_curvatura_paper(curvaturas_menos1, ds_m=1.0,
                                         este_inicial=0.0, norte_inicial=0.0,
                                         psi_inicial_rad=0.0):
    """Integra las tres ecuaciones (1a–1c) de geometría del paper.

    psi_r(s) = psi_inicial + integral(K(z) dz)
    E(s) = E_inicial - integral(sin(psi_r(z)) dz)
    N(s) = N_inicial + integral(cos(psi_r(z)) dz)

    Se usa el método del punto medio en cada sección de longitud ds_m.
    Se supone K constante dentro de cada sección.
    """
    k = np.asarray(curvaturas_menos1, dtype=float)  # Curvatura de cada tramo.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if k.ndim != 1 or not np.all(np.isfinite(k)) or ds_m <= 0:
        raise ValueError('Curvaturas 1D finitas y ds_m positivo requeridos')
    cambios_psi = k * ds_m  # Cambio angular de cada segmento: dpsi = K ds.
    psi = psi_inicial_rad + np.r_[0.0, np.cumsum(cambios_psi)]  # Ángulos en nodos.
    psi_medio = psi[:-1] + cambios_psi / 2  # Ángulo en la mitad de cada tramo.
    delta_este = -np.sin(psi_medio) * ds_m  # (1a) dE ≈ -sin(psi) ds.
    delta_norte = np.cos(psi_medio) * ds_m  # (1b) dN ≈ cos(psi) ds.
    este = este_inicial + np.r_[0.0, np.cumsum(delta_este)]  # Integra Este.
    norte = norte_inicial + np.r_[0.0, np.cumsum(delta_norte)]  # Integra Norte.
    return np.column_stack((este, norte)), psi  # Puntos (N+1,2) y ángulos.


# Acumula longitudes euclidianas entre puntos consecutivos.
def distancias_acumuladas(puntos):
    """Calcula s_i = suma de distancias entre puntos consecutivos."""
    p = validar_puntos(puntos)  # Obtiene un array (N,2).
    tramos = np.diff(p, axis=0)  # Resta puntos consecutivos: P[i+1]-P[i].
    ds = np.linalg.norm(tramos, axis=1)  # Distancia euclidiana de cada tramo.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(ds < EPS):  # Evita segmentos de longitud cero.
        raise ValueError('Existen puntos consecutivos coincidentes')
    return np.r_[0.0, np.cumsum(ds)]  # Distancia acumulada desde el inicio.


# Estima las tangentes de la trayectoria con diferencias entre segmentos vecinos.
def orientaciones_de_puntos(puntos):
    """Estima el ángulo tangente desde +x de cada punto de pista."""
    p = validar_puntos(puntos)  # Puntos de forma (N,2).
    segmentos = np.diff(p, axis=0)  # Vectores tangentes entre pares vecinos.
    angulos_segmento = np.arctan2(segmentos[:, 1], segmentos[:, 0])  # atan2(dy,dx).
    angulos_segmento = np.unwrap(angulos_segmento)  # Evita saltos de +pi a -pi.
    psi = np.empty(len(p))  # Reserva un ángulo por cada punto.
    psi[0] = angulos_segmento[0]  # En el inicio usa el primer segmento.
    psi[-1] = angulos_segmento[-1]  # En la salida usa el último segmento.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if len(p) > 2:
        psi[1:-1] = 0.5 * (angulos_segmento[:-1] + angulos_segmento[1:])
        # En el interior promedia los ángulos vecinos (suficiente con muestreo fino).
    return psi  # Ángulos en radianes, desde el eje +x.


# Estima la curvatura firmada de cada triple de puntos mediante el círculo circunscrito.
def curvaturas_de_puntos(puntos):
    """Curvatura con signo de círculos por cada tres puntos consecutivos.

    k = 2*det(B-A,C-A)/(|B-A| |C-B| |C-A|).
    Los extremos toman el valor de su vecino (aproximación).
    """
    p = validar_puntos(puntos, minimo=3)  # Necesitamos triples de puntos.
    ab = p[1:-1] - p[:-2]  # Vector A->B para todos los triples.
    bc = p[2:] - p[1:-1]  # Vector B->C para todos los triples.
    ac = p[2:] - p[:-2]  # Vector A->C para todos los triples.
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]  # Determinante 2D.
    denom = (np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)
             * np.linalg.norm(ac, axis=1))  # Producto de longitudes.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(denom < EPS):
        raise ValueError('Segmento degenerado en curvatura')
    interior = 2 * det / denom  # Curvatura en los puntos interiores.
    return np.r_[interior[0], interior, interior[-1]]  # Completa los extremos.


# Obtiene vectores unitarios normales a la tangente de la ruta.
def normales_izquierdas(puntos):
    """Vector normal izquierdo N=(-sin(psi), cos(psi)) en cada punto."""
    psi = orientaciones_de_puntos(puntos)  # Dirección tangente de cada punto.
    return np.column_stack((-np.sin(psi), np.cos(psi)))  # Una normal por fila.


# Aplica desplazamientos laterales dentro del ancho útil permitido de la pista.
def desplazar_puntos(puntos_centro, desplazamientos_m, ancho_pista_m,
                     ancho_coche_m, margen_m=0.15):
    """Genera P_i = C_i + d_i*N_i sin superar el ancho útil aproximado.

    Este filtro geométrico NO sustituye comprobar las esquinas del carro.
    """
    centro = validar_puntos(puntos_centro)  # Array (N,2) de línea central.
    d = np.asarray(desplazamientos_m, dtype=float)  # Desplazamiento por punto.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if d.shape != (len(centro),) or not np.all(np.isfinite(d)):
        raise ValueError('Se requiere un desplazamiento finito por punto')
    max_d = (ancho_pista_m - ancho_coche_m) / 2 - margen_m  # Margen aproximado.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if max_d < 0 or np.any(np.abs(d) > max_d + EPS):
        raise ValueError('Desplazamiento fuera del ancho de pista permitido')
    n = normales_izquierdas(centro)  # Calcula todas las normales en NumPy.
    return centro + d[:, None] * n  # Broadcast: suma d_i*N_i a cada punto.


# ===========================================================================
# C. BICICLETA CINEMÁTICA: MOVIMIENTO DE UN ÚNICO VEHÍCULO
# ===========================================================================
def giro_volante_a_ruedas(theta_volante_rad, veh):
    """Convierte ángulo del volante a ángulo de rueda mediante la relación de dirección."""
    return theta_volante_rad / veh.relacion_direccion  # Modelo dirección fija.


# Convierte ángulo de rueda a ángulo del volante.
def giro_ruedas_a_volante(delta_rad, veh):
    """Convierte ángulo de rueda a ángulo del volante."""
    return delta_rad * veh.relacion_direccion  # Operación inversa a la anterior.


# Calcula la dirección requerida por la bicicleta cinemática para una curvatura dada.
def curvatura_a_direccion(kappa, veh):
    """Calcula la dirección requerida por la bicicleta cinemática para una curvatura dada."""
    return np.arctan(veh.distancia_ejes_m * np.asarray(kappa))  # delta=atan(L*k).


# Calcula el radio de giro a partir del ángulo de la rueda delantera.
def direccion_a_radio(delta_rad, veh):
    """Calcula el radio de giro a partir del ángulo de la rueda delantera."""
    tangente = math.tan(delta_rad)  # Tangente del ángulo de ruedas.
    return math.inf if abs(tangente) < EPS else veh.distancia_ejes_m / tangente


# Evalúa las cuatro derivadas de la bicicleta cinemática para un estado y aceleración.
def derivadas_bicicleta(estado, aceleracion_m_s2, veh):
    """Ecuaciones: x'=v cos(psi); y'=v sin(psi); psi'=v tan(delta)/L; v'=a."""
    v = estado.velocidad_m_s  # Velocidad instantánea del carro.
    psi = estado.psi_rad  # Orientación actual del carro.
    delta = estado.delta_rad  # Ángulo actual de las ruedas delanteras.
    dx = v * math.cos(psi)  # Velocidad en eje Este [m/s].
    dy = v * math.sin(psi)  # Velocidad en eje Norte [m/s].
    dpsi = (v / veh.distancia_ejes_m) * math.tan(delta)  # Giro [rad/s].
    dv = aceleracion_m_s2  # Cambio de rapidez [m/s²].
    return np.array([dx, dy, dpsi, dv], dtype=float)  # Vector derivada.


# Integra un paso de Euler y respeta límites de posición de dirección y su velocidad de cambio.
def paso_bicicleta(estado, aceleracion_m_s2, delta_objetivo_rad, dt_s, veh):
    """Un paso temporal de Euler; controla límite y rapidez de dirección."""
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if dt_s <= 0:
        raise ValueError('El tiempo debe ser positivo')
    objetivo = np.clip(delta_objetivo_rad, -veh.giro_ruedas_max_rad,
                       veh.giro_ruedas_max_rad)  # Límite mecánico.
    cambio_max = veh.velocidad_giro_ruedas_max_rad_s * dt_s  # Giro permitido.
    cambio = np.clip(objetivo - estado.delta_rad, -cambio_max, cambio_max)
    delta = float(estado.delta_rad + cambio)  # Nueva dirección de las ruedas.
    actual = Estado(estado.x, estado.y, estado.psi_rad, estado.velocidad_m_s,
                    delta)  # Usa el ángulo actualizado para estimar movimiento.
    derivadas = derivadas_bicicleta(actual, aceleracion_m_s2, veh)  # dx/dt, etc.
    valor_actual = np.array([actual.x, actual.y, actual.psi_rad,
                             actual.velocidad_m_s])  # Estado en vector NumPy.
    siguiente = valor_actual + derivadas * dt_s  # Euler: nuevo=actual+deriv*dt.
    return Estado(float(siguiente[0]), float(siguiente[1]), float(siguiente[2]),
                  max(0.0, float(siguiente[3])), delta)  # Prohíbe rapidez negativa.


# ===========================================================================
# D. LÍMITES DE ADHERENCIA, POTENCIA, ACELERACIÓN Y FRENADO
# ===========================================================================
def aceleracion_lateral(v_m_s, kappa):
    """Calcula aceleración lateral firmada a partir de v² y curvatura."""
    return np.asarray(v_m_s)**2 * np.asarray(kappa)  # a_lat=v²*kappa, con signo.


# Calcula aceleración lateral utilizando el ángulo de rueda y la batalla.
def aceleracion_lateral_desde_direccion(v_m_s, delta_rad, veh):
    """Calcula aceleración lateral utilizando el ángulo de rueda y la batalla."""
    return np.asarray(v_m_s)**2 * np.tan(delta_rad) / veh.distancia_ejes_m


# Comprueba el círculo simplificado de adherencia longitudinal y lateral.
def aceleracion_total_admisible(a_lateral, a_longitudinal, veh):
    # Círculo de fricción global: a_lat²+a_long² <= (mu*g)².
    """Comprueba el círculo simplificado de adherencia longitudinal y lateral."""
    return np.asarray(a_lateral)**2 + np.asarray(a_longitudinal)**2 <= (veh.coef_friccion * G)**2 + EPS


# Obtiene aceleración longitudinal restante tras reservar adherencia para la curva.
def aceleracion_longitudinal_por_agarre(v_m_s, kappa, veh):
    """Obtiene aceleración longitudinal restante tras reservar adherencia para la curva."""
    a_lat = aceleracion_lateral(v_m_s, kappa)  # Parte del agarre usada al girar.
    libre_cuadrado = (veh.coef_friccion * G)**2 - a_lat**2  # Agar­re restante².
    return np.sqrt(np.maximum(0.0, libre_cuadrado))  # Aceleración posible en x.


# Calcula la cota de velocidad por adherencia lateral para cada curvatura.
def velocidad_maxima_curva(kappa, veh):
    """v_max=sqrt(mu*g/abs(kappa)); rectas admiten infinito localmente."""
    k = np.asarray(kappa, dtype=float)  # Funciona con escalar o array.
    with np.errstate(divide='ignore', invalid='ignore'):
        limites = np.where(np.abs(k) < EPS, np.inf,
                           np.sqrt(veh.coef_friccion * G / np.maximum(np.abs(k), EPS)))
    return float(limites) if limites.ndim == 0 else limites


# Aproxima fuerza motriz disponible como P/v, regularizando v cercano a cero.
def fuerza_por_potencia(v_m_s, veh):
    """Aproxima fuerza motriz disponible como P/v, regularizando v cercano a cero."""
    return veh.potencia_util_w / max(float(v_m_s), 0.5)  # F=P/v, límite en v≈0.


# Limita aceleración longitudinal simultáneamente por potencia y por tope del vehículo.
def aceleracion_motor(v_m_s, veh):
    """Limita aceleración longitudinal simultáneamente por potencia y por tope del vehículo."""
    por_potencia = fuerza_por_potencia(v_m_s, veh) / veh.masa_kg  # a=F/m.
    return min(veh.aceleracion_max_m_s2, por_potencia)  # Tope mecánico.


# Aplica v_f² = v_i² + 2aΔs en cada segmento de aceleración o frenado.
def acelerar_en_distancia(v_inicial, a_m_s2, distancia_m):
    # Ecuación cinemática v_f²=v_i²+2*a*ds (a constante en este segmento).
    """Aplica v_f² = v_i² + 2aΔs en cada segmento de aceleración o frenado."""
    return math.sqrt(max(0.0, v_inicial**2 + 2 * a_m_s2 * distancia_m))


# Despeja la distancia de frenado bajo desaceleración constante.
def distancia_frenado(v_inicial, v_final, frenado_m_s2):
    """d=(v_i²-v_f²)/(2*b), con b>0 magnitud del frenado."""
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if frenado_m_s2 <= 0 or v_final > v_inicial or v_final < 0:
        raise ValueError('Velocidades o frenado inválidos')
    return (v_inicial**2 - v_final**2) / (2 * frenado_m_s2)


# Obtiene la rapidez máxima que permite frenar dentro de la distancia visible.
def velocidad_segura_visibilidad(velocidad_final_requerida, distancia_visible_m,
                                 frenado_m_s2):
    """Límite idealizado v <= sqrt(v_c²+2*b*D_visible)."""
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if frenado_m_s2 <= 0 or distancia_visible_m < 0:
        raise ValueError('La visibilidad y el frenado deben ser válidos')
    return math.sqrt(velocidad_final_requerida**2 +
                     2 * frenado_m_s2 * distancia_visible_m)
