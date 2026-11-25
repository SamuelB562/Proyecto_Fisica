"""Conducción física optimizada con número configurable de vueltas.

La geometría se optimiza UNA sola vez, antes de arrancar. Después se reutiliza
para tantas vueltas como se soliciten. estimaciones.py calcula un perfil de
velocidad para el recorrido COMPLETO: NO se impone parada ni velocidad cero
en las metas intermedias. El controlador limita los cambios de mando.

Importante: es un plan predictivo, no una garantía de velocidad real. Las
velocidades reales proceden de la telemetría de Assetto Corsa.
"""
import argparse
import math
import statistics
import time
import numpy as np

from telemetria_assetto import Physics, Graphics, conectar, leer
from geometria_circuito import construir_geometria, proyectar_centro
from gestor_circuito import preparar_vuelta, suavizar_geometria
from modelado_fisico import Vehiculo, normales_izquierdas, curvaturas_de_puntos
from estimaciones import evaluar_trayectoria
from agente_conductor import AgenteConductor, Orden
from puente_solo import convertir_estado, convertir_controles
from controller_vjoy import ControladorVJoy
from exportar_trayectoria import exportar_trayectoria
from leer_parametros_ac import obtener_parametros


# A. CONFIGURACIÓN
# Punto de partida experimental, no óptimo medido.
INTERVALO_CONTROL_S = 0.02
TIEMPO_MAXIMO_S = 450.0  # Límite por vuelta; se escala según cantidad solicitada.
VUELTAS_PREDETERMINADAS = 3
PASADAS = 3
MAX_CURVAS = 20
UMBRAL_CURVA = 0.005
MARGEN_BORDE_M = 0.15
MEJORA_MINIMA_S = 0.02
# El estimador original exige un límite numérico finito; no limita vJoy.
TOPE_EVALUADOR_M_S = 80.0


# Calcula la distancia recorrida acumulando la longitud de los segmentos; permite localizar cada punto mediante s.
def estaciones(p):
    """Calcula la distancia recorrida acumulando la longitud de los segmentos; permite localizar cada punto mediante s."""
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(ds < 1e-6):
        raise ValueError('Segmentos consecutivos coincidentes')
    return np.r_[0.0, np.cumsum(ds)]


# Ajusta el sentido de la geometría para que la trayectoria respete la orientación esperada.
def orientar(geo, estado):
    """Ajusta el sentido de la geometría para que la trayectoria respete la orientación esperada."""
    c, iz, de = (np.asarray(v, float).copy()
                 for v in (geo.centro, geo.izquierdo, geo.derecho))
    pos = np.array([estado.x, estado.y])
    distancia, i, _ = proyectar_centro(pos, c)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if distancia > 4.0:
        raise ValueError(f'Porsche a {distancia:.2f} m del centro; ubícalo en pista')
    tang = c[(i + 1) % len(c)] - c[i]
    tang /= np.linalg.norm(tang)
    avance = np.array([math.cos(estado.psi_rad), math.sin(estado.psi_rad)])
    prod = float(tang @ avance)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if abs(prod) < 0.7:
        raise ValueError('El Porsche no está alineado con la pista')
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if prod < 0:
        c, iz, de = c[::-1].copy(), de[::-1].copy(), iz[::-1].copy()
    n = normales_izquierdas(c)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.all(np.sum((iz - c) * n, axis=1) < 0):
        iz, de = de, iz
    return c, iz, de


# Proyecta la posición sobre la trayectoria y estima su avance longitudinal.
def progreso(posicion, linea, s, anterior=None):
    """Proyecta la posición sobre la trayectoria y estima su avance longitudinal."""
    pos = np.asarray(posicion, float)
    indices = np.arange(len(linea) - 1)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if anterior is not None:
        indices = indices[(s[:-1] >= anterior - 25) & (s[:-1] <= anterior + 125)]
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if not len(indices):
            raise ValueError('Se perdió el progreso del coche')
    a = linea[indices]
    segmentos = linea[indices + 1] - a
    t = np.clip(np.einsum('ij,ij->i', pos - a, segmentos)
                / np.maximum(np.einsum('ij,ij->i', segmentos, segmentos), 1e-12), 0, 1)
    pies = a + t[:, None] * segmentos
    j = int(np.argmin(np.sum((pies - pos) ** 2, axis=1)))
    i = int(indices[j])
    return (float(s[i] + t[j] * (s[i + 1] - s[i])),
            float(np.linalg.norm(pies[j] - pos)))


# B. OPTIMIZACIÓN DE ÁPICES (ANTES DE CONDUCIR)
def evaluar(p, veh, v0):
    """Obtiene el perfil de velocidad y el tiempo predicho por el modelo sobre la trayectoria."""
    # Protege recursos externos y permite limpieza ante errores.
    try:
        e = evaluar_trayectoria(p, veh, velocidad_entrada_m_s=v0,
                               tope_velocidad_m_s=TOPE_EVALUADOR_M_S)
        return e if np.isfinite(e.tiempo_s) else None
    except (ValueError, ArithmeticError, FloatingPointError):
        return None




# Evalúa candidatos de trazada y conserva únicamente las modificaciones que mejoran el tiempo sujeto a restricciones.
def optimizar(c, iz, de, veh, v0, s_inicio, s_fin):
    """Busca una línea rápida por toda la vuelta con ápices temprano/medio/tardío.

    Solo modifica la línea durante el PREPROCESAMIENTO. Las trayectorias
    candidatas pasan por exterior -> interior -> exterior con quinticas.
    Se conserva la candidata que reduzca el tiempo estimado de toda la vuelta.
    """
    s = estaciones(c)
    normales = normales_izquierdas(c)
    reserva = veh.ancho_m / 2.0 + MARGEN_BORDE_M
    ancho_iz = np.sum((iz - c) * normales, axis=1)
    ancho_de = -np.sum((de - c) * normales, axis=1)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(ancho_iz <= reserva) or np.any(ancho_de <= reserva):
        raise ValueError('Bordes incompatibles o pista demasiado estrecha en algún tramo')
    iz_disp = ancho_iz - reserva
    de_disp = ancho_de - reserva

    d = np.zeros(len(c), dtype=float)
    mejor = evaluar(c, veh, v0)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if mejor is None:
        raise ValueError('La línea central no es físicamente evaluable')
    t_centro = float(mejor.tiempo_s)

    # B1. Curvas: filtrado para detección, NO para la evaluación física.
    k = curvaturas_de_puntos(c)
    k_suave = np.convolve(k, np.ones(21) / 21, mode='same')
    mascara = np.abs(k_suave) >= UMBRAL_CURVA
    grupos, actual = [], []
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for i, activo in enumerate(mascara):
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if activo:
            actual.append(i)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        elif actual:
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if len(actual) >= 8:
                grupos.append(actual)
            actual = []
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if len(actual) >= 8:
        grupos.append(actual)

    picos = [max(g, key=lambda j: abs(k_suave[j])) for g in grupos]
    picos.sort(key=lambda j: -abs(k_suave[j]))
    elegidos = []
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for i in picos:
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if (s_inicio + 165 < s[i] < s_fin - 165
                and all(abs(s[i] - s[j]) >= 35 for j in elegidos)):
            elegidos.append(i)
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if len(elegidos) >= MAX_CURVAS:
                break
    elegidos.sort()

    # Gestiona la etapa «q» del flujo del módulo, validando los datos que procesa.
    def q(t):
        """Interpolación quintica, C2 en los puntos de control aislados."""
        t = np.clip(t, 0.0, 1.0)
        return t**3 * (10.0 - 15.0*t + 6.0*t*t)

    # Gestiona la etapa «generar_perfil» del flujo del módulo, validando los datos que procesa.
    def generar_perfil(s_apice, signo, interior, exterior, extension):
        """Deformación de línea con cinco controles y extremos sin salto."""
        controles_s = s_apice + extension * np.array([-1., -.53, 0., .53, 1.])
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if controles_s[0] <= s_inicio + 25 or controles_s[-1] >= s_fin - 25:
            return None
        anch_iz = np.interp(controles_s, s, iz_disp)
        anch_de = np.interp(controles_s, s, de_disp)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if signo > 0:
            dentro, fuera = anch_iz, anch_de
        else:
            dentro, fuera = anch_de, anch_iz
        # Las amplitudes siguen los bordes locales, no el mínimo en 130 metros.
        controles_d = np.array([
            float(np.interp(controles_s[0], s, d)),
            -signo * exterior * fuera[1],
            signo * interior * dentro[2],
            -signo * exterior * fuera[3],
            float(np.interp(controles_s[4], s, d)),
        ])
        propuesta = d.copy()
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for j in range(4):
            tramo = (s >= controles_s[j]) & (s <= controles_s[j + 1])
            t = (s[tramo] - controles_s[j]) / (controles_s[j + 1] - controles_s[j])
            propuesta[tramo] = controles_d[j] + (controles_d[j+1] - controles_d[j]) * q(t)
        # Rechazar, en vez de recortar, cualquier trayectoria que invada bordes.
        if np.any(propuesta < -de_disp - 1e-8) or np.any(propuesta > iz_disp + 1e-8):
            return None
        return propuesta

    # Gestiona la etapa «huella_cabe» del flujo del módulo, validando los datos que procesa.
    def huella_cabe(ruta):
        """Verificación discreta de las cuatro esquinas de la carrocería.

        Proyecta las esquinas sobre segmentos próximos del centro original.
        No modela elevación, altura ni geometría real de los pianos.
        """
        largo2 = veh.largo_m / 2.0
        ancho2 = veh.ancho_m / 2.0
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for j in range(0, len(ruta), 3):
            a = max(0, j-1)
            b = min(len(ruta)-1, j+1)
            tang = ruta[b] - ruta[a]
            norma = np.linalg.norm(tang)
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if norma < 1e-9:
                return False
            tang /= norma
            lat = np.array([-tang[1], tang[0]])
            caja_centro = ruta[j] + (veh.distancia_ejes_m/2.0) * tang
            # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
            for u in (-largo2, largo2):
                for v in (-ancho2, ancho2):
                    esquina = caja_centro + u*tang + v*lat
                    indices = np.arange(max(0,j-12), min(len(c)-1,j+13))
                    e0 = c[indices]
                    seg = c[indices+1] - e0
                    seg2 = np.sum(seg*seg, axis=1)
                    t = np.clip(np.sum((esquina-e0)*seg,axis=1)/np.maximum(seg2,1e-12),0,1)
                    proy = e0 + t[:,None]*seg
                    r = int(np.argmin(np.sum((proy-esquina)**2,axis=1)))
                    i = indices[r]
                    tangent = seg[r]/np.sqrt(seg2[r])
                    normal = np.array([-tangent[1],tangent[0]])
                    offset = float((esquina-proy[r]) @ normal)
                    lim_i = float(ancho_iz[i]*(1-t[r])+ancho_iz[i+1]*t[r])
                    lim_d = float(ancho_de[i]*(1-t[r])+ancho_de[i+1]*t[r])
                    if offset > lim_i-MARGEN_BORDE_M or offset < -lim_d+MARGEN_BORDE_M:
                        return False
        return True

    # B2. Búsqueda: se favorecen candidatos próximos al piano, pero no se
    # premia artificialmente tocarlos: solo importa el tiempo físico calculado.
    posiciones = [('temprano', -20.), ('medio', 0.), ('tardio', 20.)]
    interiores = (.85, .95, .99)
    exteriores = (.70, .95)
    extensiones = (85., 130.)
    total, mejoras, rechazadas_huella = 1, 0, 0
    print(f'Curvas propuestas: {len(elegidos)}')
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for pasada in range(PASADAS):
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for numero, idx in enumerate(elegidos, 1):
            signo = float(np.sign(k_suave[idx]))
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if signo == 0:
                continue
            mejor_curva = None
            evaluacion_curva = mejor
            etiqueta = None
            # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
            for tipo, corrimiento in posiciones:
                for extension in extensiones:
                    for fi in interiores:
                        for fe in exteriores:
                            d_nueva = generar_perfil(
                                s[idx]+corrimiento, signo, fi, fe, extension
                            )
                            if d_nueva is None:
                                continue
                            ruta = c + d_nueva[:, None] * normales
                            if np.any(np.linalg.norm(np.diff(ruta, axis=0), axis=1) < 1e-5):
                                continue
                            e = evaluar(ruta, veh, v0)
                            total += 1
                            if e is None or e.tiempo_s >= evaluacion_curva.tiempo_s - MEJORA_MINIMA_S:
                                continue
                            if not huella_cabe(ruta):
                                rechazadas_huella += 1
                                continue
                            mejor_curva, evaluacion_curva = d_nueva, e
                            etiqueta = (tipo, fi, fe, extension)
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if mejor_curva is not None:
                d, mejor = mejor_curva, evaluacion_curva
                mejoras += 1
                tipo, fi, fe, extension = etiqueta
                print(f'Pasada {pasada+1}, curva {numero}: {tipo} | '
                      f'interior={fi:.0%}, exterior={fe:.0%}, '
                      f'extensión={extension:.0f} m | T={mejor.tiempo_s:.3f} s')

    ruta_final = c + d[:, None] * normales
    print(f'Evaluaciones físicas: {total}; mejoras aceptadas: {mejoras}; '
          f'rechazos por huella: {rechazadas_huella}')
    print(f'Centro: {t_centro:.3f} s | Optimizada: {mejor.tiempo_s:.3f} s | '
          f'Ganancia: {t_centro-mejor.tiempo_s:.3f} s')
    print(f'Desplazamiento lateral máximo: {np.max(np.abs(d)):.3f} m')
    return ruta_final, mejor


# C. CONDUCTOR: REFERENCIAS POR POSICIÓN, SIN NUEVAS TRAYECTORIAS
def controlar(conductor, evaluacion, s_ruta, s_actual, t):
    """Convierte el objetivo de seguimiento en órdenes de acelerador, freno y dirección para vJoy."""
    i = int(np.clip(np.searchsorted(s_ruta, s_actual, side='right') - 1,
                    0, len(s_ruta) - 2))
    velocidad = float(evaluacion.velocidades_m_s[i + 1])
    conductor.agregar_ordenes([Orden(
        inicio_s=t, fin_s=t + 0.001,
        direccion_rad=float(evaluacion.direcciones_rad[i]),
        velocidad_objetivo_m_s=velocidad
    )])
    return velocidad


# Lleva los controles a reposo para evitar órdenes residuales al finalizar.
def detener(controlador, memoria):
    """Lleva los controles a reposo para evitar órdenes residuales al finalizar."""
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for _ in range(160):
        controlador.enviar(acelerador=0.0, freno=1.0, direccion=0.0)
        # Protege recursos externos y permite limpieza ante errores.
        try:
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if abs(float(leer(memoria, Physics).speedKmh)) < 1:
                break
        except Exception:
            break
        time.sleep(0.05)
    controlador.reposo()


# D. CONSTRUCCIÓN NUMÉRICA DEL RECORRIDO DE VARIAS VUELTAS

def repetir_vuelta(linea, s_inicio, s_fin, vueltas, *, nombre='trayectoria'):
    """Repite el MISMO trazado, sin crear un salto espacial en la meta.

    D1. Conserva la extensión anterior al inicio. Sirve para que el perfil
        predictivo tenga una entrada desde la posición de partida.
    D2. Extrae exactamente una vuelta entre s_inicio y s_fin. Ambos extremos
        son el mismo lugar físico del circuito.
    D3. Repite dicha vuelta (sin duplicar el punto compartido de la meta).
    D4. Conecta la extensión posterior al terminar la última vuelta, para
        que el estimador no interprete el último cruce como una parada.
    D5. Devuelve las estaciones de todas las metas en la nueva polilínea.

    El resultado es una representación espacial de la vuelta recorrida varias
    veces: las coordenadas pueden repetirse, pero NO hay segmentos de longitud
    cero consecutivos ni un teletransporte de fin de vuelta a inicio.
    """
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if vueltas < 1:
        raise ValueError('Se necesita al menos una vuelta')
    linea = np.asarray(linea, dtype=float)
    s = estaciones(linea)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not (0 < s_inicio < s_fin < s[-1]):
        raise ValueError(f'Intervalo de vuelta fuera de {nombre}: {s_inicio:.2f}, {s_fin:.2f}, {s[-1]:.2f}')

    # D1. Interpolación lineal sobre un segmento; respeta la malla existente.
    def posicion_en(s_obj):
        """Gestiona la etapa «posicion_en» del flujo del módulo, validando los datos que procesa."""
        i = int(np.clip(np.searchsorted(s, s_obj, side='right') - 1, 0, len(s)-2))
        t = (s_obj-s[i])/(s[i+1]-s[i])
        return linea[i]*(1-t)+linea[i+1]*t

    inicio = posicion_en(s_inicio)
    fin = posicion_en(s_fin)
    separacion = float(np.linalg.norm(fin-inicio))
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if separacion > 2.0:
        raise ValueError(
            f'No se puede unir la meta de {nombre}: los extremos se separan '
            f'{separacion:.2f} m. Revisar geometría de la vuelta antes de conducir.'
        )
    # El punto de llegada coincide EXACTAMENTE con el punto de salida.
    # Evitamos un salto de posición o un segmento artificial en el enlace.
    interior = linea[(s > s_inicio) & (s < s_fin)]
    vuelta = np.vstack((inicio, interior, inicio))
    prefijo = linea[s < s_inicio]
    sufijo = linea[s > s_fin]

    # D2. La primera copia mantiene la aproximación original desde boxes.
    partes = [prefijo, vuelta]
    metas = []
    dist_inicio = float(estaciones(np.vstack((prefijo, vuelta[:1])))[-1])
    largo_vuelta = float(estaciones(vuelta)[-1])
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for numero in range(vueltas):
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if numero:
            # Saltamos el punto inicial, ya compartido con la meta previa.
            partes.append(vuelta[1:])
        metas.append(dist_inicio + (numero+1)*largo_vuelta)
    partes.append(sufijo)
    resultado = np.concatenate(partes, axis=0)
    estaciones(resultado)  # Verifica segmentos degenerados ANTES de conducir.
    return resultado, dist_inicio, np.asarray(metas, dtype=float)


# Resume tiempos observados sin confundir intentos, vueltas completas y predicciones.
def resumen_vueltas(tiempos, solicitadas):
    """Imprime resultados sin confundir vueltas completas con intentos."""
    print(f'Vueltas completas: {len(tiempos)}/{solicitadas}')
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for n, dt in enumerate(tiempos, 1):
        print(f'  Vuelta {n}: {formato_tiempo(dt)} ({dt:.3f} s)')
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if tiempos:
        print(f'Promedio: {formato_tiempo(statistics.mean(tiempos))}')
        print(f'Mejor: {formato_tiempo(min(tiempos))}')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(tiempos) > 1:
            print(f'Desviación estándar muestral: {statistics.stdev(tiempos):.3f} s')


# E. EJECUCIÓN

def configurar_argumentos():
    """Interpreta opciones de ejecución y valida sus valores antes de iniciar el experimento."""
    parser = argparse.ArgumentParser(description="Conducción física optimizada por vueltas")
    parser.add_argument('--vueltas', type=int, default=VUELTAS_PREDETERMINADAS,
                        help=f"Número de vueltas consecutivas (predeterminado: {VUELTAS_PREDETERMINADAS})")
    return parser.parse_args()


# Convierte segundos a una representación legible de minutos y segundos.
def formato_tiempo(segundos):
    """Convierte segundos a una representación legible de minutos y segundos."""
    return f"{int(segundos // 60)}:{segundos % 60:06.3f}"


# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main(vueltas=None):
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if vueltas is None:
        vueltas = configurar_argumentos().vueltas
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not isinstance(vueltas, int) or isinstance(vueltas, bool) or vueltas < 1:
        raise ValueError("El número de vueltas debe ser un entero positivo")

    mfis = mgraf = mando = None
    tiempos_vueltas = []
    motivo_fin = "Finalización desconocida"
    # Protege recursos externos y permite limpieza ante errores.
    try:
        # E0. Confirmar las vueltas efectivas, tanto si vienen de CLI como del puente.
        print(f'[COMPARADOR] VUELTAS_CONFIGURADAS {vueltas}', flush=True)
        nombre, variante, geo = construir_geometria()
        print(f'Conducción física optimizada | {nombre} / {variante}')
        mfis = conectar('acpmf_physics', Physics)
        mgraf = conectar('acpmf_graphics', Graphics)
        f, g = leer(mfis, Physics), leer(mgraf, Graphics)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if g.status != 2 or abs(float(f.speedKmh)) > 2:
            raise RuntimeError('Comienza en sesión LIVE con el coche detenido')
        e0 = convertir_estado(f, g)
        c, iz, de = orientar(geo, e0)
        c, iz, de = suavizar_geometria(c, iz, de)
        c, iz, de, longitud, _ = preparar_vuelta(
            c, iz, de, np.array([e0.x, e0.y]),
            margen_atras_m=200.0, margen_despues_m=650.0
        )
        # Deduplicación alineada de centro y bordes, sin cambiar los originales.
        mantener = np.r_[True, np.linalg.norm(np.diff(c, axis=0), axis=1) > 1e-5]
        c, iz, de = c[mantener], iz[mantener], de[mantener]
        sc = estaciones(c)
        s_inicio, _ = progreso([e0.x, e0.y], c, sc)
        p = obtener_parametros()

        veh = Vehiculo(
            masa_kg=p["masa_kg"],
            potencia_util_w=p["potencia_maxima_telemetria"],
            distancia_ejes_m=p["distancia_ejes_m"],
            relacion_direccion=p["direccion_steer_ratio"],
            aceleracion_max_m_s2= 6.35,
            frenado_max_m_s2= 9.93,
            coef_friccion=p["coef_friccion_modelo"]
        )
        
        print('Optimizando la vuelta completa antes de arrancar...')
        ruta, evaluacion = optimizar(
            c, iz, de, veh, e0.velocidad_m_s,
            s_inicio, s_inicio + longitud
            )

        # E1. Reutilizamos una geometría optimizada, sin buscar nuevos ápices.
        #     La repetición se realiza ANTES de calcular los perfiles de velocidad.
        #     Los puntos de centro y ruta tienen la misma correspondencia de índices.
        sc_base = estaciones(c)
        sr_base = estaciones(ruta)
        s_meta_base = s_inicio + longitud
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if s_meta_base >= sc_base[-1]:
            raise ValueError('La geometría no incluye el margen posterior de meta')
        s_inicio_ruta = float(np.interp(s_inicio, sc_base, sr_base))
        s_meta_ruta = float(np.interp(s_meta_base, sc_base, sr_base))

        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if vueltas == 1:
            # Compatibilidad: una vuelta mantiene exactamente la geometría y
            # la evaluación del programa original que ya funcionaba.
            centro_total, ruta_total = c, ruta
            evaluacion_total = evaluacion
            s_primera_meta = s_meta_base
            metas_centro = np.array([s_primera_meta], dtype=float)
            s_inicio_total = s_inicio
        else:
            # E2. Construimos N vueltas consecutivas en la misma posición física.
            centro_total, s_inicio_total, metas_centro = repetir_vuelta(
                c, s_inicio, s_meta_base, vueltas, nombre='centro')
            ruta_total, s_inicio_ruta_total, metas_ruta = repetir_vuelta(
                ruta, s_inicio_ruta, s_meta_ruta, vueltas, nombre='ruta')

            # E3. Una sola evaluación de velocidad en TODAS las vueltas.
            # NO se fuerza frenar en las metas. La velocidad calculada al
            # final de una vuelta es la de entrada a la siguiente.
            # La pasada inversa del evaluador puede anticipar una curva de la
            # siguiente vuelta, y la pasada directa respeta aceleración.
            evaluacion_total = evaluar(ruta_total, veh, e0.velocidad_m_s)
            if evaluacion_total is None:
                raise ValueError(
                    'La trayectoria de varias vueltas no resulta físicamente '
                    'evaluable. Revisar continuidad geométrica en la meta.'
                )
            print('Perfil multivuelta calculado sin parada en las metas:')
            sr_total_temp = estaciones(ruta_total)
            # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
            for n, meta_ruta in enumerate(metas_ruta, 1):
                ind = int(np.clip(np.searchsorted(sr_total_temp, meta_ruta),
                                  0, len(sr_total_temp)-1))
                v_est = float(evaluacion_total.velocidades_m_s[ind])
                print(f'  Meta {n}: velocidad estimada {v_est*3.6:.1f} km/h')

        # E4. Telemetría de progreso sobre el centro, y órdenes sobre la
        # trayectoria de carrera, ambos extendidos durante el mismo número de vueltas.
        sc_total = estaciones(centro_total)
        sr_total = estaciones(ruta_total)
        # E4a. La visualización muestra SOLO la primera vuelta.
        #       Mantiene el formato de archivo existente para el HTML.
        if vueltas == 1:
            mascara_visual = ((sr_total >= s_inicio_ruta) &
                              (sr_total <= s_inicio_ruta + longitud))
        else:
            mascara_visual = ((sr_total >= s_inicio_ruta_total) &
                              (sr_total <= metas_ruta[0]))
        ruta_visual = ruta_total[mascara_visual]
        exportar_trayectoria(ruta_visual)

        # E5. AgenteConductor se conserva COMO CONTROLADOR de actuadores.
        # No planifica ápices, solo aplica controles con limitaciones de giro.
        conductor = AgenteConductor(veh, periodo_control_s=INTERVALO_CONTROL_S)
        # La primera trayectoria ya tiene extensiones; la ruta total agrega
        # copias de la misma vuelta con una meta geométricamente continua.
        conductor.establecer_trayectoria(ruta_total)
        print(f'Una única línea optimizada para {vueltas} vuelta(s).')
        input('ENTER para iniciar la conducción...')
        mando = ControladorVJoy()
        mando.reposo()
        reloj = time.monotonic()
        instante_ultima_meta = reloj
        tiempos_vueltas = []
        vueltas_completadas = 0
        log_ultimo = -1.0
        s_anterior = s_inicio_total
        s_objetivo_anterior = float(np.interp(s_inicio, sc_base, sr_base))
        recorrido_anterior = 0.0
        t_anterior = 0.0

        # E6. Se mantiene el estado del automóvil, reloj y controlador.
        # No hay pausas, reinicios ni frenadas al cruzar una meta intermedia.
        while True:
            tic = time.monotonic()
            t = tic - reloj
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if t >= TIEMPO_MAXIMO_S * vueltas:
                motivo_fin = 'Tiempo máximo alcanzado'
                print(motivo_fin)
                break
            f, g = leer(mfis, Physics), leer(mgraf, Graphics)
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if g.status != 2:
                motivo_fin = 'La sesión ya no está LIVE'
                print(motivo_fin)
                break
            estado = convertir_estado(f, g)

            # E7. Progreso MONÓTONO APROXIMADO sobre una polilínea completa.
            # Limitar la búsqueda espacial al tramo anterior impide que el
            # vehículo salte virtualmente a otra vuelta de la misma geometría.
            s_coche, error_centro = progreso(
                [estado.x, estado.y], centro_total, sc_total, s_anterior)
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if error_centro > 10.0:
                motivo_fin = f'Desviación excesiva: {error_centro:.2f} m'
                print(motivo_fin)
                break
            # En caso de oscilación por muestreo cerca de la meta, no
            # retrocedemos el índice de búsqueda a otra vuelta completa.
            s_anterior = max(s_anterior, s_coche)
            recorrido = s_coche - s_inicio_total

            # E8. Cronometraje de cada cruce, usando interpolación temporal
            # dentro del intervalo de control para evitar error de hasta 20 ms.
            while vueltas_completadas < vueltas and s_coche >= metas_centro[vueltas_completadas]:
                s_meta = metas_centro[vueltas_completadas]
                fraccion = (s_meta - (s_inicio_total + recorrido_anterior)) / max(
                    s_coche - (s_inicio_total + recorrido_anterior), 1e-9)
                fraccion = float(np.clip(fraccion, 0.0, 1.0))
                instante_cruce_relativo = t_anterior + fraccion*(t - t_anterior)
                instante_cruce = reloj + instante_cruce_relativo
                t_vuelta = instante_cruce - instante_ultima_meta
                instante_ultima_meta = instante_cruce
                tiempos_vueltas.append(t_vuelta)
                vueltas_completadas += 1
                # E8a. Evento inequívoco para registros y servidores.
                print(f'[COMPARADOR] VUELTA_COMPLETADA numero={vueltas_completadas} '
                      f'total={vueltas} tiempo_s={t_vuelta:.3f}', flush=True)
                print(f'¡VUELTA {vueltas_completadas}/{vueltas} COMPLETADA! '
                      f'Tiempo: {formato_tiempo(t_vuelta)} ({t_vuelta:.3f} s)')
                if vueltas_completadas >= vueltas:
                    motivo_fin = 'Número de vueltas solicitado completado'
                    break
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if vueltas_completadas >= vueltas:
                break

            # E9. La velocidad objetivo viene del perfil global. En cada
            # nueva vuelta se avanza a la SIGUIENTE copia del trazado.
            s_objetivo, _ = progreso([estado.x, estado.y], ruta_total,
                                     sr_total, s_objetivo_anterior)
            s_objetivo_anterior = max(s_objetivo_anterior, s_objetivo)
            objetivo = controlar(conductor, evaluacion_total, sr_total, s_objetivo, t)
            informe = conductor.actualizar(t, estado)
            controles = convertir_controles(informe.controles, veh)
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if not np.all(np.isfinite(list(controles.values()))):
                raise ValueError('Control inválido')
            mando.enviar(**controles)
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if t - log_ultimo >= 0.5:
                log_ultimo = t
                print(f'vuelta={vueltas_completadas+1}/{vueltas} | '
                      f't={t:.2f}s | recorrido={recorrido:.1f}m | '
                      f'v_real={f.speedKmh:.1f} km/h | '
                      f'v_objetivo={objetivo*3.6:.1f} km/h | '
                      f'error={informe.error_lateral_m}')
            recorrido_anterior, t_anterior = recorrido, t
            time.sleep(max(0, INTERVALO_CONTROL_S - (time.monotonic() - tic)))
    except KeyboardInterrupt:
        motivo_fin = 'Interrumpido manualmente'
        print(motivo_fin)
    except Exception as ex:
        motivo_fin = f'ERROR: {type(ex).__name__}: {ex}'
        print(motivo_fin)
    finally:
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if mando is not None:
            # Protege recursos externos y permite limpieza ante errores.
            try:
                detener(mando, mfis)
            finally:
                mando.reposo()
                mando.cerrar()
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if mfis is not None:
            mfis.close()
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if mgraf is not None:
            mgraf.close()
        print(f'Experimento finalizado: {motivo_fin}')
        print(f'[COMPARADOR] FIN vueltas_completadas={len(tiempos_vueltas)} '
              f'vueltas_solicitadas={vueltas}', flush=True)
        resumen_vueltas(tiempos_vueltas, vueltas)


if __name__ == '__main__':
    main()
