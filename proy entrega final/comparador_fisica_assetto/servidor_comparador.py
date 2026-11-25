"""A. SERVIDOR DEL COMPARADOR: BICICLETA CINEMÁTICA VS ASSETTO CORSA.
Se conserva el comparador web; el overlay se gestiona desde Assetto Corsa.
- La ruta se exporta una sola vez mediante puente_plan.py (intercepta optimizar).
- Python modela la bicicleta y Assetto Corsa ejecuta el conductor independiente.
- N vueltas en ambos sistemas; la geometría se optimiza una sola vez.
- La señal speed real procede de la memoria compartida de Assetto Corsa cuando
  esté disponible; los logs del conductor son una segunda fuente etiquetada.
- El JSON del plan contiene predicciones, NUNCA telemetría del simulador real.
"""
from __future__ import annotations
import argparse
import csv  # Lee registros de telemetría AC a frecuencia de control.
import json
import math
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from collections import deque
from urllib.parse import urlsplit
ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'plan_compartido.json'
GRAFICAS = ROOT / 'graficas'
# A1. Historial numérico: solo muestras reportadas, nunca predicciones del plan.
HISTORIAL = {'model': deque(maxlen=100000), 'assetto': deque(maxlen=100000)}
GRAFICAS_LOCK = threading.Lock()
ULTIMAS_GRAFICAS = {'generadas': 0.0, 'respuesta': None}
LOCK = threading.RLock()
STOP_MODEL = threading.Event()
MODEL_THREAD = None
PROCESS = None
ASSETTO_THREAD = None
ARGS = None
# B. ESTADO PÚBLICO. No sustituir mediciones faltantes por predicciones.
STATE = {
    'model': {'status': 'listo', 'laps_completed': 0, 'lap_times': []},
    'assetto': {'status': 'listo', 'laps_completed': 0, 'lap_times': []},
    'settings': {'velocidad': 1.0, 'vueltas': None},
    'plan_ready': False,
}
# Actualiza estados y mantiene el historial de observaciones finitas.
def setstate(which, **values):
    # B0. Guardar series del experimento para que matplotlib las grafique.
    # Se registran puntos solamente cuando cambia el tiempo o la distancia;
    # valores no disponibles NO se sustituyen por datos de estimaciones.py.
    """Actualiza estados y mantiene el historial de observaciones finitas."""
    with LOCK:
        STATE[which].update(values)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if which in HISTORIAL and STATE[which].get('status') in ('corriendo', 'completado'):
            obj = STATE[which]
            # Protege recursos externos y permite limpieza ante errores.
            try:
                punto = {k: float(obj[k]) for k in ('time', 'distance', 'speed')}
                if all(math.isfinite(v) for v in punto.values()):
                    prev = HISTORIAL[which][-1] if HISTORIAL[which] else None
                    if prev is None or (punto['time'] != prev['time'] or punto['distance'] != prev['distance']):
                        HISTORIAL[which].append(punto)
            except (TypeError, ValueError, KeyError):
                pass
# Obtiene el factor de reproducción del modelo dentro de un rango válido.
def get_speed_factor():
    """Obtiene el factor de reproducción del modelo dentro de un rango válido."""
    with LOCK:
        return max(0.1, float(STATE['settings']['velocidad']))
# Lee únicamente planes JSON completos con geometría válida.
def load_plan():
    """B1. Solo devuelve planes completos; el puente escribe atómicamente."""
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not PLAN.is_file():
        return None
    # Protege recursos externos y permite limpieza ante errores.
    try:
        obj = json.loads(PLAN.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if len(obj.get('ruta', [])) < 4:
        return None
    return obj
# Valida la cantidad solicitada de vueltas.
def requested_laps(value):
    """B2. Valores ausentes: se conserva el predeterminado del conductor."""
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if value is None:
        return None
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError('vueltas debe ser un entero entre 1 y 100')
    return value
# Calcula la distancia recorrida acumulando la longitud de los segmentos; permite localizar cada punto mediante s.
def estaciones(puntos):
    """C0. Distancia acumulada del trazado a partir de segmentos discretos."""
    import numpy as np
    ds = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(ds < 1e-6):
        raise ValueError('Segmentos consecutivos coincidentes')
    return np.r_[0.0, np.cumsum(ds)]
# Concatena copias del recorrido de referencia manteniendo continuidad geométrica entre vueltas.
def repetir_vuelta(linea, s_inicio, s_fin, vueltas):
    """C0a. Misma unión geométrica que el conductor, sin importar vJoy.
    Repite exactamente la geometría base, deja un prefijo de entrada y un
    sufijo final y elimina el punto duplicado de cada meta intermedia.
    """
    import numpy as np
    s = estaciones(linea)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not 0 < s_inicio < s_fin < s[-1]:
        raise ValueError('Estaciones de salida/meta inválidas')
    # Gestiona la etapa «at» del flujo del módulo, validando los datos que procesa.
    def at(distancia):
        """Gestiona la etapa «at» del flujo del módulo, validando los datos que procesa."""
        idx = int(np.clip(np.searchsorted(s, distancia, side='right') - 1,
                          0, len(s)-2))
        peso = (distancia-s[idx])/(s[idx+1]-s[idx])
        return linea[idx]*(1-peso) + linea[idx+1]*peso
    inicio, fin = at(s_inicio), at(s_fin)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if float(np.linalg.norm(inicio-fin)) > 2.0:
        raise ValueError('El tramo de vuelta no cierra geométricamente')
    vuelta = np.vstack((inicio, linea[(s>s_inicio)&(s<s_fin)], inicio))
    prefijo, sufijo = linea[s<s_inicio], linea[s>s_fin]
    dist_inicio = float(estaciones(np.vstack((prefijo, vuelta[:1])))[-1])
    longitud = float(estaciones(vuelta)[-1])
    piezas = [prefijo, vuelta]
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for _ in range(1, vueltas):
        piezas.append(vuelta[1:])
    piezas.append(sufijo)
    extendida = np.concatenate(piezas, axis=0)
    estaciones(extendida)
    metas = dist_inicio + np.arange(1, vueltas+1, dtype=float)*longitud
    return extendida, dist_inicio, metas
# Integra el modelo de bicicleta con perfil de velocidad, control propio y cronómetro independiente.
def model_run(vueltas):
    """C. Modelo numérico independiente, con vueltas y perfil continuos.
    La salida de optimizar() contiene UNA vuelta de referencia y márgenes de
    aproximación. Para N>1 se repite la ruta, NO se optimizan nuevos ápices.
    Las velocidades se reevalúan juntas para impedir frenadas artificiales en
    las metas intermedias; se aplica el mismo evaluador que usa el conductor.
    """
    global MODEL_THREAD
    # Protege recursos externos y permite limpieza ante errores.
    try:
        setstate('model', status='esperando trayectoria', message='Esperando el plan exportado por Assetto Corsa')
        plan = None
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for _ in range(1500):
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if STOP_MODEL.is_set():
                setstate('model', status='detenido', message='Detenido antes de recibir el plan')
                return
            plan = load_plan()
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if plan is not None:
                break
            time.sleep(0.2)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if plan is None:
            raise RuntimeError('No se recibió el plan optimizado en cinco minutos')
        import numpy as np
        # C1. Los imports de física usan los archivos originales del proyecto.
        if str(ARGS.script.parent) not in sys.path:
            sys.path.insert(0, str(ARGS.script.parent))
        from modelado_fisico import (Estado, Vehiculo, paso_bicicleta,
                                    aceleracion_motor, aceleracion_longitudinal_por_agarre)
        from estimaciones import evaluar_trayectoria
        base = np.asarray(plan['ruta'], dtype=float)
        sb = np.asarray(plan['s'], dtype=float)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(sb) != len(base):
            raise ValueError('Las estaciones del JSON no corresponden con la ruta')
        inicio_base = float(plan.get('s_inicio_ruta', plan['s_inicio']))
        fin_base = float(plan.get('s_fin_ruta', plan['s_fin']))
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if not (0 < inicio_base < fin_base < sb[-1]):
            raise ValueError('El plan compartido no define correctamente una vuelta en coordenadas de ruta')
        vueltas = vueltas if vueltas is not None else plan.get('vueltas_solicitadas')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if vueltas is None:
            # Misma fuente que el conductor (configuración Python).
            # La configuración del conductor se lee como texto, sin importar
            # sus dependencias de vJoy en este proceso independiente.
            import ast
            arbol = ast.parse(ARGS.script.read_text(encoding='utf-8'))
            defaults = [n.value.value for n in arbol.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'VUELTAS_PREDETERMINADAS'
                                for t in n.targets)
                        and isinstance(n.value, ast.Constant) and type(n.value.value) is int]
            vueltas = defaults[-1] if defaults else 1
        vueltas = requested_laps(vueltas)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if vueltas is None:
            raise ValueError('No se pudo determinar el número de vueltas')
        # C1a. El comparador usa exactamente los parámetros del vehículo

        # que generó el plan; no vuelve a los predeterminados de Vehiculo().

        parametros = plan.get('vehiculo') or {}

        claves = ('masa', 'potencia', 'L', 'mu', 'freno', 'aceleracion')

        ausentes = [clave for clave in claves if parametros.get(clave) is None]

        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if ausentes:

            raise ValueError(f'El plan no contiene parámetros físicos: {ausentes}')

        veh = Vehiculo(

            masa_kg=float(parametros['masa']),

            potencia_util_w=float(parametros['potencia']),

            distancia_ejes_m=float(parametros['L']),

            coef_friccion=float(parametros['mu']),

            frenado_max_m_s2=float(parametros['freno']),

            aceleracion_max_m_s2=float(parametros['aceleracion']),
            # Igualar también la dirección que utilizó el plan exportado.
            relacion_direccion=float(parametros.get('relacion_direccion',15.0)),
            giro_ruedas_max_rad=float(parametros.get('giro_ruedas_max_rad', math.radians(30.0))),

        )
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if vueltas == 1:
            ruta, inicio_total = base, inicio_base
            metas = np.asarray([fin_base], dtype=float)
            perfil = np.asarray(plan['v'], dtype=float)
            curv = np.asarray(plan['k'], dtype=float)
        else:
            # C2. Se reaprovecha la MISMA función de repetición usada en Assetto.
            ruta, inicio_total, metas = repetir_vuelta(
                base, inicio_base, fin_base, vueltas)
            evaluacion = evaluar_trayectoria(
                ruta, veh, velocidad_entrada_m_s=0.0, tope_velocidad_m_s=80.0)
            perfil = np.asarray(evaluacion.velocidades_m_s, dtype=float)
            curv = np.asarray(evaluacion.curvaturas_menos1, dtype=float)
        s = estaciones(ruta)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(perfil) != len(ruta) or len(curv) != len(ruta):
            raise ValueError('Longitud incompatible entre ruta y perfil físico')
        # C3. Condiciones iniciales COMPARABLES: ambos automóviles arrancan
        # detenidos en la salida real de la sesión, no 200 m antes en el prefijo.
        # El prefijo se conserva dentro del plan para evaluar la optimización,
        # pero NO constituye un recorrido ni una vuelta del comparador.
        ix = int(np.clip(np.searchsorted(s, inicio_total), 0, len(s)-2))
        tangente = ruta[ix+1] - ruta[ix]
        psi = math.atan2(float(tangente[1]), float(tangente[0]))
        estado = Estado(float(ruta[ix,0]), float(ruta[ix,1]), psi, 0.0, 0.0)
        avance = float(inicio_total)
        # C3a. Misma referencia espacial: ambos progresos se proyectan
        # sobre la geometría central, NO sobre longitudes de líneas diferentes.
        centro_base = np.asarray(plan['centro'], dtype=float)
        if vueltas == 1:
            centro_total, inicio_centro = centro_base, float(plan['s_inicio'])
        else:
            centro_total, inicio_centro, _ = repetir_vuelta(
                centro_base, float(plan['s_inicio']), float(plan['s_fin']), vueltas)
        sc_total = estaciones(centro_total)
        if len(centro_total) != len(ruta):
            raise ValueError('La correspondencia centro/ruta por índice no es válida')
        # La progresión central también es monótona para conservar el número
        # de vuelta cuando se repiten las mismas coordenadas XY.
        avance_centro = float(inicio_centro)
        elapsed = 0.0
        dt = 0.02
        ultimo = -1.0
        vueltas_completadas = 0
        inicio_ultima_vuelta = 0.0  # Ambos parten detenidos desde la salida de referencia.
        lap_times = []
        s_prev = avance
        t_prev = 0.0
        setstate('model', status='corriendo', time=0.0, speed=0.0,
                 distance=0.0, progress=0.0, laps_completed=0,
                 laps_requested=vueltas, lap_times=[], message='Modelo bicicleta: Euler 20 ms')
        # C4. Se utiliza la distancia acumulada CONTINUA entre vueltas para
        # distinguir la misma coordenada XY de vuelta 1, 2, ...
        while not STOP_MODEL.is_set() and vueltas_completadas < vueltas and elapsed < 500.0 * vueltas:
            lo = max(0, int(np.searchsorted(s, avance - 17)) - 1)
            hi = min(len(s) - 2, int(np.searchsorted(s, avance + 85)) + 1)
            a = ruta[lo:hi+1]
            b = ruta[lo+1:hi+2]
            ab = b - a
            xy = np.array([estado.x, estado.y])
            tt = np.clip(np.sum((xy - a)*ab, axis=1) /
                         np.maximum(np.sum(ab*ab, axis=1), 1e-9), 0, 1)
            pies = a + tt[:, None]*ab
            errs = np.sum((pies - xy)**2, axis=1)
            j = int(np.argmin(errs))
            idx = lo + j
            prog = float(s[idx] + tt[j]*(s[idx+1]-s[idx]))
            avance = max(avance, prog)
            # C4a. Proyectar el vehículo modelo REAL sobre la misma línea central
            # usada por el conductor de Assetto Corsa. La búsqueda se restringe
            # alrededor del índice homólogo para no saltar entre vueltas.
            ci0 = max(0, idx-12)
            ci1 = min(len(centro_total)-2, idx+12)
            ca = centro_total[ci0:ci1+1]
            cb = centro_total[ci0+1:ci1+2]
            segmento = cb-ca
            norma2 = np.maximum(np.sum(segmento*segmento,axis=1),1e-9)
            t_c = np.clip(np.sum((xy-ca)*segmento,axis=1)/norma2,0,1)
            pies_c = ca+t_c[:,None]*segmento
            j_c = int(np.argmin(np.sum((pies_c-xy)**2,axis=1)))
            estacion_centro = float(sc_total[ci0+j_c]
                       + t_c[j_c]*(sc_total[ci0+j_c+1]-sc_total[ci0+j_c]))
            avance_centro = max(avance_centro, estacion_centro)
            error = math.sqrt(float(errs[j]))
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if error > 22:
                raise RuntimeError(f'Error lateral excesivo: {error:.2f} m')
            # C5. No se cronometra el prefijo de aproximación: el experimento
            # parte en la salida y sus relojes arrancan en cero.
            # Cronometraje de cada meta por interpolación temporal.
            while vueltas_completadas < vueltas and avance >= metas[vueltas_completadas]:
                meta = float(metas[vueltas_completadas])
                frac = (meta-s_prev) / max(avance-s_prev, 1e-9)
                cruza = t_prev + float(np.clip(frac, 0, 1))*(elapsed-t_prev)
                lap_times.append(round(cruza-inicio_ultima_vuelta, 3))
                inicio_ultima_vuelta = cruza
                vueltas_completadas += 1
                setstate('model', laps_completed=vueltas_completadas,
                         lap_times=lap_times.copy(), last_lap_s=lap_times[-1])
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if vueltas_completadas >= vueltas:
                break
            # C6. Control cinemático ORIGINAL del comparador: seguimiento
            # con anticipación, referencia de curvatura y límite de dirección.
            look = min(float(metas[-1]), avance + max(6., estado.velocidad_m_s * .35))
            target = min(len(s)-2, int(np.searchsorted(s, look)-1))
            dx = float(ruta[target, 0]-estado.x)
            dy = float(ruta[target, 1]-estado.y)
            look_angle = math.atan2(dy, dx)
            errpsi = math.atan2(math.sin(look_angle-estado.psi_rad),
                                math.cos(look_angle-estado.psi_rad))
            delta = float(np.clip(math.atan(veh.distancia_ejes_m * float(curv[idx])) + .55*errpsi,
                                  -veh.giro_ruedas_max_rad, veh.giro_ruedas_max_rad))
            vgoal = float(min(perfil[min(len(perfil)-1, idx+2)],
                              perfil[min(len(perfil)-1, target)] + 3.0))
            acel = float(np.clip(1.5*(vgoal-estado.velocidad_m_s),
                                 -veh.frenado_max_m_s2, float(aceleracion_motor(estado.velocidad_m_s, veh))))
            grip = float(aceleracion_longitudinal_por_agarre(
                estado.velocidad_m_s, float(curv[idx]), veh))
            acel = float(np.clip(acel, -min(veh.frenado_max_m_s2, grip),
                                 min(veh.aceleracion_max_m_s2, grip)))
            estado = paso_bicicleta(estado, acel, delta, dt, veh)
            elapsed += dt
            s_prev, t_prev = avance, elapsed
            # C7. Visualización repite el índice de la vuelta base.
            # No mezclar este índice gráfico con la estación multivuelta.
            if elapsed-ultimo >= .075:
                ultimo = elapsed
                dist = max(0., avance_centro-inicio_centro)
                # Visualización en el mismo índice de ruta, con cronómetro
                # y distancia común basados en las estaciones del centro.
                s_base = inicio_base + (avance-inicio_total) % (fin_base-inicio_base)
                idx_base = int(np.clip(np.searchsorted(sb, s_base)-1, 0, len(base)-2))
                setstate('model', time=round(elapsed, 3),
                         speed=round(estado.velocidad_m_s*3.6, 2),
                         distance=round(dist, 2),
                         progress=round(100.0*min(1.0, dist/max(1e-9, metas[-1]-inicio_total)), 2),
                         x=float(estado.x), y=float(estado.y), heading=float(estado.psi_rad),
                         delta=math.degrees(estado.delta_rad),
                         accel_lat=estado.velocidad_m_s**2*math.tan(estado.delta_rad)/veh.distancia_ejes_m,
                         error_lateral=error, plan_index=idx_base)
            time.sleep(dt/get_speed_factor())
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if STOP_MODEL.is_set():
            setstate('model', status='detenido', message='Detenido manualmente')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        elif vueltas_completadas >= vueltas:
            setstate('model', status='completado', time=round(elapsed, 3), progress=100.0,
                     message=f'{vueltas_completadas}/{vueltas} vueltas completas')
        else:
            setstate('model', status='incompleto', time=round(elapsed, 3),
                     message='No completó las vueltas antes del tiempo máximo')
    except Exception as exc:
        setstate('model', status='error', message=f'{type(exc).__name__}: {exc}')
    finally:
        with LOCK:
            MODEL_THREAD = None
# Reconoce mensajes del conductor y extrae observaciones y cruces de meta verificables.
def parse_driver_log(line):
    """D. Traducir la versión ACTUAL del conductor a métricas para el HTML.
    Los logs son muestras reales tomadas por el conductor de la telemetría AC
    (cada ~0.5 s); se evita confundirlos con el perfil calculado por Python.
    """
    match = re.search(r'\[COMPARADOR\] VUELTAS_CONFIGURADAS\s+(\d+)', line)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if match:
        setstate('assetto', laps_requested=int(match.group(1)))
    match = re.search(r'\bvuelta=(\d+)/(\d+)\s*\|\s*t=([\d.]+)s\s*\|\s*recorrido=([-\d.]+)m\s*\|\s*v_real=([-\d.]+)\s*km/h(?:\s*\|\s*v_objetivo=([-\d.]+))?', line)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if match:
        lap, total, elapsed, distance, speed, target = match.groups()
        values = dict(status='corriendo', time=float(elapsed), distance=float(distance),
                      speed=float(speed), laps_requested=int(total), current_lap=int(lap),
                      telemetry_source='conductor: speedKmh (AC)')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if target is not None:
            values['speed_target'] = float(target)
        setstate('assetto', **values)
        return
    # D1. Compatibilidad con el formato original de una vuelta.
    match = re.search(r'\bt=([\d.]+)s\s*\|\s*recorrido=([-\d.]+)m\s*\|\s*v=([-\d.]+)', line)
    if match:
        t, d, v = map(float, match.groups())
        setstate('assetto', status='corriendo', time=t, distance=d,
                 speed=v, telemetry_source='conductor: speedKmh (AC)')
        return
    # D2. Eventos inequívocos de meta. NO declaramos completada toda la prueba
    # hasta que haya alcanzado el total solicitado.
    match = re.search(r'\[COMPARADOR\] VUELTA_COMPLETADA\s+numero=(\d+)\s+total=(\d+)\s+tiempo_s=([\d.]+)', line)
    if match:
        numero, total, lap_time = match.groups()
        numero, total = int(numero), int(total)
        with LOCK:
            old = STATE['assetto'].get('lap_times', [])
            times = old[:] if isinstance(old, list) else []
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if len(times) < numero:
                times.append(float(lap_time))
            STATE['assetto'].update(laps_completed=numero, laps_requested=total,
                                   lap_times=times,
                                   last_lap_s=float(lap_time),
                                   status='completado' if numero >= total else 'corriendo')
        return
    match = re.search(r'\[COMPARADOR\] FIN vueltas_completadas=(\d+)\s+vueltas_solicitadas=(\d+)', line)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if match:
        completed, requested = map(int, match.groups())
        setstate('assetto', laps_completed=completed, laps_requested=requested)
        return
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if 'VUELTA GEOMÉTRICA COMPLETADA' in line:
        match = re.search(r'\(([\d.]+)\s*s\)', line)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if match:
            setstate('assetto', laps_completed=1, status='completado',
                     time=float(match.group(1)), lap_times=[float(match.group(1))])
        return
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if 'ERROR:' in line:
        setstate('assetto', status='error', message=line)
# Lee velocidad desde memoria compartida sin sustituir una medición ausente por predicción.
def poll_assetto_telemetry():
    """E. Lectura adicional de velocidad REAL desde la memoria compartida AC.
    Opcional: si mmap aún no existe, dejamos visibles las lecturas de logs.
    La falta de acceso a la memoria no debe romper el proceso de conducción.
    """
    memory = None
    # Protege recursos externos y permite limpieza ante errores.
    try:
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if str(ARGS.script.parent) not in sys.path:
            sys.path.insert(0, str(ARGS.script.parent))
        from telemetria_assetto import Physics, conectar, leer
        # Repite el proceso hasta alcanzar la condición de parada o el límite de seguridad.
        while True:
            with LOCK:
                active = PROCESS is not None
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if not active:
                break
            # Protege recursos externos y permite limpieza ante errores.
            try:
                if memory is None:
                    memory = conectar('acpmf_physics', Physics)
                physics = leer(memory, Physics)
                speed = float(physics.speedKmh)
                if math.isfinite(speed):
                    with LOCK:
                        activo = STATE['assetto'].get('status') in ('corriendo', 'preparado')
                    if activo:
                        setstate('assetto', speed=speed, telemetry_source='Assetto Corsa: memoria compartida')
            except Exception:
                if memory is not None:
                    try:
                        memory.close()
                    except Exception:
                        pass
                    memory = None
            time.sleep(0.17)
    except (ImportError, AttributeError):
        pass
    finally:
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if memory is not None:
            # Protege recursos externos y permite limpieza ante errores.
            try:
                memory.close()
            except Exception:
                pass
# Ejecuta el conductor por el puente y registra telemetría y eventos de vuelta.
def run_assetto(vueltas):
    """F. Lanzar UNA instancia del conductor mediante el puente."""
    global PROCESS, ASSETTO_THREAD
    # Protege recursos externos y permite limpieza ante errores.
    try:
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if not ARGS.script.is_file():
            raise FileNotFoundError(f'No encuentro {ARGS.script}')
        cmd = [sys.executable, '-u', str(ROOT/'puente_plan.py'),
               '--script', str(ARGS.script), '--plan', str(PLAN)]
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if vueltas is not None:
            cmd.extend(['--vueltas', str(vueltas)])
        # F0. Nunca mezclar telemetría del ensayo anterior con el actual.
        csv_anterior = ARGS.script.parent / 'datos' / 'telemetria_ac.csv'
        if csv_anterior.exists():
            csv_anterior.unlink()
        process = subprocess.Popen(cmd, cwd=str(ARGS.script.parent),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding='utf-8',
                                   errors='replace', bufsize=1)
        with LOCK:
            PROCESS = process
        threading.Thread(target=poll_assetto_telemetry, daemon=True).start()
        autoenter = False
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for raw in process.stdout:
            line = raw.rstrip('\r\n')
            with LOCK:
                assetto = STATE['assetto']
                assetto['logs'] = (assetto.get('logs') or [])[-79:] + [line]
                assetto['message'] = line
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if '[COMPARADOR] PLAN_EXPORTADO' in line:
                with LOCK:
                    STATE['plan_ready'] = True
                # F1. Se puede escribir antes de que input() lo solicite.
                if process.stdin and not autoenter:
                    try:
                        process.stdin.write('\n')
                        process.stdin.flush()
                        autoenter = True
                    except (BrokenPipeError, OSError):
                        pass
                setstate('assetto', status='preparado')
            parse_driver_log(line)
        code = process.wait()
        with LOCK:
            current_status = STATE['assetto'].get('status')
            completed = STATE['assetto'].get('laps_completed', 0)
            wanted = STATE['assetto'].get('laps_requested')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if current_status not in ('completado', 'error'):
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if wanted is not None and completed >= wanted:
                setstate('assetto', status='completado')
            else:
                setstate('assetto', status='error' if code else 'incompleto',
                         message=f'Proceso terminó (código {code}); {completed}/{wanted} vueltas')
    except Exception as exc:
        setstate('assetto', status='error', message=f'{type(exc).__name__}: {exc}')
    finally:
        with LOCK:
            PROCESS = None
            ASSETTO_THREAD = None
# G0. DATOS FÍSICOS Y GRÁFICAS.
# Los valores son parámetros del modelo, no mediciones independientes de AC.
def parametros_fisicos():

    """G0. Mostrar SOLO parámetros efectivamente exportados en el plan.
    El plan es la fuente para los valores utilizados por el optimizador.

    Los datos del automóvil y neumático en sesión se consultan aparte y

    no se confunden con la procedencia del plan.

    """

    plan = load_plan()

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if plan is None:

        return {'ready': False, 'message': 'Aún no hay plan exportado. Inicia Assetto para obtener los parámetros.'}
    v = plan.get('vehiculo') or {}
    # Gestiona la etapa «numero» del flujo del módulo, validando los datos que procesa.
    def numero(clave):

        """Gestiona la etapa «numero» del flujo del módulo, validando los datos que procesa."""
        # Protege recursos externos y permite limpieza ante errores.
        try:

            valor = float(v[clave])

            return valor if math.isfinite(valor) else None

        except (KeyError, TypeError, ValueError):

            return None
    masa = numero('masa')

    potencia = numero('potencia')

    datos = {

        'ready': True,

        'origen': 'Valores utilizados por el optimizador, exportados en plan_compartido.json',

        'masa_kg': masa,

        'peso_n': masa * 9.81 if masa is not None else None,

        'coef_friccion': numero('mu'),

        'distancia_ejes_m': numero('L'),

        'potencia_util_kw': potencia / 1000 if potencia is not None else None,

        'frenado_max_m_s2': numero('freno'),

        'aceleracion_max_m_s2': numero('aceleracion'),

        'tiempo_estimado_s': plan.get('tiempo_estimado_s'),

        'automovil': None,

        'compuesto': None,

        'origen_automovil': None,

    }
    # Identificación opcional de la sesión ACTUAL: no certifica la procedencia

    # del plan si éste fue exportado con otra sesión.

    try:

        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if str(ARGS.script.parent) not in sys.path:

            sys.path.insert(0, str(ARGS.script.parent))

        from telemetria_assetto import Graphics, Static, conectar, leer

        gm = conectar('acpmf_graphics', Graphics)

        sm = conectar('acpmf_static', Static)

        # Protege recursos externos y permite limpieza ante errores.
        try:

            g = leer(gm, Graphics)

            s = leer(sm, Static)

            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if g.status == 2:

                datos['automovil'] = str(s.carModel).strip('\x00 ')

                datos['compuesto'] = str(g.tyreCompound).strip('\x00 ')

                datos['origen_automovil'] = 'Sesión LIVE actual (independiente del plan)'

        finally:

            gm.close()

            sm.close()

    except (OSError, ImportError, AttributeError, ValueError, TypeError):

        pass
    return datos
# Genera imágenes a partir de muestras registradas y evita recalcularlas demasiado seguido.
def cargar_telemetria_ac():
    """G0a. Leer muestras LIVE de AC sin interpolarlas a frecuencia ficticia.

    Se omiten las muestras incompletas mientras el conductor escribe el CSV.
    Los logs de 2 Hz permanecen disponibles como respaldo identificado.
    """
    if ARGS is None:
        return []
    archivo = ARGS.script.parent / 'datos' / 'telemetria_ac.csv'
    if not archivo.is_file():
        return []
    muestras = []
    try:
        with archivo.open(newline='', encoding='utf-8') as f:
            for fila in csv.DictReader(f):
                try:
                    t = float(fila['time_s'])
                    d = float(fila['distance_m'])
                    v = float(fila['speed_kmh'])
                except (KeyError, ValueError, TypeError):
                    continue
                if all(math.isfinite(x) for x in (t,d,v)) and d >= 0:
                    muestras.append({'time':t, 'distance':d, 'speed':v})
    except OSError:
        return []
    return muestras


def generar_graficas_api():
    """Generar PNG a demanda, como máximo cada 4 s, sin bloquear la telemetría."""
    with GRAFICAS_LOCK:
        ahora = time.monotonic()
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if (ULTIMAS_GRAFICAS['respuesta'] is not None
                and ahora - ULTIMAS_GRAFICAS['generadas'] < 4):
            return ULTIMAS_GRAFICAS['respuesta']
        with LOCK:
            datos = {'model': list(HISTORIAL['model']), 'assetto': list(HISTORIAL['assetto']),
                     'lap_times_model': list(STATE['model'].get('lap_times') or []),
                     'lap_times_assetto': list(STATE['assetto'].get('lap_times') or [])}
        # Datos AC registrados por tick tienen prioridad sobre el log cada 0.5 s.
        # Si aún no hay muestras completas, permanecen los datos de respaldo.
        alta_frecuencia = cargar_telemetria_ac()
        if alta_frecuencia:
            datos['assetto'] = alta_frecuencia
        # Mantener las tres series separadas: perfil, bicicleta y AC.
        datos['plan'] = load_plan()
        from graficas_comparacion import generar
        resultado = generar(datos, GRAFICAS)
        estampilla = int(time.time()*1000)
        resultado['imagenes'] = {clave: f'/graficas/{archivo}?v={estampilla}'
                                for clave, archivo in resultado['archivos'].items()}
        resultado['mensaje'] = ('Esperando telemetría para crear gráficas'
                                if not resultado['archivos'] else 'Gráficas generadas con las muestras registradas')
        ULTIMAS_GRAFICAS.update(generadas=ahora, respuesta=resultado)
        return resultado
# Publica endpoints HTTP para el comparador y sirve su interfaz web.
class API(BaseHTTPRequestHandler):
    """G. API y páginas del comparador web (sin overlay Electron)."""
    # Gestiona la etapa «log_message» del flujo del módulo, validando los datos que procesa.
    def log_message(self, *args):
        """Gestiona la etapa «log_message» del flujo del módulo, validando los datos que procesa."""
        pass
    # Gestiona la etapa «respond» del flujo del módulo, validando los datos que procesa.
    def respond(self, obj, code=200):
        """Gestiona la etapa «respond» del flujo del módulo, validando los datos que procesa."""
        payload = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    # Gestiona la etapa «page» del flujo del módulo, validando los datos que procesa.
    def page(self, name):
        """Gestiona la etapa «page» del flujo del módulo, validando los datos que procesa."""
        file = ROOT/name
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if not file.is_file():
            return self.respond({'error': f'No encuentro {name}'}, 404)
        content = file.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)
    # Gestiona la etapa «do_GET» del flujo del módulo, validando los datos que procesa.
    def do_GET(self):
        """Gestiona la etapa «do_GET» del flujo del módulo, validando los datos que procesa."""
        ruta = urlsplit(self.path).path
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if ruta == '/api/parametros':
            return self.respond(parametros_fisicos())
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if ruta == '/api/graficas':
            # Protege recursos externos y permite limpieza ante errores.
            try:
                return self.respond(generar_graficas_api())
            except Exception as exc:
                return self.respond({'error': f'{type(exc).__name__}: {exc}'}, 500)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if ruta.startswith('/graficas/'):
            archivo = ruta[len('/graficas/'):]
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if archivo not in {'velocidad_tiempo.png', 'velocidad_distancia.png',
                               'diferencia_velocidad.png', 'tiempos_vuelta.png'}:
                return self.respond({'error': 'Imagen no permitida'}, 404)
            fichero = GRAFICAS / archivo
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if not fichero.is_file():
                return self.respond({'error': 'Gráfica todavía no disponible'}, 404)
            contenido = fichero.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(contenido)))
            self.end_headers()
            self.wfile.write(contenido)
            return
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self.path == '/':
            return self.page('index.html')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self.path == '/api/state':
            with LOCK:
                snapshot = json.loads(json.dumps(STATE))
            return self.respond(snapshot)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if self.path == '/api/plan':
            plan = load_plan()
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if plan is None:
                return self.respond({'ready': False})
            return self.respond({
                'ready': True,
                'ruta': plan['ruta'], 'centro': plan['centro'],
                'izquierdo': plan['izquierdo'], 'derecho': plan['derecho'],
                's': plan['s'], 's_inicio': plan['s_inicio'], 's_fin': plan['s_fin'],
                's_inicio_ruta': plan.get('s_inicio_ruta'),
                's_fin_ruta': plan.get('s_fin_ruta'),
                'longitud': plan['longitud'],
                'tiempo_estimado_s': plan['tiempo_estimado_s'],
            })
        return self.respond({'error': 'No encontrado'}, 404)
    # Gestiona la etapa «do_POST» del flujo del módulo, validando los datos que procesa.
    def do_POST(self):
        """Gestiona la etapa «do_POST» del flujo del módulo, validando los datos que procesa."""
        global MODEL_THREAD, ASSETTO_THREAD
        # Protege recursos externos y permite limpieza ante errores.
        try:
            size = int(self.headers.get('Content-Length', 0))
            request = json.loads(self.rfile.read(size) or b'{}')
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if self.path == '/api/start-assetto':
                vueltas = requested_laps(request.get('vueltas'))
                with LOCK:
                    if ASSETTO_THREAD is not None or PROCESS is not None:
                        return self.respond({'ok': False, 'message': 'Assetto ya está iniciado'}, 409)
                    # G1. Resetear PLAN antes del hilo para que el simulador
                    # propio no cargue por error la trayectoria anterior.
                    PLAN.unlink(missing_ok=True)
                    STATE['plan_ready'] = False
                    STATE['settings']['vueltas'] = vueltas
                    HISTORIAL['assetto'].clear()
                    ULTIMAS_GRAFICAS['generadas'] = 0.0
                    STATE['assetto'] = {
                        'status': 'iniciando', 'time': None, 'distance': None,
                        'speed': None, 'laps_completed': 0, 'lap_times': [],
                        'laps_requested': vueltas,
                        'logs': [], 'message': 'Preparando trayectoria...',
                    }
                    ASSETTO_THREAD = threading.Thread(target=run_assetto, args=(vueltas,), daemon=True)
                    ASSETTO_THREAD.start()
                return self.respond({'ok': True, 'vueltas': vueltas})
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if self.path == '/api/start-model':
                vueltas = requested_laps(request.get('vueltas'))
                with LOCK:
                    if MODEL_THREAD is not None and MODEL_THREAD.is_alive():
                        return self.respond({'ok': False, 'message': 'Modelo ya iniciado'}, 409)
                    if vueltas is None:
                        vueltas = STATE['settings'].get('vueltas')
                    STOP_MODEL.clear()
                    HISTORIAL['model'].clear()
                    ULTIMAS_GRAFICAS['generadas'] = 0.0
                    STATE['model'] = {'status': 'esperando trayectoria',
                                      'laps_completed': 0, 'lap_times': [],
                                      'laps_requested': vueltas}
                    MODEL_THREAD = threading.Thread(target=model_run, args=(vueltas,), daemon=True)
                    MODEL_THREAD.start()
                return self.respond({'ok': True, 'vueltas': vueltas})
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if self.path == '/api/stop-model':
                STOP_MODEL.set()
                return self.respond({'ok': True})
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if self.path == '/api/settings':
                with LOCK:
                    velocity = float(request.get('velocidad', 1.0))
                    if not math.isfinite(velocity) or velocity <= 0:
                        raise ValueError('velocidad debe ser positiva y finita')
                    STATE['settings']['velocidad'] = velocity
                return self.respond({'ok': True})
            return self.respond({'error': 'No encontrado'}, 404)
        except (ValueError, TypeError, OSError) as exc:
            return self.respond({'ok': False, 'message': str(exc)}, 400)
# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main():
    """H. Servidor HTTP local del experimento."""
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', default='../conducir_fisica_optimizado.py')
    parser.add_argument('--port', type=int, default=8771)
    ARGS = parser.parse_args()
    ARGS.script = (ROOT/ARGS.script).resolve()
    print(f'Comparador: http://127.0.0.1:{ARGS.port}/', flush=True)
    print(f'Script de conducción: {ARGS.script}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', ARGS.port), API).serve_forever()
if __name__ == '__main__':
    main()
