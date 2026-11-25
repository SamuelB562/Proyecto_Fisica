"""Registro experimental independiente para Assetto Corsa.

Ejecutar mientras el conductor da vueltas SIN optimización (PASADAS=0).
No modifica el conductor, los parámetros ni el simulador.

Produce muestras.csv, resumen_por_velocidad.csv y curvas.png (si hay matplotlib).
Las curvas son capacidad OBSERVADA con pedal alto en recta; no necesariamente
capacidad máxima física. El perfil no se debe estimar promediando toda la vuelta.
"""
import argparse
import csv
import math
import statistics
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from telemetria_assetto import Physics, Graphics, conectar, leer


# Interpola linealmente un percentil de los valores ordenados.
def percentil(valores, porcentaje):
    """Interpola linealmente un percentil de los valores ordenados."""
    valores = sorted(valores)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not valores:
        return None
    posicion = (len(valores)-1)*porcentaje/100
    i = int(posicion)
    fraccion = posicion-i
    return valores[i] if i == len(valores)-1 else valores[i]*(1-fraccion)+valores[i+1]*fraccion


# Dibuja curvas experimentales por velocidad sin crear observaciones inexistentes.
def crear_grafica(grupos, destino):
    """Dibuja curvas experimentales por velocidad sin crear observaciones inexistentes."""
    # Protege recursos externos y permite limpieza ante errores.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Gráfica omitida: instala matplotlib con python -m pip install matplotlib')
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for tipo, etiqueta in [('acelerando', 'Aceleración observada'), ('frenando', 'Frenado observado')]:
        serie = sorted((g['velocidad_centro_kmh'], g['aceleracion_representativa_m_s2'])
                       for g in grupos if g['tipo'] == tipo)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if serie:
            xs, ys = zip(*serie)
            ax.plot(xs, ys, marker='o', label=etiqueta)
    ax.set(xlabel='Velocidad (km/h)', ylabel='Magnitud de aceleración (m/s²)',
           title='Assetto Corsa: aceleración y frenado observados en recta')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    print('Gráfica:', destino)


# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main():
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    parser = argparse.ArgumentParser(description='Medición de aceleración y frenado en Assetto Corsa')
    parser.add_argument('--duracion', type=float, default=180, help='Segundos de captura, por defecto 180')
    parser.add_argument('--hz', type=float, default=20, help='Muestreo aproximado, por defecto 20 Hz')
    parser.add_argument('--direccion-max', type=float, default=5.0,
                        help='Ángulo absoluto de dirección en unidades del campo steerAngle (por defecto 5)')
    parser.add_argument('--bin-kmh', type=float, default=10, help='Anchura de las bandas de velocidad')
    parser.add_argument('--min-muestras', type=int, default=5, help='Mínimo para mostrar banda')
    parser.add_argument('--carpeta', default='medicion_ac', help='Carpeta de salida')
    args = parser.parse_args()
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if args.duracion <= 0 or args.hz <= 0 or args.bin_kmh <= 0:
        parser.error('duracion, hz y bin-kmh deben ser positivos')

    salida = Path(args.carpeta) / datetime.now().strftime('%Y%m%d_%H%M%S')
    salida.mkdir(parents=True, exist_ok=True)
    fisica = conectar('acpmf_physics', Physics)
    graficos = conectar('acpmf_graphics', Graphics)
    historial = deque()
    muestras = []
    inicio = time.monotonic()
    print('Capturando datos LIVE de Assetto Corsa. Detén con Ctrl+C.')
    print('Para medir límites, el conductor debe alcanzar acelerador/freno alto en rectas.')

    # Protege recursos externos y permite limpieza ante errores.
    try:
        # Repite el proceso hasta alcanzar la condición de parada o el límite de seguridad.
        while time.monotonic()-inicio < args.duracion:
            tic = time.monotonic()
            p = leer(fisica, Physics)
            g = leer(graficos, Graphics)
            t = tic-inicio
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if g.status == 2:
                v_kmh = float(p.speedKmh)
                gas = float(p.gas)
                brake = float(p.brake)
                steer = float(p.steerAngle)
                if all(math.isfinite(x) for x in (v_kmh, gas, brake, steer)):
                    v = v_kmh/3.6
                    historial.append((t, v))
                    while historial and t-historial[0][0] > 0.40:
                        historial.popleft()
                    # Derivada promedio de velocidad durante ~0.3 s para reducir ruido.
                    if historial and (t-historial[0][0]) >= 0.25:
                        t0, v0 = historial[0]
                        acc = (v-v0)/(t-t0)
                        recta = abs(steer) <= args.direccion_max
                        # Clasificación EXCLUSIVA para no mezclar frenada y aceleración.
                        tipo = ('acelerando' if recta and gas >= .90 and brake <= .05 and acc > .1
                                else 'frenando' if recta and brake >= .80 and gas <= .05 and acc < -.1
                                else 'otro')
                        muestras.append({'tiempo_s': round(t, 3), 'velocidad_kmh': round(v_kmh, 3),
                                         'acelerador': round(gas, 3), 'freno': round(brake, 3),
                                         'steerAngle': round(steer, 3),
                                         'aceleracion_firmada_m_s2': round(acc, 4),
                                         'tramo_recto_aprox': int(recta), 'tipo': tipo})
            else:
                historial.clear()
            time.sleep(max(0.0, 1/args.hz-(time.monotonic()-tic)))
    except KeyboardInterrupt:
        print('\nCaptura interrumpida; se guardan las muestras obtenidas.')
    finally:
        fisica.close()
        graficos.close()

    archivo = salida/'muestras.csv'
    campos = ['tiempo_s', 'velocidad_kmh', 'acelerador', 'freno', 'steerAngle',
              'aceleracion_firmada_m_s2', 'tramo_recto_aprox', 'tipo']
    with archivo.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(muestras)
    print('Muestras:', len(muestras), '| archivo:', archivo)

    agrupados = {}
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for muestra in muestras:
        tipo = muestra['tipo']
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if tipo == 'otro':
            continue
        bin_id = int(muestra['velocidad_kmh']//args.bin_kmh)
        clave = (tipo, bin_id)
        # Magnitud positiva en frenado; positivo en aceleración.
        valor = abs(muestra['aceleracion_firmada_m_s2'])
        agrupados.setdefault(clave, []).append(valor)

    grupos = []
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for (tipo, bin_id), valores in sorted(agrupados.items()):
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(valores) < args.min_muestras:
            continue
        grupos.append({'tipo': tipo, 'velocidad_centro_kmh': (bin_id+.5)*args.bin_kmh,
                       'muestras': len(valores),
                       'aceleracion_media_m_s2': round(statistics.mean(valores), 4),
                       'aceleracion_representativa_m_s2': round(percentil(valores, 80), 4)})

    resumen = salida/'resumen_por_velocidad.csv'
    with resumen.open('w', newline='', encoding='utf-8-sig') as f:
        campos_resumen = ['tipo', 'velocidad_centro_kmh', 'muestras', 'aceleracion_media_m_s2',
                          'aceleracion_representativa_m_s2']
        writer = csv.DictWriter(f, fieldnames=campos_resumen)
        writer.writeheader()
        writer.writerows(grupos)
    print('Resumen:', resumen)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if grupos:
        crear_grafica(grupos, salida/'curvas.png')
    else:
        print('No hubo suficientes muestras de pedal alto en recta para graficar.')
    print('ATENCIÓN: los percentiles son estimaciones observadas; no se deben tratar como máximos garantizados.')


if __name__ == '__main__':
    main()
