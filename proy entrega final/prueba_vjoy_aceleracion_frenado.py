"""Prueba controlada de aceleración y frenado de Assetto Corsa con vJoy.

Requisitos: Assetto Corsa LIVE, pista recta despejada, vJoy mapeado,
controller_vjoy.py y medir_aceleracion_frenado_ac.py en la misma carpeta.

El registrador genera muestras.csv, resumen_por_velocidad.csv y curvas.png.
No usa el optimizador ni modifica archivos del coche.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import time

from controller_vjoy import ControladorVJoy
from telemetria_assetto import Physics, Graphics, conectar, leer


BASE = Path(__file__).resolve().parent
MEDIDOR = BASE / 'medir_aceleracion_frenado_ac.py'


# Espera durante la fase de control conservando una duración limitada.
def esperar_segundos(duracion, tiempo_muestra=0.05):
    """Espera durante la fase de control conservando una duración limitada."""
    fin = time.monotonic() + duracion
    # Repite el proceso hasta alcanzar la condición de parada o el límite de seguridad.
    while time.monotonic() < fin:
        time.sleep(min(tiempo_muestra, max(0.0, fin - time.monotonic())))


# Lee velocidad y estado LIVE para decidir la fase de una prueba experimental.
def tomar_estado(fisica, graficos):
    """Lee velocidad y estado LIVE para decidir la fase de una prueba experimental."""
    g = leer(graficos, Graphics)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if g.status != 2:
        raise RuntimeError('Assetto Corsa dejó de estar LIVE. Prueba abortada.')
    p = leer(fisica, Physics)
    velocidad = float(p.speedKmh)
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not math.isfinite(velocidad):
        raise RuntimeError('La telemetría de velocidad no es válida.')
    return velocidad, float(p.gas), float(p.brake)


# Aplica una orden de control hasta que se alcance el objetivo o expire el tramo.
def tramo_hasta(fisica, graficos, mando, *, etapa, acelerador, freno,
                objetivo_kmh, max_segundos, comprobacion_pedal=True):
    """Aplica una orden de control hasta que se alcance el objetivo o expire el tramo."""
    mando.enviar(acelerador=acelerador, freno=freno, direccion=0.0)
    comienzo = time.monotonic()
    ultima_impresion = -1.0
    pedal_comprobado = not comprobacion_pedal
    vel_inicial, _, _ = tomar_estado(fisica, graficos)
    print(f'\n{etapa}: inicio {vel_inicial:.1f} km/h; objetivo {objetivo_kmh:.1f} km/h', flush=True)

    # Repite el proceso hasta alcanzar la condición de parada o el límite de seguridad.
    while True:
        ahora = time.monotonic() - comienzo
        velocidad, gas_real, freno_real = tomar_estado(fisica, graficos)

        # Comprobación de mapeo: el juego DEBE reconocer el pedal pedido.
        if not pedal_comprobado and ahora >= 0.8:
            valor = gas_real if acelerador > 0 else freno_real
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if valor < 0.70:
                raise RuntimeError(
                    f'Assetto Corsa no registra el pedal de {etapa.lower()} '
                    f'(lectura={valor:.2f}). Revisa la asignación de ejes vJoy. '
                    'Se liberan los mandos.'
                )
            pedal_comprobado = True

        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if ahora - ultima_impresion >= 1.0:
            print(f'  t={ahora:5.1f}s | v={velocidad:6.1f} km/h | '
                  f'gas={gas_real:.2f} | freno={freno_real:.2f}', flush=True)
            ultima_impresion = ahora

        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if etapa == 'ACELERACIÓN' and velocidad >= objetivo_kmh:
            print('  Velocidad objetivo alcanzada.', flush=True)
            return
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if etapa == 'FRENADO' and velocidad <= objetivo_kmh:
            print('  Vehículo detenido.', flush=True)
            return
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if ahora >= max_segundos:
            raise RuntimeError(
                f'Tiempo máximo de {etapa.lower()} ({max_segundos:.0f}s) alcanzado '
                'sin cumplir el objetivo. Se liberan los mandos.'
            )
        time.sleep(0.05)


# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main():
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    parser = argparse.ArgumentParser(description='Prueba de aceleración/frenado usando vJoy')
    parser.add_argument('--objetivo-kmh', type=float, default=120.0,
                        help='Velocidad donde comienza la frenada (predeterminado 120)')
    parser.add_argument('--max-acelerar-s', type=float, default=20.0)
    parser.add_argument('--max-frenar-s', type=float, default=15.0)
    parser.add_argument('--preparacion-s', type=float, default=4.0)
    parser.add_argument('--pausa-s', type=float, default=0.4,
                        help='Pausa entre acelerar y frenar, con pedales liberados')
    parser.add_argument('--carpeta', default='medicion_ac_vjoy')
    args = parser.parse_args()
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not (10 <= args.objetivo_kmh <= 250):
        parser.error('--objetivo-kmh debe estar entre 10 y 250')
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if min(args.max_acelerar_s, args.max_frenar_s, args.preparacion_s) <= 0 or args.pausa_s < 0:
        parser.error('Los tiempos deben ser positivos, salvo pausa que puede ser cero')
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not MEDIDOR.is_file():
        raise FileNotFoundError(f'Falta el registrador: {MEDIDOR}')

    fisica = None
    graficos = None
    mando = None
    registrador = None
    error = None
    # Protege recursos externos y permite limpieza ante errores.
    try:
        fisica = conectar('acpmf_physics', Physics)
        graficos = conectar('acpmf_graphics', Graphics)
        v, _, _ = tomar_estado(fisica, graficos)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if v > 3.0:
            raise RuntimeError(f'Inicia con el vehículo detenido (ahora: {v:.1f} km/h).')

        mando = ControladorVJoy()
        mando.reposo()
        # Medición inicia antes de aplicar controles.
        tiempo_registro = (args.preparacion_s + args.max_acelerar_s +
                           args.pausa_s + args.max_frenar_s + 3.0)
        comando = [sys.executable, '-u', str(MEDIDOR), '--duracion', str(tiempo_registro),
                   '--carpeta', args.carpeta]
        print('Iniciando registrador:', ' '.join(comando), flush=True)
        registrador = subprocess.Popen(comando, cwd=str(BASE))
        print('No ejecutes el conductor automático durante esta prueba.', flush=True)
        print('Ctrl+C libera vJoy y cancela la prueba.', flush=True)
        print(f'Preparación: {args.preparacion_s:.0f}s para confirmar que la recta está despejada.', flush=True)
        esperar_segundos(args.preparacion_s)
        # Un Popen recién lanzado puede fallar inmediatamente por una importación rota.
        if registrador.poll() is not None:
            raise RuntimeError('El registrador terminó prematuramente; revisa el error anterior.')

        tramo_hasta(fisica, graficos, mando, etapa='ACELERACIÓN', acelerador=1.0,
                    freno=0.0, objetivo_kmh=args.objetivo_kmh,
                    max_segundos=args.max_acelerar_s)
        mando.reposo()
        esperar_segundos(args.pausa_s)
        tramo_hasta(fisica, graficos, mando, etapa='FRENADO', acelerador=0.0,
                    freno=1.0, objetivo_kmh=2.0, max_segundos=args.max_frenar_s)
        print('\nPrueba finalizada; controles liberados.', flush=True)
    except KeyboardInterrupt:
        error = 'Interrumpido por Ctrl+C.'
        print('\n' + error, flush=True)
    except Exception as exc:
        error = str(exc)
        print('\nERROR:', error, flush=True)
    finally:
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if mando is not None:
            # Protege recursos externos y permite limpieza ante errores.
            try:
                mando.reposo()
                mando.cerrar()
            except Exception as exc:
                print('ADVERTENCIA: no se pudo confirmar el reposo de vJoy:', exc, flush=True)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if fisica is not None:
            fisica.close()
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if graficos is not None:
            graficos.close()
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if registrador is not None:
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if registrador.poll() is None:
                print('El registrador está guardando las muestras; controles en reposo.', flush=True)
                try:
                    registrador.wait(timeout=max(5.0, tiempo_registro + 5.0))
                except subprocess.TimeoutExpired:
                    print('El registrador excedió el tiempo; se solicita terminarlo.', flush=True)
                    registrador.terminate()
                    try:
                        registrador.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        registrador.kill()
            # Comprueba una condición para separar datos válidos y casos excepcionales.
            if registrador.returncode not in (None, 0):
                print('ADVERTENCIA: el registrador terminó con código', registrador.returncode)

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if error:
        sys.exit(1)
    print('Resultados: revisa la carpeta', args.carpeta, flush=True)


if __name__ == '__main__':
    main()
