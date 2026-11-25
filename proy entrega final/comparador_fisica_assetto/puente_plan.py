"""A. PUENTE ENTRE EL COMPARADOR Y LA CONDUCCIÓN FÍSICA OPTIMIZADA.

Carga el programa de conducción dentro del MISMO proceso e intercepta únicamente
la función optimizar() para exportar el plan que realmente devolvió el
optimizador. No ejecuta la optimización por segunda vez ni sustituye el archivo
original. Conserva las órdenes de control y la telemetría de Assetto Corsa.
"""

import argparse
import json
import os
import pathlib
import runpy
import sys
import tempfile


# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main():
    # B. ARGUMENTOS DEL PUENTE
    # B1. --script y --plan son exclusivos del puente.
    # B2. --vueltas es opcional: si no se envía, el conductor conserva SU valor
    #     VUELTAS_PREDETERMINADAS. Esto evita sobrescribir un 3 con un 1 oculto.
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    ap = argparse.ArgumentParser(description='Puente del comparador físico')
    ap.add_argument('--script', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--vueltas', type=int, default=None)
    args = ap.parse_args()
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if args.vueltas is not None and args.vueltas < 1:
        ap.error('--vueltas debe ser un entero positivo')

    script = pathlib.Path(args.script).resolve()
    destino = pathlib.Path(args.plan).resolve()
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not script.is_file():
        raise FileNotFoundError(f'No encuentro el script: {script}')

    # C. EJECUCIÓN EN EL MISMO PROCESO
    # C1. Se preserva el directorio para imports y recursos relativos.
    directorio_original = pathlib.Path.cwd()
    argv_original = sys.argv[:]
    path_original = sys.path[:]
    # Protege recursos externos y permite limpieza ante errores.
    try:
        sys.path.insert(0, str(script.parent))
        os.chdir(script.parent)
        namespace = runpy.run_path(str(script), run_name='__comparador__')
        ejecutar = namespace['main']
        globales = ejecutar.__globals__
        optimizador_original = globales['optimizar']

        # Gestiona la etapa «interceptar» del flujo del módulo, validando los datos que procesa.
        def interceptar(c, iz, de, veh, v0, s_inicio, s_fin):
            """D. Captura exactamente la salida del optimizador real."""
            # D1. Se invoca una sola vez por llamada del programa principal.
            ruta, evaluacion = optimizador_original(c, iz, de, veh, v0, s_inicio, s_fin)
            import numpy as np

            # D2. El plan que compartimos es una SOLA vuelta de referencia.
            #     El conductor puede repetirla N veces sin calcular nuevos ápices.
            #     No confundimos distancias a lo largo del centro con distancias
            #     a lo largo de la ruta optimizada.
            centros = np.asarray(c, dtype=float)
            trayectoria = np.asarray(ruta, dtype=float)
            segmentos = np.linalg.norm(np.diff(trayectoria, axis=0), axis=1)
            est = np.r_[0.0, np.cumsum(segmentos)]
            centro_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centros, axis=0), axis=1))]
            s_inicio_ruta = float(np.interp(s_inicio, centro_s, est))
            s_fin_ruta = float(np.interp(s_fin, centro_s, est))

            # D3. Los valores 'v' son PREDICCIONES de estimaciones.py,
            #     no velocidades reales provenientes de Assetto Corsa.
            record = {
                'source': 'optimizar() original interceptado en memoria',
                'ruta': trayectoria.tolist(),
                'centro': centros.tolist(),
                'izquierdo': np.asarray(iz, dtype=float).tolist(),
                'derecho': np.asarray(de, dtype=float).tolist(),
                's': est.tolist(),
                's_centro': centro_s.tolist(),
                'v': np.asarray(evaluacion.velocidades_m_s, dtype=float).tolist(),
                'k': np.asarray(evaluacion.curvaturas_menos1, dtype=float).tolist(),
                'delta': np.asarray(evaluacion.direcciones_rad, dtype=float).tolist(),
                's_inicio': float(s_inicio),
                's_fin': float(s_fin),
                's_inicio_ruta': s_inicio_ruta,
                's_fin_ruta': s_fin_ruta,
                'longitud': float(s_fin - s_inicio),
                'longitud_ruta': float(s_fin_ruta - s_inicio_ruta),
                'vueltas_solicitadas': args.vueltas,  # None significa usar el predeterminado del conductor.
                'tiempo_estimado_s': float(evaluacion.tiempo_s),
                'vehiculo': {
                    'L': float(veh.distancia_ejes_m),
                    'masa': float(veh.masa_kg),
                    'potencia': float(veh.potencia_util_w),
                    'mu': float(veh.coef_friccion),
                    'freno': float(veh.frenado_max_m_s2),
                    'aceleracion': float(veh.aceleracion_max_m_s2),
                },
            }

            # D4. Escritura atómica: el HTML nunca debe leer un JSON a medias.
            destino.parent.mkdir(parents=True, exist_ok=True)
            temporal = None
            # Protege recursos externos y permite limpieza ante errores.
            try:
                with tempfile.NamedTemporaryFile(
                    'w', encoding='utf-8', dir=destino.parent,
                    delete=False, suffix='.tmp'
                ) as tmp:
                    json.dump(record, tmp, ensure_ascii=False, separators=(',', ':'))
                    temporal = tmp.name
                os.replace(temporal, destino)
            finally:
                if temporal is not None and os.path.exists(temporal):
                    os.unlink(temporal)

            print(f'[COMPARADOR] PLAN_EXPORTADO {destino} '
                  f'puntos={len(trayectoria)} '
                  f'vueltas={args.vueltas if args.vueltas is not None else "predeterminadas"}',
                  flush=True)
            return ruta, evaluacion

        # E. INTERCEPCIÓN Y CONFIGURACIÓN DE VUELTAS
        globales['optimizar'] = interceptar
        sys.argv = [str(script)]
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if args.vueltas is not None:
            sys.argv += ['--vueltas', str(args.vueltas)]
        # E1. No se suministran --script/--plan al parser del conductor.
        # E2. Sin --vueltas, su configuración predeterminada permanece vigente.
        ejecutar()
    finally:
        # F. Restauración del estado del proceso Python.
        sys.argv = argv_original
        sys.path[:] = path_original
        os.chdir(directorio_original)


if __name__ == '__main__':
    main()
