"""A. GRÁFICAS EXPERIMENTALES — MODELO BICICLETA FRENTE A ASSETTO CORSA.

No simula el automóvil ni inventa telemetría. Recibe registros numéricos
obtenidos de ambos procesos y genera gráficos PNG con matplotlib (Agg).

Uso independiente: python graficas_comparacion.py datos.json --salida graficas
El servidor invoca ``generar`` al consultar /api/graficas.

Los parámetros del vehículo no son mediciones experimentales de Assetto Corsa:
son las constantes del modelo que el puente exporta en plan_compartido.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# B. PREPARACIÓN Y VALIDACIÓN DE DATOS

def serie(muestras, eje, columna):
    """Devuelve solo pares numéricos, finitos y ordenados por el eje."""
    pares = []
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for m in muestras:
        # Protege recursos externos y permite limpieza ante errores.
        try:
            x, y = float(m[eje]), float(m[columna])
        except (KeyError, TypeError, ValueError):
            continue
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if np.isfinite(x) and np.isfinite(y):
            pares.append((x, y))
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not pares:
        return np.array([]), np.array([])
    pares.sort(key=lambda p: p[0])
    x, y = np.array(pares, dtype=float).T
    return x, y


# Aplica un estilo consistente a los ejes de las gráficas.
def estilo(titulo, xlabel, ylabel):
    """Aplica un estilo consistente a los ejes de las gráficas."""
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=130)
    fig.patch.set_facecolor('#151c2a')
    ax.set_facecolor('#1d2738')
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for spine in ax.spines.values():
        spine.set_color('#5c6b83')
    ax.tick_params(colors='#e2e8f0')
    ax.xaxis.label.set_color('#e2e8f0')
    ax.yaxis.label.set_color('#e2e8f0')
    ax.set_title(titulo, color='#f2f5fc', fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.22, color='#c3d0df')
    fig.tight_layout(pad=2)
    return fig, ax


# Construye la leyenda disponible sin inventar series.
def leyenda(ax):
    """Construye la leyenda disponible sin inventar series."""
    l = ax.legend(loc='best', facecolor='#263247', edgecolor='#61728d')
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if l:
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for t in l.get_texts():
            t.set_color('white')


# C. GENERACIÓN DE GRÁFICAS

def generar(datos: dict, destino: str | Path) -> dict:
    """Genera únicamente las gráficas sustentadas por muestras reales.

    Se devuelven nombres de archivos y conteos, no URLs. El servidor compone
    las URLs locales para servir los PNG; no altera los archivos del conductor.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    modelo = datos.get('model', [])
    assetto = datos.get('assetto', [])
    vuelta_modelo = datos.get('lap_times_model', [])
    vuelta_assetto = datos.get('lap_times_assetto', [])
    archivos = {}
    plan = datos.get('plan')  # Perfil calculado, tercera señal distinta del conductor.
    pc, pv, pk = np.array([]), np.array([]), np.array([])
    if isinstance(plan, dict):
        try:
            sc = np.asarray(plan['s_centro'], dtype=float)
            v = np.asarray(plan['v'], dtype=float)*3.6
            k = np.asarray(plan['k'], dtype=float)
            inicio, fin = float(plan['s_inicio']), float(plan['s_fin'])
            validos = (sc >= inicio)&(sc <= fin)
            if len(sc)==len(v)==len(k) and np.any(validos):
                pc, pv, pk = sc[validos]-inicio, v[validos], k[validos]
        except (KeyError, TypeError, ValueError):
            pass

    # C1. Velocidad contra tiempo: fuentes cronometradas independientemente.
    mt, mv = serie(modelo, 'time', 'speed')
    at, av = serie(assetto, 'time', 'speed')
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if len(mt) or len(at):
        fig, ax = estilo('Velocidad durante el experimento', 'Tiempo propio de cada simulación (s)', 'Velocidad (km/h)')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(mt): ax.plot(mt, mv, color='#4ed5da', label='Bicicleta (Python)', lw=1.8)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(at): ax.plot(at, av, color='#b681ff', label='Assetto Corsa (medida)', lw=1.8)
        leyenda(ax)
        fig.savefig(destino/'velocidad_tiempo.png', bbox_inches='tight')
        plt.close(fig)
        archivos['velocidad_tiempo'] = 'velocidad_tiempo.png'

    # C2. Velocidad contra distancia: mismo eje físico, aunque los relojes difieran.
    md, mvd = serie(modelo, 'distance', 'speed')
    ad, avd = serie(assetto, 'distance', 'speed')
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if len(md) or len(ad):
        fig, ax = estilo('Velocidad según el recorrido', 'Distancia acumulada (m)', 'Velocidad (km/h)')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(md): ax.plot(md, mvd, color='#4ed5da', label='Bicicleta (Python)', lw=1.8)
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(ad): ax.plot(ad, avd, color='#b681ff', label='Assetto Corsa (medida)', lw=1.8)
        if len(pc): ax.plot(pc, pv, color='#fbbf24', ls='--', lw=1.25,
                            label='Perfil predictivo del plan (no medido)')
        leyenda(ax)
        fig.savefig(destino/'velocidad_distancia.png', bbox_inches='tight')
        plt.close(fig)
        archivos['velocidad_distancia'] = 'velocidad_distancia.png'

    # C3. Errores de velocidad por distancia: solo puntos de rango compartido.
    #     No se reportan diferencias si las muestras no se solapan.
    if len(md) >= 2 and len(ad) >= 2:
        md_unique, inds = np.unique(md, return_index=True)
        mv_unique = mvd[inds]
        mask = (ad >= md_unique[0]) & (ad <= md_unique[-1])
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if len(md_unique) >= 2 and np.count_nonzero(mask) >= 2:
            err = avd[mask] - np.interp(ad[mask], md_unique, mv_unique)
            fig, ax = estilo('Diferencia de velocidad por distancia', 'Distancia acumulada (m)', 'AC − bicicleta (km/h)')
            ax.plot(ad[mask], err, color='#fca5a5', lw=1.6)
            ax.axhline(0, color='#e2e8f0', lw=1, ls='--', alpha=.7)
            fig.savefig(destino/'diferencia_velocidad.png', bbox_inches='tight')
            plt.close(fig)
            archivos['diferencia_velocidad'] = 'diferencia_velocidad.png'

    # C3a. Curvatura de la línea optimizada, en estaciones del centro común.
    # Los picos corresponden a la discretización existente; no se suavizan
    # aquí para no esconder problemas físicos de la geometría.
    if len(pc) >= 3:
        fig, ax = estilo('Curvatura del plan en Tocancipá',
                         'Distancia sobre la línea central (m)', 'κ (1/m)')
        ax.plot(pc, pk, color='#fbbf24', lw=1.2)
        ax.axhline(0, color='#e2e8f0', lw=1, alpha=.5)
        fig.savefig(destino/'curvatura_plan.png',bbox_inches='tight')
        plt.close(fig)
        archivos['curvatura_plan'] = 'curvatura_plan.png'

    # C4. Cronometría por vueltas: no fabricar una barra si la vuelta no existe.
    if vuelta_modelo or vuelta_assetto:
        fig, ax = estilo('Tiempo por vuelta completada', 'Número de vuelta', 'Tiempo (s)')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if vuelta_modelo:
            ax.plot(np.arange(1, len(vuelta_modelo)+1), vuelta_modelo, 'o-', color='#4ed5da', label='Bicicleta (Python)')
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if vuelta_assetto:
            ax.plot(np.arange(1, len(vuelta_assetto)+1), vuelta_assetto, 's-', color='#b681ff', label='Assetto Corsa')
        ax.xaxis.get_major_locator().set_params(integer=True)
        leyenda(ax)
        fig.savefig(destino/'tiempos_vuelta.png', bbox_inches='tight')
        plt.close(fig)
        archivos['tiempos_vuelta'] = 'tiempos_vuelta.png'

    return {'archivos': archivos, 'muestras_modelo': len(modelo), 'muestras_assetto': len(assetto),
            'vueltas_modelo': len(vuelta_modelo), 'vueltas_assetto': len(vuelta_assetto)}


# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main():
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    ap = argparse.ArgumentParser(description='Graficar la comparación experimental a partir de datos reales')
    ap.add_argument('datos', type=Path, help='JSON con model, assetto, lap_times_model y lap_times_assetto')
    ap.add_argument('--salida', type=Path, default=Path('graficas'))
    args = ap.parse_args()
    resultado = generar(json.loads(args.datos.read_text(encoding='utf-8')), args.salida)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
