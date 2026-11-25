"""DOCUMENTACIÓN DEL MÓDULO — PROYECTO FINAL.

Los comentarios explican entradas, transformaciones y supuestos. Se conserva
la lógica preexistente; los límites estimados no son mediciones directas de AC.
"""

import math
import numpy as np

import modelado_fisico as mf

# ===========================================================================
# E. PERFIL DE VELOCIDAD Y TIEMPO DE UNA TRAYECTORIA COMPLETA
# ===========================================================================
def evaluar_trayectoria(puntos, veh, velocidad_entrada_m_s=80/3.6,
                        velocidad_salida_max_m_s=math.inf, tope_velocidad_m_s=80.0,
                        iteraciones=30):
    """Compara una trayectoria usando frenado atrás y aceleración adelante.

    La curvatura limita la velocidad local (a_lat <= mu*g). Una pasada inversa
    comprueba qué velocidades permiten frenar antes de una curva; la pasada
    directa limita la aceleración según potencia y agarre restantes.

    Nota: es una aproximación discreta, NO una solución exacta de dinámica.
    El inicio debe quedar suficientemente ANTES de la primera curva.
    """
    p = mf.validar_puntos(puntos, minimo=3)  # Coordenadas (N,2).
    s = mf.distancias_acumuladas(p)  # Metros recorridos desde el inicio.
    ds = np.diff(s)  # Distancia de cada intervalo, calculada de una vez.
    k = mf.curvaturas_de_puntos(p)  # Curvatura de cada punto.
    direcciones = mf.curvatura_a_direccion(k, veh)  # Ángulo necesario por bicicleta.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(np.abs(direcciones) > veh.giro_ruedas_max_rad + mf.EPS):
        raise ValueError('La trayectoria requiere demasiado giro de ruedas')
    vlim = np.minimum(tope_velocidad_m_s, mf.velocidad_maxima_curva(k, veh))
    vlim[-1] = min(vlim[-1], velocidad_salida_max_m_s)  # Restringe salida.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if velocidad_entrada_m_s > vlim[0] + 1e-7:
        raise ValueError('La entrada supera el límite de curvatura inicial')
    # Comprobación previa aproximada del cambio de dirección entre puntos.
    dt_local = 2 * ds / np.maximum(vlim[:-1] + vlim[1:], 0.2)
    # vlim puede incluir inf en rectas, pero el tope de velocidad es finito.
    delta_diff = np.abs(np.diff(direcciones))  # Cambios de ángulo entre puntos.
    if np.any(delta_diff > veh.velocidad_giro_ruedas_max_rad_s * dt_local + 1e-6):
        raise ValueError('Los giros consecutivos exigen una dirección demasiado rápida')
    v = vlim.copy()  # Empezamos con la velocidad límite de cada punto.
    v[0] = velocidad_entrada_m_s  # Impone la condición inicial del experimento.
    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for _ in range(iteraciones):  # Itera porque la fricción depende de la velocidad.
        anterior = v.copy()  # Permite comprobar convergencia.
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for i in range(len(v)-2, -1, -1):  # Recorre los puntos hacia atrás.
            libre_i = mf.aceleracion_longitudinal_por_agarre(v[i], k[i], veh)
            libre_j = mf.aceleracion_longitudinal_por_agarre(v[i+1], k[i+1], veh)
            b = min(veh.frenado_max_m_s2, float(libre_i), float(libre_j))
            # Si en el punto siguiente debemos ir lento, aquí necesitamos frenar.
            v[i] = min(v[i], mf.acelerar_en_distancia(v[i+1], b, ds[i]))
        if v[0] < velocidad_entrada_m_s - 1e-5:
            raise ValueError('Falta distancia para frenar desde la velocidad de entrada')
        v[0] = velocidad_entrada_m_s  # Mantiene fija la velocidad de entrada.
        # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
        for i in range(len(v)-1):  # Recorre la trayectoria hacia adelante.
            libre_i = mf.aceleracion_longitudinal_por_agarre(v[i], k[i], veh)
            libre_j = mf.aceleracion_longitudinal_por_agarre(v[i+1], k[i+1], veh)
            a = min(mf.aceleracion_motor(v[i], veh), float(libre_i), float(libre_j))
            v[i+1] = min(v[i+1], mf.acelerar_en_distancia(v[i], a, ds[i]))
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if np.max(np.abs(v - anterior)) < 1e-5:  # Cambio despreciable.
            break
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(v[:-1] + v[1:] <= mf.EPS):
        raise ValueError('Un tramo tiene velocidad nula y tiempo no finito')
    tiempo_por_tramo = 2 * ds / (v[:-1] + v[1:])  # dt=2*ds/(v_i+v_j).
    tiempo = float(np.sum(tiempo_por_tramo))  # Integral discreta: suma de dt.
    uso = np.abs(mf.aceleracion_lateral(v, k)) / (veh.coef_friccion * mf.G)
    return mf.Evaluacion(tiempo, float(s[-1]), v, k, direcciones, float(np.max(uso)))


# ===========================================================================
# F. RECTÁNGULO PARA CONTACTOS Y LÍMITES DE PISTA
# ===========================================================================
def esquinas_rectangulo(estado, veh):
    """Calcula cuatro esquinas; centro de carrocería supuesto L/2 delante del eje trasero."""
    psi = estado.psi_rad  # Orientación del automóvil.
    adelante = np.array([math.cos(psi), math.sin(psi)])  # Eje longitudinal.
    izquierda = np.array([-math.sin(psi), math.cos(psi)])  # Eje transversal.
    centro = np.array([estado.x, estado.y]) + veh.distancia_ejes_m / 2 * adelante
    longitudinal = np.array([-veh.largo_m/2, veh.largo_m/2])  # Frente y atrás.
    transversal = np.array([-veh.ancho_m/2, veh.ancho_m/2])  # Izquierda y derecha.
    aa, bb = np.meshgrid(longitudinal, transversal, indexing='ij')  # 2x2 combinaciones.
    return centro + aa.reshape(-1, 1)*adelante + bb.reshape(-1, 1)*izquierda
    # Resultado (4,2). Este rectángulo no cambia el movimiento de la bicicleta.


# Mide distancia mínima a segmentos de la línea central para comprobar pertenencia al corredor.
def puntos_en_corredor(puntos_consulta, centro_pista, ancho_m):
    """Compara puntos frente a TODOS los segmentos centrales, con broadcasting.

    La pista se aproxima a un corredor de ancho constante con extremos redondos.
    """
    q = mf.validar_puntos(puntos_consulta, minimo=1)  # Puntos a inspeccionar (M,2).
    p = mf.validar_puntos(centro_pista)  # Línea central (N,2).
    a = p[:-1]  # Comienzos de segmentos (N-1,2).
    seg = np.diff(p, axis=0)  # Vectores de segmentos (N-1,2).
    longitud2 = np.sum(seg**2, axis=1)  # Longitudes al cuadrado.
    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if np.any(longitud2 < mf.EPS):
        raise ValueError('El corredor contiene segmentos degenerados')
    diferencia = q[:, None, :] - a[None, :, :]  # (M,N-1,2) por broadcasting.
    t = np.sum(diferencia * seg[None, :, :], axis=2) / longitud2[None, :]
    t = np.clip(t, 0.0, 1.0)  # Proyección dentro de cada segmento (0 a 1).
    proyecciones = a[None, :, :] + t[:, :, None]*seg[None, :, :]
    distancia2 = np.sum((q[:, None, :] - proyecciones)**2, axis=2)
    minima2 = np.min(distancia2, axis=1)  # Cercanía al mejor segmento.
    return minima2 <= (ancho_m/2)**2 + mf.EPS  # Booleano para cada punto consultado.


# Comprueba que las cuatro esquinas de la carrocería permanezcan dentro del corredor.
def carro_dentro_de_pista(estado, veh, centro_pista, ancho_m):
    """Comprueba que las cuatro esquinas de la carrocería permanezcan dentro del corredor."""
    esquinas = esquinas_rectangulo(estado, veh)  # Obtiene los cuatro vértices.
    return bool(np.all(puntos_en_corredor(esquinas, centro_pista, ancho_m)))
    # Solo revisa esquinas, no todos los puntos de los bordes del rectángulo.


# Revisa discretamente las esquinas del vehículo a lo largo de la ruta candidata.
def validar_huella_trayectoria(puntos, centro_pista, veh, ancho_m,
                               salto=1, margen_m=0.0):
    """Comprueba huella rectangular en muestras de la línea propuesta.

    Conservador respecto al margen; la comprobación es DISCRETA (no continua).
    """
    p = mf.validar_puntos(puntos, minimo=3)  # Línea propuesta.
    psi = mf.orientaciones_de_puntos(p)  # Orientación tangente en cada punto.
    # Para esta comprobación, los puntos de la línea se interpretan como eje trasero.
    for i in list(range(0, len(p), salto)) + [len(p)-1]:
        e = mf.Estado(float(p[i, 0]), float(p[i, 1]), float(psi[i]), 0.0)
        esquinas = esquinas_rectangulo(e, veh)  # Cuatro esquinas de ese estado.
        # Comprueba una condición para separar datos válidos y casos excepcionales.
        if not np.all(puntos_en_corredor(esquinas, centro_pista, ancho_m - 2*margen_m)):
            return False  # Una esquina toca o sobrepasa el límite permitido.
    return True
