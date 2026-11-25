# Proyecto final — repositorio de trabajo definitivo

Se conservan los módulos activos del proyecto original. La carpeta `disabled/` se eliminó por solicitud expresa. Se retiraron archivos `__pycache__` generados, ya que Python puede recrearlos automáticamente.

## Arquitectura

- `modelado_fisico.py`: bicicleta cinemática, geometría y límites simplificados.
- `estimaciones.py`: evaluación de trayectoria, frenado inverso y aceleración directa.
- `conducir_fisica_optimizado.py` y `agente_conductor.py`: optimización y seguimiento en AC.
- `leer_parametros_ac.py` y `telemetria_assetto.py`: configuración física y memoria compartida del vehículo activo.
- `comparador_fisica_assetto/`: puente de plan, servidor HTTP, HTML y gráficas comparativas.
- `medir_aceleracion_frenado_ac.py` y `prueba_vjoy_aceleracion_frenado.py`: medición experimental con CSV y curva.
- `Ecuaciones_proyecto_final.docx`: guía de las ecuaciones implementadas y sus supuestos.

## Precauciones experimentales

Los valores `DX_REF`/`DY_REF` son coeficientes de referencia, no máximos garantizados durante una vuelta. Los límites longitudinales derivados de archivos INI son aproximaciones, mientras que `medicion_ac_vjoy/` contiene observaciones experimentales. No se ha ejecutado un ensayo LIVE en esta entrega.

La documentación añadida es explicativa y no intenta alterar el comportamiento del código.
