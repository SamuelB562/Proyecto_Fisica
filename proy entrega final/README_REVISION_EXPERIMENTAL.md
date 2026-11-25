# Revisión de comparabilidad — ejecución y límites

## Qué se modificó

- La bicicleta comienza en la estación de salida de la vuelta, detenida, igual que la sesión AC. El prefijo de aproximación se sigue usando en la planificación pero no en su cronometraje.
- El optimizador compara tiempos de la vuelta entre salida y meta, y guarda el tiempo del recorrido completo por separado.
- El progreso de la bicicleta se proyecta sobre el centro geométrico, igual que el automóvil AC.
- La telemetría del conductor AC se guarda en `datos/telemetria_ac.csv`, nominalmente una muestra por ciclo de control (objetivo: 50 Hz). La frecuencia real depende del PC; usar las marcas de tiempo del CSV.
- Las gráficas usan ese CSV cuando está presente y conservan los logs como respaldo de menor resolución. La gráfica de curvatura muestra κ(s) para examinar posibles picos.
- Las cotas longitudinales del Porsche HR se leen del resumen experimental existente; su origen y población de muestras son trazables. No extrapolar ni llamarlas máximos certificados.
- La dirección exporta sus parámetros efectivos a la bicicleta del comparador. No se cambia el giro máximo de ruedas de 30° sin verificar cómo interpreta AC `STEER_LOCK`.

## Prueba en Windows

1. Con el Porsche HR en la variante correcta de Tocancipá, detenido y alineado sobre la salida, ejecutar el servidor del comparador.
2. Iniciar ambos desde la página. Registrar si ambos estados parten en 0 m y 0 km/h.
3. Tras finalizar, revisar `datos/telemetria_ac.csv` y pulsar actualizar gráficas. Comprobar el tiempo entre filas, que la primera vuelta empiece donde corresponde y que el eje de distancia sea el centro.
4. Comparar por separado perfil predictivo, bicicleta y AC; no atribuir todo el error del controlador a la física.
5. Ensayar el escalado de giro y validar la convención de `STEER_LOCK` mediante telemetría antes de cambiar el límite nominal de ruedas.
6. Repetir las pruebas de aceleración/frenado a más velocidad si la instalación dispone de una recta suficiente; no extrapolar el ensayo actual de 121 km/h hasta 187 km/h.

## Limitaciones vigentes

- No se dispuso de una sesión LIVE ni vJoy en el entorno de edición. No se declara la conducción validada.
- El controlador de Assetto sigue siendo proporcional y la bicicleta usa uno distinto: sus errores se comparan, pero no equivalen a una diferencia física pura.
- La predicción inicial de `optimizar()` todavía se evalúa sobre el recorrido completo con prefijo y sufijo; el **tiempo que se reporta como vuelta** sí corresponde exclusivamente al intervalo salida/meta.
- El notebook no se adjuntó. Debe corregirse allí «Euler semiimplícito» por «Euler explícito» y el espaciamiento por el valor que se mida en el plan, no por 1 m supuesto.
- La medición experimental suministrada corresponde a un único ciclo de aceleración/frenada y no determina máximos universales.
