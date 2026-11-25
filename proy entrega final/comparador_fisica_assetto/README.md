# Comparador de física — edición corregida

**No sobrescribe los archivos de la raíz.** Descomprime esta carpeta dentro de `Proyecto final`:

```
Proyecto final/
  conducir_fisica_optimizado.py
  geometria_circuito.py
  gestor_circuito.py
  modelado_fisico.py
  estimaciones.py
  agente_conductor.py
  ... (otros módulos y carpeta de pista que tu script ya usa)
  comparador_fisica_assetto/
    servidor_comparador.py
    puente_plan.py
    index.html
    graficas_comparacion.py
```

## Ejecutar desde PowerShell

```powershell
cd "C:\Users\jerin\OneDrive\Documents\Proyecto final\comparador_fisica_assetto"
python servidor_comparador.py
```

Abrir `http://127.0.0.1:8771/`. **No pongas el script en la carpeta del comparador:** busca `../conducir_fisica_optimizado.py` por defecto.

Inicia Assetto Corsa por Content Manager, coche inmóvil, sesión LIVE y sobre el circuito real que conoce `geometria_circuito.py`. Pulsa **Iniciar ambos**. El puente arranca el script original en otro proceso. Cuando su optimizador termina, captura exactamente `ruta` y su `evaluacion`, guarda `plan_compartido.json` (archivo nuevo en esta subcarpeta) y pulsa ENTER para iniciar la ejecución real. El modelo de bicicleta propio empieza al detectar el plan y mantiene su cronómetro separado. Ningún CSV ni archivo madre se modifica.

**Vista 3D:** la perspectiva 2.5D se genera con Canvas a partir de `centro`, `izquierdo`, `derecho` y `ruta` del script original; usa una proyección de cámara con horizonte, pista, dos ruedas, dirección delantera y un esquema superior. No pretende copiar la textura fotográfica del juego. Tiene proporción **16:9** y dos cámaras esquemáticas (exterior/interior).

**Visualización en Assetto Corsa:** se utiliza la app instalada en el propio juego. Se retiraron el overlay Electron, su HTML, package.json y el endpoint `/overlay`. La página web mantiene su panel de telemetría de Assetto Corsa.

## Restricciones y precisión científica

- Se importa la **geometría REAL del proyecto del usuario**, **no** se usa `circuito_tocancipa.json` del prototipo antiguo (este archivo se eliminó de este paquete).
- El script original se invoca mediante `runpy.run_path` desde `puente_plan.py`, y sólo se intercepta `optimizar` **en memoria**, para exportar ruta/perfil. El archivo original NO se edita.
- El modelo izquierdo utiliza `paso_bicicleta()` y `Vehiculo` originales e integra Euler con dt=0.02 s, aplicando un controlador geométrico sencillo y el perfil de la ruta. Puede diferir del original y fallar al seguir curvas; se reporta ese fallo en vez de simular un tiempo ficticio.
- **No hay garantías de comparar el mismo recorrido hasta calibrar salida/meta**. La referencia de progresión se proyecta sobre la ruta de forma aproximada; conviene contrastar ambos recorridos usando telemetría de posición.
- No se ha podido probar Assetto/vJoy ni cargar tus módulos de geometría en este entorno porque ejecutan en tu equipo Windows. Aquí se ha comprobado sintaxis, servidor y conectividad HTTP; la integración con el coche debe probarse en tu PC.
