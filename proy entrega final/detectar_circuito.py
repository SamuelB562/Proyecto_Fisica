
"""
DETECTOR AUTOMÁTICO DE CIRCUITOS

Identifica la pista cargada en Assetto Corsa
y busca sus archivos de geometría.

No controla el coche ni modifica archivos.
"""

from pathlib import Path

from telemetria_assetto import Static, conectar, leer


# Cambia esta ruta únicamente si Assetto Corsa
# está instalado en otra ubicación.
CARPETA_ASSETTO = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
)


# Consulta la memoria compartida para identificar el circuito cargado y su variante.
def detectar_circuito():
    """Lee el circuito y su variante desde Shared Memory."""

    memoria = conectar("acpmf_static", Static)

    # Protege recursos externos y permite limpieza ante errores.
    try:
        datos = leer(memoria, Static)

        return (
            datos.track.strip(),
            datos.trackConfiguration.strip()
        )
    finally:
        memoria.close()


# Busca los archivos de geometría disponibles para la combinación circuito/variante.
def buscar_geometria(circuito, variante):
    """Encuentra los archivos disponibles de la pista."""

    # Evitar rutas arbitrarias procedentes de los nombres.
    if not circuito or Path(circuito).name != circuito:
        raise ValueError("Nombre de circuito inválido")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if variante and Path(variante).name != variante:
        raise ValueError("Variante inválida")

    carpeta = (
        CARPETA_ASSETTO
        / "content"
        / "tracks"
        / circuito
    )

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not carpeta.is_dir():
        raise FileNotFoundError(
            f"No se encontró el circuito: {carpeta}"
        )

    # Algunas pistas tienen variantes y otras no.
    carpetas = []

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if variante:
        carpetas.append(carpeta / variante)

    carpetas.append(carpeta)

    # Priorizar la variante activa.
    carpeta_activa = next(
        (p for p in carpetas if p.is_dir()),
        carpeta
    )

    archivos = {
        "fast_lane": carpeta_activa / "ai" / "fast_lane.ai",
        "pit_lane": carpeta_activa / "ai" / "pit_lane.ai",
        "side_l": carpeta_activa / "data" / "side_l.csv",
        "side_r": carpeta_activa / "data" / "side_r.csv",
    }

    return carpeta_activa, archivos


# Organiza el flujo principal del módulo y gestiona su salida controlada.
def main():
    """Organiza el flujo principal del módulo y gestiona su salida controlada."""
    print("=== DETECTOR DE CIRCUITOS ===")

    circuito, variante = detectar_circuito()

    print(f"Circuito: {circuito}")
    print(f"Variante: {variante or 'Sin variante'}")

    # Comprueba una condición para separar datos válidos y casos excepcionales.
    if not CARPETA_ASSETTO.is_dir():
        print("\nNo encontré la instalación de Assetto Corsa.")
        print("Configura CARPETA_ASSETTO en este archivo.")
        return

    carpeta, archivos = buscar_geometria(
        circuito, variante
    )

    print(f"\nCarpeta: {carpeta}")
    print("\nArchivos encontrados:")

    # Itera sobre las muestras o segmentos, conservando su orden de procesamiento.
    for nombre, ruta in archivos.items():
        estado = "ENCONTRADO" if ruta.is_file() else "NO ENCONTRADO"
        print(f"{nombre}: {estado}")

    print("\nDetección terminada.")
    print("No se modificó ningún archivo.")


if __name__ == "__main__":
    main()
