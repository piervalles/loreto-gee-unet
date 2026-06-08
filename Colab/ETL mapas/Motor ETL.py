# =============================================================================
# CELDA 2: ETL ESPACIAL PARA LORETO - PRE-CÁLCULO DE BUFFERS
# Ejecuta esta celda solo cuando la Celda 1 haya finalizado.
# =============================================================================

import geopandas as gpd
import logging
import time
import os
import sys

# Configurar logging con colores simulados (solo texto)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ETL_LORETO")

# Ruta a tu carpeta en Drive (cámbiala si usas otro nombre)
RUTA_DRIVE = "/content/drive/MyDrive/Mapas_Loreto/"
os.makedirs(RUTA_DRIVE, exist_ok=True)  # crea la carpeta si no existe

# CRS de trabajo: UTM 18S (metros, adecuado para Loreto)
CRS_BASE = "EPSG:32718"

# Diccionario de tareas: nombre, archivo de entrada, archivo de salida, distancia buffer (metros)
TAREAS = {
    "rios": {
        "input": "rios_osm.gpkg",
        "output": "rios_buffer_500m.gpkg",
        "buffer": 500,
        "desc": "Zona de minería aluvial"
    },
    "vias": {
        "input": "vias_osm.gpkg",
        "output": "vias_buffer_1000m.gpkg",
        "buffer": 1000,
        "desc": "Zona de deforestación por acceso vial"
    },
    "urbano": {
        "input": "urbano_osm.gpkg",
        "output": "urbano_buffer_2000m.gpkg",
        "buffer": 2000,
        "desc": "Zona de expansión urbana"
    }
}

# -----------------------------
# 2. FUNCIÓN DE PROCESAMIENTO (MEJORADA)
# -----------------------------
def procesar_capa(nombre, config):
    """Carga, calcula buffer y guarda resultado."""
    logger.info(f"\n{'='*50}")
    logger.info(f"INICIANDO: {nombre.upper()} ({config['desc']})")
    logger.info(f"Buffer de {config['buffer']} metros")
    inicio = time.time()

    input_path = os.path.join(RUTA_DRIVE, config['input'])
    output_path = os.path.join(RUTA_DRIVE, config['output'])

    # Validación crítica: ¿existe el archivo de entrada?
    if not os.path.exists(input_path):
        logger.error(f"❌ Archivo NO encontrado: {input_path}")
        logger.error("   Verifica que el archivo esté en tu Google Drive dentro de la carpeta 'Mapas_Loreto'")
        return False

    try:
        # 1. Cargar y reproyectar
        logger.info(f"📂 Leyendo: {config['input']}")
        gdf = gpd.read_file(input_path)
        logger.info(f"   Geometrías originales: {len(gdf):,}")
        logger.info(f"   CRS origen: {gdf.crs}")

        # Reproject to metric CRS
        if gdf.crs != CRS_BASE:
            logger.info(f"   Reproyectando a {CRS_BASE} (UTM 18S)...")
            gdf = gdf.to_crs(CRS_BASE)

        # 2. Disolver todas las geometrías en una sola (más eficiente que unary_union directo)
        logger.info("🔀 Fusionando geometrías (dissolve)...")
        # dissolve crea un GeoDataFrame con una fila y geometría MultiLineString o MultiPolygon
        dissolved = gdf.dissolve()
        geometria_unica = dissolved.geometry.iloc[0]
        logger.info(f"   Geometría fusionada: {geometria_unica.geom_type}")

        # 3. Aplicar buffer (operación pesada)
        logger.info(f"📏 Calculando buffer de {config['buffer']} m...")
        buffer_geom = geometria_unica.buffer(config['buffer'])

        # 4. Simplificar la geometría para reducir peso (tolerancia de 10 metros es suficiente para Loreto)
        logger.info("✂️  Simplificando geometría (tolerancia 10 m)...")
        buffer_geom = buffer_geom.simplify(tolerance=10, preserve_topology=True)

        # 5. Crear GeoDataFrame de salida
        gdf_buffer = gpd.GeoDataFrame(
            geometry=[buffer_geom],
            crs=CRS_BASE,
            data={"capa": [nombre], "buffer_m": [config['buffer']]}
        )

        # 6. Guardar a archivo GeoPackage
        logger.info(f"💾 Guardando resultado: {config['output']}")
        gdf_buffer.to_file(output_path, driver="GPKG")

        # 7. Limpiar variables pesadas para liberar RAM
        del gdf, dissolved, geometria_unica, buffer_geom, gdf_buffer

        elapsed = (time.time() - inicio) / 60
        logger.info(f"✅ {nombre.upper()} COMPLETADO en {elapsed:.2f} minutos")
        return True

    except Exception as e:
        logger.error(f"🔥 Error catastrófico en {nombre}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# -----------------------------
# 3. EJECUCIÓN PRINCIPAL
# -----------------------------
def main():
    logger.info("🚀 INICIANDO ETL ESPACIAL PARA LORETO (MODO PRODUCCIÓN)")
    logger.info(f"📁 Carpeta de trabajo: {RUTA_DRIVE}")

    # Verificar que la carpeta existe y es accesible
    if not os.path.isdir(RUTA_DRIVE):
        logger.error(f"❌ La carpeta {RUTA_DRIVE} no existe. Cámbiala o créala.")
        sys.exit(1)

    # Mostrar archivos presentes en la carpeta (debug)
    logger.info("📋 Archivos encontrados en Drive:")
    for f in os.listdir(RUTA_DRIVE):
        if f.endswith('.gpkg'):
            logger.info(f"   - {f}")

    resultados = {}
    for nombre, config in TAREAS.items():
        ok = procesar_capa(nombre, config)
        resultados[nombre] = ok

    logger.info("\n" + "="*50)
    logger.info("📊 RESUMEN FINAL")
    for nombre, ok in resultados.items():
        estado = "✅ ÉXITO" if ok else "❌ FALLÓ"
        logger.info(f"   {nombre.upper()}: {estado}")

    if all(resultados.values()):
        logger.info("🎉 TODOS LOS BUFFERS FUERON GENERADOS CORRECTAMENTE")
        logger.info(f"📁 Los archivos .gpkg están en: {RUTA_DRIVE}")
        logger.info("   Ahora descárgalos y colócalos en tu backend/data/precomputed/")
    else:
        logger.warning("⚠️ Algunos buffers fallaron. Revisa los mensajes de error.")

# Ejecutar
if __name__ == "__main__":
    main()