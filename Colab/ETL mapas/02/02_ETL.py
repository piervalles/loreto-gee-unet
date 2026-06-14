# =============================================================================
# ETL ESPACIAL PARA LORETO - VERSIÓN RESILIENTE (OPTIMIZADO PARA RAM)
# =============================================================================
# CONFIGURACIÓN DE RUTAS GCS (MODIFICABLES)
GCS_BUCKET = "mapas_loreto"
CARPETA_ORIGEN = "obtener_mapa/01"
CARPETA_DESTINO = "ETL/01"
CRS_BASE = "EPSG:32718"
TOLERANCIA_SIMPLIFICACION = 10   # metros

# Tareas: (nombre, archivo_entrada, archivo_salida, buffer_m)
TAREAS = [
    ("rios",   "rios_osm.gpkg",   "rios_buffer_500m.gpkg",   500),
    ("vias",   "vias_osm.gpkg",   "vias_buffer_1000m.gpkg", 1000),
    ("urbano", "urbano_osm.gpkg", "urbano_buffer_2000m.gpkg",2000)
]

# -----------------------------------------------------------------------------
# 1. INSTALACIÓN, AUTENTICACIÓN Y PREPARACIÓN
# -----------------------------------------------------------------------------
!pip install -q geopandas
from google.colab import auth
auth.authenticate_user()
print("✅ Autenticación GCP completada.")

import geopandas as gpd
import logging
import os
import time
import subprocess
import gc  # Recolector de basura para liberar RAM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ETL_RESILIENTE")

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE VERIFICACIÓN Y PROCESAMIENTO
# -----------------------------------------------------------------------------
def verificar_integridad_gcs(gcs_out, tmp_out):
    """Comprueba si el archivo existe en GCS y si es un GeoPackage válido."""
    if subprocess.run(f"gsutil -q stat {gcs_out}", shell=True).returncode != 0:
        return False

    logger.info(f"🔍 Archivo encontrado en GCS: {gcs_out}. Descargando para verificar integridad...")
    
    if subprocess.run(f"gsutil -q cp {gcs_out} {tmp_out}", shell=True).returncode != 0:
        return False

    try:
        test_gdf = gpd.read_file(tmp_out)
        if len(test_gdf) > 0 and test_gdf.crs == CRS_BASE:
            logger.info("✅ Integridad confirmada: El archivo existe y es válido.")
            os.remove(tmp_out)
            return True
    except Exception as e:
        logger.warning(f"⚠️ El archivo existe pero está CORRUPTO. Se volverá a procesar. Error: {e}")
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        return False

    return False

def procesar_capa(nombre, archivo_in, archivo_out, buffer_dist):
    logger.info(f"\n{'='*50}")
    logger.info(f"INICIANDO: {nombre.upper()} (buffer: {buffer_dist}m)")
    inicio = time.time()

    gcs_in = f"gs://{GCS_BUCKET}/{CARPETA_ORIGEN}/{archivo_in}"
    gcs_out = f"gs://{GCS_BUCKET}/{CARPETA_DESTINO}/{archivo_out}"
    tmp_in = f"/tmp/{archivo_in}"
    tmp_out = f"/tmp/{archivo_out}"

    if verificar_integridad_gcs(gcs_out, tmp_out):
        logger.info(f"⏭️ {nombre.upper()} ya está procesado y es válido. Saltando tarea.")
        return True

    try:
        logger.info(f"📥 Descargando crudo: {gcs_in} → {tmp_in}")
        subprocess.run(f"gsutil -q cp {gcs_in} {tmp_in}", shell=True, check=True)

        logger.info(f"🔧 Leyendo geometrías...")
        gdf = gpd.read_file(tmp_in)
        if gdf.crs != CRS_BASE:
            gdf = gdf.to_crs(CRS_BASE)

        # NUEVA LÓGICA DE MEMORIA EFICIENTE
        logger.info(f"📏 Calculando Buffer individual ({buffer_dist}m)...")
        gdf['geometry'] = gdf.geometry.buffer(buffer_dist)
        
        logger.info(f"✂️ Simplificando geometrías antes de fusionar...")
        gdf['geometry'] = gdf.geometry.simplify(TOLERANCIA_SIMPLIFICACION, preserve_topology=True)

        logger.info(f"🔀 Fusionando polígonos superpuestos (Dissolve seguro)...")
        gdf['capa'] = nombre # Creamos una columna común para agrupar
        gdf_out = gdf.dissolve(by='capa').reset_index() # Fusiona todo lo que se toca

        logger.info(f"💾 Guardando temporalmente en {tmp_out}")
        gdf_out.to_file(tmp_out, driver="GPKG")

        logger.info(f"☁️ Subiendo a {gcs_out}")
        subprocess.run(f"gsutil -q cp {tmp_out} {gcs_out}", shell=True, check=True)

        # LIMPIEZA
        os.remove(tmp_in)
        os.remove(tmp_out)
        
        del gdf, gdf_out
        gc.collect() 
        
        elapsed = (time.time() - inicio) / 60
        logger.info(f"✅ {nombre.upper()} COMPLETADO EXITOSAMENTE en {elapsed:.2f} minutos")
        return True

    except Exception as e:
        logger.error(f"🔥 Fallo crítico al procesar {nombre}: {str(e)}")
        if os.path.exists(tmp_in): os.remove(tmp_in)
        if os.path.exists(tmp_out): os.remove(tmp_out)
        gc.collect()
        return False

# -----------------------------------------------------------------------------
# 3. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
logger.info(f"🚀 INICIANDO ETL RESILIENTE (GCS: {GCS_BUCKET}/{CARPETA_ORIGEN} → {CARPETA_DESTINO})")

resultados = {}
for nombre, in_file, out_file, distancia in TAREAS:
    ok = procesar_capa(nombre, in_file, out_file, distancia)
    resultados[nombre] = ok

logger.info("\n" + "="*50)
logger.info("📊 RESUMEN FINAL DEL ETL")
for nombre, ok in resultados.items():
    estado = "✅ ÉXITO / YA EXISTÍA" if ok else "❌ FALLÓ"
    logger.info(f"   {nombre.upper()}: {estado}")

if all(resultados.values()):
    logger.info("\n🎉 TODOS LOS BUFFERS ESTÁN LISTOS EN LA NUBE.")
else:
    logger.warning("\n⚠️ ALGUNAS TAREAS FALLARON. Revisa los logs.")