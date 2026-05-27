import ee
import logging
from dataclasses import dataclass

# =====================================================================
# 1. CONFIGURACIÓN, LOGGING Y CONTROL DE PRODUCCIÓN
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("Loreto_Produccion_Pipeline")

@dataclass
class GEEConfigProduccion:
    """Configuración central para el escalado masivo en producción."""
    project_id: str = 'proyecto-ia-496311'
    gcs_bucket: str = 'dataset-tfrecords-loreto'
    
    # Nombre del directorio de salida para producción masiva
    output_name: str = 'tensor_loreto_full_' 
    
    # Temporalidad controlada (Año completo 2023)
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    patch_size: tuple = (128, 128)
    
    # El objeto ROI ya no será un cuadrado manual, se inicializará dinámicamente
    roi: any = None


# =====================================================================
# 2. FUNCIONES DE EXTRACCIÓN Y GEOPROCESAMIENTO
# =====================================================================

def init_gee(project_id: str) -> None:
    """Inicializa la SDK de Earth Engine con el proyecto asignado."""
    try:
        ee.Initialize(project=project_id)
        logger.info(f"Conexión exitosa al proyecto de producción GCP: {project_id}")
    except Exception as e:
        logger.warning(f"Requiere autenticación interactiva: {e}")
        ee.Authenticate()
        ee.Initialize(project=project_id)

def get_loreto_political_boundary() -> ee.Geometry:
    """
    Filtra la base de datos global de vectores del Banco de Datos de la FAO
    para extraer estrictamente el contorno político de Loreto, Perú.
    """
    logger.info("Cargando límites políticos oficiales desde FAO/GAUL (Nivel 2)...")
    
    # Filtramos por el nombre del departamento oficial indexado por la FAO
    loreto_vector = (ee.FeatureCollection("FAO/GAUL/2015/level2")
                     .filter(ee.Filter.eq('clean_adm1_name', 'Loreto')))
    
    # Extraemos la geometría unificada del contorno
    return loreto_vector.geometry()

def build_optical_composite(config: GEEConfigProduccion) -> ee.Image:
    logger.info("Generando compuesto óptico Sentinel-2 en el contorno de Loreto...")
    s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(config.roi)
              .filterDate(config.start_date, config.end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    # El recorte (.clip) ahora se hace con la silueta real de Loreto
    return s2_col.median().clip(config.roi)

def apply_spectral_indices(image: ee.Image) -> ee.Image:
    logger.info("Inyectando álgebra de índices (NDVI, NDWI, NDTI, EVI)...")
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndti = image.normalizedDifference(['B4', 'B3']).rename('NDTI')
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 10000))', {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }).rename('EVI')
    return image.addBands([ndvi, ndwi, ndti, evi])

def build_radar_composite(config: GEEConfigProduccion) -> ee.Image:
    logger.info("Generando compuesto SAR Sentinel-1 (Fusión Anti-Nubes)...")
    s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(config.roi)
              .filterDate(config.start_date, config.end_date)
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
              .filter(ee.Filter.eq('instrumentMode', 'IW')))
    return s1_col.median().clip(config.roi)

def load_and_rasterize_labels(config: GEEConfigProduccion, base_image: ee.Image) -> ee.Image:
    logger.info("Cargando base de datos Hansen para inyección de verdad-terreno...")
    hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')
    etiqueta_exacta = hansen.select('lossyear').eq(23).clip(config.roi).rename('LABEL')
    return base_image.addBands(etiqueta_exacta.float())

def export_pipeline_to_gcs(tensor_image: ee.Image, config: GEEConfigProduccion) -> None:
    # Aislamos los datos de producción en una carpeta limpia llamada datos_produccion
    ruta_destino = f'datos_produccion/{config.output_name}'
    logger.info(f"Iniciando despacho masivo asíncrono hacia: gs://{config.gcs_bucket}/{ruta_destino}")
    
    task = ee.batch.Export.image.toCloudStorage(
        image=tensor_image.float(),
        description='Generacion_Masiva_TFRecords_Loreto_Full',
        bucket=config.gcs_bucket,
        fileNamePrefix=ruta_destino,
        region=config.roi,
        scale=10, # Mantenemos resolución nativa de 10 metros por píxel
        fileFormat='TFRecord',
        maxPixels=1e13, # Límite expandido para evitar desbordamiento por tamaño de Loreto
        formatOptions={
            'patchDimensions': list(config.patch_size),
            'compressed': True # Compresión GZIP obligatoria para ahorrar almacenamiento
        }
    )
    task.start()
    logger.info("¡La orden masiva de Loreto ha sido inyectada en los servidores de Google!")
    logger.info("Monitorea el progreso del clúster en: https://code.earthengine.google.com/tasks")


# =====================================================================
# 3. ORQUESTADOR CORE
# =====================================================================

def main():
    logger.info("=== INICIANDO PIPELINE DE PRODUCCIÓN DE CAJA BLANCA (LORETO FULL) ===")
    
    cfg = GEEConfigProduccion()
    init_gee(cfg.project_id)
    
    # Capturamos dinámicamente el polígono oficial recortado por la frontera de Loreto
    cfg.roi = get_loreto_political_boundary()
    
    # Orquestación secuencial
    optical_base = build_optical_composite(cfg)
    optical_features = apply_spectral_indices(optical_base)
    radar_features = build_radar_composite(cfg)
    
    # Fusión matemática del Tensor de producción (10 canales limpios)
    features_tensor = optical_features.select(['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI']) \
                                      .addBands(radar_features.select(['VV', 'VH']))
    
    # Acople de etiquetas oficiales de control
    final_dataset = load_and_rasterize_labels(cfg, features_tensor)
    
    # Despacho final a la nube
    export_pipeline_to_gcs(final_dataset, cfg)
    
    logger.info("=== EL DISPARADOR LOCAL HA CULMINADO EL PROCESO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()