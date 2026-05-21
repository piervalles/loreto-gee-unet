import ee
import logging
from dataclasses import dataclass

# =====================================================================
# 1. CONFIGURACIÓN Y LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("Loreto_Pipeline")

@dataclass
class GEEConfig:
    """Configuración central. Modifica los parámetros del pipeline desde aquí."""
    project_id: str = 'proyecto-ia-496311'
    gcs_bucket: str = 'dataset-tfrecords-loreto'
    output_name: str = 'tensor_loreto_' 
    
    # ---> FECHAS CORREGIDAS AL 2023 <---
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    patch_size: tuple = (128, 128)
    
    roi_coords: tuple = (
        ([-74.0, -4.0], [-73.0, -4.0], [-73.0, -3.0], [-74.0, -3.0])
    )
    
    roi: any = None


# =====================================================================
# 2. FUNCIONES MODULARES (Separation of Concerns)
# =====================================================================

def init_gee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
        logger.info(f"Sesión activa conectada al proyecto: {project_id}")
    except Exception as e:
        logger.warning(f"Requiere autenticación OAuth: {e}")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        logger.info("Autenticación completada exitosamente.")

def build_optical_composite(config: GEEConfig) -> ee.Image:
    logger.info("Procesando Sentinel-2 (Óptico) y filtrando nubosidad...")
    s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(config.roi)
              .filterDate(config.start_date, config.end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    return s2_col.median().clip(config.roi)

def apply_spectral_indices(image: ee.Image) -> ee.Image:
    logger.info("Calculando índices algebraicos (NDVI, NDWI, NDTI, EVI)...")
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

def build_radar_composite(config: GEEConfig) -> ee.Image:
    logger.info("Procesando Sentinel-1 (Radar SAR) para penetración de nubes...")
    s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(config.roi)
              .filterDate(config.start_date, config.end_date)
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
              .filter(ee.Filter.eq('instrumentMode', 'IW')))
    return s1_col.median().clip(config.roi)

def load_and_rasterize_labels(config: GEEConfig, base_image: ee.Image) -> ee.Image:
    logger.info("Generando banda de etiquetas (Deforestación EXACTA del año 2023)...")
    
    # ---> DATASET ACTUALIZADO Y FILTRO EXACTO PARA 2023 <---
    hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')
    etiqueta_exacta = hansen.select('lossyear').eq(23).clip(config.roi).rename('LABEL')
    
    return base_image.addBands(etiqueta_exacta.float())

def export_pipeline_to_gcs(tensor_image: ee.Image, config: GEEConfig) -> None:
    ruta_destino = f'datos_entrenamiento/{config.output_name}'
    logger.info(f"Iniciando exportación hacia gs://{config.gcs_bucket}/{ruta_destino}")
    
    task = ee.batch.Export.image.toCloudStorage(
        image=tensor_image.float(),
        description='Generacion_TFRecords_Loreto',
        bucket=config.gcs_bucket,
        fileNamePrefix=ruta_destino,
        region=config.roi,
        scale=10,
        fileFormat='TFRecord',
        maxPixels=1e13,
        formatOptions={
            'patchDimensions': list(config.patch_size),
            'compressed': True
        }
    )
    task.start()
    logger.info("¡Orden enviada exitosamente a los clústeres de Google!")
    logger.info("Puedes monitorear la tarea en: https://code.earthengine.google.com/tasks")

# =====================================================================
# 3. ORQUESTADOR PRINCIPAL
# =====================================================================

def main():
    logger.info("=== INICIANDO PIPELINE DE CAJA BLANCA ===")
    
    cfg = GEEConfig()
    init_gee(cfg.project_id)
    cfg.roi = ee.Geometry.Polygon([list(cfg.roi_coords)])
    
    optical_base = build_optical_composite(cfg)
    optical_features = apply_spectral_indices(optical_base)
    radar_features = build_radar_composite(cfg)
    
    features_tensor = optical_features.select(['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI']) \
                                      .addBands(radar_features.select(['VV', 'VH']))
    
    final_dataset = load_and_rasterize_labels(cfg, features_tensor)
    export_pipeline_to_gcs(final_dataset, cfg)
    
    logger.info("=== EL ORQUESTADOR LOCAL HA TERMINADO SU TRABAJO ===")

if __name__ == "__main__":
    main()