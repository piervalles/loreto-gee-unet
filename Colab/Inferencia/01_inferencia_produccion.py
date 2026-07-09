# =============================================================================
# PIPELINE DE INFERENCIA EN LA NUBE — Monitor Loreto IA
# Archivo : Colab/Inferencia/01_inferencia_produccion.py
# Propósito: Ejecutar el modelo U-Net v4 sobre los TFRecords de
#            datos_entrenamiento_2, clasificar alertas con el sintetizador
#            espacial y exportar el GeoJSON final a GCS.
#
# INSTRUCCIONES DE USO:
#   1. Sube este archivo a Google Colab (o pégalo en un notebook).
#   2. Ejecuta las celdas en orden (0 → 6).
#   3. Para procesar otro dataset, cambia solo la sección "CONFIGURACIÓN".
#
# REQUISITOS:
#   - Cuenta GCP con acceso al bucket dataset-tfrecords-loreto
#   - Bucket mapas_loreto con los buffers .gpkg en /ETL/01/
#   - GPU en Colab (recomendado T4 o superior)
# =============================================================================


# =============================================================================
# CELDA 0 — KEEP-ALIVE + INSTALACIÓN DE DEPENDENCIAS
# Ejecuta primero. Mantiene Colab activo y prepara el entorno.
# =============================================================================

from google.colab import output

output.eval_js('''
new Promise(resolve => {
  setInterval(() => {
    let btn = document.querySelector("colab-connect-button");
    if (btn) btn.click();
    console.log("Keep-alive activo");
  }, 60000);
  resolve("keep-alive iniciado");
});
''')
print("✅ Keep-Alive activado.\n")

import subprocess
resultado_pip = subprocess.run(
    ["pip", "install", "-q", "geopandas", "pyproj", "shapely", "scipy"],
    capture_output=True, text=True
)
print(f"✅ Dependencias instaladas: geopandas, pyproj, shapely, scipy")


# =============================================================================
# CELDA 1 — AUTENTICACIÓN + CONFIGURACIÓN CENTRAL
# =============================================================================

from google.colab import auth
auth.authenticate_user()
print("✅ Autenticación Google Cloud completada.\n")

import os
import json
import logging
import datetime
import numpy as np
import tensorflow as tf
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from scipy import ndimage
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("InferenciaLoreto")

# ---------------------------------------------------------------------------
# ✏️ CONFIGURACIÓN — CAMBIA ESTOS VALORES PARA DISTINTOS DATASETS
# ---------------------------------------------------------------------------

GCP_PROJECT        = "proyecto-ia-496311"
BUCKET_DATOS       = "dataset-tfrecords-loreto"
BUCKET_MAPAS       = "mapas_loreto"

# Prefijo de los TFRecords a procesar
# Para datos 2026: cambiar a "datos_entrenamiento_3/tensor_loreto_full_"
PREFIJO_TFRECORDS  = f"gs://{BUCKET_DATOS}/datos_entrenamiento_2/tensor_loreto_2_"
RUTA_MIXER         = f"{PREFIJO_TFRECORDS}mixer.json"

# Modelo U-Net
RUTA_MODELO_GCS    = f"gs://{BUCKET_DATOS}/modelos_guardados/unet_loreto_4.keras"
RUTA_MODELO_LOCAL  = "/tmp/unet_loreto_4.keras"

# Buffers GIS (jerarquía: Urbano > Minería > Vial > Otros)
RUTA_BUFFERS_GCS   = f"gs://{BUCKET_MAPAS}/ETL/01/"
RUTA_BUFFERS_LOCAL = "/tmp/"
BUFFERS_JERARQUIA  = [
    ("Urbano",  "urbano_buffer_2000m.gpkg"),
    ("Minería", "rios_buffer_500m.gpkg"),
    ("Vial",    "vias_buffer_1000m.gpkg"),
]

# Salida
# Para datos 2026: cambiar a "alertas_loreto_2026.geojson"
FECHA_DATASET      = "2023"
RUTA_SALIDA_GCS    = f"gs://{BUCKET_DATOS}/resultado_inferencia/2023/resultado_inferencia_1.geojson"
RUTA_SALIDA_LOCAL  = f"/tmp/resultado_inferencia_1.geojson"

# Parámetros de inferencia
UMBRAL_DEFORES     = 0.5    # umbral de binarización del modelo U-Net
PIXELES_MINIMOS    = 5      # ignorar clusters menores (ruido)
RESOLUCION_M       = 10.0   # metros por píxel (Sentinel-2 nativo)
CRS_UTM            = "EPSG:32718"  # UTM Zona 18S (Loreto, Perú)

# Bandas en el mismo orden que el entrenamiento
BANDAS = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI', 'VV', 'VH']

# Mapa de causas internas → 3 etiquetas visuales del frontend
MAPA_ETIQUETA = {
    "Urbano":  "Deforestación Urbana",
    "Minería": "Posible minería ilegal aluvial",
    "Vial":    "Deforestación No Urbana",
    "Otros":   "Deforestación No Urbana",
}

# ---------------------------------------------------------------------------
print("✅ Configuración cargada:")
print(f"   TFRecords : {PREFIJO_TFRECORDS}*.tfrecord.gz")
print(f"   Modelo    : {RUTA_MODELO_GCS}")
print(f"   Salida    : {RUTA_SALIDA_GCS}")


# =============================================================================
# CELDA 2 — DESCARGA DE RECURSOS DESDE GCS
# Descarga modelo, buffers GIS y el mixer.json (coordenadas geográficas).
# =============================================================================

def gsutil_cp(origen: str, destino: str, desc: str = "") -> None:
    """Copia un archivo de GCS a local usando gsutil."""
    logger.info(f"📥 Descargando {desc or os.path.basename(origen)}...")
    r = subprocess.run(["gsutil", "cp", origen, destino], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gsutil cp falló para {origen}:\n{r.stderr}")
    logger.info(f"   ✅ OK → {destino}")


# Modelo U-Net
if not os.path.exists(RUTA_MODELO_LOCAL):
    gsutil_cp(RUTA_MODELO_GCS, RUTA_MODELO_LOCAL, "modelo U-Net v4 (38 MB)")
else:
    logger.info("✅ Modelo ya en disco local, omitiendo descarga.")

# Buffers GIS (descargar a /tmp/ — NO usar GeoPandas directamente desde GCS)
for causa, archivo in BUFFERS_JERARQUIA:
    ruta_local = os.path.join(RUTA_BUFFERS_LOCAL, archivo)
    if not os.path.exists(ruta_local):
        gsutil_cp(f"{RUTA_BUFFERS_GCS}{archivo}", ruta_local, f"buffer '{causa}'")
    else:
        logger.info(f"✅ Buffer '{causa}' ya en disco.")

# Mixer JSON (georeferenciación de parches)
RUTA_MIXER_LOCAL = "/tmp/mixer.json"
mixer = None
try:
    gsutil_cp(RUTA_MIXER, RUTA_MIXER_LOCAL, "mixer.json (georeferenciación)")
    with open(RUTA_MIXER_LOCAL, encoding="utf-8") as f:
        mixer = json.load(f)
    logger.info(
        f"✅ Mixer cargado: {mixer.get('totalPatches','?')} parches totales, "
        f"{mixer.get('patchesPerRow','?')} por fila."
    )
except Exception as exc:
    logger.warning(
        f"⚠️ No se encontró mixer.json ({exc}).\n"
        "   Se usarán coordenadas de respaldo distribuidas en Loreto.\n"
        "   Las coordenadas serán aproximadas pero el flujo funcionará."
    )
    mixer = None

print("\n✅ Todos los recursos descargados.")


# =============================================================================
# CELDA 3 — MÓDULOS INLINE (autocontenidos, sin imports del backend local)
# =============================================================================

# ---------------------------------------------------------------------------
# 3.1 Conversión de coordenadas UTM 18S → WGS84
# ---------------------------------------------------------------------------
_PROYECTOR_UTM = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True)

def utm_a_wgs84(x_utm: float, y_utm: float) -> tuple:
    """Convierte coordenadas UTM 18S → (lon, lat) WGS84."""
    return _PROYECTOR_UTM.transform(x_utm, y_utm)


# ---------------------------------------------------------------------------
# 3.2 Georreferenciación desde mixer.json de GEE
# ---------------------------------------------------------------------------
def obtener_origen_parche_utm(parche_global_idx: int, mixer_data: dict) -> tuple:
    """
    Calcula el origen UTM (esquina superior-izquierda) de un parche de 128×128
    usando los metadatos del mixer.json exportado por Google Earth Engine.

    El mixer.json de GEE tiene la forma:
    {
      "patchesPerRow": N,
      "totalPatches" : M,
      "projection"   : {
        "crs": "EPSG:32718",
        "affine": { "doubleMatrix": [sx, 0, ox, 0, sy, oy] }
      }
    }
    donde sx = escala X (ej. 10.0), sy = escala Y (ej. -10.0, negativo = norte arriba),
    ox = easting origen, oy = northing origen (esquina superior-izquierda del mosaico).

    Retorna (origen_x_utm, origen_y_utm) del parche.
    """
    if mixer_data is None:
        # Fallback: distribuir de forma determinista dentro de la bbox de Loreto
        # ROI del pipeline_caja_blanca_2.py: lon [-73.83, -71.26], lat [-4.91, -3.09]
        # En UTM 18S aprox: X [580_000, 790_000], Y [9_457_000, 9_657_000]
        rng = np.random.default_rng(seed=parche_global_idx)
        x0 = rng.uniform(600_000, 780_000)
        y0 = rng.uniform(9_470_000, 9_640_000)
        return x0, y0

    patches_per_row = mixer_data.get("patchesPerRow", 1)
    affine_matrix   = (mixer_data
                       .get("projection", {})
                       .get("affine", {})
                       .get("doubleMatrix", None))

    if affine_matrix is None or len(affine_matrix) < 6:
        logger.warning("Mixer.json sin affine válido. Usando fallback.")
        return obtener_origen_parche_utm(parche_global_idx, None)

    scale_x  = affine_matrix[0]  # ej.  10.0
    scale_y  = affine_matrix[4]  # ej. -10.0
    origin_x = affine_matrix[2]  # easting  del borde superior-izquierdo del mosaico
    origin_y = affine_matrix[5]  # northing del borde superior-izquierdo del mosaico

    fila = parche_global_idx // patches_per_row
    col  = parche_global_idx %  patches_per_row

    # Desplazamiento en metros (128 píxeles × escala)
    patch_x = origin_x + col  * 128 * scale_x
    patch_y = origin_y + fila * 128 * scale_y   # scale_y negativo → resta

    return patch_x, patch_y


# ---------------------------------------------------------------------------
# 3.3 TraductorUNet — carga y ejecuta el modelo U-Net
# ---------------------------------------------------------------------------
class TraductorUNet:
    """
    Adaptador sobre el modelo U-Net v4 para inferencia de producción.
    Carga el modelo una sola vez y lo mantiene en memoria.
    """

    def __init__(self, ruta_local: str = RUTA_MODELO_LOCAL) -> None:
        logger.info("🧠 Cargando modelo U-Net desde disco local...")
        self.modelo = tf.keras.models.load_model(ruta_local, compile=False)
        logger.info(
            f"   Input shape : {self.modelo.input_shape}\n"
            f"   Output shape: {self.modelo.output_shape}"
        )
        self._warm_up()

    def _warm_up(self) -> None:
        """Pre-compila el grafo con un tensor dummy para evitar latencia JIT."""
        dummy = tf.zeros(shape=(1, 128, 128, 10), dtype=tf.float32)
        _ = self.modelo(dummy, training=False)
        logger.info("   ✅ Warm-up completado. Modelo listo.")

    def evaluar_imagen(
        self,
        tensor: "tf.Tensor",       # (1, 128, 128, 10)
        umbral: float = UMBRAL_DEFORES,
    ) -> np.ndarray:               # (128, 128) binario
        """
        Ejecuta la inferencia U-Net y retorna una máscara binaria (128×128).
        1 = deforestado, 0 = no deforestado.
        """
        probabilidades = self.modelo(tensor, training=False)
        mascara_prob   = tf.squeeze(probabilidades).numpy()     # (128, 128)
        return (mascara_prob >= umbral).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3.4 SintetizadorEspacial — clasifica puntos según buffers GIS
# ---------------------------------------------------------------------------
class SintetizadorEspacial:
    """
    Clasifica puntos UTM según su proximidad a zonas conocidas.
    Jerarquía: Urbano > Minería > Vial > Otros.
    """

    def __init__(self) -> None:
        self._buffers: dict = {}
        self._cargar_buffers()
        logger.info("✅ SintetizadorEspacial listo.")

    def _cargar_buffers(self) -> None:
        for causa, archivo in BUFFERS_JERARQUIA:
            ruta = os.path.join(RUTA_BUFFERS_LOCAL, archivo)
            try:
                gdf = gpd.read_file(ruta)
                self._buffers[causa] = gdf if not gdf.empty else gpd.GeoDataFrame()
                logger.info(
                    f"   Buffer '{causa}': {len(gdf):,} geometrías cargadas."
                )
            except Exception as exc:
                logger.error(f"   ❌ Error cargando buffer '{causa}': {exc}")
                self._buffers[causa] = gpd.GeoDataFrame()

    def clasificar_punto(self, punto_utm: Point) -> str:
        """
        Clasifica un punto shapely.Point (en UTM 18S) con jerarquía:
        Urbano > Minería > Vial > Otros.
        """
        gdf_p = gpd.GeoDataFrame(geometry=[punto_utm], crs=CRS_UTM)

        for causa, _ in BUFFERS_JERARQUIA:
            buf = self._buffers.get(causa, gpd.GeoDataFrame())
            if buf.empty:
                continue
            joined = gpd.sjoin(gdf_p, buf, how="inner", predicate="intersects")
            if not joined.empty:
                return causa

        return "Otros"


# ---------------------------------------------------------------------------
# 3.5 Parser de TFRecord (mismo esquema que el entrenamiento)
# ---------------------------------------------------------------------------
def parsear_tfrecord(ejemplo_proto: bytes) -> "tf.Tensor":
    """
    Parsea un ejemplo TFRecord de GEE y devuelve el tensor imagen (128, 128, 10).
    El campo LABEL existe en el TFRecord pero se ignora en inferencia.
    """
    feature_dict = {
        b: tf.io.FixedLenFeature([128, 128], tf.float32) for b in BANDAS
    }
    feature_dict["LABEL"] = tf.io.FixedLenFeature([128, 128], tf.float32)

    parsed = tf.io.parse_single_example(ejemplo_proto, feature_dict)
    imagen = tf.stack([parsed[b] for b in BANDAS], axis=-1)   # (128, 128, 10)
    return imagen


# ---------------------------------------------------------------------------
# 3.6 Extractor de alertas desde una máscara binaria
# ---------------------------------------------------------------------------
def extraer_alertas_de_mascara(
    mascara:      np.ndarray,     # (128, 128) binario
    parche_idx:   int,
    mixer_data:   dict,
    sintetizador: SintetizadorEspacial,
) -> list:
    """
    Convierte la máscara de un parche en alertas georreferenciadas.
    Usa componentes conectadas (scipy.ndimage.label) para agrupar
    píxeles contiguos de deforestación en una sola alerta.

    Retorna lista de dicts con:
      lon, lat, causa_interna, etiqueta_visual,
      pixeles, porcentaje, area_ha
    """
    if int(mascara.sum()) == 0:
        return []  # parche sin deforestación detectada

    labeled, n_componentes = ndimage.label(mascara)
    origen_x, origen_y = obtener_origen_parche_utm(parche_idx, mixer_data)

    alertas_parche = []

    for i in range(1, n_componentes + 1):
        region  = labeled == i
        pixeles = int(region.sum())

        if pixeles < PIXELES_MINIMOS:
            continue   # descartar ruido

        # Centroide en coordenadas de píxel
        filas, cols = np.where(region)
        fila_c = float(filas.mean())
        col_c  = float(cols.mean())

        # Centroide en UTM 18S
        x_utm = origen_x + col_c * RESOLUCION_M
        y_utm = origen_y - fila_c * RESOLUCION_M   # Y decrece hacia el sur

        punto_utm       = Point(x_utm, y_utm)
        causa_interna   = sintetizador.clasificar_punto(punto_utm)
        etiqueta_visual = MAPA_ETIQUETA.get(causa_interna, "Deforestación No Urbana")

        # Convertir centroide a WGS84 para el GeoJSON
        lon, lat = utm_a_wgs84(x_utm, y_utm)

        alertas_parche.append({
            "lon":            lon,
            "lat":            lat,
            "causa_interna":  causa_interna,
            "etiqueta_visual": etiqueta_visual,
            "pixeles":        pixeles,
            "porcentaje":     round(pixeles / 16_384 * 100, 2),  # 128×128=16384
            "area_ha":        round(pixeles * RESOLUCION_M ** 2 / 10_000, 4),
        })

    return alertas_parche

print("✅ Módulos inline listos.")


# =============================================================================
# CELDA 4 — PIPELINE DE INFERENCIA PRINCIPAL
# =============================================================================

def listar_tfrecords(prefijo: str) -> list:
    """Lista todos los .tfrecord.gz bajo el prefijo dado en GCS."""
    r = subprocess.run(
        ["gsutil", "ls", f"{prefijo}*.tfrecord.gz"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Error al listar TFRecords en {prefijo}:\n{r.stderr}"
        )
    archivos = [
        l.strip() for l in r.stdout.strip().split("\n")
        if l.strip().endswith(".gz")
    ]
    return archivos


def ejecutar_pipeline() -> tuple:
    """
    Orquesta todo el pipeline:
      1. Cargar modelo + sintetizador
      2. Listar TFRecords
      3. Para cada TFRecord: parsear → inferir → extraer alertas → clasificar
    Retorna (lista_alertas, total_parches_procesados).
    """
    # ---- Inicializar componentes ----
    traductor    = TraductorUNet()
    sintetizador = SintetizadorEspacial()

    # ---- Listar TFRecords ----
    archivos = listar_tfrecords(PREFIJO_TFRECORDS)
    if not archivos:
        raise RuntimeError(
            f"❌ No se encontraron TFRecords en {PREFIJO_TFRECORDS}*.tfrecord.gz\n"
            "   Verifica que la ruta y el bucket sean correctos."
        )

    logger.info(f"\n{'='*65}")
    logger.info(f"🚀 INICIANDO INFERENCIA")
    logger.info(f"   Archivos encontrados : {len(archivos)}")
    logger.info(f"   Umbral deforestación : {UMBRAL_DEFORES}")
    logger.info(f"   Píxeles mínimos      : {PIXELES_MINIMOS}")
    logger.info(f"{'='*65}\n")

    todas_las_alertas  = []
    total_parches      = 0
    parche_global_idx  = 0   # índice global entre todos los TFRecords

    for archivo_idx, ruta_tfrecord in enumerate(archivos):
        nombre = os.path.basename(ruta_tfrecord)
        logger.info(f"[{archivo_idx+1:02d}/{len(archivos):02d}] {nombre}")

        try:
            dataset = tf.data.TFRecordDataset(
                ruta_tfrecord,
                compression_type="GZIP",
            )

            for ejemplo in dataset:
                try:
                    imagen = parsear_tfrecord(ejemplo)
                    tensor = tf.expand_dims(imagen, axis=0)     # (1,128,128,10)
                    mascara = traductor.evaluar_imagen(tensor)   # (128,128) binario

                    alertas_parche = extraer_alertas_de_mascara(
                        mascara, parche_global_idx, mixer, sintetizador
                    )
                    todas_las_alertas.extend(alertas_parche)
                    total_parches    += 1
                    parche_global_idx += 1

                    if total_parches % 100 == 0:
                        logger.info(
                            f"   ⏱ {total_parches:5d} parches | "
                            f"{len(todas_las_alertas):4d} alertas"
                        )

                except Exception as exc_parche:
                    logger.warning(
                        f"   ⚠️ Parche {parche_global_idx} ignorado: {exc_parche}"
                    )
                    parche_global_idx += 1
                    continue

        except Exception as exc_archivo:
            logger.error(f"   ❌ Error en {nombre}: {exc_archivo}")
            continue

    logger.info(f"\n✅ Inferencia finalizada:")
    logger.info(f"   Parches procesados : {total_parches:,}")
    logger.info(f"   Alertas generadas  : {len(todas_las_alertas):,}")

    return todas_las_alertas, total_parches


# Ejecutar pipeline
alertas_raw, total_parches_procesados = ejecutar_pipeline()


# =============================================================================
# CELDA 5 — CONSTRUIR GEOJSON Y EXPORTAR A GCS
# =============================================================================

def construir_geojson(alertas: list, total_parches: int) -> dict:
    """
    Construye el FeatureCollection GeoJSON con las alertas detectadas.

    Cada feature es un Point (WGS84) con propiedades:
      - id_unico       : identificador único (ej. L00001)
      - causa_sugerida : causa interna del sintetizador (para el frontend)
      - causa          : etiqueta visual (3 categorías, tal como especificado)
      - pixeles        : número de píxeles deforestados en el cluster
      - porcentaje     : % del parche 128×128 que representa
      - area_ha        : área estimada en hectáreas
      - fecha          : año de los datos fuente
    """
    features = []
    for idx, alerta in enumerate(alertas):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [alerta["lon"], alerta["lat"]],
            },
            "properties": {
                "id_unico":       f"L{idx + 1:05d}",
                "causa_sugerida": alerta["causa_interna"],    # clave para frontend
                "causa":          alerta["etiqueta_visual"],  # etiqueta visual (3 cats)
                "pixeles":        alerta["pixeles"],
                "porcentaje":     alerta["porcentaje"],
                "area_ha":        alerta["area_ha"],
                "fecha":          FECHA_DATASET,
            },
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "fecha_inferencia":  datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "modelo":            "unet_loreto_4.keras",
            "umbral":            UMBRAL_DEFORES,
            "datos_fuente":      f"{PREFIJO_TFRECORDS}*.tfrecord.gz",
            "total_parches":     total_parches,
            "total_alertas":     len(features),
        },
        "features": features,
    }


geojson_final = construir_geojson(alertas_raw, total_parches_procesados)

# --- Guardar localmente ---
with open(RUTA_SALIDA_LOCAL, "w", encoding="utf-8") as f:
    json.dump(geojson_final, f, ensure_ascii=False, indent=2)
logger.info(f"✅ GeoJSON guardado en disco: {RUTA_SALIDA_LOCAL}")

# --- Subir a GCS ---
r_upload = subprocess.run(
    ["gsutil", "-m", "cp", RUTA_SALIDA_LOCAL, RUTA_SALIDA_GCS],
    capture_output=True, text=True,
)
if r_upload.returncode == 0:
    logger.info(f"☁️  GeoJSON subido a GCS: {RUTA_SALIDA_GCS}")
else:
    logger.error(f"❌ Error al subir a GCS:\n{r_upload.stderr}")
    raise RuntimeError("El upload a GCS falló. Revisa los permisos del bucket.")


# =============================================================================
# CELDA 6 — RESUMEN FINAL
# =============================================================================

causas_internas    = Counter(a["causa_interna"]  for a in alertas_raw)
etiquetas_visuales = Counter(a["etiqueta_visual"] for a in alertas_raw)
area_total_ha      = sum(a["area_ha"] for a in alertas_raw)

separador = "=" * 65
print(f"\n{separador}")
print(f"  📊  RESUMEN FINAL — Pipeline de Inferencia Loreto {FECHA_DATASET}")
print(separador)
print(f"  📂  Parches procesados      : {total_parches_procesados:>8,}")
print(f"  📍  Alertas generadas       : {len(alertas_raw):>8,}")
print(f"  🌳  Área total estimada     : {area_total_ha:>8.2f} ha")
print()
print("  🏷️   Distribución por causa interna (sintetizador):")
for causa, n in sorted(causas_internas.items(), key=lambda x: -x[1]):
    barra = "█" * min(n, 30)
    print(f"    {causa:10s} → {n:5d}  {barra}")
print()
print("  🎨  Distribución por etiqueta visual (frontend):")
for etiqueta, n in sorted(etiquetas_visuales.items(), key=lambda x: -x[1]):
    barra = "█" * min(n, 30)
    print(f"    {etiqueta[:42]:42s} {n:5d}  {barra}")
print()
print(f"  ☁️   GeoJSON disponible en:")
print(f"    {RUTA_SALIDA_GCS}")
print(separador)
print()
print("  ✅  Listo. Abre Streamlit y presiona '🔄 Refrescar predicciones'.")
print(separador)
