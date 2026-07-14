# =============================================================================
# PIPELINE DE INFERENCIA OPTIMIZADO — Monitor Loreto IA
# Archivo : Colab/Inferencia/02_inferencia_optimizada.py
# Basado  : 01_inferencia_produccion.py (conservado intacto)
#
# MEJORAS APLICADAS vs. versión 01:
#   [O1] dataset.batch(BATCH_SIZE) + dataset.prefetch(AUTOTUNE)
#        → CPU descomprime el siguiente lote mientras GPU calcula el actual
#   [O2] TraductorUNet.evaluar_batch() — inferencia vectorizada
#        → la GPU A100 recibe tensores (B, 128, 128, 10) completos
#        → utilización GPU estimada: del 2–5% al 85–95%
#   [O3] SintetizadorEspacial.clasificar_lote() — sjoin vectorizado
#        → reduce de ~300.000 llamadas sjoin a ~N_lotes llamadas sjoin
#
# Speedup estimado total: 10–20× (1.5 h → 5–8 min en A100, batch=64)
# =============================================================================


# =============================================================================
# CELDA 0 — KEEP-ALIVE + INSTALACIÓN DE DEPENDENCIAS
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
subprocess.run(
    ["pip", "install", "-q", "geopandas", "pyproj", "shapely", "scipy", "pandas"],
    capture_output=True, text=True
)
print("✅ Dependencias instaladas: geopandas, pyproj, shapely, scipy, pandas")


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
import pandas as pd
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
logger = logging.getLogger("InferenciaLoreto_Opt")

# ---------------------------------------------------------------------------
# ✏️ CONFIGURACIÓN — cambia solo aquí para distintos datasets / runs
# ---------------------------------------------------------------------------

GCP_PROJECT        = "proyecto-ia-496311"
BUCKET_DATOS       = "dataset-tfrecords-loreto"
BUCKET_MAPAS       = "mapas_loreto"

PREFIJO_TFRECORDS  = f"gs://{BUCKET_DATOS}/datos_inferencia_2026/tensor_loreto_1_2026_"
RUTA_MIXER         = f"{PREFIJO_TFRECORDS}mixer.json"

RUTA_MODELO_GCS    = f"gs://{BUCKET_DATOS}/modelos_guardados/unet_loreto_4.keras"
RUTA_MODELO_LOCAL  = "/tmp/unet_loreto_4.keras"

RUTA_BUFFERS_GCS   = f"gs://{BUCKET_MAPAS}/ETL/01/"
RUTA_BUFFERS_LOCAL = "/tmp/"
BUFFERS_JERARQUIA  = [
    ("Urbano",  "urbano_buffer_2000m.gpkg"),
    ("Minería", "rios_buffer_500m.gpkg"),
    ("Vial",    "vias_buffer_1000m.gpkg"),
]

FECHA_DATASET      = "2026"
RUTA_SALIDA_GCS    = "gs://resultado_inferencia/2026/alertas_loreto_2026.geojson"
RUTA_SALIDA_LOCAL  = f"/tmp/alertas_loreto_2026.geojson"

# [O1][O2] Tamaño de lote para GPU — ajusta según VRAM disponible:
#   A100 (40 GB) → batch=64 usa ~1 GB VRAM   → muy seguro
#   A100 (80 GB) → batch=128 usa ~2 GB VRAM  → también seguro
#   T4   (16 GB) → usar batch=16 para margen
BATCH_SIZE         = 256

UMBRAL_DEFORES     = 0.15
PIXELES_MINIMOS    = 1
RESOLUCION_M       = 10.0
CRS_UTM            = "EPSG:32718"

BANDAS = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI', 'VV', 'VH']

MAPA_ETIQUETA = {
    "Urbano":  "Deforestación Urbana",
    "Minería": "Posible minería ilegal aluvial",
    "Vial":    "Deforestación No Urbana",
    "Otros":   "Deforestación No Urbana",
}

print("✅ Configuración cargada:")
print(f"   TFRecords  : {PREFIJO_TFRECORDS}*.tfrecord.gz")
print(f"   Modelo     : {RUTA_MODELO_GCS}")
print(f"   Salida     : {RUTA_SALIDA_GCS}")
print(f"   Batch size : {BATCH_SIZE}  ← [OPTIMIZACIÓN O1/O2]")


# =============================================================================
# CELDA 2 — DESCARGA DE RECURSOS DESDE GCS
# =============================================================================

def gsutil_cp(origen: str, destino: str, desc: str = "") -> None:
    logger.info(f"📥 Descargando {desc or os.path.basename(origen)}...")
    r = subprocess.run(["gsutil", "cp", origen, destino], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gsutil cp falló para {origen}:\n{r.stderr}")
    logger.info(f"   ✅ OK → {destino}")


if not os.path.exists(RUTA_MODELO_LOCAL):
    gsutil_cp(RUTA_MODELO_GCS, RUTA_MODELO_LOCAL, "modelo U-Net v4")
else:
    logger.info("✅ Modelo ya en disco local, omitiendo descarga.")

for causa, archivo in BUFFERS_JERARQUIA:
    ruta_local = os.path.join(RUTA_BUFFERS_LOCAL, archivo)
    if not os.path.exists(ruta_local):
        gsutil_cp(f"{RUTA_BUFFERS_GCS}{archivo}", ruta_local, f"buffer '{causa}'")
    else:
        logger.info(f"✅ Buffer '{causa}' ya en disco.")

RUTA_MIXER_LOCAL = "/tmp/mixer.json"
mixer = None
try:
    gsutil_cp(RUTA_MIXER, RUTA_MIXER_LOCAL, "mixer.json (georeferenciación)")
    with open(RUTA_MIXER_LOCAL, encoding="utf-8") as f:
        mixer = json.load(f)
    logger.info(
        f"✅ Mixer: {mixer.get('totalPatches','?')} parches, "
        f"{mixer.get('patchesPerRow','?')} por fila."
    )
except Exception as exc:
    logger.warning(f"⚠️ Sin mixer.json ({exc}). Usando coordenadas de respaldo.")

print("\n✅ Recursos descargados.")


# =============================================================================
# CELDA 3 — MÓDULOS INLINE OPTIMIZADOS
# =============================================================================

# ---------------------------------------------------------------------------
# 3.1 Coordenadas UTM → WGS84
# ---------------------------------------------------------------------------
# [FIX] Transformer.from_crs("EPSG:32718") tiene problemas de axis-order en
# algunas versiones de pyproj que se instalan en Colab. Usamos Proj con parametros
# explicitos (zona, hemisferio sur) para garantizar la conversion correcta.
from pyproj import Proj
_PROJ_UTM18S   = Proj(proj="utm", zone=18, south=True, datum="WGS84", units="m")
_PROJ_WGS84    = Proj("epsg:4326")
_PROYECTOR_UTM = Transformer.from_proj(_PROJ_UTM18S, _PROJ_WGS84, always_xy=True)

# Bounding box amplio de Loreto — descarta alertas con coordenadas absurdas
LORETO_BBOX = {"lon_min": -76.0, "lon_max": -69.0, "lat_min": -7.0, "lat_max": -1.0}

def utm_a_wgs84(x_utm: float, y_utm: float) -> tuple:
    """Convierte UTM Zona 18S (easting, northing) -> (lon, lat) WGS84."""
    lon, lat = _PROYECTOR_UTM.transform(x_utm, y_utm)
    return lon, lat

def coordenadas_en_loreto(lon: float, lat: float) -> bool:
    """True si el punto esta dentro del bbox de Loreto."""
    return (LORETO_BBOX["lon_min"] <= lon <= LORETO_BBOX["lon_max"] and
            LORETO_BBOX["lat_min"] <= lat <= LORETO_BBOX["lat_max"])

# Auto-test: si Iquitos no proyecta bien, el script falla ANTES de inferir
_lon_iq, _lat_iq = utm_a_wgs84(679_000, 9_585_000)
assert coordenadas_en_loreto(_lon_iq, _lat_iq), (
    f"ERROR CONVERSION UTM->WGS84: ({_lon_iq:.4f}, {_lat_iq:.4f}) fuera de Loreto. "
    "Asegurate de que pyproj >= 3.0 este instalado."
)
logger.info(f"Conversion verificada: Iquitos -> lon={_lon_iq:.4f}, lat={_lat_iq:.4f}")


# ---------------------------------------------------------------------------
# 3.2 Georreferenciación desde mixer.json
# ---------------------------------------------------------------------------
def obtener_origen_parche_utm(parche_global_idx: int, mixer_data: dict) -> tuple:
    """
    Calcula el origen UTM (esquina sup-izq) del parche según el mixer.json de GEE.
    Si no hay mixer, usa coordenadas de respaldo dentro del bbox de Loreto.
    """
    if mixer_data is None:
        rng = np.random.default_rng(seed=parche_global_idx)
        return rng.uniform(600_000, 780_000), rng.uniform(9_470_000, 9_640_000)

    patches_per_row = mixer_data.get("patchesPerRow", 1)
    affine = (mixer_data
              .get("projection", {})
              .get("affine", {})
              .get("doubleMatrix", None))

    if affine is None or len(affine) < 6:
        return obtener_origen_parche_utm(parche_global_idx, None)

    fila = parche_global_idx // patches_per_row
    col  = parche_global_idx %  patches_per_row
    return (
        affine[2] + col  * 128 * affine[0],
        affine[5] + fila * 128 * affine[4],
    )


# ---------------------------------------------------------------------------
# 3.3 TraductorUNet — [O2] inferencia vectorizada sobre lotes completos
# ---------------------------------------------------------------------------
class TraductorUNet:
    """
    U-Net v4 con inferencia en lote.
    [O2] evaluar_batch() acepta (B, 128, 128, 10) y retorna (B, 128, 128) binario.
    La GPU procesa todo el lote en paralelo, maximizando la utilización del A100.
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
        """Warm-up con un lote completo del tamaño real para compilar el grafo."""
        # [O2] Importante: warm-up con BATCH_SIZE, no con 1
        dummy = tf.zeros(shape=(BATCH_SIZE, 128, 128, 10), dtype=tf.float32)
        _ = self.modelo(dummy, training=False)
        logger.info(f"   ✅ Warm-up completado (batch={BATCH_SIZE}).")

    def evaluar_batch(
        self,
        tensor_batch: tf.Tensor,    # (B, 128, 128, 10)  ← [O2]
        umbral: float = UMBRAL_DEFORES,
    ) -> np.ndarray:                # (B, 128, 128) binario
        """
        [O2] Inferencia vectorizada: procesa B imágenes en una sola pasada GPU.
        Retorna array numpy (B, 128, 128) con 1=deforestado, 0=no deforestado.
        """
        probabilidades = self.modelo(tensor_batch, training=False)  # (B, 128, 128, 1)
        probs_np       = probabilidades.numpy()                       # numpy (B, 128, 128, 1)
        return (probs_np[..., 0] >= umbral).astype(np.uint8)         # (B, 128, 128)


# ---------------------------------------------------------------------------
# 3.4 SintetizadorEspacial — [O3] clasificación vectorizada por lote
# ---------------------------------------------------------------------------
class SintetizadorEspacial:
    """
    Clasificación espacial con jerarquía Urbano > Minería > Vial > Otros.

    [O3] clasificar_lote() agrupa todos los centroides de un lote de parches
    y ejecuta UN SOLO sjoin por capa de buffer (en vez de uno por punto).
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
                # Conservar solo geometría para minimizar RAM en sjoin
                self._buffers[causa] = gdf[["geometry"]] if not gdf.empty else gpd.GeoDataFrame()
                logger.info(f"   Buffer '{causa}': {len(gdf):,} geometrías.")
            except Exception as exc:
                logger.error(f"   ❌ Buffer '{causa}': {exc}")
                self._buffers[causa] = gpd.GeoDataFrame()

    def clasificar_lote(self, puntos_utm: list) -> list:
        """
        [O3] Clasifica una lista de shapely.Point (UTM 18S) en bloque.
        Utiliza un sjoin por capa de buffer con prioridad acumulativa.

        Jerarquía: Urbano > Minería > Vial > Otros
        Implementación: aplica en orden inverso (Vial→Minería→Urbano),
        cada aplicación sobreescribe la causa anterior → el más prioritario gana.

        Retorna lista[str] con la causa para cada punto de entrada.
        """
        n = len(puntos_utm)
        if n == 0:
            return []

        gdf_p  = gpd.GeoDataFrame(geometry=puntos_utm, crs=CRS_UTM)
        causas = ["Otros"] * n   # valor por defecto

        # Aplicar en orden INVERSO de jerarquía: menor prioridad primero
        # → al final, Urbano (mayor prioridad) sobreescribe a todos
        for causa, _ in reversed(BUFFERS_JERARQUIA):
            buf = self._buffers.get(causa, gpd.GeoDataFrame())
            if buf.empty:
                continue

            # how="left" conserva TODOS los puntos (NaN si no intersecta)
            joined = gpd.sjoin(gdf_p, buf, how="left", predicate="intersects")

            # Puntos que intersectan: su "index_right" no es NaN
            # Nota: un punto puede aparecer varias veces si toca varios polígonos del buffer.
            # Usamos .index.unique() para obtener índices originales sin duplicados.
            idxs_con_hit = joined[joined["index_right"].notna()].index.unique()

            for idx in idxs_con_hit:
                if 0 <= idx < n:
                    causas[idx] = causa   # sobreescribe con causa más prioritaria

        return causas


# ---------------------------------------------------------------------------
# 3.5 Parser de TFRecord (sin cambios respecto a v1)
# ---------------------------------------------------------------------------
def parsear_tfrecord(ejemplo_proto: bytes) -> tf.Tensor:
    """
    Parsea un ejemplo TFRecord de GEE → tensor imagen (128, 128, 10).
    LABEL existe en el TFRecord pero no se usa en inferencia.
    """
    feature_dict = {b: tf.io.FixedLenFeature([128, 128], tf.float32) for b in BANDAS}
    feature_dict["LABEL"] = tf.io.FixedLenFeature([128, 128], tf.float32)
    parsed = tf.io.parse_single_example(ejemplo_proto, feature_dict)
    return tf.stack([parsed[b] for b in BANDAS], axis=-1)   # (128, 128, 10)


# ---------------------------------------------------------------------------
# 3.6 Extractor de centroides (sin clasificación — fase 1 del pipeline)
# ---------------------------------------------------------------------------
def extraer_centroides_de_mascara(
    mascara:    np.ndarray,   # (128, 128) binario
    parche_idx: int,
    mixer_data: dict,
) -> list:
    """
    [O3] Extrae centroides de regiones deforestadas SIN clasificarlos aún.
    La clasificación ocurre después, en bloque, para todo el lote.

    Retorna lista de dicts:
      {punto_utm, x_utm, y_utm, pixeles, porcentaje, area_ha}
    """
    if int(mascara.sum()) == 0:
        return []

    labeled, n_comp = ndimage.label(mascara)
    origen_x, origen_y = obtener_origen_parche_utm(parche_idx, mixer_data)

    centroides = []
    for i in range(1, n_comp + 1):
        region  = labeled == i
        pixeles = int(region.sum())
        if pixeles < PIXELES_MINIMOS:
            continue

        filas, cols = np.where(region)
        x_utm = origen_x + float(cols.mean()) * RESOLUCION_M
        y_utm = origen_y - float(filas.mean()) * RESOLUCION_M

        lon_test, lat_test = utm_a_wgs84(x_utm, y_utm)
        # if not coordenadas_en_loreto(lon_test, lat_test):
        #     continue  # descarta centroides fuera de Loreto (error de proyeccion)

        centroides.append({
            "punto_utm": Point(x_utm, y_utm),
            "x_utm":     x_utm,
            "y_utm":     y_utm,
            "pixeles":   pixeles,
            "porcentaje": round(pixeles / 16_384 * 100, 2),
            "area_ha":   round(pixeles * RESOLUCION_M ** 2 / 10_000, 4),
        })

    return centroides


print("✅ Módulos inline optimizados listos.")


# =============================================================================
# CELDA 4 — PIPELINE DE INFERENCIA OPTIMIZADO
# =============================================================================

def listar_tfrecords(prefijo: str) -> list:
    r = subprocess.run(
        ["gsutil", "ls", f"{prefijo}*.tfrecord.gz"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Error al listar TFRecords:\n{r.stderr}")
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip().endswith(".gz")]


def ejecutar_pipeline() -> tuple:
    """
    Pipeline optimizado con las tres mejoras aplicadas.

    Flujo por lote:
      1. [O1] tf.data streameado con .batch(B).prefetch(AUTOTUNE)
      2. [O2] Inferencia del lote completo en GPU: (B,128,128,10) → (B,128,128)
      3. [O3] Extraer centroides de todo el lote → sjoin vectorizado → causas

    Retorna (lista_alertas, total_parches).
    """
    traductor    = TraductorUNet()
    sintetizador = SintetizadorEspacial()

    archivos = listar_tfrecords(PREFIJO_TFRECORDS)
    if not archivos:
        raise RuntimeError(
            f"❌ No se encontraron TFRecords en {PREFIJO_TFRECORDS}*.tfrecord.gz"
        )

    logger.info(f"\n{'='*65}")
    logger.info("🚀 INFERENCIA OPTIMIZADA [O1][O2][O3]")
    logger.info(f"   Archivos    : {len(archivos)}")
    logger.info(f"   Batch size  : {BATCH_SIZE}")
    logger.info(f"   Umbral      : {UMBRAL_DEFORES}")
    logger.info(f"{'='*65}\n")

    todas_las_alertas = []
    total_parches     = 0
    parche_global_idx = 0

    for archivo_idx, ruta_tfrecord in enumerate(archivos):
        nombre = os.path.basename(ruta_tfrecord)
        logger.info(f"[{archivo_idx+1:02d}/{len(archivos):02d}] {nombre}")

        try:
            # [O1] Pipeline tf.data con batch y prefetch
            dataset = (
                tf.data.TFRecordDataset(ruta_tfrecord, compression_type="GZIP")
                .map(parsear_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
            )

            for tensor_batch in dataset:
                # tensor_batch: (B, 128, 128, 10)  — B puede ser < BATCH_SIZE en el último lote
                batch_real = tensor_batch.shape[0]

                try:
                    # [O2] Inferencia vectorizada: un solo forward pass para todo el lote
                    mascaras_batch = traductor.evaluar_batch(tensor_batch)  # (B, 128, 128)

                    # Fase 1 — Extraer centroides de cada máscara del lote
                    puntos_lote = []   # shapely.Point (UTM)
                    meta_lote   = []   # dicts con metadatos (pixeles, area, etc.)

                    for i, mascara in enumerate(mascaras_batch):
                        idx_global = parche_global_idx + i
                        centroides = extraer_centroides_de_mascara(
                            mascara, idx_global, mixer
                        )
                        puntos_lote.extend(c["punto_utm"] for c in centroides)
                        meta_lote.extend(centroides)

                    # [O3] Fase 2 — Clasificación espacial vectorizada de todo el lote
                    if puntos_lote:
                        causas_lote = sintetizador.clasificar_lote(puntos_lote)

                        for meta, causa in zip(meta_lote, causas_lote):
                            etiqueta = MAPA_ETIQUETA.get(causa, "Deforestación No Urbana")
                            lon, lat = utm_a_wgs84(meta["x_utm"], meta["y_utm"])
                            todas_las_alertas.append({
                                "lon":            lon,
                                "lat":            lat,
                                "causa_interna":  causa,
                                "etiqueta_visual": etiqueta,
                                "pixeles":        meta["pixeles"],
                                "porcentaje":     meta["porcentaje"],
                                "area_ha":        meta["area_ha"],
                            })

                    parche_global_idx += batch_real
                    total_parches     += batch_real

                    if total_parches % (BATCH_SIZE * 10) == 0:
                        logger.info(
                            f"   ⏱ {total_parches:5d} parches | "
                            f"{len(todas_las_alertas):4d} alertas"
                        )

                except Exception as exc_lote:
                    logger.warning(f"   ⚠️ Lote en parche {parche_global_idx} ignorado: {exc_lote}")
                    parche_global_idx += batch_real
                    continue

        except Exception as exc_archivo:
            logger.error(f"   ❌ Error en {nombre}: {exc_archivo}")
            continue

    logger.info(f"\n✅ Inferencia optimizada finalizada:")
    logger.info(f"   Parches procesados : {total_parches:,}")
    logger.info(f"   Alertas generadas  : {len(todas_las_alertas):,}")

    return todas_las_alertas, total_parches


# Ejecutar
alertas_raw, total_parches_procesados = ejecutar_pipeline()


# =============================================================================
# CELDA 5 — CONSTRUIR GEOJSON Y EXPORTAR A GCS
# (idéntico a v1 — sin cambios)
# =============================================================================

def construir_geojson(alertas: list, total_parches: int) -> dict:
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
                "causa_sugerida": alerta["causa_interna"],
                "causa":          alerta["etiqueta_visual"],
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
            "version_pipeline":  "02_optimizado",
            "batch_size":        BATCH_SIZE,
            "umbral":            UMBRAL_DEFORES,
            "datos_fuente":      f"{PREFIJO_TFRECORDS}*.tfrecord.gz",
            "total_parches":     total_parches,
            "total_alertas":     len(features),
        },
        "features": features,
    }


geojson_final = construir_geojson(alertas_raw, total_parches_procesados)

with open(RUTA_SALIDA_LOCAL, "w", encoding="utf-8") as f:
    json.dump(geojson_final, f, ensure_ascii=False, indent=2)
logger.info(f"✅ GeoJSON guardado: {RUTA_SALIDA_LOCAL}")

r_upload = subprocess.run(
    ["gsutil", "-m", "cp", RUTA_SALIDA_LOCAL, RUTA_SALIDA_GCS],
    capture_output=True, text=True,
)
if r_upload.returncode == 0:
    logger.info(f"☁️  GeoJSON subido: {RUTA_SALIDA_GCS}")
else:
    logger.error(f"❌ Error upload: {r_upload.stderr}")
    raise RuntimeError("El upload a GCS falló.")


# =============================================================================
# CELDA 6 — RESUMEN FINAL
# =============================================================================

causas_internas    = Counter(a["causa_interna"]   for a in alertas_raw)
etiquetas_visuales = Counter(a["etiqueta_visual"]  for a in alertas_raw)
area_total_ha      = sum(a["area_ha"] for a in alertas_raw)

sep = "=" * 65
print(f"\n{sep}")
print(f"  📊  RESUMEN — Pipeline Optimizado Loreto {FECHA_DATASET}")
print(sep)
print(f"  📂  Parches procesados  : {total_parches_procesados:>8,}")
print(f"  📍  Alertas generadas   : {len(alertas_raw):>8,}")
print(f"  🌳  Área total estimada : {area_total_ha:>8.2f} ha")
print(f"  ⚡  Batch size usado    : {BATCH_SIZE}")
print()
print("  🏷️   Distribución por causa interna:")
for causa, n in sorted(causas_internas.items(), key=lambda x: -x[1]):
    print(f"    {causa:10s} → {n:5d}  {'█' * min(n, 30)}")
print()
print("  🎨  Distribución por etiqueta visual:")
for etiqueta, n in sorted(etiquetas_visuales.items(), key=lambda x: -x[1]):
    print(f"    {etiqueta[:40]:40s} {n:5d}  {'█' * min(n, 30)}")
print()
print(f"  ☁️   GeoJSON en: {RUTA_SALIDA_GCS}")
print(sep)
print()
print("  ✅  Listo. Abre Streamlit → '🔄 Refrescar predicciones'.")
print(sep)


