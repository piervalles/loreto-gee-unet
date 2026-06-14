"""
sintetizador.py
===============
Módulo de contextualización espacial para el sistema de detección
de deforestación de Loreto.

Responsabilidad única: dado un punto geográfico, determinar la causa
probable de la deforestación consultando capas vectoriales OSM
pre-procesadas (ETL Offline) como zonas de influencia (buffers).
"""

import logging
import pathlib

import geopandas as gpd
import numpy as np
import tensorflow as tf
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Configuración de logging del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas base resueltas de forma absoluta (Apuntando a los ETL procesados)
# ---------------------------------------------------------------------------
_DIR_RAIZ = pathlib.Path(__file__).resolve().parents[2]   # …/Monitoreo_Loreto_IA/
_DIR_MAPAS: pathlib.Path = _DIR_RAIZ / "backend" / "data" / "Mapas_Loreto"

# Rutas de los archivos "gordos" generados por tu script de pre-procesamiento
_GPKG_RIOS_BUFFER: pathlib.Path   = _DIR_MAPAS / "rios_buffer_500m.gpkg"
_GPKG_VIAS_BUFFER: pathlib.Path   = _DIR_MAPAS / "vias_buffer_1000m.gpkg"
_GPKG_URBANO_BUFFER: pathlib.Path = _DIR_MAPAS / "urbano_buffer_2000m.gpkg"

# Sistema de Referencia de Coordenadas proyectado en metros (UTM zona 18S)
_CRS_BASE = "EPSG:32718"

# Etiquetas de causa de alerta (exportadas para uso en otros módulos)
CAUSA_URBANA   = "Expansión Urbana"
CAUSA_MINERA   = "Minería Aluvial (Alerta Alta)"
CAUSA_AGRICOLA = "Expansión Agrícola / Tala"
CAUSA_AISLADA  = "Tala Aislada / Pista Clandestina"


class SintetizadorEspacial:
    """
    Motor de clasificación espacial de alertas de deforestación.

    Carga capas vectoriales OSM PRE-PROCESADAS en el CRS proyectado ``EPSG:32718``.
    Las zonas de influencia (buffers y unary_union) ya fueron calculadas
    offline en un proceso ETL, por lo que la carga es instantánea.

    - **Minería aluvial** → zona de influencia de ríos (500 m).
    - **Expansión agrícola / tala** → zona de influencia de vías (1 000 m).
    - **Expansión urbana** → zona de influencia de áreas urbanas (2 000 m).

    Atributos
    ---------
    zona_minera   : gpd.GeoDataFrame  Buffer unificado de ríos.
    zona_agricola : gpd.GeoDataFrame  Buffer unificado de vías.
    zona_urbana   : gpd.GeoDataFrame  Buffer unificado de zonas urbanas.
    """

    def __init__(self) -> None:
        """
        Carga de forma instantánea las tres capas GeoPackage ya procesadas.

        Lanza
        -----
        FileNotFoundError
            Si alguno de los archivos GeoPackage pre-procesados no existe.
        """
        self._verificar_archivos()

        logger.info("Cargando capas vectoriales OSM PRE-PROCESADAS para Loreto…")

        # ----------------------------------------------------------------
        # Carga instantánea: Geometrías ya disueltas en el script ETL
        # ----------------------------------------------------------------
        self.zona_minera:   gpd.GeoDataFrame = gpd.read_file(_GPKG_RIOS_BUFFER)
        self.zona_agricola: gpd.GeoDataFrame = gpd.read_file(_GPKG_VIAS_BUFFER)
        self.zona_urbana:   gpd.GeoDataFrame = gpd.read_file(_GPKG_URBANO_BUFFER)

        logger.info("Zonas de influencia cargadas. SintetizadorEspacial inicializado ultrarrápido.")

    # ------------------------------------------------------------------
    # Métodos privados de soporte
    # ------------------------------------------------------------------

    @staticmethod
    def _verificar_archivos() -> None:
        """Valida que los tres GeoPackages pre-procesados existan en disco."""
        for ruta in (_GPKG_RIOS_BUFFER, _GPKG_VIAS_BUFFER, _GPKG_URBANO_BUFFER):
            if not ruta.exists():
                raise FileNotFoundError(
                    f"GeoPackage pre-procesado no encontrado: {ruta}\n"
                    "Asegúrate de haber ejecutado el script ETL espacial "
                    "para generar los archivos *_buffer_*.gpkg."
                )

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def clasificar_alerta(self, punto_shapely: Point) -> str:
        """
        Determina la causa probable de deforestación para un punto dado.

        La lógica de prioridad es jerárquica y mutuamente excluyente:

        1. **Expansión Urbana** → el punto intersecta la zona urbana.
        2. **Minería Aluvial** → intersecta zona minera y *no* zona agrícola.
        3. **Expansión Agrícola / Tala** → intersecta zona agrícola (y no urbana).
        4. **Tala Aislada / Pista Clandestina** → no intersecta ninguna zona.

        Parámetros
        ----------
        punto_shapely : shapely.geometry.Point
            Coordenada del píxel deforestado en el CRS ``EPSG:32718``
            (metros, UTM zona 18S).

        Retorna
        -------
        str
            Etiqueta de causa de alerta.
        """
        # Construir un GeoDataFrame de un solo punto para usar sjoin
        gdf_punto = gpd.GeoDataFrame(
            geometry=[punto_shapely], crs=_CRS_BASE
        )

        # ----------------------------------------------------------------
        # Consultas espaciales ultrarrápidas con gpd.sjoin
        # ----------------------------------------------------------------
        en_urbano   = not gpd.sjoin(
            gdf_punto, self.zona_urbana,  how="inner", predicate="intersects"
        ).empty

        en_minera   = not gpd.sjoin(
            gdf_punto, self.zona_minera,  how="inner", predicate="intersects"
        ).empty

        en_agricola = not gpd.sjoin(
            gdf_punto, self.zona_agricola, how="inner", predicate="intersects"
        ).empty

        logger.debug(
            "Clasificación espacial → urbano=%s | minera=%s | agrícola=%s",
            en_urbano, en_minera, en_agricola,
        )

        # ----------------------------------------------------------------
        # Lógica de prioridad estricta (jerárquica)
        # ----------------------------------------------------------------
        if en_urbano:
            causa = CAUSA_URBANA
        elif en_minera and not en_agricola:
            causa = CAUSA_MINERA
        elif en_agricola:
            causa = CAUSA_AGRICOLA
        else:
            causa = CAUSA_AISLADA

        logger.info("Causa clasificada para %s → '%s'", punto_shapely, causa)
        return causa

def clasificar_bloque_alertas(self, lista_puntos: list[Point]) -> list[str]:
    """
    Clasifica un lote masivo de puntos de deforestación como Urbano o No urbano.

    Parámetros
    ----------
    lista_puntos : list de shapely.geometry.Point
        Lista con las coordenadas UTM de cada píxel deforestado detectado.

    Retorna
    -------
    list de str
        Lista con "Urbano" o "No urbano" para cada punto.
    """
    if not lista_puntos:
        return []

    # 1. Crear GeoDataFrame con los puntos
    gdf_puntos = gpd.GeoDataFrame(geometry=lista_puntos, crs=_CRS_BASE)

    # 2. Inicializar columna con "No urbano" por defecto
    gdf_puntos["causa"] = "No urbano"

    # 3. Si hay zona urbana cargada, hacer un sjoin para marcar los que intersectan
    if not self.zona_urbana.empty:
        idx_urbano = gpd.sjoin(gdf_puntos, self.zona_urbana, how="inner", predicate="intersects").index
        gdf_puntos.loc[idx_urbano, "causa"] = "Urbano"

    logger.info("Clasificación urbano/no urbano completada para %d puntos.", len(lista_puntos))
    return gdf_puntos["causa"].tolist()


# ---------------------------------------------------------------------------
# Bloque de ejecución local para pruebas de integración rápida
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Configuración de logging para ejecución standalone
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    log = logging.getLogger("pipeline_integracion")

    # ----------------------------------------------------------------
    # 1. Inicializar el modelo de inferencia
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("INICIANDO PIPELINE DE INTEGRACIÓN — Detección de Deforestación Loreto")
    log.info("=" * 60)

    # Importación diferida para evitar dependencia circular si inferencia.py
    # también importa sintetizador en el futuro.
    from backend.core.inferencia import TraductorUNet  # noqa: E402

    log.info("[1/4] Cargando modelo U-Net…")
    traductor = TraductorUNet()
    log.info("[1/4] Modelo listo.")

    # ----------------------------------------------------------------
    # 2. Simular un tensor de entrada aleatorio (1, 128, 128, 10)
    # ----------------------------------------------------------------
    log.info("[2/4] Generando tensor de prueba aleatorio (1, 128, 128, 10)…")
    tensor_prueba = tf.random.uniform(
        shape=(1, 128, 128, 10), minval=0.0, maxval=1.0, dtype=tf.float32
    )
    log.info("[2/4] Tensor generado con forma: %s", tensor_prueba.shape)

    # ----------------------------------------------------------------
    # 3. Inferencia: obtener máscara binaria
    # ----------------------------------------------------------------
    log.info("[3/4] Ejecutando inferencia sobre el tensor de prueba…")
    mascara = traductor.evaluar_imagen(tensor_prueba, umbral=0.5)

    pixeles_total       = mascara.size                     # 128 × 128 = 16 384
    pixeles_deforest    = int(mascara.sum())
    porcentaje_deforest = 100.0 * pixeles_deforest / pixeles_total

    log.info(
        "[3/4] Resultados de inferencia:\n"
        "      Píxeles totales    : %d\n"
        "      Píxeles deforest.  : %d\n"
        "      Porcentaje         : %.1f%%",
        pixeles_total, pixeles_deforest, porcentaje_deforest,
    )

    # ----------------------------------------------------------------
    # 4. Clasificación espacial de un punto simulado
    # ----------------------------------------------------------------
    log.info("[4/4] Inicializando sintetizador espacial…")
    sintetizador = SintetizadorEspacial()

    punto_prueba = Point(670_000, 9_600_000)   # Coordenadas UTM 18S en Loreto
    log.info("[4/4] Clasificando punto: %s", punto_prueba)

    causa = sintetizador.clasificar_alerta(punto_prueba)

    log.info(
        "[4/4] Causa de deforestación determinada → '%s'", causa
    )

    log.info("=" * 60)
    log.info("PIPELINE COMPLETADO EXITOSAMENTE")
    log.info("=" * 60)