"""
sintetizador.py
===============
Módulo de contextualización espacial para el sistema de detección
de deforestación de Loreto.

Responsabilidad única: dado un punto geográfico, determinar la causa
probable de la deforestación consultando capas vectoriales OSM
pre-procesadas como zonas de influencia (buffers).
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
# Rutas base resueltas de forma absoluta
# ---------------------------------------------------------------------------
_DIR_RAIZ = pathlib.Path(__file__).resolve().parents[2]   # …/Monitoreo_Loreto_IA/
_DIR_MAPAS: pathlib.Path = _DIR_RAIZ / "backend" / "data" / "Mapas_Loreto"

_GPKG_RIOS: pathlib.Path    = _DIR_MAPAS / "rios_osm.gpkg"
_GPKG_VIAS: pathlib.Path    = _DIR_MAPAS / "vias_osm.gpkg"
_GPKG_URBANO: pathlib.Path  = _DIR_MAPAS / "urbano_osm.gpkg"

# Sistema de Referencia de Coordenadas proyectado en metros (UTM zona 18S)
_CRS_BASE = "EPSG:32718"

# Distancias de influencia por capa [metros]
_BUFFER_RIOS_M:   int = 500
_BUFFER_VIAS_M:   int = 1_000
_BUFFER_URBANO_M: int = 2_000

# Etiquetas de causa de alerta (exportadas para uso en otros módulos)
CAUSA_URBANA  = "Expansión Urbana"
CAUSA_MINERA  = "Minería Aluvial (Alerta Alta)"
CAUSA_AGRICOLA = "Expansión Agrícola / Tala"
CAUSA_AISLADA  = "Tala Aislada / Pista Clandestina"


class SintetizadorEspacial:
    """
    Motor de clasificación espacial de alertas de deforestación.

    Carga capas vectoriales OSM en el CRS proyectado ``EPSG:32718`` y
    pre-calcula zonas de influencia (buffers) para las tres categorías
    de presión antrópica identificadas en Loreto:

    - **Minería aluvial** → zona de influencia de ríos (500 m).
    - **Expansión agrícola / tala** → zona de influencia de vías (1 000 m).
    - **Expansión urbana** → zona de influencia de áreas urbanas (2 000 m).

    El pre-cálculo en ``__init__`` garantiza que las consultas en
    ``clasificar_alerta`` sean O(1) respecto al tiempo de carga de datos.

    Atributos
    ---------
    zona_minera   : gpd.GeoDataFrame  Buffer de 500 m sobre ríos.
    zona_agricola : gpd.GeoDataFrame  Buffer de 1 000 m sobre vías.
    zona_urbana   : gpd.GeoDataFrame  Buffer de 2 000 m sobre urbano.
    """

    def __init__(self) -> None:
        """
        Carga las tres capas GeoPackage y pre-calcula los buffers.

        Lanza
        -----
        FileNotFoundError
            Si alguno de los archivos GeoPackage no existe.
        """
        self._verificar_archivos()

        logger.info("Cargando capas vectoriales OSM para Loreto…")

        # ----------------------------------------------------------------
        # Carga y reproyección al CRS base (metros)
        # HACK DE PROTOTIPO: limitamos a 50 geometrías para evitar asfixia de RAM
        # ----------------------------------------------------------------
        rios   = gpd.read_file(_GPKG_RIOS).to_crs(_CRS_BASE).head(50)
        vias   = gpd.read_file(_GPKG_VIAS).to_crs(_CRS_BASE).head(50)
        urbano = gpd.read_file(_GPKG_URBANO).to_crs(_CRS_BASE).head(50)

        logger.info(
            "Capas cargadas → Ríos: %d geometrías | Vías: %d geometrías | "
            "Urbano: %d geometrías",
            len(rios), len(vias), len(urbano),
        )

        # ----------------------------------------------------------------
        # Pre-cálculo de zonas de influencia (buffers)
        # ----------------------------------------------------------------
        logger.info("Pre-calculando zonas de influencia (buffers)…")

        self.zona_minera:   gpd.GeoDataFrame = self._calcular_buffer(
            rios, _BUFFER_RIOS_M, "zona_minera"
        )
        self.zona_agricola: gpd.GeoDataFrame = self._calcular_buffer(
            vias, _BUFFER_VIAS_M, "zona_agricola"
        )
        self.zona_urbana:   gpd.GeoDataFrame = self._calcular_buffer(
            urbano, _BUFFER_URBANO_M, "zona_urbana"
        )

        logger.info("Zonas de influencia listas. SintetizadorEspacial inicializado.")

    # ------------------------------------------------------------------
    # Métodos privados de soporte
    # ------------------------------------------------------------------

    @staticmethod
    def _verificar_archivos() -> None:
        """Valida que los tres GeoPackages existan en disco."""
        for ruta in (_GPKG_RIOS, _GPKG_VIAS, _GPKG_URBANO):
            if not ruta.exists():
                raise FileNotFoundError(
                    f"GeoPackage no encontrado: {ruta}\n"
                    "Verifica que los archivos OSM estén en "
                    "'backend/data/Mapas_Loreto/'."
                )

    @staticmethod
    def _calcular_buffer(
        gdf: gpd.GeoDataFrame,
        distancia_m: int,
        nombre_zona: str,
    ) -> gpd.GeoDataFrame:
        """
        Disuelve la capa de entrada y calcula un buffer en metros.

        Usar ``unary_union`` + dissolve antes del buffer reduce drásticamente
        el número de polígonos resultantes y acelera las consultas sjoin.

        Parámetros
        ----------
        gdf          : GeoDataFrame de entrada (ya en CRS métrico).
        distancia_m  : Radio del buffer en metros.
        nombre_zona  : Nombre descriptivo para logging.

        Retorna
        -------
        gpd.GeoDataFrame
            GeoDataFrame con una sola geometría: el buffer disuelto.
        """
        geometria_unida = gdf.geometry.unary_union
        buffer_geom     = geometria_unida.buffer(distancia_m)
        resultado = gpd.GeoDataFrame(
            geometry=[buffer_geom], crs=_CRS_BASE
        )
        logger.debug(
            "Buffer '%s' calculado: %d m, tipo geométrico resultante: %s",
            nombre_zona, distancia_m, buffer_geom.geom_type,
        )
        return resultado

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
        # Consultas espaciales con gpd.sjoin
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
