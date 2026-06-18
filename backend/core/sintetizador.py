"""
Módulo SintetizadorEspacial
Clasifica puntos de deforestación como Urbanos o No Urbanos.
Diseñado para ser simple, robusto y fácilmente extensible.
"""
import logging
import os
import pathlib
from typing import List, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACIÓN DE RUTAS (Flexible y configurable)
# =============================================================================
# Usa variable de entorno o ruta relativa como fallback
_RUTA_BASE = os.getenv(
    "MAPAS_LORETO_PATH",
    pathlib.Path(__file__).resolve().parents[2] / "backend" / "data" / "Mapas_Loreto"
)

_GPKG_URBANO = _RUTA_BASE / "ETL" / "01" / "ETL_01_urbano_buffer_2000m.gpkg"
# _GPKG_RIOS  = _RUTA_BASE / "rios_buffer_500m.gpkg"    # Reservado para futura extensión
# _GPKG_VIAS  = _RUTA_BASE / "vias_buffer_1000m.gpkg"   # Reservado para futura extensión

_CRS_BASE = "EPSG:32718"

# =============================================================================
# DEFINICIÓN DE CAUSAS (Jerarquía: urbano tiene prioridad)
# =============================================================================
CAUSA_URBANO = "Urbano"
CAUSA_NO_URBANO = "No urbano"

# Lista de causas en orden de prioridad (solo urbano por ahora)
# Cuando añadas más causas, solo agrega aquí y descomenta las líneas correspondientes.
CAUSAS_POR_PRIORIDAD = [
    (CAUSA_URBANO, _GPKG_URBANO),
    # (CAUSA_MINERIA, _GPKG_RIOS),   # Añadir cuando esté listo
    # (CAUSA_VIAL, _GPKG_VIAS),      # Añadir cuando esté listo
]

# =============================================================================
# CLASE PRINCIPAL
# =============================================================================
class SintetizadorEspacial:
    """
    Clasifica puntos de deforestación según su ubicación relativa a zonas urbanas.
    Actualmente solo soporta dos categorías, pero está preparado para añadir más.
    """

    def __init__(self) -> None:
        self._buffers: dict = {}
        self._verificar_archivos()
        self._cargar_buffers()
        logger.info("SintetizadorEspacial inicializado correctamente.")

    def _verificar_archivos(self) -> None:
        """Verifica que existan los archivos GPKG necesarios."""
        for causa, ruta in CAUSAS_POR_PRIORIDAD:
            if not ruta.exists():
                raise FileNotFoundError(
                    f"No se encuentra el archivo para '{causa}': {ruta}. "
                    "Asegúrate de descargar los buffers desde GCS."
                )

    def _cargar_buffers(self) -> None:
        """Carga los GeoDataFrames de los buffers en memoria."""
        for causa, ruta in CAUSAS_POR_PRIORIDAD:
            try:
                gdf = gpd.read_file(ruta)
                if gdf.empty:
                    logger.warning(f"El buffer '{causa}' está vacío. Se omitirá en la clasificación.")
                    self._buffers[causa] = gpd.GeoDataFrame()  # vacío seguro
                else:
                    self._buffers[causa] = gdf
            except Exception as e:
                raise RuntimeError(f"Error al leer {ruta}: {e}")

    def clasificar_alerta(self, punto: Point) -> str:
        """
        Clasifica un único punto.
        """
        if not isinstance(punto, Point):
            raise TypeError("punto debe ser un objeto shapely.geometry.Point")
        gdf_punto = gpd.GeoDataFrame(geometry=[punto], crs=_CRS_BASE)
        return self._clasificar_gdf(gdf_punto)[0]

    def clasificar_bloque(self, puntos: List[Point]) -> List[str]:
        """
        Clasifica múltiples puntos de forma vectorizada.
        """
        if not puntos:
            return []
        gdf_puntos = gpd.GeoDataFrame(geometry=puntos, crs=_CRS_BASE)
        return self._clasificar_gdf(gdf_puntos)

    # =========================================================================
    # CORREGIDO: Método interno seguro con índices de Pandas
    # =========================================================================
    def _clasificar_gdf(self, gdf: gpd.GeoDataFrame) -> List[str]:
        """
        Lógica interna de clasificación (aplica jerarquía).
        """
        # Trabajamos sobre una copia para no alterar el DataFrame original
        df_trabajo = gdf.copy()
        df_trabajo["causa"] = CAUSA_NO_URBANO

        # Iterar sobre las causas en orden de prioridad (urbano primero)
        for causa, buffer_gdf in self._buffers.items():
            if buffer_gdf.empty:
                continue  # buffer vacío, saltar

            # Encontrar cuáles siguen siendo "No Urbano"
            mask_no_clasificados = df_trabajo["causa"] == CAUSA_NO_URBANO
            if not mask_no_clasificados.any():
                break  # todos clasificados

            gdf_restantes = df_trabajo[mask_no_clasificados]

            # sjoin con how='inner' devuelve solo los que intersectan
            joined = gpd.sjoin(gdf_restantes, buffer_gdf, how='inner', predicate='intersects')

            if not joined.empty:
                # Asignación segura por índice nativo de Pandas
                df_trabajo.loc[joined.index, "causa"] = causa

        return df_trabajo["causa"].tolist()

    # =========================================================================
    # MÉTODO ESTÁTICO DE UTILIDAD: conversión de píxeles a coordenadas
    # =========================================================================
    @staticmethod
    def pixeles_a_coordenadas(
        filas: np.ndarray,
        columnas: np.ndarray,
        origen_x: float,
        origen_y: float,
        resolucion: float = 10.0
    ) -> List[Point]:
        """
        Convierte índices de fila/columna de una máscara a puntos geográficos (UTM).
        - filas, columnas: arrays 1D con índices de píxeles con deforestación.
        - origen_x, origen_y: coordenadas UTM de la esquina superior izquierda del parche.
        - resolucion: tamaño del píxel en metros (por defecto 10 m para Sentinel-2).
        Retorna lista de objetos Point en CRS EPSG:32718.
        """
        if len(filas) == 0:
            return []
        xs = origen_x + columnas * resolucion
        ys = origen_y - filas * resolucion  # coordenada Y decrece hacia abajo
        return [Point(x, y) for x, y in zip(xs, ys)]


# =============================================================================
# EJEMPLO DE USO (solo si se ejecuta directamente, no al importar)
# =============================================================================
if __name__ == "__main__":
    # Este bloque solo se ejecuta para pruebas rápidas (no afecta la importación)
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    log = logging.getLogger("test_sintetizador")

    # Ejemplo de uso con puntos sintéticos (coordenadas UTM ficticias)
    puntos_ejemplo = [
        Point(680000, 9500000),
        Point(700000, 9480000),
        Point(720000, 9460000),
    ]

    sintetizador = SintetizadorEspacial()
    resultados = sintetizador.clasificar_bloque(puntos_ejemplo)
    for p, c in zip(puntos_ejemplo, resultados):
        log.info(f"Punto {p.wkt[:30]}... → {c}")