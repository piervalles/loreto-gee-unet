import logging
import pathlib
import geopandas as gpd
import numpy as np
import tensorflow as tf
from shapely.geometry import Point

logger = logging.getLogger(__name__)

_DIR_RAIZ = pathlib.Path(__file__).resolve().parents[2]
_DIR_MAPAS = _DIR_RAIZ / "backend" / "data" / "Mapas_Loreto"

_GPKG_RIOS_BUFFER = _DIR_MAPAS / "rios_buffer_500m.gpkg"
_GPKG_VIAS_BUFFER = _DIR_MAPAS / "vias_buffer_1000m.gpkg"
_GPKG_URBANO_BUFFER = _DIR_MAPAS / "urbano_buffer_2000m.gpkg"

_CRS_BASE = "EPSG:32718"

CAUSA_URBANA = "Expansión Urbana"
CAUSA_MINERA = "Minería Aluvial (Alerta Alta)"
CAUSA_AGRICOLA = "Expansión Agrícola / Tala"
CAUSA_AISLADA = "Tala Aislada / Pista Clandestina"

class SintetizadorEspacial:
    def __init__(self) -> None:
        self._verificar_archivos()
        self.zona_minera = gpd.read_file(_GPKG_RIOS_BUFFER)
        self.zona_agricola = gpd.read_file(_GPKG_VIAS_BUFFER)
        self.zona_urbana = gpd.read_file(_GPKG_URBANO_BUFFER)

    @staticmethod
    def _verificar_archivos() -> None:
        for ruta in (_GPKG_RIOS_BUFFER, _GPKG_VIAS_BUFFER, _GPKG_URBANO_BUFFER):
            if not ruta.exists():
                raise FileNotFoundError(f"GeoPackage no encontrado: {ruta}")

    def clasificar_alerta(self, punto_shapely: Point) -> str:
        gdf_punto = gpd.GeoDataFrame(geometry=[punto_shapely], crs=_CRS_BASE)
        en_urbano = not gpd.sjoin(gdf_punto, self.zona_urbana, how="inner", predicate="intersects").empty
        en_minera = not gpd.sjoin(gdf_punto, self.zona_minera, how="inner", predicate="intersects").empty
        en_agricola = not gpd.sjoin(gdf_punto, self.zona_agricola, how="inner", predicate="intersects").empty

        if en_urbano: return CAUSA_URBANA
        elif en_minera and not en_agricola: return CAUSA_MINERA
        elif en_agricola: return CAUSA_AGRICOLA
        return CAUSA_AISLADA

    def clasificar_bloque_alertas(self, lista_puntos: list[Point]) -> list[str]:
        if not lista_puntos:
            return []

        gdf_puntos = gpd.GeoDataFrame(geometry=lista_puntos, crs=_CRS_BASE)
        gdf_puntos["urbano"] = False
        gdf_puntos["minera"] = False
        gdf_puntos["agricola"] = False

        if not self.zona_urbana.empty:
            idx_urbano = gpd.sjoin(gdf_puntos, self.zona_urbana, how="inner", predicate="intersects").index
            gdf_puntos.loc[idx_urbano, "urbano"] = True

        if not self.zona_minera.empty:
            idx_minera = gpd.sjoin(gdf_puntos, self.zona_minera, how="inner", predicate="intersects").index
            gdf_puntos.loc[idx_minera, "minera"] = True

        if not self.zona_agricola.empty:
            idx_agricola = gpd.sjoin(gdf_puntos, self.zona_agricola, how="inner", predicate="intersects").index
            gdf_puntos.loc[idx_agricola, "agricola"] = True

        condiciones = [
            gdf_puntos["urbano"],
            gdf_puntos["minera"] & ~gdf_puntos["agricola"],
            gdf_puntos["agricola"]
        ]
        opciones = [CAUSA_URBANA, CAUSA_MINERA, CAUSA_AGRICOLA]
        
        gdf_puntos["causa"] = np.select(condiciones, opciones, default=CAUSA_AISLADA)
        return gdf_puntos["causa"].tolist()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    log = logging.getLogger("pipeline")

    from backend.core.inferencia import TraductorUNet

    log.info("Cargando modelo y ejecutando inferencia de prueba...")
    traductor = TraductorUNet()
    tensor_prueba = tf.random.uniform(shape=(1, 128, 128, 10), minval=0.0, maxval=1.0, dtype=tf.float32)
    mascara = traductor.evaluar_imagen(tensor_prueba, umbral=0.5)

    log.info("Inicializando sintetizador espacial...")
    sintetizador = SintetizadorEspacial()
    
    filas, columnas = np.where(mascara == 1)
    
    # Simulación de georreferenciación de la máscara completa
    puntos_deforestados = [Point(670_000 + (c * 30), 9_600_000 - (f * 30)) for f, c in zip(filas, columnas)]

    if puntos_deforestados:
        causas = sintetizador.clasificar_bloque_alertas(puntos_deforestados)
        log.info(f"Se detectaron y clasificaron masivamente {len(causas)} píxeles de deforestación.")
    else:
        log.info("No se detectó deforestación en el tensor de prueba.")