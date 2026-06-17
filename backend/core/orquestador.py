"""
orquestador.py
==============
Punto de integración entre la U-Net y el Sintetizador Espacial.
Expone funciones de alto nivel para el frontend.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from shapely.geometry import Point

from backend.core.inferencia import TraductorUNet
from backend.core.sintetizador import SintetizadorEspacial

logger = logging.getLogger(__name__)

# Inicialización perezosa (singleton) de los componentes pesados
_traductor = None
_sintetizador = None

def obtener_traductor() -> TraductorUNet:
    global _traductor
    if _traductor is None:
        _traductor = TraductorUNet()
    return _traductor

def obtener_sintetizador() -> SintetizadorEspacial:
    global _sintetizador
    if _sintetizador is None:
        _sintetizador = SintetizadorEspacial()
    return _sintetizador

def procesar_parche(
    tensor_imagen: tf.Tensor,
    origen_x: float,
    origen_y: float,
    resolucion: float = 10.0,
    umbral: float = 0.5
) -> List[Tuple[Point, str]]:
    """
    Procesa un parche de 128x128x10:
      1. Infiere máscara de deforestación con U-Net.
      2. Convierte píxeles a puntos geográficos.
      3. Clasifica cada punto con el sintetizador.
    Retorna lista de tuplas (punto, causa).
    """
    traductor = obtener_traductor()
    sintetizador = obtener_sintetizador()

    # 1. Inferencia
    mascara = traductor.evaluar_imagen(tensor_imagen, umbral=umbral)

    # 2. Obtener índices de píxeles deforestados
    filas, columnas = np.where(mascara == 1)
    if len(filas) == 0:
        return []  # sin deforestación en este parche

    # 3. Convertir a puntos
    puntos = sintetizador.pixeles_a_coordenadas(filas, columnas, origen_x, origen_y, resolucion)

    # 4. Clasificar
    causas = sintetizador.clasificar_bloque(puntos)

    # 5. Empaquetar resultados
    return list(zip(puntos, causas))