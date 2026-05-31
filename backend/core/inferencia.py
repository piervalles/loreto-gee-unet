"""
inferencia.py
=============
Módulo de inferencia para el sistema de detección de deforestación de Loreto.

Responsabilidad única: envolver el modelo U-Net entrenado y exponerlo
mediante una interfaz limpia para el resto del backend.
"""

import logging
import pathlib
from typing import Optional

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuración de logging del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ruta base resuelta de forma absoluta relativa a este archivo,
# para que el módulo funcione independientemente del directorio de trabajo.
# ---------------------------------------------------------------------------
_DIR_RAIZ = pathlib.Path(__file__).resolve().parents[2]          # …/Monitoreo_Loreto_IA/
_RUTA_MODELO: pathlib.Path = _DIR_RAIZ / "backend" / "data" / "Models" / "unet_loreto_1.keras"


class TraductorUNet:
    """
    Adaptador de alto nivel sobre el modelo U-Net entrenado para Loreto.

    Responsabilidades
    -----------------
    - Cargar el modelo desde disco una única vez (patrón *singleton de sesión*).
    - Pre-compilar el grafo de inferencia con un warm-up silencioso.
    - Proveer un método público de evaluación que valida entradas y retorna
      máscaras binarias listas para consumo por capas superiores.

    Atributos
    ---------
    modelo : tf.keras.Model
        Modelo cargado en memoria, listo para inferencia.
    """

    def __init__(self, ruta_modelo: Optional[pathlib.Path] = None) -> None:
        """
        Carga el modelo U-Net y ejecuta el warm-up del grafo.

        Parámetros
        ----------
        ruta_modelo : pathlib.Path, opcional
            Ruta al archivo `.keras`. Si se omite, se usa la ruta canónica
            del proyecto (``backend/data/Models/unet_loreto_1.keras``).

        Lanza
        -----
        FileNotFoundError
            Si el archivo del modelo no existe en la ruta indicada.
        """
        ruta_efectiva = pathlib.Path(ruta_modelo) if ruta_modelo else _RUTA_MODELO

        if not ruta_efectiva.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {ruta_efectiva}\n"
                "Verifica que el archivo 'unet_loreto_1.keras' esté en "
                "'backend/data/Models/'."
            )

        logger.info("Cargando modelo U-Net desde: %s", ruta_efectiva)
        # compile=False evita recompilar el optimizador guardado; solo necesitamos
        # el grafo de inferencia hacia adelante.
        self.modelo: tf.keras.Model = tf.keras.models.load_model(
            str(ruta_efectiva), compile=False
        )
        logger.info(
            "Modelo cargado. Entradas esperadas: %s | Salidas esperadas: %s",
            self.modelo.input_shape,
            self.modelo.output_shape,
        )

        self._warm_up()

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _warm_up(self) -> None:
        """
        Pre-compila el grafo XLA/TF ejecutando una pasada dummy.

        Pasar un tensor de ceros por el modelo en el momento de la carga
        evita la latencia de compilación JIT en la primera inferencia real.
        El resultado se descarta.
        """
        logger.info("Ejecutando warm-up del grafo de inferencia…")
        tensor_dummy = tf.zeros(shape=(1, 128, 128, 10), dtype=tf.float32)
        # Invocación directa (NO .predict()) para activar la compilación del grafo.
        _ = self.modelo(tensor_dummy, training=False)
        logger.info("Warm-up completado. El modelo está listo para inferencia.")

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def evaluar_imagen(
        self,
        tensor_imagen: tf.Tensor,
        umbral: float = 0.5,
    ) -> np.ndarray:
        """
        Evalúa un tensor multibanda y retorna una máscara binaria de deforestación.

        Parámetros
        ----------
        tensor_imagen : tf.Tensor
            Tensor de entrada con forma ``(Lote, Alto, Ancho, 10)``.
            El modelo espera 10 bandas espectrales (p. ej. Sentinel-2 + máscaras).
        umbral : float, opcional
            Umbral de probabilidad para binarizar la salida del modelo.
            Por defecto 0.5.

        Retorna
        -------
        np.ndarray
            Máscara binaria 2-D con forma ``(128, 128)`` donde:
            - ``1`` indica píxel clasificado como **deforestado**.
            - ``0`` indica píxel clasificado como **no deforestado**.

        Lanza
        -----
        ValueError
            Si el tensor de entrada no tiene exactamente 4 dimensiones o
            si la última dimensión no es 10 (bandas espectrales).
        """
        # ----------------------------------------------------------------
        # Validación estricta de la forma del tensor
        # ----------------------------------------------------------------
        forma = tensor_imagen.shape
        if len(forma) != 4:
            raise ValueError(
                f"Se esperaba un tensor de 4 dimensiones (Lote, Alto, Ancho, 10), "
                f"pero se recibió uno de {len(forma)} dimensiones con forma {forma}."
            )
        if forma[-1] != 10:
            raise ValueError(
                f"La última dimensión del tensor debe ser 10 (bandas espectrales), "
                f"pero se recibió {forma[-1]}. Forma completa: {forma}."
            )

        logger.debug(
            "Evaluando tensor con forma %s usando umbral=%.2f", forma, umbral
        )

        # ----------------------------------------------------------------
        # Inferencia: invocación directa del modelo (no .predict())
        # ----------------------------------------------------------------
        probabilidades: tf.Tensor = self.modelo(tensor_imagen, training=False)

        # ----------------------------------------------------------------
        # Binarización y aplanamiento a (128, 128)
        # ----------------------------------------------------------------
        # probabilidades tiene forma (Lote, 128, 128, 1) → squeeze a (128, 128)
        mascara_prob = tf.squeeze(probabilidades).numpy()          # (128, 128)
        mascara_binaria: np.ndarray = (mascara_prob >= umbral).astype(np.uint8)

        pixeles_deforestados = int(mascara_binaria.sum())
        logger.debug(
            "Píxeles deforestados detectados: %d / %d (%.1f%%)",
            pixeles_deforestados,
            mascara_binaria.size,
            100.0 * pixeles_deforestados / mascara_binaria.size,
        )

        return mascara_binaria
