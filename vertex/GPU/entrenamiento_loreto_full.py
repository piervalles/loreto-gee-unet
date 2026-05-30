import os
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, optimizers, callbacks, models, mixed_precision
import logging

# =====================================================================
# 0. CONFIGURACIÓN INICIAL Y LOGGING (Vertex AI Style)
# =====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("Vertex_Training_Job")

logger.info(f"TensorFlow Version: {tf.__version__}")
logger.info("Verificando aceleradores de hardware disponibles en la máquina virtual...")

gpus = tf.config.list_physical_devices('GPU')
logger.info(f"GPUs detectadas por el contenedor: {gpus}")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # ACTIVA LA PRECISIÓN MIXTA SI HAY GPU
        mixed_precision.set_global_policy('mixed_float16')
        
        logger.info("✅ Configuración 'Memory Growth' habilitada.")
        logger.info("⚡ Precisión Mixta (float16) ACTIVADA: Explotando Tensor Cores.")
    except RuntimeError as e:
        logger.error(f"❌ Error al configurar la GPU (Continuando de todos modos): {e}")
else:
    logger.warning("⚠️ ADVERTENCIA: El trabajo fue asignado a una máquina CPU. El entrenamiento será significativamente más lento.")
    # No activamos mixed_precision en CPU porque puede ser contraproducente.

# =====================================================================
# 1. RUTAS DE INFRAESTRUCTURA (EN EL BUCKET)
# =====================================================================
# En Vertex, asegúrate de que el Service Account tenga permisos en este bucket
BUCKET_NAME = "dataset-tfrecords-loreto"
RUTA_DATOS = f"gs://{BUCKET_NAME}/datos_produccion/*.tfrecord.gz"
RUTA_MODELO_FINAL = f"gs://{BUCKET_NAME}/modelos_guardados/unet_loreto_produccion.keras"
RUTA_LOCAL_TEMP = "/tmp/unet_loreto_produccion.keras"

# =====================================================================
# 2. TUBERÍA DE DATOS MULTI-HILO (Anti-Cuellos de Botella)
# =====================================================================
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

def parsear_tfrecord(example_proto):
    bandas = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI', 'VV', 'VH']
    feature_dict = {b: tf.io.FixedLenFeature([128, 128], tf.float32) for b in bandas}
    feature_dict['LABEL'] = tf.io.FixedLenFeature([128, 128], tf.float32)

    parsed_features = tf.io.parse_single_example(example_proto, feature_dict)
    
    X = tf.stack([parsed_features[b] for b in bandas], axis=-1)
    Y = tf.expand_dims(parsed_features['LABEL'], axis=-1)
    return X, Y

logger.info("Construyendo tubería de lectura entrelazada (Interleave)...")

archivos = tf.data.Dataset.list_files(RUTA_DATOS)

dataset_entrenamiento = (archivos
           .interleave(
               lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
               cycle_length=AUTOTUNE,
               num_parallel_calls=AUTOTUNE,
               deterministic=False 
           )
           .map(parsear_tfrecord, num_parallel_calls=AUTOTUNE)
           .shuffle(2000)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

# =====================================================================
# 3. DEFINICIÓN DE BLOQUES CONVOLUCIONALES
# =====================================================================
def doble_convolucion(x, filtros: int):
    """Bloque estándar de la U-Net: Dos convoluciones seguidas de Batch Normalization y ReLU."""
    x = layers.Conv2D(filtros, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filtros, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

# =====================================================================
# 4. ARQUITECTURA DE LA U-NET (Funcional API)
# =====================================================================
def construir_unet(input_shape=(128, 128, 10)):
    inputs = layers.Input(shape=input_shape)

    # --- ENCODER ---
    c1 = doble_convolucion(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1) 

    c2 = doble_convolucion(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2) 

    c3 = doble_convolucion(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3) 

    # --- CUELLO DE BOTELLA ---
    c4 = doble_convolucion(p3, 256)

    # --- DECODER ---
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = layers.concatenate([u5, c3]) 
    c5 = doble_convolucion(u5, 128)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = doble_convolucion(u6, 64)

    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = doble_convolucion(u7, 32)

    # --- CAPA DE SALIDA ---
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype=tf.float32)(c7)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="UNet_Loreto_Senior")
    return model

# =====================================================================
# 5. COMPILACIÓN Y ESTRATEGIA DE ENTRENAMIENTO
# =====================================================================
logger.info("Compilando modelo U-Net...")

if gpus:
    with tf.device('/GPU:0'):
        modelo_unet = construir_unet()
else:
    modelo_unet = construir_unet()

modelo_unet.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss=losses.BinaryCrossentropy(),
    metrics=[
        metrics.BinaryAccuracy(name="accuracy"),
        metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="IoU")
    ]
)

modelo_unet.summary(print_fn=logger.info)

# =====================================================================
# 6. ORQUESTACIÓN Y GUARDIANES (CALLBACKS)
# =====================================================================
logger.info("Configurando guardianes de entrenamiento (Callbacks)...")
early_stopping = callbacks.EarlyStopping(
    monitor='loss', 
    patience=10, 
    restore_best_weights=True
)

# =====================================================================
# 7. DISPARAR EL ENTRENAMIENTO (FUEGO)
# =====================================================================
logger.info("=== INICIANDO ENTRENAMIENTO MASIVO ===")
historial = modelo_unet.fit(
    dataset_entrenamiento,
    epochs=100,
    callbacks=[early_stopping],
    verbose=2 # Vertex prefiere verbose=2 (una línea por época) para no saturar los logs
)
logger.info("=== ENTRENAMIENTO FINALIZADO CON ÉXITO ===")

# =====================================================================
# 8. GUARDADO SEGURO (DISCO LOCAL -> CLOUD STORAGE)
# =====================================================================
logger.info("Iniciando transferencia segura al Bucket...")

modelo_unet.save(RUTA_LOCAL_TEMP)
logger.info("Paso 1: Modelo guardado temporalmente en /tmp/ exitoso.")

tf.io.gfile.copy(RUTA_LOCAL_TEMP, RUTA_MODELO_FINAL, overwrite=True)
logger.info(f"🚀 Paso 2: Modelo guardado de forma permanente y segura en: {RUTA_MODELO_FINAL}")