import os
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, optimizers, callbacks, models, mixed_precision

print(f"TensorFlow Version: {tf.__version__}")

# =====================================================================
# 0. CONFIGURACIÓN EXTREMA Y AUTO-BATCHING DE GPU
# =====================================================================
gpus = tf.config.list_physical_devices('GPU')
BATCH_SIZE = 32  # Default conservador
CYCLE_LENGTH = 4 # Paralelismo de lectura por defecto

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        mixed_precision.set_global_policy('mixed_float16')
        
        # Heurística dinámica de hardware para escalar memoria y tuberías
        gpu_name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', '')
        if 'A100' in gpu_name or 'H100' in gpu_name or 'A10g' in gpu_name:
            BATCH_SIZE = 128
            CYCLE_LENGTH = 10
            print(f"🚀 GPU Premium detectada ({gpu_name}). Batch Size: {BATCH_SIZE} | Cycle Length: {CYCLE_LENGTH}")
        else:
            BATCH_SIZE = 64
            CYCLE_LENGTH = 4
            print(f"✅ GPU Estándar ({gpu_name}). Batch Size: {BATCH_SIZE} | Cycle Length: {CYCLE_LENGTH}")

    except RuntimeError as e:
        print(f"❌ Error al configurar la GPU: {e}")
else:
    print("⚠️ ADVERTENCIA: Entrenando en CPU. Esto será sumamente lento.")

# =====================================================================
# 1. RUTAS DE INFRAESTRUCTURA (CLOUD STORAGE)
# =====================================================================
BUCKET_NAME = "dataset-tfrecords-loreto"
RUTA_DATOS = f"gs://{BUCKET_NAME}/datos_entrenamiento_3/*.tfrecord.gz"
RUTA_MODELO_FINAL = f"gs://{BUCKET_NAME}/modelos_guardados/unet_loreto_4.keras"
RUTA_LOCAL_TEMP = "/tmp/unet_loreto_4.keras"

# =====================================================================
# 2. TUBERÍA DE DATOS (CON PARALELISMO AVANZADO)
# =====================================================================
AUTOTUNE = tf.data.AUTOTUNE

def parsear_tfrecord(example_proto):
    bandas = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI', 'VV', 'VH']
    feature_dict = {b: tf.io.FixedLenFeature([128, 128], tf.float32) for b in bandas}
    feature_dict['LABEL'] = tf.io.FixedLenFeature([128, 128], tf.float32)

    parsed_features = tf.io.parse_single_example(example_proto, feature_dict)
    X = tf.stack([parsed_features[b] for b in bandas], axis=-1)
    Y = tf.expand_dims(parsed_features['LABEL'], axis=-1)
    return X, Y

def augment(X, Y):
    if tf.random.uniform(shape=[]) > 0.5:
        X = tf.image.flip_left_right(X)
        Y = tf.image.flip_left_right(Y)
    if tf.random.uniform(shape=[]) > 0.5:
        X = tf.image.flip_up_down(X)
        Y = tf.image.flip_up_down(Y)
    
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    X = tf.image.rot90(X, k)
    Y = tf.image.rot90(Y, k)
    return X, Y

archivos_lista = tf.io.gfile.glob(RUTA_DATOS)
total_archivos = len(archivos_lista)

if total_archivos == 0:
    print(f"❌ ERROR CRÍTICO: No se encontraron archivos en {RUTA_DATOS}.")
    import sys; sys.exit(1)

archivos = tf.data.Dataset.list_files(RUTA_DATOS, shuffle=True)
num_val = max(1, int(total_archivos * 0.2)) 

archivos_val = archivos.take(num_val)
archivos_train = archivos.skip(num_val)

# --- Cálculo ROBUSTO dinámico de steps_per_epoch ---
print("\nCalculando volumen exacto del dataset para orquestación de épocas...")
archivos_muestra = archivos_lista[:min(5, total_archivos)] # Promediamos hasta 5 archivos
total_parches_muestra = 0

for archivo in archivos_muestra:
    dataset_temporal = tf.data.TFRecordDataset(archivo, compression_type='GZIP')
    total_parches_muestra += sum(1 for _ in dataset_temporal)

num_parches_promedio = total_parches_muestra // len(archivos_muestra)
archivos_entrenamiento_reales = total_archivos - num_val

steps_per_epoch = (archivos_entrenamiento_reales * num_parches_promedio) // BATCH_SIZE
print(f"📊 Parches promedio por archivo: {num_parches_promedio} | Pasos por época (Steps per Epoch): {steps_per_epoch}")

# =====================================================================
# Tubería base SIN repeat (para sanity check)
# =====================================================================
dataset_entrenamiento_base = (archivos_train
    .interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
        cycle_length=CYCLE_LENGTH, 
        block_length=1, 
        num_parallel_calls=AUTOTUNE
    )
    .map(parsear_tfrecord, num_parallel_calls=AUTOTUNE)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .shuffle(2000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

# --- SANITY CHECK (verificación defensiva) ---
print("\n=== SANITY CHECK: Verificando un lote del dataset ===")
for X_batch, Y_batch in dataset_entrenamiento_base.take(1):
    print(f"✅ X shape: {X_batch.shape}, dtype: {X_batch.dtype}")
    print(f"✅ Y shape: {Y_batch.shape}, dtype: {Y_batch.dtype}")
    assert X_batch.shape == (BATCH_SIZE, 128, 128, 10), "Error: Dimensiones de X incorrectas"
    assert Y_batch.shape == (BATCH_SIZE, 128, 128, 1), "Error: Dimensiones de Y incorrectas"
    assert X_batch.dtype == tf.float32, "X debe ser float32"
    assert Y_batch.dtype == tf.float32, "Y debe ser float32"
print("✅ Sanity check superado. El pipeline produce datos correctos.\n")

# --- Tubería de entrenamiento definitiva (con repeat) ---
dataset_entrenamiento = dataset_entrenamiento_base.repeat()

# Tubería de Validación (sin repeat, sin sanity check)
dataset_validacion = (archivos_val
    .interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
        cycle_length=CYCLE_LENGTH, 
        block_length=1, 
        num_parallel_calls=AUTOTUNE
    )
    .map(parsear_tfrecord, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

# =====================================================================
# Extraer muestra pura (sin aumentos) para adaptar la Normalización
# =====================================================================
print("\nCalculando estadísticas de normalización del dataset (Raw Data)...")

muestra_normalizacion = (archivos_train
    .interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
        cycle_length=CYCLE_LENGTH, 
        block_length=1, 
        num_parallel_calls=AUTOTUNE
    )
    .map(parsear_tfrecord, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .take(50)
    .map(lambda x, y: tf.cast(x, tf.float32)))   

capa_normalizacion = layers.Normalization(axis=-1, dtype=tf.float32)   
capa_normalizacion.adapt(muestra_normalizacion)
print("✅ Normalización dinámica adaptada exitosamente (Modo Ultra-Robusto).")

# =====================================================================
# 3. ARQUITECTURA U-NET (Con Normalización Integrada)
# =====================================================================
def doble_convolucion(x, filtros: int):
    x = layers.Conv2D(filtros, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filtros, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def construir_unet(input_shape=(128, 128, 10)):
    inputs = layers.Input(shape=input_shape)
    norm_inputs = capa_normalizacion(inputs)

    # Encoder
    c1 = doble_convolucion(norm_inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = doble_convolucion(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = doble_convolucion(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = doble_convolucion(p3, 256)

    # Decoder
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = doble_convolucion(u5, 128)
    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = doble_convolucion(u6, 64)
    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = doble_convolucion(u7, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype=tf.float32)(c7)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="UNet_Loreto_V4_Definitiva")

modelo_unet = construir_unet()

modelo_unet.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss=losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.75), 
    metrics=[
        metrics.BinaryAccuracy(name="accuracy"),
        metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="IoU")
    ]
)

modelo_unet.summary()

# =====================================================================
# 4. ORQUESTACIÓN Y DISPARO
# =====================================================================
print("\nConfigurando guardianes de entrenamiento...")
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-4, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(filepath="/tmp/mejor_modelo.keras", monitor='val_loss', save_best_only=True, verbose=1)
]

print("\n=== INICIANDO ENTRENAMIENTO ACELERADO (V4.1) ===")
historial = modelo_unet.fit(
    dataset_entrenamiento,
    validation_data=dataset_validacion,
    epochs=100,
    steps_per_epoch=steps_per_epoch, 
    callbacks=callbacks_list
)

# =====================================================================
# 5. GUARDADO DEFINITIVO EN NUBE
# =====================================================================
print("\nTransfiriendo modelo a Cloud Storage...")
if os.path.exists("/tmp/mejor_modelo.keras"):
    tf.io.gfile.copy("/tmp/mejor_modelo.keras", RUTA_MODELO_FINAL, overwrite=True)
    print("✅ Checkpoint de validación guardado con éxito.")
else:
    print("⚠️ No se encontró checkpoint de validación. Guardando estado actual...")
    modelo_unet.save(RUTA_LOCAL_TEMP)
    tf.io.gfile.copy(RUTA_LOCAL_TEMP, RUTA_MODELO_FINAL, overwrite=True)

print(f"🚀 Misión Cumplida: Modelo V4 guardado en {RUTA_MODELO_FINAL}")