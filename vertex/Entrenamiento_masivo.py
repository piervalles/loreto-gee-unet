import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

print(f"TensorFlow Version: {tf.__version__}")
print("Verificando aceleradores de hardware...")
print(tf.config.list_physical_devices('GPU'))

# =====================================================================
# 1. RUTAS DE INFRAESTRUCTURA (DIRECTO EN EL BUCKET)
# =====================================================================
BUCKET_NAME = "dataset-tfrecords-loreto"
RUTA_DATOS = f"gs://{BUCKET_NAME}/datos_produccion/*.tfrecord.gz"
RUTA_MODELO_FINAL = f"gs://{BUCKET_NAME}/modelos_guardados/unet_loreto_produccion.keras"

# =====================================================================
# 2. TUBERÍA DE DATOS DE ALTO RENDIMIENTO (DATA PIPELINE)
# =====================================================================
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32  # Aprovechamos la memoria RAM de la GPU de Vertex AI

def parsear_tfrecord(example_proto):
    bandas = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI', 'VV', 'VH']
    feature_dict = {b: tf.io.FixedLenFeature([128, 128], tf.float32) for b in bandas}
    feature_dict['LABEL'] = tf.io.FixedLenFeature([128, 128], tf.float32)

    parsed_features = tf.io.parse_single_example(example_proto, feature_dict)
    
    X = tf.stack([parsed_features[b] for b in bandas], axis=-1)
    Y = tf.expand_dims(parsed_features['LABEL'], axis=-1)
    return X, Y

print("Construyendo tubería de lectura distribuida...")
archivos = tf.io.gfile.glob(RUTA_DATOS)
dataset_crudo = tf.data.TFRecordDataset(archivos, compression_type='GZIP', num_parallel_reads=AUTOTUNE)

# Optimizaciones de nivel empresarial para alimentar la GPU a máxima velocidad
dataset = (dataset_crudo
           .map(parsear_tfrecord, num_parallel_calls=AUTOTUNE)
           .shuffle(2000) # Mezcla 2000 imágenes para evitar sesgos
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE)) # Pre-carga el siguiente lote mientras la GPU procesa el actual

# =====================================================================
# 3. ARQUITECTURA DE LA RED NEURONAL (U-NET MASIVA)
# =====================================================================
def construir_unet(input_shape=(128, 128, 10)):
    inputs = layers.Input(shape=input_shape)

    # BAJADA (Encoder)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # CUELLO DE BOTELLA
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # SUBIDA (Decoder)
    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    modelo = models.Model(inputs=[inputs], outputs=[outputs])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

print("Compilando arquitectura U-Net...")
modelo = construir_unet()

# =====================================================================
# 4. ORQUESTACIÓN DEL ENTRENAMIENTO Y GUARDIANES (CALLBACKS)
# =====================================================================
print("Configurando guardianes de entrenamiento (Callbacks)...")

# Guardián 1: Detiene el proceso si la IA deja de aprender en 10 épocas seguidas (Ahorra USD)
early_stopping = callbacks.EarlyStopping(
    monitor='loss', 
    patience=10, 
    restore_best_weights=True
)

# Guardián 2: Guarda automáticamente el modelo en tu Bucket al terminar
model_checkpoint = callbacks.ModelCheckpoint(
    filepath=RUTA_MODELO_FINAL,
    save_best_only=True,
    monitor='loss'
)

print("\n=== INICIANDO ENTRENAMIENTO MASIVO EN VERTEX AI ===")
# Arrancamos el motor con un máximo de 100 épocas
historial = modelo.fit(
    dataset,
    epochs=100,
    callbacks=[early_stopping, model_checkpoint]
)

print(f"\n=== ENTRENAMIENTO COMPLETADO ===")
print(f"Modelo guardado exitosamente de forma permanente en: {RUTA_MODELO_FINAL}")