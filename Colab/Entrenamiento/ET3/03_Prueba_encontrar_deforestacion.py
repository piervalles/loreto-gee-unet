# =====================================================================
# SCRIPT DE QA VISUAL - FILTRADO DE PARCHE CON DEFORESTACIÓN
# Busca activamente parches que contengan al menos un píxel de pérdida
# =====================================================================
import tensorflow as tf
import matplotlib.pyplot as plt

# Autenticación necesaria para acceder al bucket
from google.colab import auth
auth.authenticate_user()

print("Iniciando búsqueda de parches con deforestación...")

BUCKET_NAME = "dataset-tfrecords-loreto"
RUTA_DATOS_QA = f"gs://{BUCKET_NAME}/datos_entrenamiento_3/*.tfrecord.gz"

def parsear_qa(example_proto):
    bandas = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI', 'VV', 'VH']
    feature_dict = {b: tf.io.FixedLenFeature([128, 128], tf.float32) for b in bandas}
    feature_dict['LABEL'] = tf.io.FixedLenFeature([128, 128], tf.float32)
    parsed_features = tf.io.parse_single_example(example_proto, feature_dict)
    
    rgb = tf.stack([parsed_features['B4'], parsed_features['B3'], parsed_features['B2']], axis=-1)
    label = parsed_features['LABEL']
    return rgb, label

# Obtener lista de archivos
archivos_qa = tf.io.gfile.glob(RUTA_DATOS_QA)
if not archivos_qa:
    print(f"❌ No se encontraron archivos en {RUTA_DATOS_QA}. Ejecuta primero el pipeline GEE.")
    exit()

print(f"✅ Archivos encontrados: {len(archivos_qa)}. Procesando el primero...")

# Crear dataset del primer archivo, filtrar parches con etiqueta positiva
dataset_raw = tf.data.TFRecordDataset(archivos_qa[0], compression_type='GZIP').map(parsear_qa)

# Aplicar filtro: solo parches donde la máscara tenga al menos un píxel > 0
dataset_filtrado = (dataset_raw
                    .filter(lambda img, label: tf.reduce_sum(label) > 0)
                    .batch(3))

# Intentar tomar un lote de 3 parches con deforestación
for imagenes_rgb, etiquetas in dataset_filtrado.take(1):
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle("QA Visual: Parches con Deforestación Confirmada", fontsize=16)

    for i in range(3):
        # Escalar RGB de 0-10000 a 0-1 para visualización
        img_visual = tf.clip_by_value(imagenes_rgb[i] / 3000.0, 0.0, 1.0)
        mascara = etiquetas[i]

        axs[i, 0].imshow(img_visual)
        axs[i, 0].set_title(f"Parche {i+1}: Imagen RGB (zona talada)")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mascara, cmap='Reds')
        axs[i, 1].set_title(f"Parche {i+1}: Máscara Hansen")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
    print("✅ Inspección completada. Las manchas rojas deben coincidir con claros en el bosque.")
    break
else:
    print("⚠️ No se encontró ningún parche con deforestación en el primer archivo.")
    print("   Puede que los parches positivos estén en otros archivos. Revisa la exportación de GEE.")