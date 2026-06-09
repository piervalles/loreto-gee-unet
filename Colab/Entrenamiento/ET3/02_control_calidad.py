# =====================================================================
# SCRIPT DE QA VISUAL: VERIFICACIÓN POST-EXPORTACIÓN (MATPLOTLIB)
# Ejecutar para confirmar que GEE exportó correctamente los datos
# =====================================================================
import tensorflow as tf
import matplotlib.pyplot as plt

print("Iniciando control de calidad visual...")

BUCKET_NAME = "dataset-tfrecords-loreto"
RUTA_DATOS_QA = f"gs://{BUCKET_NAME}/datos_entrenamiento_3/*.tfrecord.gz"

# 1. Función de parseo (idéntica a la de producción)
def parsear_qa(example_proto):
    bandas = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI', 'NDTI', 'EVI', 'VV', 'VH']
    feature_dict = {b: tf.io.FixedLenFeature([128, 128], tf.float32) for b in bandas}
    feature_dict['LABEL'] = tf.io.FixedLenFeature([128, 128], tf.float32)
    parsed_features = tf.io.parse_single_example(example_proto, feature_dict)
    
    # Extraemos solo el RGB (B4=Rojo, B3=Verde, B2=Azul) para visualización humana
    rgb = tf.stack([parsed_features['B4'], parsed_features['B3'], parsed_features['B2']], axis=-1)
    label = parsed_features['LABEL']
    return rgb, label

# 2. Tomar solo el primer archivo que encuentre para no saturar la red
archivos_qa = tf.io.gfile.glob(RUTA_DATOS_QA)
if not archivos_qa:
    print(f"❌ No se encontraron archivos en {RUTA_DATOS_QA}. Espera a que GEE termine.")
else:
    dataset_qa = tf.data.TFRecordDataset(archivos_qa[0], compression_type='GZIP').map(parsear_qa).batch(3)

    # 3. Extraer 3 ejemplos y graficarlos
    for imagenes_rgb, etiquetas in dataset_qa.take(1):
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        fig.suptitle("QA Visual: RGB (Sentinel-2) vs Etiqueta de Deforestación", fontsize=16)
        
        for i in range(3):
            # Sentinel-2 viene en valores 0-10000. Dividimos por 3000 para dar brillo a la imagen en la gráfica
            img_visual = tf.clip_by_value(imagenes_rgb[i] / 3000.0, 0.0, 1.0)
            mascara = etiquetas[i]
            
            axs[i, 0].imshow(img_visual)
            axs[i, 0].set_title(f"Parche {i+1}: Imagen RGB")
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(mascara, cmap='Reds')
            axs[i, 1].set_title(f"Parche {i+1}: Máscara Hansen (Rojo=Pérdida)")
            axs[i, 1].axis('off')
            
        plt.tight_layout()
        plt.show()
        print("✅ Inspección visual completada. Revisa los gráficos arriba.")