from google.cloud import aiplatform

# =====================================================================
# 1. CREDENCIALES Y UBICACIÓN
# =====================================================================
PROJECT_ID = "proyecto-ia-496311"
REGION = "us-central1" # La región más barata y con más GPUs disponibles
BUCKET_STAGING = "gs://dataset-tfrecords-loreto"

print("Inicializando conexión con Vertex AI...")
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_STAGING)

# =====================================================================
# 2. DEFINICIÓN DEL TRABAJO DE ENTRENAMIENTO (CUSTOM JOB)
# =====================================================================
print("Empaquetando el código de entrenamiento...")

job = aiplatform.CustomTrainingJob(
    display_name="entrenamiento-unet-loreto-full",
    script_path="entrenamiento_masivo.py", # El archivo que creamos en el paso anterior
    
    # Usamos un contenedor pre-configurado de Google que ya tiene TensorFlow 2.x optimizado para GPU
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest",
    
    # Requisitos de hardware y librerías extra
    requirements=["google-cloud-storage", "numpy"] 
)

# =====================================================================
# 3. LANZAMIENTO A LA SUPERCOMPUTADORA
# =====================================================================
print("\n=== SOLICITANDO HARDWARE A GOOGLE CLOUD ===")
print("Solicitando: 1 máquina n1-standard-4 con 1 GPU NVIDIA Tesla T4...")

# El script se pausa aquí en tu terminal local, pero el trabajo arranca en la nube
modelo = job.run(
    machine_type="n1-standard-4", # 4 vCPUs y 15 GB de RAM
    accelerator_type="NVIDIA_TESLA_T4", # GPU dedicada para Deep Learning
    accelerator_count=1,
    sync=True # True significa que verás los logs del entrenamiento aquí mismo en tu terminal
)

print("\n=== ¡EL TRABAJO EN VERTEX AI HA CULMINADO! ===")
print("El modelo debería estar ya a salvo en tu Bucket de Storage.")