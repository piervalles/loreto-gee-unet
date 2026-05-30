from google.cloud import aiplatform

# =====================================================================
# 1. CREDENCIALES Y UBICACIÓN
# =====================================================================
PROJECT_ID = "proyecto-ia-496311"
REGION = "us-central1" 
BUCKET_STAGING = "gs://dataset-tfrecords-loreto"

print("Inicializando conexión con Vertex AI...")
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_STAGING)

# =====================================================================
# 2. DEFINICIÓN DEL TRABAJO DE ENTRENAMIENTO (CUSTOM JOB)
# =====================================================================
print("Empaquetando el código de entrenamiento...")

job = aiplatform.CustomTrainingJob(
    display_name="entrenamiento-unet-loreto-cpu",
    script_path="entrenamiento_masivo.py", 
    
    # CRÍTICO: Cambiamos al contenedor de CPU oficial de Google
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest",
    
    requirements=["google-cloud-storage", "numpy"] 
)

# =====================================================================
# 3. LANZAMIENTO A LA SUPERCOMPUTADORA
# =====================================================================
print("\n=== SOLICITANDO HARDWARE A GOOGLE CLOUD ===")
print("Solicitando: 1 máquina n1-highcpu-16 (16 vCPUs sin GPU)...")

# Fuerza bruta de procesador, cero bloqueos de cuota
modelo = job.run(
    machine_type="n1-highcpu-16", 
    sync=True 
)

print("\n=== ¡EL TRABAJO EN VERTEX AI HA CULMINADO! ===")
print("El modelo debería estar ya a salvo en tu Bucket de Storage.")