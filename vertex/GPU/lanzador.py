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
print("Empaquetando el código de entrenamiento (Enterprise)...")

# Asegúrate de que el nombre de tu archivo Python coincida aquí
ARCHIVO_ENTRENAMIENTO = "task.py" # o "entrenamiento_masivo.py"

job = aiplatform.CustomTrainingJob(
    display_name="entrenamiento-unet-loreto-GPU",
    script_path=ARCHIVO_ENTRENAMIENTO, 
    
    # CRÍTICO: Cambiamos al contenedor de GPU oficial de Google para TensorFlow
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest",
    
    requirements=["google-cloud-storage", "numpy"] 
)

# =====================================================================
# 3. LANZAMIENTO A LA NUBE (INTENTO DE GPU)
# =====================================================================
print("\n=== SOLICITANDO HARDWARE A GOOGLE CLOUD ===")
print("Solicitando: 1 máquina n1-standard-8 con 1 NVIDIA T4...")

try:
    # Intento 1: Pedir máquina estándar con GPU
    modelo = job.run(
        machine_type="n1-standard-8", 
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        sync=True 
    )
    print("\n=== ¡EL TRABAJO EN VERTEX AI (CON GPU) HA CULMINADO! ===")

except Exception as e:
    # Si Google rechaza la solicitud de GPU (por cuota de Free Trial), lo atrapamos y lanzamos CPU
    print(f"\n❌ Google Cloud rechazó la solicitud de GPU: {e}")
    print("\n⚠️ Aplicando Plan B: Lanzando con fuerza bruta de CPU (n1-highcpu-16)...")
    
    # Intento 2: Caer con estilo a la CPU que ya sabías que funcionaba
    # (Nota: Mantenemos el contenedor tf-gpu, funciona igual en CPU, solo ignorará la GPU que no existe)
    modelo = job.run(
        machine_type="n1-highcpu-16", 
        sync=True 
    )
    print("\n=== ¡EL TRABAJO EN VERTEX AI (CON CPU) HA CULMINADO! ===")

print("Revisa tu Bucket de Storage para confirmar el guardado.")