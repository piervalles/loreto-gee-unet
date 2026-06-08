# =============================================================================
# CELDA 1: DEPENDENCIAS Y PERMISOS
# Ejecuta esta celda primero, espera que instale y acepta los permisos de Drive.
# =============================================================================

!pip install -q geopandas

from google.colab import drive
print("Solicitando conexión a Google Drive...")
drive.mount('/content/drive')
print("¡Entorno listo!")