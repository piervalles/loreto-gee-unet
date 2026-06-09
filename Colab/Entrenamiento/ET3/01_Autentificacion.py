# =====================================================================
# 0. AUTENTICACIÓN GOOGLE CLOUD (Obligatorio cada vez que abres Colab)
# =====================================================================
from google.colab import auth
print("Solicitando permisos VIP para acceder a Cloud Storage...")
auth.authenticate_user()
print("¡Autenticación exitosa! Tienes luz verde para leer el Bucket.")