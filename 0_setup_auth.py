# 0_setup_auth.py
import ee

print("Iniciando flujo de autenticación de Google Earth Engine...")
# Esto abrirá el navegador y guardará el token en tu disco duro
ee.Authenticate() 
print("¡Token guardado exitosamente en la computadora!")