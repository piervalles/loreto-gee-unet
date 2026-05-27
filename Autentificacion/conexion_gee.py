import ee

# 1. Autenticación (Solo la primera vez te abrirá el navegador para pedirte permisos)
print("Iniciando autenticación con Google Earth Engine...")
ee.Authenticate()

# 2. Inicialización de la sesión vinculada a tu proyecto
# Reemplaza 'proyecto-ia-496311' con el ID exacto si es diferente
print("Inicializando sesión...")
ee.Initialize(project='proyecto-ia-496311')

print("¡Conexión exitosa! El orquestador local está listo para enviar instrucciones a la nube.")