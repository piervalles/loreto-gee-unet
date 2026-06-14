# =============================================================================
# DESCARGA, CLIPPING Y SUBIDA DIRECTA A GCS (bucket: mapas_loreto)
# =============================================================================

!pip install -q geopandas osmnx gcsfs

from google.colab import auth
auth.authenticate_user()
print("✅ Autenticación GCP completada.")

!wget -q https://download.geofabrik.de/south-america/peru-latest-free.shp.zip
!unzip -q -o peru-latest-free.shp.zip -d peru_data
print("✅ Shapefile de Perú descargado.")

import os
import geopandas as gpd
import osmnx as ox

# --- CONFIGURACIÓN ---
BUCKET = "mapas_loreto"
GCS_FOLDER = "01"          # carpeta dentro del bucket
CRS_METRICO = "EPSG:32718"

# Crear carpeta temporal local
os.makedirs("temp_gpkg", exist_ok=True)

# Obtener silueta de Loreto
print("Obteniendo molde de Loreto...")
loreto_molde = ox.geocode_to_gdf("Loreto, Peru")
print("✅ Silueta cargada.")

def procesar_y_subir(shp_path, output_name, filtro=None, columna_filtro='fclass'):
    print(f"\n--- Procesando {output_name} ---")
    gdf = gpd.read_file(shp_path)
    if filtro:
        gdf = gdf[gdf[columna_filtro].isin(filtro)]
        print(f"   Filtrado: {len(gdf)} geometrías")
    
    gdf_loreto = gpd.clip(gdf, loreto_molde)
    gdf_loreto = gdf_loreto.to_crs(CRS_METRICO)
    
    # Guardar temporal local
    tmp_path = f"temp_gpkg/{output_name}"
    gdf_loreto.to_file(tmp_path, driver='GPKG')
    
    # Subir a GCS (bucket mapas_loreto, carpeta 01/)
    gcs_path = f"gs://{BUCKET}/{GCS_FOLDER}/{output_name}"
    !gsutil cp {tmp_path} {gcs_path}
    print(f"✅ Subido a {gcs_path}")
    
    os.remove(tmp_path)

# --- Ejecutar capas ---
procesar_y_subir('peru_data/gis_osm_roads_free_1.shp', 'vias_osm.gpkg')
procesar_y_subir('peru_data/gis_osm_waterways_free_1.shp', 'rios_osm.gpkg')
procesar_y_subir('peru_data/gis_osm_landuse_a_free_1.shp', 'urbano_osm.gpkg',
                 filtro=['residential', 'commercial', 'industrial'])

print("\n🚀 Archivos guardados en:")
print(f"   gs://{BUCKET}/{GCS_FOLDER}/")
print("   - vias_osm.gpkg")
print("   - rios_osm.gpkg")
print("   - urbano_osm.gpkg")