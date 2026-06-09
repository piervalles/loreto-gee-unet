import os
import geopandas as gpd
import osmnx as ox

DATA_DIR = '/content/drive/MyDrive/Mapas_Loreto'
CRS_METRICO = "EPSG:32718"

print("1. Obteniendo el molde fronterizo de Loreto...")
# Hacemos una única petición minúscula solo para obtener la silueta de la región
loreto_molde = ox.geocode_to_gdf("Loreto, Peru")

print("2. Procesando Red Vial (Cortando carreteras de Perú a Loreto)...")
vias_peru = gpd.read_file('peru_data/gis_osm_roads_free_1.shp')
vias_loreto = gpd.clip(vias_peru, loreto_molde)
vias_loreto.to_crs(CRS_METRICO).to_file(os.path.join(DATA_DIR, 'vias_osm.gpkg'), driver='GPKG')
del vias_peru # Vaciamos la memoria RAM para no saturar la máquina
print("✅ Vías de Loreto guardadas en Drive.")

print("3. Procesando Hidrografía (Ríos)...")
rios_peru = gpd.read_file('peru_data/gis_osm_waterways_free_1.shp')
rios_loreto = gpd.clip(rios_peru, loreto_molde)
rios_loreto.to_crs(CRS_METRICO).to_file(os.path.join(DATA_DIR, 'rios_osm.gpkg'), driver='GPKG')
del rios_peru
print("✅ Ríos de Loreto guardados en Drive.")

print("4. Procesando Zonas Urbanas...")
usos_peru = gpd.read_file('peru_data/gis_osm_landuse_a_free_1.shp')
# Filtramos solo las categorías urbanas que nos interesan
urbanos_peru = usos_peru[usos_peru['fclass'].isin(['residential', 'commercial', 'industrial'])]
urbanos_loreto = gpd.clip(urbanos_peru, loreto_molde)
urbanos_loreto.to_crs(CRS_METRICO).to_file(os.path.join(DATA_DIR, 'urbano_osm.gpkg'), driver='GPKG')
del usos_peru
print("✅ Zonas Urbanas guardadas en Drive.")

print("\n🚀 ¡PLAN C COMPLETADO CON ÉXITO! Base de datos regional offline creada.")