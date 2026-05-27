import streamlit as st
import folium
from streamlit_folium import st_folium

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Monitor IA - Amazonía",
    page_icon="🌳",
    layout="wide"
)

# Título de la plataforma
st.title("🛰️ Sistema de Detección de Deforestación y Minería Ilegal")
st.markdown("""
Esta plataforma utiliza Inteligencia Artificial (Redes Neuronales U-Net) e imágenes 
satelitales de radar y ópticas (Sentinel-1 y 2) para identificar la pérdida de bosque en la región de Loreto.
""")

st.divider()

# =====================================================================
# 2. PANEL LATERAL (Filtros y Controles)
# =====================================================================
with st.sidebar:
    st.header("⚙️ Panel de Control")
    st.info("El sistema está analizando el departamento de Loreto, Perú.")
    
    # Aquí en el futuro pondremos botones para cargar predicciones
    capa_mostrar = st.radio(
        "Seleccionar Capa de Visualización:",
        ("Mapa Base", "Indicios de Minería (IA)", "Deforestación Agrícola (IA)")
    )
    
    st.markdown("---")
    st.write("Desarrollado por Piero")

# =====================================================================
# 3. EL MAPA INTERACTIVO (Folium / Leaflet)
# =====================================================================
# Coordenadas centrales de Loreto (Latitud, Longitud)
LORETO_COORDS = [-4.0, -74.0]

st.subheader("📍 Mapa Georreferenciado")

# Creamos el mapa base con un estilo satelital/terreno
mapa = folium.Map(
    location=LORETO_COORDS, 
    zoom_start=6,
    tiles="CartoDB positron" # Estilo de mapa limpio
)

# Un marcador de ejemplo (Más adelante serán los polígonos de tu IA)
folium.Marker(
    location=[-3.7491, -73.2538], # Iquitos
    popup="Iquitos (Capital de Loreto)",
    icon=folium.Icon(color="green", icon="info-sign")
).add_to(mapa)

# Renderizamos el mapa en la página web
st_folium(mapa, width=1200, height=600)