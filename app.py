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

st.title("🛰️ Sistema de Detección de Deforestación y Minería Ilegal")
st.markdown("""
Esta plataforma utiliza Inteligencia Artificial (Redes Neuronales U-Net) e imágenes 
satelitales de radar y ópticas para identificar la pérdida de bosque en la región de Loreto.
""")
st.divider()

# =====================================================================
# 2. PANEL LATERAL E INTERACTIVIDAD
# =====================================================================
with st.sidebar:
    st.header("⚙️ Panel de Control")
    st.info("El sistema está analizando el departamento de Loreto, Perú.")
    
    capa_mostrar = st.radio(
        "Seleccionar Capa de Visualización:",
        ("Mapa Base", "Indicios de Minería (IA)", "Deforestación Agrícola (IA)")
    )
    
    st.markdown("---")
    st.write("Desarrollado por Piero")

# =====================================================================
# 3. LÓGICA ESPACIAL (El Mapa Interactivo)
# =====================================================================
LORETO_COORDS = [-4.0, -74.0]

# Inicializamos el mapa
mapa = folium.Map(
    location=LORETO_COORDS, 
    zoom_start=6,
    tiles="CartoDB positron" 
)

# Patrón Mock: Simulamos las coordenadas que escupirá tu IA en el futuro
COORDENADAS_MINERIA = [[-3.8, -73.5], [-3.8, -73.4], [-3.9, -73.4], [-3.9, -73.5]]
COORDENADAS_TALA = [[-4.5, -74.5], [-4.5, -74.3], [-4.7, -74.3], [-4.7, -74.5]]

# Lógica de renderizado según el botón que presione el usuario
if capa_mostrar == "Indicios de Minería (IA)":
    folium.Polygon(
        locations=COORDENADAS_MINERIA,
        color="orange",
        weight=2,
        fill=True,
        fill_opacity=0.5,
        popup="⚠️ ALERTA IA: Posible Minería Ilegal (Cerca a cuerpo de agua)"
    ).add_to(mapa)

elif capa_mostrar == "Deforestación Agrícola (IA)":
    folium.Polygon(
        locations=COORDENADAS_TALA,
        color="red",
        weight=2,
        fill=True,
        fill_opacity=0.5,
        popup="🪓 ALERTA IA: Deforestación por tala detectada"
    ).add_to(mapa)

# Renderizamos el mapa actualizado
st_folium(mapa, width=1200, height=600)