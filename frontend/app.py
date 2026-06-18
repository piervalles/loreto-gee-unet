"""
frontend/app.py
===============
Interfaz de usuario principal del Sistema Inteligente de Alerta de Deforestación
para la región de Loreto, Perú.

Ejecutar desde la raíz del proyecto:
    streamlit run frontend/app.py

Requiere:
    streamlit, folium, streamlit-folium, tensorflow, geopandas, numpy, shapely
"""

import sys
import pathlib
import logging

# ---------------------------------------------------------------------------
# Ajuste de sys.path — DEBE ejecutarse antes de cualquier import del backend.
# Resuelve el raíz del proyecto independientemente del directorio de trabajo
# actual, por lo que funciona tanto con `streamlit run frontend/app.py` como
# ejecutando el script directamente desde cualquier ubicación.
# ---------------------------------------------------------------------------
_DIR_RAIZ = pathlib.Path(__file__).resolve().parent.parent
if str(_DIR_RAIZ) not in sys.path:
    sys.path.insert(0, str(_DIR_RAIZ))

import numpy as np
import tensorflow as tf
import folium
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import Point
from backend.core.inferencia import TraductorUNet
from backend.core.sintetizador import SintetizadorEspacial



# ---------------------------------------------------------------------------
# Configuración de logging (Streamlit captura stdout; usamos el logger)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes geográficas
# ---------------------------------------------------------------------------
IQUITOS_LAT:  float = -3.7491
IQUITOS_LON:  float = -73.2538
PUNTO_UTM_X:  int   = 685_000   # UTM Zona 18S (EPSG:32718)
PUNTO_UTM_Y:  int   = 9_585_000

# Paleta de colores semántica
COLOR_ALTA   = "#FF4136"   # Minería / Alerta alta
COLOR_MEDIA  = "#FF851B"   # Expansión urbana / Agrícola
COLOR_BAJA   = "#2ECC40"   # Sin deforestación significativa
COLOR_ACENTO = "#00B4D8"   # Azul cian para UI

# ===========================================================================
# CONFIGURACIÓN DE PÁGINA
# ===========================================================================
st.set_page_config(
    page_title="Sistema Inteligente de Alerta de Deforestación - Loreto",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Monitor satelital de deforestación para la Amazonía de Loreto, Perú.",
    },
)

# ===========================================================================
# CSS PERSONALIZADO — Estética premium
# ===========================================================================
st.markdown(
    """
    <style>
    /* ---- Variables de diseño ---- */
    :root {
        --verde-oscuro:  #0d2818;
        --verde-medio:   #1a4731;
        --verde-claro:   #2d7d46;
        --acento-cian:   #00b4d8;
        --acento-gold:   #f4a261;
        --fondo:         #0a1f0f;
        --superficie:    #112318;
        --borde:         #1e3a25;
        --texto-ppal:    #e8f5e9;
        --texto-sec:     #81c784;
        --rojo-alerta:   #ff4136;
        --naranja-warn:  #ff851b;
    }

    /* ---- Fondo global ---- */
    .stApp {
        background: linear-gradient(135deg, var(--fondo) 0%, #0b1f14 60%, #091a10 100%);
        color: var(--texto-ppal);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--verde-oscuro) 0%, #0c1e12 100%);
        border-right: 1px solid var(--borde);
    }
    [data-testid="stSidebar"] * { color: var(--texto-ppal) !important; }

    /* ---- Encabezado hero ---- */
    .hero-header {
        background: linear-gradient(135deg, rgba(0,180,216,0.12) 0%, rgba(45,125,70,0.18) 100%);
        border: 1px solid rgba(0,180,216,0.25);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00b4d8, #81c784, #f4a261);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #81c784;
        font-size: 0.95rem;
        margin-top: 6px;
        letter-spacing: 0.3px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0,180,216,0.15);
        border: 1px solid rgba(0,180,216,0.4);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #00b4d8;
        margin-top: 10px;
    }

    /* ---- Tarjetas de métricas ---- */
    .metric-card {
        background: linear-gradient(145deg, rgba(17,35,24,0.95) 0%, rgba(13,40,24,0.9) 100%);
        border: 1px solid var(--borde);
        border-radius: 14px;
        padding: 22px 24px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        height: 100%;
    }
    .metric-card:hover {
        border-color: rgba(0,180,216,0.45);
        box-shadow: 0 6px 28px rgba(0,180,216,0.15);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64b5f6;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 6px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #81c784;
        opacity: 0.85;
    }
    .metric-icon { font-size: 1.6rem; margin-bottom: 8px; }

    /* ---- Tarjeta de causa ---- */
    .causa-card {
        border-radius: 14px;
        padding: 22px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        height: 100%;
        border-width: 1px;
        border-style: solid;
    }
    .causa-alta {
        background: linear-gradient(145deg, rgba(255,65,54,0.15), rgba(180,20,10,0.12));
        border-color: rgba(255,65,54,0.45);
    }
    .causa-media {
        background: linear-gradient(145deg, rgba(255,133,27,0.15), rgba(180,90,10,0.12));
        border-color: rgba(255,133,27,0.45);
    }
    .causa-baja {
        background: linear-gradient(145deg, rgba(46,204,64,0.1), rgba(20,140,30,0.1));
        border-color: rgba(46,204,64,0.3);
    }

    /* ---- Sección de mapa ---- */
    .map-section {
        background: rgba(10,31,15,0.7);
        border: 1px solid var(--borde);
        border-radius: 16px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }
    .map-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--acento-cian);
        letter-spacing: 0.5px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ---- Estado idle ---- */
    .idle-message {
        background: rgba(17,35,24,0.7);
        border: 1px dashed rgba(45,125,70,0.4);
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        color: #64b5f6;
    }
    .idle-icon { font-size: 3rem; margin-bottom: 12px; }

    /* ---- Botón sidebar ---- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #007bbd 0%, #00b4d8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 700;
        font-size: 0.92rem;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,180,216,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0099cc 0%, #48cae4 100%);
        box-shadow: 0 6px 22px rgba(0,180,216,0.45);
        transform: translateY(-1px);
    }

    /* ---- Ocultar elementos Streamlit por defecto ---- */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# CACHÉ DE RECURSOS — Carga única del modelo y motor geográfico
# ===========================================================================

@st.cache_resource(show_spinner="🔄 Cargando modelo U-Net (esto ocurre solo una vez)…")
def cargar_modelo() -> TraductorUNet:
    """Instancia y cachea el TraductorUNet."""
    logger.info("Inicializando TraductorUNet desde caché de Streamlit…")
    return TraductorUNet()


@st.cache_resource(show_spinner="🗺️ Cargando capas GIS y calculando zonas de influencia…")
def cargar_sintetizador() -> SintetizadorEspacial:
    """Instancia y cachea el SintetizadorEspacial."""
    logger.info("Inicializando SintetizadorEspacial desde caché de Streamlit…")
    return SintetizadorEspacial()


# ===========================================================================
# HELPERS DE UI
# ===========================================================================

def _color_causa(causa: str) -> tuple[str, str]:
    """Devuelve (clase_css, emoji) según la causa clasificada."""
    if "Minería" in causa:
        return "causa-alta", "⛏️"
    if "Urbana" in causa or "Agrícola" in causa or "Tala" in causa and "Aislada" not in causa:
        return "causa-media", "🌾"
    if "Aislada" in causa or "Clandestina" in causa:
        return "causa-media", "🪓"
    return "causa-baja", "✅"


def _construir_mapa(lat: float, lon: float, zoom: int = 11) -> folium.Map:
    """Crea y retorna un mapa Folium base centrado en Iquitos."""
    mapa = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles=None,
        prefer_canvas=True,
    )
    # Capa base satelital — ESRI World Imagery
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri | NASA, NGA, USGS",
        name="Satélite ESRI",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(mapa)
    # Capa de etiquetas superpuesta
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors, © CARTO",
        name="Etiquetas",
        overlay=True,
        control=True,
    ).add_to(mapa)
    folium.LayerControl().add_to(mapa)
    return mapa


def _agregar_marcador(
    mapa: folium.Map,
    lat: float,
    lon: float,
    causa: str,
    pixeles: int,
    porcentaje: float,
) -> None:
    """Añade un marcador de alerta rojo al mapa con popup informativo."""
    popup_html = f"""
    <div style="
        font-family: 'Segoe UI', sans-serif;
        background: #1a1a2e;
        color: #e0e0e0;
        border-radius: 10px;
        padding: 16px 20px;
        min-width: 240px;
        border: 1px solid rgba(255,65,54,0.5);
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    ">
        <div style="
            font-size:0.7rem;
            text-transform:uppercase;
            letter-spacing:2px;
            color:#ff4444;
            font-weight:700;
            margin-bottom:8px;
        ">⚠ ALERTA DE DEFORESTACIÓN</div>
        <div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:12px;">
            {causa}
        </div>
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:8px 0;">
        <div style="font-size:0.82rem;color:#aaaaaa;">
            📍 <b>Lat:</b> {lat:.4f} | <b>Lon:</b> {lon:.4f}<br>
            🔴 <b>Píxeles deforestados:</b> {pixeles:,}<br>
            📊 <b>Cobertura del cuadrante:</b> {porcentaje:.1f}%
        </div>
    </div>
    """
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"🚨 {causa}",
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
    ).add_to(mapa)

    # Círculo de influencia visual
    folium.Circle(
        location=[lat, lon],
        radius=3500,
        color="#FF4136",
        weight=2,
        fill=True,
        fill_color="#FF4136",
        fill_opacity=0.10,
    ).add_to(mapa)

    # Pulso animado (ring exterior)
    folium.Circle(
        location=[lat, lon],
        radius=5500,
        color="#FF851B",
        weight=1.5,
        fill=False,
        dash_array="6 4",
    ).add_to(mapa)


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(0,180,216,0.1), rgba(45,125,70,0.12));
            border: 1px solid rgba(0,180,216,0.2);
            border-radius: 12px;
            padding: 18px 16px;
            margin-bottom: 24px;
            text-align: center;
        ">
            <div style="font-size:2.2rem;">🛰️</div>
            <div style="font-weight:800;font-size:1.05rem;color:#00b4d8;margin-top:6px;">
                Monitor Loreto
            </div>
            <div style="font-size:0.72rem;color:#81c784;margin-top:4px;letter-spacing:1px;">
                SISTEMA DE DETECCIÓN IA
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### ⚙️ Panel de Control")
    st.markdown(
        "<div style='color:#81c784;font-size:0.82rem;margin-bottom:16px;line-height:1.5;'>"
        "Simula el escaneo de un cuadrante de 128×128 px sobre la "
        "Amazonía de Loreto usando el modelo U-Net entrenado."
        "</div>",
        unsafe_allow_html=True,
    )

    boton_escanear = st.button(
        "🛰️ Simular Escaneo Satelital (Sentinel-2/1)",
        use_container_width=True,
    )

    st.divider()

    st.markdown("#### 📋 Leyenda de Causas")
    causas_leyenda = [
        ("⛏️", "#FF4136", "Minería Aluvial", "Alerta Alta"),
        ("🏘️", "#FF851B", "Expansión Urbana", "Alerta Media"),
        ("🌾", "#FF851B", "Expansión Agrícola", "Alerta Media"),
        ("🪓", "#FFDC00", "Tala Aislada", "Alerta Baja"),
    ]
    for emoji, color, nombre, nivel in causas_leyenda:
        st.markdown(
            f"""<div style="
                display:flex;align-items:center;gap:10px;
                padding:8px 10px;margin-bottom:6px;
                background:rgba(17,35,24,0.7);
                border-left:3px solid {color};
                border-radius:0 8px 8px 0;
                font-size:0.82rem;
            ">
                <span>{emoji}</span>
                <div>
                    <div style="color:#e8f5e9;font-weight:600;">{nombre}</div>
                    <div style="color:{color};font-size:0.7rem;">{nivel}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<div style='font-size:0.7rem;color:#546e7a;text-align:center;line-height:1.6;'>"
        "Modelo: U-Net Loreto v1<br>"
        "CRS: EPSG:32718 (UTM 18S)<br>"
        "Fuente: Sentinel-2 / OSM Loreto"
        "</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# ÁREA PRINCIPAL — Hero Header
# ===========================================================================
st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">🌿 Sistema Inteligente de Alerta de Deforestación</h1>
        <div class="hero-subtitle">
            Detección automática de cambios de cobertura forestal en la Amazonía de Loreto, Perú
            mediante Deep Learning (U-Net) y análisis geoespacial.
        </div>
        <span class="hero-badge">🛰️ Sentinel-2 · Sentinel-1 · U-Net · GIS</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# CARGA DE RECURSOS (con caché)
# ===========================================================================
with st.spinner("Inicializando recursos del sistema…"):
    try:
        traductor    = cargar_modelo()
        sintetizador = cargar_sintetizador()
        recursos_ok  = True
    except Exception as exc:
        st.error(f"❌ Error al cargar recursos del sistema: {exc}")
        recursos_ok = False

# ===========================================================================
# ESTADO DE SESIÓN — Persistir resultados entre reruns
# ===========================================================================
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None

# ===========================================================================
# LÓGICA DEL BOTÓN — Simulación de escaneo satelital
# ===========================================================================
if boton_escanear and recursos_ok:
    with st.spinner("📡 Procesando imagen satelital…"):
        try:
            # 1. Generar tensor simulado (imagen óptica/radar multibanda)
            tensor_simulado = tf.random.uniform(
                shape=(1, 128, 128, 10),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )
            logger.info("Tensor simulado generado: %s", tensor_simulado.shape)

            # 2. Inferencia → máscara binaria (128, 128)
            mascara = traductor.evaluar_imagen(tensor_simulado, umbral=0.5)

            # 3. Métricas de deforestación
            total_pixeles       = mascara.size                        # 16 384
            pixeles_deforest    = int(mascara.sum())
            porcentaje_deforest = round(100.0 * pixeles_deforest / total_pixeles, 2)

            logger.info(
                "Inferencia completada: %d/%d píxeles deforestados (%.1f%%)",
                pixeles_deforest, total_pixeles, porcentaje_deforest,
            )

            # 4. Clasificación espacial con coordenadas UTM 18S cerca de Iquitos
            punto_utm = Point(PUNTO_UTM_X, PUNTO_UTM_Y)
            causa = sintetizador.clasificar_alerta(punto_utm)

            logger.info("Causa clasificada: %s", causa)

            # 5. Guardar resultado en session_state
            st.session_state["resultado"] = {
                "pixeles_deforest":    pixeles_deforest,
                "total_pixeles":       total_pixeles,
                "porcentaje_deforest": porcentaje_deforest,
                "causa":               causa,
                "hay_deforestacion":   pixeles_deforest > 0,
            }

        except Exception as exc:
            st.error(f"❌ Error durante el escaneo: {exc}")
            logger.exception("Error en pipeline de escaneo")

# ===========================================================================
# SECCIÓN DE MÉTRICAS
# ===========================================================================
res = st.session_state.get("resultado")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    pixeles_val = f"{res['pixeles_deforest']:,}" if res else "—"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🔴</div>
            <div class="metric-label">Píxeles Afectados</div>
            <div class="metric-value" style="color:#ff4136;">{pixeles_val}</div>
            <div class="metric-sub">de 16,384 px totales (128×128)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    pct_val = f"{res['porcentaje_deforest']:.1f}%" if res else "—"
    pct_color = "#ff4136" if res and res["porcentaje_deforest"] > 20 else (
        "#ff851b" if res and res["porcentaje_deforest"] > 5 else "#2ecc40"
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-label">Porcentaje del Cuadrante</div>
            <div class="metric-value" style="color:{pct_color};">{pct_val}</div>
            <div class="metric-sub">Cobertura deforestada detectada</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    if res:
        causa_val  = res["causa"]
        css_clase, causa_emoji = _color_causa(causa_val)
        st.markdown(
            f"""
            <div class="metric-card causa-card {css_clase}">
                <div class="metric-icon">{causa_emoji}</div>
                <div class="metric-label">Causa Clasificada</div>
                <div style="font-size:1.05rem;font-weight:800;line-height:1.3;
                            color:#ffffff;margin-bottom:6px;">
                    {causa_val}
                </div>
                <div class="metric-sub">Análisis por SintetizadorEspacial GIS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-icon">🧩</div>
                <div class="metric-label">Causa Clasificada</div>
                <div class="metric-value" style="color:#546e7a;">—</div>
                <div class="metric-sub">Pendiente de escaneo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Alerta semántica adicional bajo las métricas
if res:
    if res["porcentaje_deforest"] > 20 or "Minería" in res["causa"]:
        st.error(
            f"🚨 **Alerta Crítica** — {res['causa']} detectada con "
            f"{res['porcentaje_deforest']:.1f}% del cuadrante comprometido."
        )
    elif res["porcentaje_deforest"] > 5:
        st.warning(
            f"⚠️ **Alerta Moderada** — {res['causa']} con "
            f"{res['porcentaje_deforest']:.1f}% de cobertura afectada."
        )
    else:
        st.success(
            f"✅ **Cobertura baja** — {res['porcentaje_deforest']:.1f}% afectado. "
            f"Causa: {res['causa']}."
        )

# ===========================================================================
# SECCIÓN DEL MAPA INTERACTIVO
# ===========================================================================
st.markdown(
    """
    <div style="margin-top:24px;">
        <div class="map-title">
            🗺️ Mapa de Monitoreo — Región Iquitos, Loreto
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mapa = _construir_mapa(IQUITOS_LAT, IQUITOS_LON, zoom=11)

if res and res["hay_deforestacion"]:
    _agregar_marcador(
        mapa=mapa,
        lat=IQUITOS_LAT,
        lon=IQUITOS_LON,
        causa=res["causa"],
        pixeles=res["pixeles_deforest"],
        porcentaje=res["porcentaje_deforest"],
    )
elif not res:
    # Marcador informativo (sin alerta) cuando no se ha ejecutado el escaneo
    folium.Marker(
        location=[IQUITOS_LAT, IQUITOS_LON],
        tooltip="📍 Iquitos — Zona de monitoreo",
        icon=folium.Icon(color="blue", icon="satellite", prefix="fa"),
    ).add_to(mapa)

# Renderizado del mapa con streamlit-folium
st_folium(
    mapa,
    width="100%",
    height=520,
    returned_objects=[],   # No retornar datos de clic (mejora rendimiento)
    key="mapa_loreto",
)

if not res:
    st.markdown(
        """
        <div class="idle-message">
            <div class="idle-icon">🛰️</div>
            <div style="font-size:1rem;font-weight:700;margin-bottom:8px;color:#64b5f6;">
                Sistema en espera
            </div>
            <div style="font-size:0.85rem;opacity:0.75;">
                Presiona <b>"Simular Escaneo Satelital"</b> en el panel lateral
                para ejecutar el modelo U-Net sobre un cuadrante de Loreto.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
