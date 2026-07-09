"""
frontend/app.py
===============
Interfaz de usuario principal del Sistema Inteligente de Alerta de Deforestación
para la región de Loreto, Perú.

Ejecutar desde la raíz del proyecto:
    streamlit run frontend/app.py

Requiere:
    streamlit, folium, streamlit-folium, pyproj
    gcloud SDK (gsutil) instalado y autenticado para lectura desde GCS.
"""

import sys
import json
import logging
import pathlib
import subprocess

# ---------------------------------------------------------------------------
# Ajuste de sys.path — DEBE ejecutarse antes de cualquier import del backend.
# ---------------------------------------------------------------------------
_DIR_RAIZ = pathlib.Path(__file__).resolve().parent.parent
if str(_DIR_RAIZ) not in sys.path:
    sys.path.insert(0, str(_DIR_RAIZ))

import folium
import streamlit as st
from streamlit_folium import st_folium
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fuente de datos — GCS (cloud-first)
# Cambia a "local" para desarrollo sin credenciales GCS.
# ---------------------------------------------------------------------------
FUENTE_DATOS: str = "gcs"
GCS_BUCKET: str = "resultado_inferencia"
GCS_RUTA_GEOJSON: str = "2023/resultado_inferencia_1.geojson"

# Fallback local (solo para desarrollo / pruebas)
RUTA_GEOJSON_LOCAL: pathlib.Path = (
    _DIR_RAIZ / "backend" / "data" / "alertas_ejemplo.geojson"
)
RESOLUCION_M: int = 10  # metros por píxel (Sentinel-2, banda 10m)

# ---------------------------------------------------------------------------
# Centro del mapa
# ---------------------------------------------------------------------------
IQUITOS_LAT: float = -3.7491
IQUITOS_LON: float = -73.2538

# ---------------------------------------------------------------------------
# Sistema de etiquetas visuales (exactamente 3 categorías en la UI)
#
# MAPA_ETIQUETA: causa interna del GeoJSON  →  etiqueta visual
#   - "Urbano"        → "Deforestación Urbana"
#   - "Minería"       → "Posible minería ilegal aluvial"
#   - "Vial" | "Otros" | cualquier otro → "Deforestación No Urbana"
# ---------------------------------------------------------------------------
MAPA_ETIQUETA: dict[str, str] = {
    "Urbano":    "Deforestación Urbana",
    "Minería":   "Posible minería ilegal aluvial",
    "Vial":      "Deforestación No Urbana",
    "Otros":     "Deforestación No Urbana",
    "No urbano": "Deforestación No Urbana",   # compatibilidad con GeoJSON de ejemplo
}

PALETA_ETIQUETA: dict[str, dict] = {
    "Deforestación Urbana":           {"fill": "#2ECC40", "border": "#27ae60", "emoji": "🏘️"},
    "Posible minería ilegal aluvial": {"fill": "#FF4136", "border": "#c0392b", "emoji": "⛏️"},
    "Deforestación No Urbana":        {"fill": "#FFDC00", "border": "#c8a800", "emoji": "🌿"},
}

COLOR_FALSO_POSITIVO: dict = {"fill": "#808080", "border": "#444444", "emoji": "🚫"}

# Lista ordenada de etiquetas (para el selectbox del panel de validación)
ETIQUETAS_VALIDAS: list[str] = list(PALETA_ETIQUETA.keys())

# ---------------------------------------------------------------------------
# Proyector UTM 18S → WGS84
# ---------------------------------------------------------------------------
_PROYECTOR = Transformer.from_crs("EPSG:32718", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# Helpers de conversión
# ---------------------------------------------------------------------------

def _causa_a_etiqueta(causa_interna: str) -> str:
    """
    Traduce la causa interna del GeoJSON a la etiqueta visual del frontend.

    Cualquier causa no reconocida cae en "Deforestación No Urbana" por defecto.
    """
    return MAPA_ETIQUETA.get(causa_interna, "Deforestación No Urbana")


def _utm_a_wgs84(coords_utm: list) -> list[list[float]]:
    """Convierte lista de pares [X_utm, Y_utm] → [[lat, lon], ...].

    Con always_xy=True: transform(easting, northing) → (lon, lat).
    Folium espera [lat, lon], así que invertimos el orden de salida.
    """
    resultado = []
    for par in coords_utm:
        x_utm, y_utm = par[0], par[1]          # easting, northing
        lon, lat = _PROYECTOR.transform(x_utm, y_utm)
        resultado.append([lat, lon])            # Folium: [lat, lon]
    return resultado


def _centroide_wgs84(coords_wgs84: list[list[float]]) -> tuple[float, float]:
    """Calcula el centroide simple de un polígono en WGS84."""
    lats = [c[0] for c in coords_wgs84]
    lons = [c[1] for c in coords_wgs84]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

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

    /* ---- Pill de estado de validación ---- */
    .pill-pendiente {
        display:inline-block;background:rgba(255,220,0,0.15);
        border:1px solid rgba(255,220,0,0.4);color:#ffdc00;
        border-radius:20px;padding:2px 10px;font-size:0.72rem;font-weight:600;
    }
    .pill-validado {
        display:inline-block;background:rgba(46,204,64,0.15);
        border:1px solid rgba(46,204,64,0.4);color:#2ecc40;
        border-radius:20px;padding:2px 10px;font-size:0.72rem;font-weight:600;
    }
    .pill-falso {
        display:inline-block;background:rgba(128,128,128,0.15);
        border:1px solid rgba(128,128,128,0.4);color:#aaaaaa;
        border-radius:20px;padding:2px 10px;font-size:0.72rem;font-weight:600;
    }

    /* ---- Panel de validación ---- */
    .val-panel {
        background: rgba(17,35,24,0.85);
        border: 1px solid var(--borde);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* ---- Sección de mapa ---- */
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

    /* ---- Sin alertas ---- */
    .no-alertas {
        background: rgba(17,35,24,0.7);
        border: 1px dashed rgba(45,125,70,0.4);
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        color: #64b5f6;
    }

    /* ---- Botones ---- */
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

    /* ---- GCS status badge ---- */
    .gcs-badge {
        background: rgba(0,180,216,0.08);
        border: 1px solid rgba(0,180,216,0.2);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-size: 0.75rem;
        color: #81c784;
        line-height: 1.6;
    }

    /* ---- Ocultar elementos Streamlit por defecto ---- */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# LECTURA DE ALERTAS (fuente: GCS o local)
# ===========================================================================

def _gsutil_timestamp(ruta_gcs: str) -> str:
    """Obtiene la fecha de actualización de un objeto GCS usando gsutil stat."""
    try:
        resultado = subprocess.run(
            ["gsutil", "stat", ruta_gcs],
            capture_output=True, text=True, timeout=15,
        )
        for linea in resultado.stdout.splitlines():
            if "Update time:" in linea:
                return linea.split("Update time:")[-1].strip()
    except Exception:
        pass
    return "—"


def _leer_alertas_geojson(
    fuente: str = "gcs",
    ruta: pathlib.Path = RUTA_GEOJSON_LOCAL,
) -> tuple[list[dict], str]:
    """
    Lee el GeoJSON de alertas y retorna (alertas, timestamp_str).

    Fuente "gcs": usa gsutil cat (requiere gcloud SDK autenticado, sin deps extra).
    Fuente "local": lee el archivo local (modo desarrollo/fallback).

    Cada dict de alerta tiene:
        id_unico, causa_sugerida, etiqueta_visual,
        pixeles, porcentaje, area_ha,
        coords_wgs84 ([[lat,lon],...]), centroide (lat, lon)
    """
    gj: dict = {}
    timestamp_str: str = "—"

    # --- Rama GCS (via gsutil, sin dependencias extra) ---
    if fuente == "gcs":
        ruta_gcs = f"gs://{GCS_BUCKET}/{GCS_RUTA_GEOJSON}"
        try:
            resultado = subprocess.run(
                ["gsutil", "cat", ruta_gcs],
                capture_output=True, text=True, timeout=30,
            )
            if resultado.returncode != 0:
                logger.warning(
                    "gsutil no pudo leer %s: %s. Usando archivo local.",
                    ruta_gcs, resultado.stderr.strip(),
                )
                fuente = "local"
            else:
                gj = json.loads(resultado.stdout)
                timestamp_str = _gsutil_timestamp(ruta_gcs)

        except subprocess.TimeoutExpired:
            logger.error("Timeout al leer GCS. Usando archivo local.")
            fuente = "local"
        except Exception as exc:
            logger.error("Error al leer GCS: %s. Usando archivo local.", exc)
            fuente = "local"

    # --- Rama local (fallback) ---
    if fuente == "local":
        if not ruta.exists():
            logger.warning("GeoJSON local no encontrado: %s", ruta)
            return [], "archivo no encontrado"
        with open(ruta, encoding="utf-8") as f:
            gj = json.load(f)
        timestamp_str = "archivo local (desarrollo)"

    features = gj.get("features", [])
    if not features:
        return [], timestamp_str

    alertas: list[dict] = []
    for feat in features:
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        tipo  = geom.get("type", "")

        # Extraer geometría — soporta Polygon, MultiPolygon y Point
        if tipo == "Point":
            # GeoJSON: coordenadas ya en WGS84 [lon, lat]
            lon, lat = geom["coordinates"][0], geom["coordinates"][1]
            coords_wgs84 = [[lat, lon]]
            centroide    = (lat, lon)
            is_point     = True
        elif tipo == "Polygon":
            anillo_utm = geom["coordinates"][0]
            try:
                coords_wgs84 = _utm_a_wgs84(anillo_utm)
            except Exception as exc:
                logger.error("Error convirtiendo coordenadas de %s: %s", props.get("id_unico", "?"), exc)
                continue
            centroide = _centroide_wgs84(coords_wgs84)
            is_point  = False
        elif tipo == "MultiPolygon":
            anillo_utm = geom["coordinates"][0][0]
            try:
                coords_wgs84 = _utm_a_wgs84(anillo_utm)
            except Exception as exc:
                logger.error("Error convirtiendo coordenadas de %s: %s", props.get("id_unico", "?"), exc)
                continue
            centroide = _centroide_wgs84(coords_wgs84)
            is_point  = False
        else:
            logger.warning("Geometría ignorada (tipo no soportado): %s", tipo)
            continue

        causa_interna = props.get("causa_sugerida", props.get("causa", "Otros"))
        etiqueta      = _causa_a_etiqueta(causa_interna)

        alertas.append({
            "id_unico":        props.get("id_unico", f"alerta_{len(alertas)+1:03d}"),
            "causa_sugerida":  causa_interna,
            "etiqueta_visual": etiqueta,
            "pixeles":         props.get("pixeles", 0),
            "porcentaje":      props.get("porcentaje", 0.0),
            "area_ha":         props.get("area_ha", 0.0),
            "coords_wgs84":    coords_wgs84,
            "centroide":       centroide,
            "is_point":        is_point,
        })

    logger.info("Alertas cargadas: %d (fuente: %s)", len(alertas), fuente)
    return alertas, timestamp_str


# ===========================================================================
# HELPERS DE MAPA
# ===========================================================================

def _etiqueta_efectiva(alerta: dict, validaciones: dict) -> str:
    """
    Retorna la etiqueta visual efectiva de una alerta,
    teniendo en cuenta la validación humana si existe.
    """
    aid = alerta["id_unico"]
    val = validaciones.get(aid, {})
    if val.get("estado") == "validado" and val.get("etiqueta_validada"):
        return val["etiqueta_validada"]
    return alerta["etiqueta_visual"]


def _estilo_poligono(etiqueta: str, estado: str) -> dict:
    """
    Devuelve dict de estilo para folium.Polygon según etiqueta visual y estado.

    estados: "pendiente" | "validado" | "falso_positivo"
    """
    if estado == "falso_positivo":
        palette = COLOR_FALSO_POSITIVO
    else:
        palette = PALETA_ETIQUETA.get(etiqueta, PALETA_ETIQUETA["Deforestación No Urbana"])

    weight       = 4 if estado == "validado" else 2
    fill_opacity = 0.35 if estado == "falso_positivo" else 0.45

    return {
        "fill_color":   palette["fill"],
        "color":        palette["border"],
        "weight":       weight,
        "fill_opacity": fill_opacity,
    }


def _popup_html(alerta: dict, estado: str, etiqueta: str) -> str:
    """Genera el HTML del popup informativo de cada polígono."""
    palette = (
        COLOR_FALSO_POSITIVO
        if estado == "falso_positivo"
        else PALETA_ETIQUETA.get(etiqueta, PALETA_ETIQUETA["Deforestación No Urbana"])
    )
    color_acento = palette["fill"]
    emoji        = palette.get("emoji", "📍")

    estado_html = {
        "pendiente":      '<span style="background:rgba(255,220,0,0.2);color:#ffdc00;border:1px solid #ffdc00;border-radius:12px;padding:2px 8px;font-size:0.7rem;">⏳ Pendiente</span>',
        "validado":       '<span style="background:rgba(46,204,64,0.2);color:#2ecc40;border:1px solid #2ecc40;border-radius:12px;padding:2px 8px;font-size:0.7rem;">✅ Validado</span>',
        "falso_positivo": '<span style="background:rgba(128,128,128,0.2);color:#aaaaaa;border:1px solid #aaaaaa;border-radius:12px;padding:2px 8px;font-size:0.7rem;">🚫 Falso positivo</span>',
    }.get(estado, "")

    return f"""
    <div style="
        font-family:'Segoe UI',sans-serif;
        background:#111c14;
        color:#e8f5e9;
        border-radius:12px;
        padding:16px 18px;
        min-width:260px;
        border:1px solid {color_acento}55;
        box-shadow:0 4px 24px rgba(0,0,0,0.6);
    ">
        <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;
                    color:{color_acento};font-weight:700;margin-bottom:6px;">
            {emoji} ALERTA DE DEFORESTACIÓN
        </div>
        <div style="font-size:1rem;font-weight:800;color:#ffffff;margin-bottom:4px;">
            {etiqueta}
        </div>
        <div style="font-size:0.72rem;color:#78909c;margin-bottom:8px;">
            Causa interna: {alerta['causa_sugerida']}
        </div>
        <div style="margin-bottom:10px;">{estado_html}</div>
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:8px 0;">
        <table style="font-size:0.8rem;color:#b0bec5;width:100%;border-collapse:collapse;">
            <tr>
                <td style="padding:3px 0;color:#78909c;">🆔 ID</td>
                <td style="padding:3px 0;color:#cfd8dc;font-weight:600;">{alerta['id_unico']}</td>
            </tr>
            <tr>
                <td style="padding:3px 0;color:#78909c;">📐 Área</td>
                <td style="padding:3px 0;color:#cfd8dc;font-weight:600;">{alerta['area_ha']:.2f} ha</td>
            </tr>
            <tr>
                <td style="padding:3px 0;color:#78909c;">🔴 Píxeles</td>
                <td style="padding:3px 0;color:#cfd8dc;font-weight:600;">{alerta['pixeles']:,}</td>
            </tr>
            <tr>
                <td style="padding:3px 0;color:#78909c;">📊 Cobertura</td>
                <td style="padding:3px 0;color:{color_acento};font-weight:700;">{alerta['porcentaje']:.1f}%</td>
            </tr>
        </table>
        <div style="margin-top:10px;font-size:0.7rem;color:#546e7a;
                    border-top:1px solid rgba(255,255,255,0.06);padding-top:8px;">
            ✏️ Usa el panel lateral para validar esta alerta
        </div>
    </div>
    """


def _construir_mapa_con_alertas(
    alertas: list[dict],
    validaciones: dict,
    filtros_activos: dict[str, bool],
) -> folium.Map:
    """
    Construye el mapa Folium con los polígonos de deforestación.

    Solo renderiza polígonos cuya etiqueta_visual esté activa en filtros_activos.
    Los polígonos marcados como falso_positivo siempre se muestran (en gris).
    """
    mapa = folium.Map(
        location=[IQUITOS_LAT, IQUITOS_LON],
        zoom_start=11,
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

    # Etiquetas superpuestas
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors, © CARTO",
        name="Etiquetas",
        overlay=True,
        control=True,
    ).add_to(mapa)

    if not alertas:
        folium.LayerControl(collapsed=False).add_to(mapa)
        return mapa

    # FeatureGroups — uno por cada etiqueta visual + uno para falsos positivos
    grupos: dict[str, folium.FeatureGroup] = {}
    for etiqueta, pal in PALETA_ETIQUETA.items():
        grupos[etiqueta] = folium.FeatureGroup(
            name=f"{pal['emoji']} {etiqueta}",
            show=filtros_activos.get(etiqueta, True),
        )
    grupos["falso_positivo"] = folium.FeatureGroup(
        name="🚫 Falsos Positivos",
        show=True,
    )

    # Bounding box para ajustar zoom automáticamente
    todas_lats, todas_lons = [], []

    for alerta in alertas:
        aid = alerta["id_unico"]
        val = validaciones.get(aid, {})
        estado = val.get("estado", "pendiente")

        # Etiqueta efectiva (validada si existe, sino la sugerida por la IA)
        if estado == "validado" and val.get("etiqueta_validada"):
            etiqueta_eff = val["etiqueta_validada"]
        else:
            etiqueta_eff = alerta["etiqueta_visual"]

        # Aplicar filtro (falsos positivos siempre se muestran)
        if estado != "falso_positivo" and not filtros_activos.get(etiqueta_eff, True):
            continue

        estilo = _estilo_poligono(etiqueta_eff, estado)
        popup  = folium.Popup(
            _popup_html(alerta, estado, etiqueta_eff),
            max_width=310,
        )
        tooltip_txt = (
            f"{PALETA_ETIQUETA.get(etiqueta_eff, COLOR_FALSO_POSITIVO)['emoji']} "
            f"{etiqueta_eff} — {alerta['area_ha']:.2f} ha"
        )

        # Renderizar como CircleMarker (Point) o Polygon según geometría
        grupo_destino = "falso_positivo" if estado == "falso_positivo" else etiqueta_eff
        if alerta.get("is_point", False):
            # Radio proporcional al área (mínimo 5, máximo 20)
            radio = max(5, min(20, int(alerta["area_ha"] * 4 + 5)))
            folium.CircleMarker(
                location=alerta["centroide"],
                radius=radio,
                popup=popup,
                tooltip=tooltip_txt,
                fill=True,
                fill_color=estilo["fill_color"],
                fill_opacity=estilo["fill_opacity"],
                color=estilo["color"],
                weight=estilo["weight"],
            ).add_to(grupos.get(grupo_destino, grupos["falso_positivo"]))
        else:
            folium.Polygon(
                locations=alerta["coords_wgs84"],
                popup=popup,
                tooltip=tooltip_txt,
                fill=True,
                **estilo,
            ).add_to(grupos.get(grupo_destino, grupos["falso_positivo"]))

        # Icono de check para validados
        if estado == "validado":
            folium.Marker(
                location=alerta["centroide"],
                icon=folium.DivIcon(
                    html='<div style="font-size:1.2rem;text-shadow:0 0 4px #000;">✅</div>',
                    icon_size=(24, 24),
                    icon_anchor=(12, 12),
                ),
                tooltip=f"✅ Validado: {etiqueta_eff}",
            ).add_to(grupos.get(grupo_destino, grupos["falso_positivo"]))

        # Acumular para bounding box
        for coord in alerta["coords_wgs84"]:
            todas_lats.append(coord[0])
            todas_lons.append(coord[1])

    # Añadir grupos al mapa
    for fg in grupos.values():
        fg.add_to(mapa)

    # Ajustar bounds si hay polígonos visibles
    if todas_lats:
        mapa.fit_bounds(
            [[min(todas_lats), min(todas_lons)],
             [max(todas_lats), max(todas_lons)]],
            padding=(40, 40),
        )

    folium.LayerControl(collapsed=False).add_to(mapa)
    return mapa


# ===========================================================================
# ESTADO DE SESIÓN
# ===========================================================================
if "validaciones" not in st.session_state:
    st.session_state["validaciones"] = {}


# ===========================================================================
# CARGA DE ALERTAS
# ===========================================================================
@st.cache_data(show_spinner="📡 Descargando alertas desde GCS…", ttl=300)
def _cargar_alertas_cached() -> tuple[list[dict], str]:
    return _leer_alertas_geojson(fuente=FUENTE_DATOS, ruta=RUTA_GEOJSON_LOCAL)


alertas, gcs_timestamp = _cargar_alertas_cached()


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    # Branding
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(0,180,216,0.1), rgba(45,125,70,0.12));
            border: 1px solid rgba(0,180,216,0.2);
            border-radius: 12px;
            padding: 18px 16px;
            margin-bottom: 20px;
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

    # -----------------------------------------------------------------------
    # Estado de conexión GCS
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="gcs-badge">
            ☁️ <b>Fuente:</b> Google Cloud Storage<br>
            🪣 <code>{GCS_BUCKET}</code><br>
            🕒 <b>Última actualización:</b> {gcs_timestamp}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Botón de recarga
    if st.button("🔄 Refrescar predicciones", key="btn_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # -----------------------------------------------------------------------
    # Filtros de visualización — 3 checkboxes
    # -----------------------------------------------------------------------
    st.markdown("#### 🔍 Filtrar por tipo")

    filtros_activos: dict[str, bool] = {}
    for etiqueta, pal in PALETA_ETIQUETA.items():
        filtros_activos[etiqueta] = st.checkbox(
            f"{pal['emoji']}  {etiqueta}",
            value=True,
            key=f"chk_{etiqueta}",
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Panel de validación humana
    # -----------------------------------------------------------------------
    st.markdown("#### ✍️ Validación Humana")

    if not alertas:
        st.info("No hay alertas activas para validar.")
    else:
        ids_alertas = [a["id_unico"] for a in alertas]
        id_seleccionado = st.selectbox(
            "Seleccionar alerta",
            ids_alertas,
            key="sb_alerta_sel",
            help="Elige el ID del polígono que deseas validar en el mapa.",
        )

        alerta_sel = next(
            (a for a in alertas if a["id_unico"] == id_seleccionado), None
        )

        if alerta_sel:
            val_actual     = st.session_state["validaciones"].get(id_seleccionado, {})
            estado_actual  = val_actual.get("estado", "pendiente")
            etiqueta_actual = val_actual.get("etiqueta_validada", alerta_sel["etiqueta_visual"])
            pal_sel         = PALETA_ETIQUETA.get(etiqueta_actual, PALETA_ETIQUETA["Deforestación No Urbana"])

            # Info rápida de la alerta seleccionada
            st.markdown(
                f"""
                <div class="val-panel">
                    <div style="font-size:0.7rem;color:#78909c;text-transform:uppercase;
                                letter-spacing:1.5px;margin-bottom:8px;">Alerta seleccionada</div>
                    <div style="font-weight:700;font-size:0.95rem;color:#e8f5e9;">
                        {pal_sel['emoji']} {id_seleccionado}
                    </div>
                    <div style="font-size:0.8rem;color:#81c784;margin-top:4px;">
                        IA sugiere: <b>{alerta_sel['etiqueta_visual']}</b> &nbsp;·&nbsp;
                        {alerta_sel['area_ha']:.2f} ha &nbsp;·&nbsp;
                        {alerta_sel['porcentaje']:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Corrección de etiqueta visual (3 opciones)
            idx_etiqueta = ETIQUETAS_VALIDAS.index(etiqueta_actual) \
                if etiqueta_actual in ETIQUETAS_VALIDAS else 0

            etiqueta_corregida = st.selectbox(
                "Tipo a validar",
                ETIQUETAS_VALIDAS,
                index=idx_etiqueta,
                key="sb_etiqueta_val",
            )

            comentario = st.text_input(
                "Comentario (opcional)",
                value=val_actual.get("comentario", ""),
                placeholder="Ej: confirmado con imagen Planet...",
                key="txt_comentario",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Confirmar", key="btn_confirmar", use_container_width=True):
                    st.session_state["validaciones"][id_seleccionado] = {
                        "estado":           "validado",
                        "etiqueta_validada": etiqueta_corregida,
                        "comentario":       comentario,
                    }
                    st.success(f"Alerta {id_seleccionado} validada.")
                    st.rerun()

            with col_b:
                if st.button("🚫 Falso positivo", key="btn_falso", use_container_width=True):
                    st.session_state["validaciones"][id_seleccionado] = {
                        "estado":           "falso_positivo",
                        "etiqueta_validada": alerta_sel["etiqueta_visual"],
                        "comentario":       comentario,
                    }
                    st.warning(f"Alerta {id_seleccionado} marcada como falso positivo.")
                    st.rerun()

            # Estado actual
            estados_pill = {
                "pendiente":      "⏳ Pendiente",
                "validado":       "✅ Validado",
                "falso_positivo": "🚫 Falso positivo",
            }
            st.markdown(
                f"<div style='font-size:0.75rem;color:#546e7a;margin-top:6px;'>"
                f"Estado actual: <b style='color:#81c784'>"
                f"{estados_pill.get(estado_actual, estado_actual)}</b></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # -----------------------------------------------------------------------
    # Leyenda de colores (informativa)
    # -----------------------------------------------------------------------
    st.markdown("#### 📋 Leyenda")
    for etiqueta, pal in PALETA_ETIQUETA.items():
        st.markdown(
            f"""<div style="
                display:flex;align-items:center;gap:10px;
                padding:7px 10px;margin-bottom:5px;
                background:rgba(17,35,24,0.7);
                border-left:3px solid {pal['fill']};
                border-radius:0 8px 8px 0;
                font-size:0.82rem;
            ">
                <span>{pal['emoji']}</span>
                <div>
                    <div style="color:#e8f5e9;font-weight:600;">{etiqueta}</div>
                    <div style="color:{pal['fill']};font-size:0.7rem;">■ {pal['fill']}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""<div style="
            display:flex;align-items:center;gap:10px;
            padding:7px 10px;margin-bottom:5px;
            background:rgba(17,35,24,0.7);
            border-left:3px solid #808080;
            border-radius:0 8px 8px 0;
            font-size:0.82rem;
        ">
            <span>🚫</span>
            <div>
                <div style="color:#e8f5e9;font-weight:600;">Falso Positivo</div>
                <div style="color:#808080;font-size:0.7rem;">■ #808080</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Exportar validaciones
    # -----------------------------------------------------------------------
    st.markdown("#### 📥 Exportar")

    export_payload = {
        "fuente_gcs":       f"gs://{GCS_BUCKET}/{GCS_RUTA_GEOJSON}",
        "ultima_actualizacion": gcs_timestamp,
        "validaciones":     st.session_state.get("validaciones", {}),
        "total_alertas":    len(alertas),
        "alertas_validadas": sum(
            1 for v in st.session_state.get("validaciones", {}).values()
            if v.get("estado") == "validado"
        ),
        "falsos_positivos": sum(
            1 for v in st.session_state.get("validaciones", {}).values()
            if v.get("estado") == "falso_positivo"
        ),
    }

    st.download_button(
        label="📥 Exportar validaciones a JSON",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2),
        file_name="validaciones_loreto.json",
        mime="application/json",
        use_container_width=True,
        key="btn_exportar",
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.7rem;color:#546e7a;text-align:center;line-height:1.6;'>"
        "CRS: EPSG:32718 (UTM 18S)<br>"
        f"Resolución: {RESOLUCION_M} m/px<br>"
        "Modelo: U-Net Loreto v4 · GCS"
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
            mediante Deep Learning (U-Net) y análisis geoespacial en tiempo real desde Google Cloud.
        </div>
        <span class="hero-badge">🛰️ Sentinel-2 · UTM 18S · U-Net v4 · GCS · Validación Humana</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# MÉTRICAS
# ===========================================================================
validaciones    = st.session_state.get("validaciones", {})
n_alertas       = len(alertas)
n_validadas     = sum(1 for v in validaciones.values() if v.get("estado") == "validado")
n_falsopositivo = sum(1 for v in validaciones.values() if v.get("estado") == "falso_positivo")
n_pendientes    = n_alertas - n_validadas - n_falsopositivo
area_total_ha   = sum(a["area_ha"] for a in alertas)

# Distribución por etiqueta visual efectiva
dist_etiqueta: dict[str, int] = {}
for a in alertas:
    val = validaciones.get(a["id_unico"], {})
    if val.get("estado") == "validado" and val.get("etiqueta_validada"):
        etiq_eff = val["etiqueta_validada"]
    else:
        etiq_eff = a["etiqueta_visual"]
    dist_etiqueta[etiq_eff] = dist_etiqueta.get(etiq_eff, 0) + 1

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🚨</div>
            <div class="metric-label">Total Alertas</div>
            <div class="metric-value" style="color:#ff4136;">{n_alertas}</div>
            <div class="metric-sub">{n_pendientes} pendientes · {n_validadas} validadas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📐</div>
            <div class="metric-label">Área Total Deforestada</div>
            <div class="metric-value" style="color:#ff851b;">{area_total_ha:.1f}</div>
            <div class="metric-sub">hectáreas detectadas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    etiqueta_top = max(dist_etiqueta, key=dist_etiqueta.get) if dist_etiqueta else "—"
    pal_top = PALETA_ETIQUETA.get(etiqueta_top, {"fill": "#546e7a", "emoji": "❓"})
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{pal_top['emoji']}</div>
            <div class="metric-label">Tipo Predominante</div>
            <div class="metric-value" style="font-size:1.1rem;color:{pal_top['fill']};">
                {etiqueta_top}
            </div>
            <div class="metric-sub">{dist_etiqueta.get(etiqueta_top, 0)} alerta(s)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    pct_validado = round(100.0 * (n_validadas + n_falsopositivo) / n_alertas, 1) if n_alertas else 0.0
    color_pct = "#2ecc40" if pct_validado >= 80 else ("#ff851b" if pct_validado >= 40 else "#ff4136")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">✍️</div>
            <div class="metric-label">Progreso de Validación</div>
            <div class="metric-value" style="color:{color_pct};">{pct_validado:.0f}%</div>
            <div class="metric-sub">{n_falsopositivo} falsos positivos descartados</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# MAPA INTERACTIVO
# ===========================================================================
st.markdown(
    """
    <div style="margin-top:24px;">
        <div class="map-title">
            🗺️ Mapa de Alertas — Región Loreto (polígonos exactos de deforestación)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not alertas:
    st.markdown(
        """
        <div class="no-alertas">
            <div style="font-size:3rem;margin-bottom:12px;">🌳</div>
            <div style="font-size:1rem;font-weight:700;margin-bottom:8px;color:#64b5f6;">
                No hay alertas activas
            </div>
            <div style="font-size:0.85rem;opacity:0.75;">
                El archivo GeoJSON no fue encontrado en GCS o está vacío.<br>
                Ejecuta el pipeline de inferencia en Colab para generar nuevas predicciones.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    mapa = _construir_mapa_con_alertas(
        alertas,
        st.session_state["validaciones"],
        filtros_activos,
    )
    st_folium(
        mapa,
        width="100%",
        height=560,
        returned_objects=[],
        key="mapa_loreto_poligonos",
    )
