import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="BeeTracker AI", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. БИРЮЗОВАЯ ТЕМА + ПЛАВНАЯ АНИМАЦИЯ ---
st.markdown("""
    <style>
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    .stDeployButton {display:none !important;}

    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    .stApp {
        background-color: #F0FDFA;
    }

    /* ФИКС КНОПКИ: прозрачный оверлей больше не мешает */
    div[data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #14B8A6 !important;
        border-radius: 20px !important;
        z-index: 100 !important;
        position: relative !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: #0D9488 !important;
        background-color: #F9FFFE !important;
        transform: translateY(-2px);
    }

    /* Заголовки */
    h1, h2, h3 {
        color: #0F766E !important;
        font-weight: 800 !important;
    }
    
    h1 {
        font-size: 3.2rem !important;
        border-left: 12px solid #14B8A6;
        padding-left: 1.5rem;
    }

    /* АНИМИРОВАННЫЕ КАРТОЧКИ */
    .info-card {
        background: #FFFFFF;
        padding: 1.8rem;
        border-radius: 24px;
        border: 1px solid #CCFBF1;
        border-top: 6px solid #14B8A6;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.05);
        height: 100%;
        color: #134E4A;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Плавный вылет */
    }

    .info-card:hover {
        transform: translateY(-10px); /* Подъем при наведении */
        box-shadow: 0 15px 30px rgba(20, 184, 166, 0.15);
        border-color: #5EEAD4;
        background: #F9FFFE;
    }

    /* Метрики */
    [data-testid="stMetricValue"] {
        color: #14B8A6 !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
    }
    
    .stMetric {
        background: white !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border: 1px solid #CCFBF1 !important;
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: scale(1.02);
    }

    /* Кнопки */
    .stDownloadButton button {
        background: linear-gradient(135deg, #14B8A6, #0D9488) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 0 #0F766E !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 0 #0D9488 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path): return None
    try: return YOLO(model_path)
    except: return None

model = load_model()

# --- 4. ШАПКА ---
st.markdown("<p style='color: #0D9488; font-weight: 600; letter-spacing: 2px; margin-bottom: 0;'>🪐 НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ</p>", unsafe_allow_html=True)

col_t, col_l = st.columns([4, 1.2])
with col_t:
    # Заменили название на BeeTracker с иконкой
    st.markdown("<h1>🐝 BeeTracker</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #115E59; margin-top: 5px;'>Интеллектуальная система мониторинга пчелиных семей</p>", unsafe_allow_html=True)
with col_l:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #14B8A6, #0D9488); color: white; padding: 12px; border-radius: 15px; text-align: center; font-weight: 800; margin-top: 15px; box-shadow: 0 4px 0 #0F766E;">
        AI CORE v11
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 5. РАБОЧАЯ ЗОНА ---
col_up, col_res = st.columns([1.5, 1])

with col_up:
    st.markdown("### 📸 Анализ снимка")
    file = st.file_uploader("upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed", key="bt_uploader")

if file is not None and model is not None:
    img = Image.open(file)
    with st.spinner('Нейросеть обрабатывает изображение...'):
        results = model(img)[0]
        annotated_img = results.plot(masks=False, kpts=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        count = len(results.boxes)
    
    with col_up:
        st.image(annotated_img, use_container_width=True)
    
    with col_res:
        st.markdown("### 📊 Статистика")
        st.metric(label="ОБНАРУЖЕНО ОСОБЕЙ", value=f"{count}")
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Аналитический вывод")
        if count == 0:
            st.error("Пчелы не обнаружены. Проверьте качество фото.")
        elif count < 15:
            st.warning("⚠️ Низкая активность. Требуется проверка улья.")
        else:
            st.success("✅ Семья в отличной форме. Плотность в норме.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Сохранить отчет", res_bytes, "bee_tracker_result.jpg", "image/jpeg", use_container_width=True)
else:
    with col_res:
        st.info("👈 Загрузите фотографию рамки или летка для автоматического подсчёта.")

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. ИНФОРМАЦИОННЫЕ БЛОКИ (С АНИМАЦИЕЙ) ---
st.header("🔬 О ПРОЕКТЕ")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>🚨 Актуальность</h3>
        Владельцам сотен ульев физически невозможно заглядывать в каждый ежедневно. 
        <b>BeeTracker</b> позволяет вовремя заметить экологическую угрозу или болезнь.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Технологии</h3>
        Пчела движется со скоростью до 30 км/ч. Наш AI на базе <b>YOLO11</b> фиксирует то, что пропускает человеческий глаз, экономя часы ручного труда.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Перспективы</h3>
        В планах — дообучение модели на распознавание <b>варроатоза</b> и запуск мобильного приложения для работы прямо на пасеке без интернета.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #0D9488; opacity: 0.6;'>BeeTracker • Всероссийский фестиваль 2026</p>", unsafe_allow_html=True)
