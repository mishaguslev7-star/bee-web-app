import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="BeeTraker AI | Бирюзовый мониторинг", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. БИРЮЗОВАЯ ТЕМА И ФИКС ИНТЕРФЕЙСА ---
st.markdown("""
    <style>
    /* Прячем служебные элементы Streamlit */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    .stDeployButton {display:none !important;}

    /* Шрифты и фон */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    .stApp {
        background-color: #F0FDFA; /* Очень легкий бирюзовый оттенок фона */
    }

    /* ФИКС КНОПКИ: убираем черное перекрытие */
    div[data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #14B8A6 !important;
        border-radius: 20px !important;
        z-index: 100 !important;
        position: relative !important;
    }

    /* Заголовки (Цвета: Глубокий бирюзовый #0F766E) */
    h1, h2, h3 {
        color: #0F766E !important;
        font-weight: 800 !important;
    }
    
    h1 {
        font-size: 3.2rem !important;
        border-left: 12px solid #14B8A6;
        padding-left: 1.5rem;
        margin-bottom: 0.5rem !important;
    }

    /* Карточки-контейнеры */
    .info-card {
        background: #FFFFFF;
        padding: 1.8rem;
        border-radius: 24px;
        border: 1px solid #CCFBF1;
        border-top: 6px solid #14B8A6;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.1);
        height: 100%;
        color: #134E4A;
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
    }

    /* Кнопки */
    .stDownloadButton button {
        background: linear-gradient(135deg, #14B8A6, #0D9488) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 0 #0F766E !important;
    }

    hr {
        border-top: 1px solid #99F6E4;
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
st.markdown("<p style='color: #0D9488; font-weight: 600; letter-spacing: 2px;'>🪐 НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ</p>", unsafe_allow_html=True)

col_t, col_l = st.columns([4, 1])
with col_t:
    st.title("БИО-МОНИТОРИНГ")
    st.markdown("<p style='font-size: 1.2rem; color: #115E59; margin-top: -10px;'>Автоматизированная детекция пчелиных семей на базе YOLO11</p>", unsafe_allow_html=True)
with col_l:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #14B8A6, #0D9488); color: white; padding: 12px; border-radius: 15px; text-align: center; font-weight: 800; margin-top: 10px; box-shadow: 0 4px 0 #0F766E;">
        AI ТРЕКЕР
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 5. РАБОЧАЯ ЗОНА ---
col_up, col_res = st.columns([1.5, 1])

with col_up:
    st.markdown("### 📸 Анализ снимка")
    file = st.file_uploader("upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed", key="bee_uploader_fixed")

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
        st.metric(label="ОБНАРУЖЕНО ПЧЁЛ", value=f"{count} шт.")
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Аналитический вывод")
        if count == 0:
            st.error("Объекты не обнаружены. Попробуйте другой снимок.")
        elif count < 15:
            st.warning("⚠️ Низкая активность. Требуется наблюдение.")
        else:
            st.success("✅ Семья активна. Показатели в норме.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Сохранить результат", res_bytes, "bee_analysis.jpg", "image/jpeg", use_container_width=True)

elif model is None:
    st.error("Файл 'best.pt' не найден в корневой папке.")
else:
    with col_res:
        st.info("👈 Загрузите фотографию для запуска системы автоматического подсчёта.")

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. ИНФОРМАЦИЯ ИЗ ПРЕЗЕНТАЦИИ ---
st.header("🔬 ПОДРОБНОСТИ ПРОЕКТА")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>💡 Решение</h3>
        Наш сервис превращает хаотичное движение пчел в ценные данные. Мы используем нейросеть <b>YOLO11</b>, чтобы мгновенно фиксировать до 100 особей на одном кадре.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>🛡️ Безопасность</h3>
        Своевременный учёт позволяет предотвратить <b>Коллапс пчелиных семей</b>. Пчеловод может заметить ослабление колонии на ранней стадии и спасти её.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>👥 Для кого?</h3>
        • Пчеловоды и фермеры<br>
        • Учёные-энтомологи<br>
        • Экологические организации<br>
        • Научные институты
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #0D9488; opacity: 0.7;'>Всероссийский фестиваль • 2026</p>", unsafe_allow_html=True)
