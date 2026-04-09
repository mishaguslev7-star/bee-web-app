import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="BeeTracker AI | Dark Edition", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. DARK TECHNOLOGY DESIGN (Сетка, Неоновые акценты) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@800&family=Montserrat:wght@300;400;700&display=swap');
    
    header, footer, #MainMenu {visibility: hidden !important;}

    /* Темная тема и паттерн сот */
    .stApp {
        background-color: #0F172A; /* Глубокий темно-синий */
        background-image: url('https://www.transparenttextures.com/patterns/honey-comb.png');
        background-attachment: fixed;
        color: #F8FAFC;
    }

    /* Ограничение ширины и сетка */
    .main .block-container {
        max-width: 1200px;
        padding: 3rem 1rem !important;
        gap: 2rem;
    }

    /* BeeTracker - Неоновое свечение */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(2.5rem, 8vw, 4.5rem);
        color: #2DD4BF; /* Яркий бирюзовый */
        text-transform: uppercase;
        letter-spacing: 5px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 20px;
        text-shadow: 0 0 20px rgba(45, 212, 191, 0.4);
    }

    /* Анимация пчелы */
    .floating-bee {
        display: inline-block;
        filter: drop-shadow(0 0 10px #2DD4BF);
        animation: bee-float 3s ease-in-out infinite;
    }
    @keyframes bee-float {
        0%, 100% { transform: translateY(0) rotate(-5deg); }
        50% { transform: translateY(-15px) rotate(10deg); }
    }

    /* Подзаголовок */
    .sub-title {
        font-family: 'Montserrat', sans-serif;
        font-size: clamp(1rem, 3vw, 1.4rem);
        color: #94A3B8;
        font-weight: 300;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    .highlight {
        color: #5EEAD4;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(94, 234, 212, 0.3);
    }

    /* Темные карточки с неоновым бордером */
    .info-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid rgba(45, 212, 191, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
    }
    .info-card:hover {
        transform: translateY(-10px);
        border-color: #2DD4BF;
        box-shadow: 0 10px 30px -10px rgba(45, 212, 191, 0.3);
        background: rgba(30, 41, 59, 0.9);
    }
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
        filter: drop-shadow(0 0 8px rgba(45, 212, 191, 0.5));
    }

    /* Стилизация загрузчика для темной темы */
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #2DD4BF !important;
        background: rgba(15, 23, 42, 0.8) !important;
        border-radius: 20px !important;
        padding: 2.5rem !important;
        color: #94A3B8 !important;
    }

    /* Метрики и текст */
    [data-testid="stMetricValue"] {
        color: #5EEAD4 !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        letter-spacing: 1px;
    }
    
    h3 { color: #5EEAD4 !important; margin-top: 0; }
    p { color: #CBD5E1; }

    /* Кнопка сохранения */
    .stDownloadButton button {
        background: linear-gradient(135deg, #2DD4BF, #0D9488) !important;
        color: #0F172A !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

# --- 4. HEADER ---
st.markdown("""
    <div class="main-title">
        <span class="floating-bee">🐝</span> BeeTracker
    </div>
    <div class="sub-title">
        Интеллектуальная система <span class="highlight">мониторинга пчел</span> 
        нового поколения.
    </div>
""", unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
col_up, col_info = st.columns([1.5, 1], gap="large")

with col_up:
    st.markdown("### 📥 Входные данные")
    file = st.file_uploader("upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if file:
        img = Image.open(file)
        if model:
            with st.spinner('Нейросеть сканирует кадр...'):
                results = model(img)[0]
                annotated_img = results.plot(masks=False, kpts=False)
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img, use_container_width=True)
                count = len(results.boxes)
        else:
            st.error("Ошибка: Файл весов нейросети не найден.")

with col_info:
    if file and model:
        st.markdown("### 📈 Аналитика")
        st.metric("ОБНАРУЖЕНО ПЧЕЛ", f"{count}")
        
        st.markdown(f"""
        <div class="info-card">
            <span class="card-icon">⚡</span>
            <h3>Экспресс-отчет</h3>
            На основе анализа снимка определена 
            <b>{"высокая" if count > 25 else "умеренная"}</b> 
            интенсивность работы семьи. Система зафиксировала все активные объекты.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("💾 Экспорт данных", res_bytes, "bee_analysis_dark.jpg", use_container_width=True)
    else:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); padding: 3rem; border-radius: 24px; border: 2px dashed rgba(148, 163, 184, 0.2); text-align: center;">
            <span style="font-size: 4rem; opacity: 0.3;">📷</span><br><br>
            <p style="color: #64748B;">Загрузите снимок рамки или летка для начала работы AI-модуля</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- 6. FEATURE GRID ---
st.markdown("## 🔍 Технологический стек")
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🏗️</span>
        <h3>Масштабируемость</h3>
        Система готова к интеграции на промышленных пасеках с тысячами ульев. Автоматизация в разы снижает риск потери семей.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🐝</span>
        <h3>Био-безопасность</h3>
        Контроль популяции в реальном времени позволяет выявлять признаки <b>коллапса семей</b> на самых ранних стадиях.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🤖</span>
        <h3>YOLO11 Core</h3>
        Сердце проекта — самая современная архитектура детекции объектов, обеспечивающая рекордную скорость обработки кадров.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-top: 5rem; color: #475569; font-size: 0.9rem; letter-spacing: 1px;">
        BEETRACKER AI • DARK EDITION • 2026
    </div>
""", unsafe_allow_html=True)
