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

# --- 2. DARK TECHNOLOGY DESIGN (UI/UX) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@800&family=Montserrat:wght@300;400;700&display=swap');
    
    header, footer, #MainMenu {visibility: hidden !important;}

    /* Фон и паттерн сот */
    .stApp {
        background-color: #0F172A;
        background-image: url('https://www.transparenttextures.com/patterns/honey-comb.png');
        background-attachment: fixed;
        color: #F8FAFC;
    }

    .main .block-container {
        max-width: 1200px;
        padding: 2.5rem 1rem !important;
        gap: 1.5rem;
    }

    /* Название BeeTracker (Space Ranger Style) */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(2.2rem, 7vw, 4rem);
        color: #2DD4BF;
        text-transform: uppercase;
        letter-spacing: 6px;
        display: flex;
        align-items: center;
        gap: 15px;
        text-shadow: 0 0 25px rgba(45, 212, 191, 0.4);
        margin-bottom: 0.2rem;
    }

    .floating-bee {
        display: inline-block;
        filter: drop-shadow(0 0 10px #2DD4BF);
        animation: bee-float 3s ease-in-out infinite;
    }
    @keyframes bee-float {
        0%, 100% { transform: translateY(0) rotate(-5deg); }
        50% { transform: translateY(-12px) rotate(10deg); }
    }

    .sub-title {
        font-family: 'Montserrat', sans-serif;
        font-size: clamp(1.1rem, 2.5vw, 1.4rem);
        color: #94A3B8;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    .highlight {
        color: #5EEAD4;
        font-weight: 700;
    }

    /* Блок ЦЕЛИ */
    .goal-card {
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.1), rgba(15, 23, 42, 0.9));
        padding: 2rem;
        border-radius: 24px;
        border: 2px solid #2DD4BF;
        margin-bottom: 3rem;
        box-shadow: 0 0 40px rgba(45, 212, 191, 0.05);
    }

    /* Общий стиль карточек с анимацией */
    .info-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.8rem;
        border-radius: 24px;
        border: 1px solid rgba(45, 212, 191, 0.1);
        backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .info-card:hover {
        transform: translateY(-10px);
        border-color: #2DD4BF;
        box-shadow: 0 15px 35px -10px rgba(45, 212, 191, 0.3);
        background: rgba(30, 41, 59, 0.9);
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }

    /* Загрузчик */
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #2DD4BF !important;
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 20px !important;
    }

    /* Метрики */
    [data-testid="stMetricValue"] {
        color: #2DD4BF !important;
        font-weight: 800 !important;
        font-family: 'Orbitron', sans-serif;
    }

    h2 { color: #5EEAD4 !important; font-family: 'Orbitron', sans-serif; letter-spacing: 2px; margin-top: 2rem !important; }
    h3 { color: #5EEAD4 !important; font-size: 1.4rem !important; margin-bottom: 12px !important; }
    p { color: #CBD5E1; line-height: 1.6; }
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

# --- 4. HEADER & GOAL ---
st.markdown("""
    <div class="main-title">
        <span class="floating-bee">🐝</span> BeeTracker
    </div>
    <div class="sub-title">
        Интеллектуальная система <span class="highlight">мониторинга пчел</span> нового поколения.
    </div>
    
    <div class="goal-card">
        <h3>🎯 Цель проекта</h3>
        <p style="font-size: 1.2rem; margin: 0;">
            Создание высокоточного инструмента для автоматизированного контроля популяции пчел. 
            Проект направлен на <b>предотвращение гибели пчелиных семей</b> и цифровую трансформацию традиционного пчеловодства с помощью ИИ.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
col_up, col_info = st.columns([1.4, 1], gap="large")

with col_up:
    st.markdown("### 📥 Анализ изображения")
    file = st.file_uploader("upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if file:
        img = Image.open(file)
        if model:
            with st.spinner('Нейросеть обрабатывает данные...'):
                results = model(img)[0]
                annotated_img = results.plot(masks=False, kpts=False)
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img, use_container_width=True)
                count = len(results.boxes)
        else:
            st.error("Критическая ошибка: Файл весов 'best.pt' не найден.")

with col_info:
    if file and model:
        st.markdown("### 📊 Результаты")
        st.metric("НАЙДЕНО ОСОБЕЙ", f"{count}")
        
        st.markdown(f"""
        <div class="info-card">
            <span class="card-icon">⚡</span>
            <h3>Отчет системы</h3>
            Детекция завершена. Плотность заселения рамки: 
            <b>{"высокая" if count > 25 else "требует внимания"}</b>. 
            Данные готовы к экспорту.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("💾 Скачать результат", res_bytes, "analysis_report.jpg", use_container_width=True)
    else:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); padding: 3rem; border-radius: 24px; border: 2px dashed rgba(148, 163, 184, 0.2); text-align: center;">
            <span style="font-size: 4rem; opacity: 0.2;">📷</span><br>
            <p style="color: #64748B; margin-top: 1rem;">Загрузите фото для автоматического подсчёта</p>
        </div>
        """, unsafe_allow_html=True)

# --- 6. БЛОК ПЕРСПЕКТИВЫ (ФИНАЛЬНЫЙ) ---
st.markdown("## 🚀 ПЕРСПЕКТИВЫ РАЗВИТИЯ")
p1, p2, p3 = st.columns(3, gap="medium")

with p1:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🔄</span>
        <h3>Real-time Tracking</h3>
        Оптимизация YOLO11 для работы с видеопотоком. Это позволит отслеживать динамику вылета и прилета пчел в реальном времени.
    </div>
    """, unsafe_allow_html=True)

with p2:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">📱</span>
        <h3>Мобильная экосистема</h3>
        Разработка автономного мобильного приложения для работы на удаленных пасеках без необходимости доступа к интернету.
    </div>
    """, unsafe_allow_html=True)

with p3:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🔬</span>
        <h3>Deep Diagnostics</h3>
        Дообучение нейросети на распознавание признаков болезней (например, варроатоза) и классификацию различных подвидов пчел.
    </div>
    """, unsafe_allow_html=True)

# Футер
st.markdown("""
    <div style="text-align: center; margin-top: 5rem; color: #475569; font-size: 0.9rem; letter-spacing: 2px;">
        BEETRACKER AI • DARK EDITION • 2026
    </div>
""", unsafe_allow_html=True)
