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

# --- 2. DARK UI/UX DESIGN ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@800&family=Montserrat:wght@300;400;700&display=swap');
    
    header, footer, #MainMenu {visibility: hidden !important;}

    .stApp {
        background-color: #0F172A;
        background-image: url('https://www.transparenttextures.com/patterns/honey-comb.png');
        background-attachment: fixed;
        color: #F8FAFC;
    }

    .main .block-container {
        max-width: 1200px;
        padding: 2rem 1rem !important;
    }

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);
        color: #2DD4BF;
        text-transform: uppercase;
        letter-spacing: 4px;
        display: flex;
        align-items: center;
        gap: 15px;
        text-shadow: 0 0 20px rgba(45, 212, 191, 0.3);
    }

    .section-header {
        font-family: 'Orbitron', sans-serif;
        color: #5EEAD4;
        letter-spacing: 2px;
        margin: 3rem 0 1.5rem 0;
        border-left: 5px solid #2DD4BF;
        padding-left: 15px;
        font-size: 1.5rem;
    }

    .info-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.8rem;
        border-radius: 24px;
        border: 1px solid rgba(45, 212, 191, 0.1);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        height: 100%;
    }
    .info-card:hover {
        transform: translateY(-5px);
        border-color: #2DD4BF;
    }

    .goal-card {
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.1), rgba(15, 23, 42, 0.9));
        padding: 2rem;
        border-radius: 24px;
        border: 2px solid #2DD4BF;
        margin-bottom: 2rem;
    }

    h3 { color: #5EEAD4 !important; font-size: 1.2rem !important; margin-bottom: 10px !important; }
    p { color: #CBD5E1; line-height: 1.6; margin: 0; }
    b { color: #5EEAD4; }

    /* Стилизация загрузчика */
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #2DD4BF !important;
        background: rgba(30, 41, 59, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. БЕЗОПАСНАЯ ЛОГИКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    # Проверка на существование и размер файла (исправление EOFError)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            return YOLO(model_path)
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
    return None

model = load_model()

# --- 4. ШАПКА ---
st.markdown("""
    <div class="main-title">
        <span style="font-size: 1.2em;">🐝</span> Пчелиный Учёт
    </div>
    <p style="color: #94A3B8; margin-bottom: 2rem;">Интеллектуальный мониторинг экосистемы пасеки</p>
    
    <div class="goal-card">
        <h3>🎯 Цель проекта</h3>
        <p>Создание системы автоматического контроля за пчелами, которая помогает вовремя заметить экологическую угрозу и спасти пасеку от гибели.</p>
    </div>
""", unsafe_allow_html=True)

# --- 5. АНАЛИЗ ---
col_up, col_res = st.columns([1.4, 1], gap="large")

with col_up:
    st.markdown("### 📥 Анализ изображения")
    file = st.file_uploader("Загрузите фото", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    if file:
        img = Image.open(file)
        if model:
            results = model(img)[0]
            # Исправленный вызов plot для исключения TypeError
            annotated_img = results.plot(labels=True, boxes=True)
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img, use_container_width=True)
            count = len(results.boxes)
        else:
            st.warning("⚠️ Модель не загружена или файл 'best.pt' пуст.")

with col_res:
    if file and model:
        st.metric("ОБНАРУЖЕНО ПЧЕЛ", f"{count}")
        st.markdown(f"""
        <div class="info-card">
            <h3>📊 Текущий отчёт</h3>
            На снимке зафиксировано <b>{count}</b> особей. 
            Это данные, которые невозможно получить вручную так быстро.
        </div>
        """, unsafe_allow_html=True)

# --- 6. НОВЫЙ БЛОК: О ПРОЕКТЕ (УПРОЩЕННО) ---
st.markdown('<div class="section-header">🔍 О ПРОЕКТЕ</div>', unsafe_allow_html=True)
a1, a2 = st.columns(2, gap="medium")

with a1:
    st.markdown("""
    <div class="info-card">
        <h3>Почему это важно?</h3>
        Пчела летает со скоростью до <b>30 км/ч</b>. Человеческий глаз не способен точно посчитать 50–100 быстро движущихся особей. 
        Владельцам сотен ульев физически невозможно заглядывать в каждый ежедневно — без автоматизации проблему замечают слишком поздно.
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="info-card">
        <h3>Как мы это решаем?</h3>
        Наш проект превращает хаотичное движение пчел в ценные данные. 
        Используя нейросеть <b>YOLO11</b>, система мгновенно фиксирует активность на летке, заменяя ручной труд точным компьютерным зрением.
    </div>
    """, unsafe_allow_html=True)

# --- 7. ПЕРСПЕКТИВЫ ---
st.markdown('<div class="section-header">🚀 ПЕРСПЕКТИВЫ РАЗВИТИЯ</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns(3, gap="medium")

with p1:
    st.markdown("""
    <div class="info-card">
        <h3>Скорость и точность</h3>
        Оптимизация нейросети для работы на более высоких скоростях и с улучшенной точностью трекинга.
    </div>
    """, unsafe_allow_html=True)

with p2:
    st.markdown("""
    <div class="info-card">
        <h3>Мобильное приложение</h3>
        Разработка версии для смартфонов, чтобы пчеловод мог навести камеру на леток и мгновенно получить отчет.
    </div>
    """, unsafe_allow_html=True)

with p3:
    st.markdown("""
    <div class="info-card">
        <h3>Диагностика</h3>
        Дообучение модели для распознавания разных подвидов пчел и определения признаков болезней.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><p style='text-align: center; color: #475569; font-size: 0.8rem;'>НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ • 2026</p>", unsafe_allow_html=True)
