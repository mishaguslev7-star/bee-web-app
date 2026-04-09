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

# --- 2. УЛУЧШЕННЫЙ ДИЗАЙН (Сетка, Анимации, Адаптивность) ---
st.markdown("""
    <style>
    /* Подключение современных шрифтов */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@800&family=Montserrat:wght@300;400;700&display=swap');
    
    header, footer, #MainMenu {visibility: hidden !important;}

    /* Фоновый узор из сот */
    .stApp {
        background-color: #F0FDFA;
        background-image: url('https://www.transparenttextures.com/patterns/honey-comb.png');
        background-attachment: fixed;
    }

    /* Сетка и отступы */
    .main .block-container {
        max-width: 1200px;
        padding: 3rem 1rem !important;
        gap: 2rem;
    }

    /* Название BeeTracker (Стилизация под Space Ranger) */
    .main-title {
        font-family: 'Orbitron', sans-serif; /* Технологичный шрифт */
        font-size: clamp(2.5rem, 8vw, 4.5rem);
        color: #0F766E;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 15px;
    }

    /* Анимация пчелы в заголовке */
    .floating-bee {
        display: inline-block;
        animation: bee-bounce 3s ease-in-out infinite;
    }
    @keyframes bee-bounce {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(5deg); }
    }

    /* Подзаголовок */
    .sub-title {
        font-family: 'Montserrat', sans-serif;
        font-size: clamp(1rem, 3vw, 1.4rem);
        color: #334155;
        font-weight: 300;
        margin-bottom: 2rem;
        line-height: 1.4;
    }
    .highlight {
        color: #14B8A6;
        font-weight: 700;
        border-bottom: 2px solid #5EEAD4;
    }

    /* Анимированные карточки с иконками */
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #CCFBF1;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    .info-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 25px -5px rgba(20, 184, 166, 0.1);
        border-color: #14B8A6;
    }
    .card-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        display: block;
    }

    /* Адаптивный загрузчик */
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #14B8A6 !important;
        background: #FFFFFF !important;
        border-radius: 15px !important;
        padding: 2rem !important;
    }

    /* Метрики */
    [data-testid="stMetricValue"] {
        color: #14B8A6 !important;
        font-weight: 800 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ЛОГИКА ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

# --- 4. ШАПКА (ВЫРАЗИТЕЛЬНАЯ) ---
st.markdown("""
    <div class="main-title">
        <span class="floating-bee">🐝</span> BeeTracker
    </div>
    <div class="sub-title">
        Интеллектуальная система <span class="highlight">мониторинга пчел</span> 
        на базе компьютерного зрения.
    </div>
""", unsafe_allow_html=True)

# --- 5. ОСНОВНОЙ КОНТЕНТ (СЕТКА) ---
col_upload, col_stats = st.columns([1.4, 1], gap="large")

with col_upload:
    st.markdown("### 📥 Загрузка данных")
    file = st.file_uploader("upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if file:
        img = Image.open(file)
        if model:
            results = model(img)[0]
            annotated_img = results.plot(masks=False, kpts=False)
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img, use_container_width=True, caption="Анализ завершен")
            count = len(results.boxes)
        else:
            st.error("Модель не найдена")

with col_stats:
    if file and model:
        st.markdown("### 📊 Отчет")
        st.metric("НАЙДЕНО ОСОБЕЙ", f"{count}")
        
        st.markdown(f"""
        <div class="info-card">
            <span class="card-icon">🍯</span>
            <strong>Статус участка:</strong><br>
            {"Оптимальная активность" if count > 20 else "Требуется проверка"}
        </div>
        """, unsafe_allow_html=True)
        
        # Кнопка
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("💾 Сохранить результат", res_bytes, "result.jpg", use_container_width=True)
    else:
        st.markdown("""
        <div style="background: white; padding: 2.5rem; border-radius: 20px; border: 2px dashed #CBD5E1; text-align: center; color: #64748B;">
            <span style="font-size: 3rem;">📸</span><br>
            Ожидание снимка для запуска детекции
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- 6. ТЕМАТИЧЕСКИЕ БЛОКИ (С ИКОНКАМИ) ---
st.markdown("## 🔍 Детали проекта")
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🏠</span>
        <h3>Для пасек</h3>
        Автоматизация учета на больших хозяйствах. Больше не нужно считать пчел вручную — ИИ сделает это за доли секунды.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🌸</span>
        <h3>Эко-контроль</h3>
        Пчела — индикатор здоровья природы. BeeTracker помогает вовремя заметить снижение популяции и спасти колонию.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🧬</span>
        <h3>Технологии</h3>
        Использование архитектуры <b>YOLO11</b> обеспечивает точность более 90% даже при быстром движении пчел.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-top: 4rem; color: #94A3B8; font-size: 0.9rem;">
        BeeTracker AI • Научная вселенная Первых • 2026
    </div>
""", unsafe_allow_html=True)
