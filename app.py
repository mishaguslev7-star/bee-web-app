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

    /* Заголовки */
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
    }

    .floating-bee {
        display: inline-block;
        animation: bee-float 3s ease-in-out infinite;
    }
    @keyframes bee-float {
        0%, 100% { transform: translateY(0) rotate(-5deg); }
        50% { transform: translateY(-12px) rotate(10deg); }
    }

    .sub-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.2rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }

    /* Блоки контента */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        color: #5EEAD4;
        letter-spacing: 2px;
        margin: 3rem 0 1.5rem 0;
        border-left: 5px solid #2DD4BF;
        padding-left: 15px;
    }

    .info-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.8rem;
        border-radius: 24px;
        border: 1px solid rgba(45, 212, 191, 0.1);
        backdrop-filter: blur(12px);
        transition: all 0.4s ease;
        height: 100%;
    }
    .info-card:hover {
        transform: translateY(-8px);
        border-color: #2DD4BF;
        box-shadow: 0 15px 35px -10px rgba(45, 212, 191, 0.3);
    }

    .goal-card {
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.1), rgba(15, 23, 42, 0.9));
        padding: 2rem;
        border-radius: 24px;
        border: 2px solid #2DD4BF;
        margin-bottom: 2rem;
    }

    .card-icon { font-size: 2.2rem; margin-bottom: 1rem; display: block; }
    h3 { color: #5EEAD4 !important; font-size: 1.3rem !important; margin-bottom: 10px !important; }
    p { color: #CBD5E1; line-height: 1.6; margin: 0; }
    b { color: #5EEAD4; }

    /* Стилизация загрузчика */
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #2DD4BF !important;
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 20px !important;
    }

    [data-testid="stMetricValue"] { color: #2DD4BF !important; font-family: 'Orbitron', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- 3. ЛОГИКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

# --- 4. ШАПКА И ЦЕЛЬ ---
st.markdown("""
    <div class="main-title">
        <span class="floating-bee">🐝</span> BeeTracker
    </div>
    <div class="sub-title">
        Интеллектуальная система мониторинга пчел
    </div>
    
    <div class="goal-card">
        <h3>🎯 Цель проекта</h3>
        <p style="font-size: 1.15rem;">
            Создание доступного ИИ-инструмента для спасения пчелиных семей. Мы помогаем пчеловодам вовремя заметить угрозу и автоматизируем сложный процесс подсчета насекомых.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 5. ИНТЕРФЕЙС АНАЛИЗА ---
col_up, col_res = st.columns([1.4, 1], gap="large")

with col_up:
    st.markdown("### 📥 Загрузка фото")
    file = st.file_uploader("upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    if file:
        img = Image.open(file)
        if model:
            with st.spinner('Нейросеть считает пчел...'):
                results = model(img)[0]
                annotated_img = results.plot(masks=False, kpts=False)
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img, use_container_width=True)
                count = len(results.boxes)
        else:
            st.error("Ошибка: Файл модели 'best.pt' не найден.")

with col_res:
    if file and model:
        st.metric("НАЙДЕНО ПЧЕЛ", f"{count}")
        st.markdown(f"""
        <div class="info-card">
            <span class="card-icon">📊</span>
            <h3>Аналитика</h3>
            Обработка завершена. Зафиксировано <b>{count}</b> объектов. 
            Это позволяет оценить активность летка без ручного вмешательства.
        </div>
        """, unsafe_allow_html=True)
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("💾 Сохранить отчет", res_bytes, "report.jpg", use_container_width=True)
    else:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); padding: 2.5rem; border-radius: 20px; border: 2px dashed rgba(148, 163, 184, 0.2); text-align: center;">
            <span style="font-size: 3rem; opacity: 0.2;">📷</span>
            <p style="color: #64748B; margin-top: 1rem;">Загрузите снимок для теста ИИ</p>
        </div>
        """, unsafe_allow_html=True)

# --- 6. НОВЫЙ БЛОК: О ПРОЕКТЕ (ПОНЯТНЫМ ЯЗЫКОМ) ---
st.markdown('<div class="section-header">🔍 О ПРОЕКТЕ</div>', unsafe_allow_html=True)
a1, a2 = st.columns(2, gap="medium")

with a1:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">❓</span>
        <h3>В чем проблема?</h3>
        Пчеловодам с сотнями ульев трудно следить за каждым. Пчелы летают очень быстро (до 30 км/ч), и человек просто не может точно их сосчитать. Если пчелы заболеют или их станет меньше, пчеловод может заметить это слишком поздно.
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">💡</span>
        <h3>Наше решение</h3>
        Мы создали умный сервис, который делает всю сложную работу за человека. Вы просто загружаете фото или видео, а наша нейросеть <b>YOLO11</b> мгновенно находит и считает каждую пчелу, помогая следить за здоровьем всей пасеки.
    </div>
    """, unsafe_allow_html=True)

# --- 7. ПЕРСПЕКТИВЫ ---
st.markdown('<div class="section-header">🚀 ПЕРСПЕКТИВЫ</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns(3, gap="medium")

with p1:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🔄</span>
        <h3>Живой поток</h3>
        Переход на видеомониторинг в реальном времени для постоянного контроля летка.
    </div>
    """, unsafe_allow_html=True)

with p2:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">📱</span>
        <h3>Приложение</h3>
        Запуск мобильной версии, чтобы пчеловод мог проверять ульи прямо со смартфона на пасеке.
    </div>
    """, unsafe_allow_html=True)

with p3:
    st.markdown("""
    <div class="info-card">
        <span class="card-icon">🔬</span>
        <h3>Здоровье</h3>
        Обучение ИИ распознаванию болезней пчел и вредителей (например, клеща Варроа).
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><p style='text-align: center; color: #475569; font-size: 0.8rem;'>BEETRACKER AI • 2026</p>", unsafe_allow_html=True)
