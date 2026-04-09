import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Пчелиный учёт | AI", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. СВЕТЛАЯ ТЕМА И УЛУЧШЕННЫЙ CSS ---
st.markdown("""
    <style>
    /* Скрываем служебные элементы Streamlit */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Шрифты и светлый фон */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    .stApp {
        background-color: #F8FAFC; /* Светлый серо-голубой фон */
        color: #1E293B; /* Темный текст для контраста */
    }
    
    /* Анимация появления контента */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main > div {
        animation: fadeInUp 0.8s ease-out;
    }

    /* Заголовки (Синий из презентации + Желтый акцент) */
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 800 !important;
    }
    
    h1 {
        font-size: 3.5rem !important;
        border-left: 10px solid #FACC15;
        padding-left: 1.5rem;
        margin-bottom: 0.5rem !important;
    }

    /* Карточки-контейнеры в светлой теме */
    .info-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #FACC15;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        transition: 0.3s;
        color: #334155;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Метрики (число пчел) */
    [data-testid="stMetricValue"] {
        color: #B45309 !important; /* Насыщенный золотистый */
        font-size: 3.5rem !important;
        font-weight: 800 !important;
    }
    
    /* Стилизация uploader */
    .stFileUploader section {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ЛОГИКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 4. ШАПКА САЙТА ---
st.markdown("""
<div style="margin-bottom: 15px;">
    <span style="color: #64748B; font-weight: 600; letter-spacing: 2px; font-size: 0.8rem; text-transform: uppercase;">
        Всероссийский фестиваль • Научная вселенная Первых
    </span>
</div>
""", unsafe_allow_html=True)

c_tit, c_logo = st.columns([3.5, 1])
with c_tit:
    st.title("🐝 ПЧЕЛИНЫЙ УЧЁТ")
    st.markdown('<p style="font-size: 1.3rem; color: #475569; margin-top: -10px;">Интеллектуальный анализ и мониторинг экологических угроз</p>', unsafe_allow_html=True)
with c_logo:
    st.markdown("""
    <div style="background: #FACC15; padding: 12px; border-radius: 40px; text-align: center; box-shadow: 0 5px 0 #B45309; margin-top: 15px;">
        <span style="color: #0F172A; font-weight: 800; font-size: 1.1rem;">AI ТРЕКЕР</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-top: 2px solid #E2E8F0;'>", unsafe_allow_html=True)

# --- 5. РАБОЧАЯ ЗОНА ---
col_up, col_stat = st.columns([1.6, 1])

with col_up:
    st.markdown("### 📸 Загрузите фото рамки")
    file = st.file_uploader("upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if file:
    img = Image.open(file)
    with st.spinner('🧠 Нейросеть YOLO11 анализирует кадр...'):
        results = model(img)[0]
        # masks=False убирает синие пятна
        annotated_img = results.plot(masks=False, kpts=False, probs=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        count = len(results.boxes)
    
    with col_up:
        st.image(annotated_img, use_container_width=True)
    
    with col_stat:
        st.markdown("### 📊 Статистика")
        st.metric(label="🐝 ОБНАРУЖЕНО ПЧЁЛ", value=f"{count} шт.")
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Рекомендация")
        if count == 0:
            st.markdown("<span style='color: #EF4444;'>❌ Пчёлы не обнаружены. Попробуйте другой ракурс.</span>", unsafe_allow_html=True)
        elif count < 20:
            st.markdown("<span style='color: #0EA5E9;'>ℹ️ Средняя плотность. Семья в норме.</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #10B981;'>✅ Высокая активность! Признак сильной семьи.</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Кнопка скачивания (синяя в светлой теме)
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Сохранить отчёт", res_bytes, "bee_report.jpg", "image/jpeg")
else:
    with col_stat:
        st.markdown("""
        <div style="background: #FFFFFF; border-radius: 20px; padding: 50px; text-align: center; border: 2px dashed #E2E8F0;">
            <span style="font-size: 50px;">🐝</span>
            <p style="color: #64748B; margin-top: 10px;">Ожидание загрузки фотографии...</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. ИНФОРМАЦИЯ ИЗ ПРЕЗЕНТАЦИИ ---
st.header("📖 О ПРОЕКТЕ")
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Актуальность</h3>
        Пчела движется со скоростью до 30 км/ч. Человеческий глаз не может точно зафиксировать 50-100 особей на лету. 
        Наш AI решает эту задачу мгновенно, помогая пчеловодам экономить часы ручного труда.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Перспективы</h3>
        Мы планируем внедрить систему распознавания болезней по внешним признакам (варроатоз) 
        и разработать мобильное приложение для работы прямо на пасеке.
    </div>
    """, unsafe_allow_html=True)
