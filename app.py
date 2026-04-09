import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="Пчелиный учёт | AI", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- КАСТОМНЫЙ ДИЗАЙН ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    /* Скрываем всё лишнее (GitHub, Menu, Deploy) */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none;}
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    .stApp {
        background: linear-gradient(145deg, #0B1120 0%, #0F172A 100%);
        color: #E2E8F0;
    }
    
    /* Анимация появления */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .main > div {
        padding: 2rem 3rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    h1, h2, h3 {
        color: #FACC15 !important;
        font-weight: 800 !important;
        text-shadow: 0 4px 12px rgba(250, 204, 21, 0.15);
    }
    
    h1 {
        font-size: 3.2rem !important;
        border-left: 8px solid #FACC15;
        padding-left: 1.5rem;
    }
    
    .stMetric {
        background: rgba(30, 41, 59, 0.8) !important;
        padding: 24px !important;
        border-radius: 20px !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 8px 0 #020617 !important;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        border-color: #FACC15 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #FACC15 !important;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
    }

    .info-card {
        background: rgba(30, 41, 59, 0.5);
        padding: 1.8rem;
        border-radius: 24px;
        border-left: 6px solid #FACC15;
        backdrop-filter: blur(4px);
        height: 100%;
        transition: all 0.3s;
        margin-bottom: 10px;
    }
    
    .info-card:hover {
        background: rgba(30, 41, 59, 0.8);
    }

    hr {
        margin: 2rem 0;
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FACC15, transparent);
    }
    </style>
""", unsafe_allow_html=True)

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- ШАПКА ---
st.markdown("""
<div style="text-align: left; margin-bottom: 10px;">
    <span style="color: #94A3B8; font-weight: 600; letter-spacing: 2px; font-size: 0.8rem;">ВСЕРОССИЙСКИЙ ФЕСТИВАЛЬ • НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ</span>
</div>
""", unsafe_allow_html=True)

col_title, col_logo = st.columns([3, 1])
with col_title:
    st.title("🐝 ПЧЕЛИНЫЙ УЧЁТ")
    st.markdown('<p style="font-size: 1.2rem; color: #CBD5E1; margin-top: -10px;">Интеллектуальный мониторинг экосистемы пасеки</p>', unsafe_allow_html=True)
with col_logo:
    st.markdown("""
    <div style="background: #FACC15; padding: 10px; border-radius: 50px; text-align: center; box-shadow: 0 4px 0 #B45309; margin-top: 15px;">
        <span style="color: #0F172A; font-weight: 800; font-size: 1rem;">YOLO11 AI</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- ИНТЕРФЕЙС ---
col_upload, col_res = st.columns([1.6, 1])

with col_upload:
    st.markdown("### 📸 Загрузка снимка")
    file = st.file_uploader("upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if file:
    img = Image.open(file)
    with st.spinner('🧠 Анализ нейросетью...'):
        results = model(img)[0]
        # masks=False убирает те самые синие пятна
        annotated_img = results.plot(masks=False, kpts=False, probs=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        count = len(results.boxes)
    
    with col_upload:
        st.image(annotated_img, use_container_width=True)
    
    with col_res:
        st.markdown("### 📊 Аналитика")
        st.metric(label="🐝 ОБНАРУЖЕНО ПЧЁЛ", value=f"{count} шт.")
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Состояние")
        if count == 0:
            st.warning("Пчёлы не найдены.")
        elif count < 15:
            st.info("Низкая плотность. Требуется наблюдение.")
        else:
            st.success("Высокая активность семьи!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Скачивание результата
        res_encoded = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Скачать отчёт", res_encoded, "bee_result.jpg", "image/jpeg")
else:
    with col_res:
        st.info("👈 Загрузите фото рамки для начала подсчёта.")
        st.markdown("""
        <div style="background: #0f172a; border-radius: 20px; padding: 40px; text-align: center; border: 1px dashed #475569;">
            <span style="font-size: 40px;">🐝</span>
            <p style="color: #94a3b8;">Система готова к работе</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- О ПРОЕКТЕ ---
st.header("📖 О ПРОЕКТЕ")
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Цель и задача</h3>
        Автоматизировать учёт пчёл для владельцев крупных пасек. 
        Человеческий глаз не может точно посчитать сотни быстродвижущихся особей. 
        BeeTraker решает эту проблему за доли секунды.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Будущее</h3>
        Разработка мобильного приложения и внедрение системы раннего 
        определения болезней пчёл по внешним признакам с помощью AI.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #64748B; margin-top: 30px;">
    <strong>Лебедянцев Лев • Гуслев Михаил</strong><br>
    г. Пермь, Лицей ПГГПУ • 2026
</div>
""", unsafe_allow_html=True)
