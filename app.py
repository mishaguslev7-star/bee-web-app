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

# --- 2. УСИЛЕННЫЙ CSS ДЛЯ ПОЛНОГО СКРЫТИЯ ИНТЕРФЕЙСА ---
st.markdown("""
    <style>
    /* Прячем верхнюю панель (хедер), меню и кнопку Deploy */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    
    /* Убираем лишние пустые отступы сверху, которые оставляет скрытый хедер */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    /* Шрифты и фон в стиле "Научная вселенная Первых" */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    .stApp {
        background: linear-gradient(145deg, #0B1120 0%, #0F172A 100%);
        color: #E2E8F0;
    }
    
    /* Анимация появления контента */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main > div {
        animation: fadeInUp 0.8s ease-out;
    }

    /* Заголовки */
    h1, h2, h3 {
        color: #FACC15 !important;
        font-weight: 800 !important;
        text-shadow: 0 4px 10px rgba(250, 204, 21, 0.2);
    }
    
    h1 {
        font-size: 3.5rem !important;
        border-left: 10px solid #FACC15;
        padding-left: 1.5rem;
        margin-bottom: 0.5rem !important;
    }

    /* Карточки-контейнеры */
    .info-card {
        background: rgba(30, 41, 59, 0.4);
        padding: 1.5rem;
        border-radius: 20px;
        border-left: 5px solid #FACC15;
        backdrop-filter: blur(5px);
        margin-bottom: 15px;
        transition: 0.3s;
    }
    .info-card:hover {
        background: rgba(30, 41, 59, 0.7);
        transform: scale(1.01);
    }

    /* Метрики (число пчел) */
    [data-testid="stMetricValue"] {
        color: #FACC15 !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
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
    <span style="color: #94A3B8; font-weight: 600; letter-spacing: 2px; font-size: 0.8rem; text-transform: uppercase;">
        Всероссийский фестиваль • Научная вселенная Первых
    </span>
</div>
""", unsafe_allow_html=True)

c_tit, c_logo = st.columns([3.5, 1])
with c_tit:
    st.title("🐝 ПЧЕЛИНЫЙ УЧЁТ")
    st.markdown('<p style="font-size: 1.3rem; color: #94A3B8; margin-top: -10px;">Интеллектуальный анализ и мониторинг экологических угроз</p>', unsafe_allow_html=True)
with c_logo:
    st.markdown("""
    <div style="background: #FACC15; padding: 12px; border-radius: 40px; text-align: center; box-shadow: 0 5px 0 #B45309; margin-top: 15px;">
        <span style="color: #0F172A; font-weight: 800; font-size: 1.1rem;">AI ТРЕКЕР</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 5. РАБОЧАЯ ЗОНА ---
col_up, col_stat = st.columns([1.6, 1])

with col_up:
    st.markdown("### 📸 Загрузите фото рамки")
    file = st.file_uploader("upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if file:
    img = Image.open(file)
    with st.spinner('🧠 Нейросеть YOLO11 анализирует кадр...'):
        results = model(img)[0]
        # masks=False убирает синие пятна сегментации
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
            st.error("Пчёлы не обнаружены. Попробуйте другой ракурс.")
        elif count < 20:
            st.info("Средняя плотность. Семья в норме.")
        else:
            st.success("Высокая активность! Признак сильной семьи.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Кнопка скачивания
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Сохранить отчёт", res_bytes, "bee_report.jpg", "image/jpeg")
else:
    with col_stat:
        st.info("👈 Загрузите фотографию для начала автоматического учёта.")
        st.markdown("""
        <div style="background: #0f172a; border-radius: 20px; padding: 50px; text-align: center; border: 1px dashed #475569;">
            <span style="font-size: 50px;">🐝</span>
            <p style="color: #64748B; margin-top: 10px;">Ожидание данных...</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

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

# Футер
st.markdown(f"""
<div style="text-align: center; color: #4B5563; margin-top: 40px; font-size: 0.9rem;">
    <strong>Михаил Гуслев • Лев Лебедянцев</strong><br>
    г. Пермь, Лицей ПГГПУ • «Научная вселенная Первых»
</div>
""", unsafe_allow_html=True)
