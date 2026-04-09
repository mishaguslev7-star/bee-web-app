import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Настройка страницы
st.set_page_config(
    page_title="BeeTraker AI",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ПОЛНОЕ СКРЫТИЕ ИНТЕРФЕЙСА И СТИЛИЗАЦИЯ ---
st.markdown("""
    <style>
    /* Скрываем вообще всё лишнее: хедер, футер, кнопки */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    
    /* Убираем лишние отступы сверху */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Фон и анимация */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stApp { 
        background-color: #0a192f; 
        color: #e6f1ff;
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Заголовки в стиле презентации */
    h1, h2, h3 { 
        color: #facc15 !important; 
        font-family: 'Segoe UI', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Карточки-контейнеры */
    .reportview-container .main .block-container {
        max-width: 1200px;
    }
    
    .info-card {
        background-color: #112240;
        border: 1px solid #233554;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        transition: 0.3s;
    }
    .info-card:hover {
        border-color: #facc15;
        background-color: #172a45;
    }

    /* Стилизация метрики */
    [data-testid="stMetricValue"] {
        color: #facc15 !important;
        font-size: 3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Загрузка YOLO11
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- ШАПКА ---
st.markdown('<h1 style="text-align: center;">🐝 BEETRAKER AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #8892b0;">Интеллектуальный анализ активности пчёл • Научная вселенная Первых</p>', unsafe_allow_html=True)

# --- ОСНОВНОЙ ФУНКЦИОНАЛ ---
st.markdown("### 🛠 Анализатор пчёл")
col_up, col_info = st.columns([1.5, 1])

with col_up:
    uploaded_file = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    with st.spinner('Анализ YOLO11...'):
        results = model(img_array)[0]
        # masks=False убирает синие дефекты сегментации
        annotated_img = results.plot(masks=False, kpts=False, probs=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        bee_count = len(results.boxes)

    with col_up:
        st.image(annotated_img, use_container_width=True)
    
    with col_info:
        st.metric("Обнаружено особей", f"{bee_count} шт.")
        st.success("Объекты зафиксированы")
        st.markdown(f"""
        <div class="info-card">
        <strong>Техническая сводка:</strong><br>
        Модель: YOLO11<br>
        Статус: Работа в полевых условиях<br>
        Локация: Пермский край 
        </div>
        """, unsafe_allow_html=True)
else:
    with col_info:
        st.info("Для начала работы загрузите фотографию рамки или летка.")

st.markdown("<br><hr><br>", unsafe_allow_html=True)

# --- ИНФОРМАЦИЯ ИЗ ПРЕЗЕНТАЦИИ ---
st.markdown("## 🔬 О ПРОЕКТЕ")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
    <h3>📢 Актуальность</h3>
    Пчела движется со скоростью до 30 км/ч. Человеческий глаз не способен точно посчитать 50-100 особей мгновенно.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
    <h3>💡 Решение</h3>
    Разработка заменяет многочасовой ручной труд и позволяет вовремя заметить экологическую угрозу.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
    <h3>🎯 Цель</h3>
    Предотвращение коллапса пчелиных семей и помощь ученым, фермерам и биологам.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p style="text-align: center; margin-top: 40px; color: #495670;">Михаил Гуслев, Лев Лебедянцев • СОШ Лицей ПГГПУ </p>', unsafe_allow_html=True)
