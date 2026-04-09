import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. CONFIG ---
st.set_page_config(
    page_title="BeeTracker AI", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. УСИЛЕННЫЙ CSS (Анимации, Фон, Шрифты) ---
st.markdown("""
    <style>
    /* Скрываем служебные элементы Streamlit */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    
    /* Основной шрифт */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

    /* ФОН С ШЕСТИУГОЛЬНИКАМИ (СОТЫ) */
    .stApp {
        background-color: #F8FAFC;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0Ny44NTgiIHZpZXdCb3g9IjAgMCA0MCA0Ny44NTgiPjxnIGZpbGw9IiNFMEYyRkUiIGZpbGwtb3BhY2l0eT0iMC4zIj48cGF0aCBkPSJNMTAgMjIuNjI0bDEwLTUuNzc0IDEwIDUuNzc0djExLjU0OGwtMTAgNS43NzQtMTAtNS43NzRWMjIuNjI0em0yLDEuMTU0djkuMjRsOCA0LjYxOCA4LTQuNjE4di05LjI0TDQyMzU2LjYzMThNMCAxMS4zMTJsMTAtNS43NzQgMTAsNS43NTR2MTEuNTQ4bC0xMCA1Ljc3NC0xMC01Ljc3NFYxMS4zMTJ6bTIsMS4xNTR2OS4yNEw4MCAyNS4xNzQgMjAtMTIuNDVWMjEuMzFMMjAyMDUuNTRMMTIgMTEuMzEyek0xMCAwVjUuNzc0TDAtLjA1MnYtMTAuMjQ2TDAtNS43NzRWMHoiLz48L2c+PC9zdmc+');
    }

    /* ЗАГОЛОВОК BeeTracker С АНИМАЦИЕЙ */
    .beetraker-header {
        color: #1E293B !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        margin-bottom: 0px !important;
        display: flex;
        align-items: center;
        gap: 15px;
    }

    /* Анимация покачивания пчелы */
    .swinging-bee {
        display: inline-block;
        animation: beeSwing 3s ease-in-out infinite;
        transform-origin: center bottom;
    }
    @keyframes beeSwing {
        0%, 100% { transform: rotate(0deg) translateY(0px); }
        50% { transform: rotate(7deg) translateY(-5px); }
    }

    /* Карточки */
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        border-top: 5px solid #FFB800;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    
    [data-testid="stMetricValue"] {
        color: #B45309 !important;
        font-weight: 800 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 4. ШАПКА ---
col_t, col_l = st.columns([4, 1])
with col_t:
    # Иконка пчелы покачивается (swinging-bee)
    st.markdown('<h1 class="beetraker-header"><span class="swinging-bee">🐝</span> BeeTracker</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; color: #475569; margin-top: -10px;">Интеллектуальная система мониторинга популяции пчел</p>', unsafe_allow_html=True)
with col_l:
    st.markdown("""
    <div style="background: #FFB800; padding: 10px; border-radius: 50px; text-align: center; font-weight: 800; color: #0F172A; margin-top: 15px;">
        AI CORE
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 5. ИНТЕРФЕЙС ---
c_up, c_stat = st.columns([1.6, 1], gap="medium")

with c_up:
    st.markdown("### 📸 Анализ снимка")
    file = st.file_uploader("upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if file:
    img = Image.open(file)
    with st.spinner('Анализ...'):
        results = model(img)[0]
        # masks=False убирает синие пятна
        annotated_img = results.plot(masks=False, kpts=False, probs=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        count = len(results.boxes)
    
    with c_up:
        st.image(annotated_img, use_container_width=True)
    
    with c_stat:
        st.markdown("### 📈 Результат")
        st.metric(label="🐝 ОБНАРУЖЕНО ПЧЁЛ", value=f"{count}")
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Рекомендация")
        if count == 0:
            st.error("Пчёлы не обнаружены.")
        elif count < 15:
            st.info("Низкая плотность популяции.")
        else:
            st.success("Высокая активность семьи!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Кнопка скачивания
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Сохранить", res_bytes, "result.jpg", "image/jpeg")
else:
    with c_stat:
        st.info("👈 Загрузите фото рамки для начала подсчёта.")

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8;'>🪐 Научная вселенная Первых</p>", unsafe_allow_html=True)
