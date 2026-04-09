import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

st.set_page_config(page_title="BeeTraker AI", page_icon="🐝", layout="wide")

# --- ЧИСТЫЙ ПРЕМИУМ СТИЛЬ ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at top, #1e293b, #020617);
    color: white;
}

/* HERO */
.hero {
    text-align: center;
    padding: 60px 0 30px 0;
}
.hero h1 {
    font-size: 64px;
    font-weight: 800;
    background: linear-gradient(90deg, #facc15, #fde68a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 20px;
    color: #cbd5f5;
}

/* КАРТОЧКИ */
.card {
    background: rgba(30, 41, 59, 0.7);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #334155;
}

/* РАЗДЕЛИТЕЛЬ */
.divider {
    height: 1px;
    background: #334155;
    margin: 40px 0;
}
</style>
""", unsafe_allow_html=True)

# --- МОДЕЛЬ ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- HERO ---
st.markdown("""
<div class="hero">
    <h1>🐝 BeeTraker AI</h1>
    <p>Интеллектуальный анализ активности пчёл</p>
</div>
""", unsafe_allow_html=True)

# КНОПКА (реальная)
st.button("🚀 Попробовать")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- ОСНОВНОЙ БЛОК (БЕЗ ПУСТОТЫ) ---
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    file = st.file_uploader("📂 Загрузите изображение", type=['jpg','png','jpeg'])

    if file:
        img = Image.open(file)

        with st.spinner("ИИ анализирует изображение..."):
            results = model(img)[0]
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            count = len(results.boxes)

        st.image(annotated, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if file:
        st.metric("🐝 Обнаружено пчёл", count)
        st.success("Анализ завершён")
    else:
        st.info("Загрузите изображение")

    st.markdown('</div>', unsafe_allow_html=True)

# --- РАЗДЕЛ ---
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- ИНФО БЛОКИ (БЕЗ ПУСТЫХ ПОЛОС) ---
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Проблема")
    st.write("Ручной подсчёт пчёл невозможен при большом количестве ульев.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Решение")
    st.write("Нейросеть автоматически считает пчёл и анализирует активность.")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Цель")
    st.write("Предотвращение гибели пчелиных колоний.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- О ПРОЕКТЕ ---
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📖 О проекте")
st.write("""
BeeTraker превращает хаотичное движение пчёл в точные данные.  
Это помогает выявлять проблемы заранее и повышать эффективность пасек.
""")
st.markdown('</div>', unsafe_allow_html=True)

# --- ФУТЕР ---
st.markdown("""
<div style="text-align:center; margin-top:40px;">
    <p style="color:#64748b;">© 2026 BeeTraker AI</p>
</div>
""", unsafe_allow_html=True)
