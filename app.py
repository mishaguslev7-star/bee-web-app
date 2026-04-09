import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

st.set_page_config(page_title="BeeTraker AI", page_icon="🐝", layout="wide")

# --- СТИЛЬ (БЕЗ ЛОМАЮЩИХ ШТУК) ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at top, #1e293b, #020617);
    color: white;
}

/* Заголовок */
.hero {
    text-align: center;
    padding: 40px 0;
}
.hero h1 {
    font-size: 60px;
    background: linear-gradient(90deg, #facc15, #fde68a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #cbd5f5;
    font-size: 18px;
}

/* Карточки */
.card {
    background: rgba(30, 41, 59, 0.6);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #334155;
    margin-bottom: 20px;
}

/* Разделитель */
.divider {
    height: 1px;
    background: #334155;
    margin: 30px 0;
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

# ✅ НОРМАЛЬНАЯ КНОПКА (Streamlit)
if st.button("🚀 Попробовать"):
    st.session_state.scroll = True

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- ДЕМО ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    file = st.file_uploader("Загрузите изображение", type=['jpg','png','jpeg'])

    if file:
        img = Image.open(file)

        with st.spinner("Анализ..."):
            results = model(img)[0]
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            count = len(results.boxes)

        st.image(annotated, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if file:
        st.metric("🐝 Пчёл", count)
        st.success("Готово")
    else:
        st.info("Загрузите фото")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- БЛОКИ ---
colA, colB, colC = st.columns(3)

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Проблема")
    st.write("Нельзя вручную отслеживать пчёл в реальном времени.")
    st.markdown('</div>', unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Решение")
    st.write("ИИ считает пчёл автоматически.")
    st.markdown('</div>', unsafe_allow_html=True)

with colC:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Цель")
    st.write("Предотвратить гибель колоний.")
    st.markdown('</div>', unsafe_allow_html=True)
