import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

st.set_page_config(page_title="BeeTraker AI", page_icon="🐝", layout="wide")

# --- СТИЛЬ ПРЕМИУМ ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at top, #1e293b, #020617);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Заголовки */
.hero-title {
    font-size: 64px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #facc15, #fde68a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    text-align: center;
    font-size: 20px;
    color: #cbd5f5;
    margin-bottom: 30px;
}

/* Кнопка */
.button {
    display: flex;
    justify-content: center;
    margin-bottom: 40px;
}
.button a {
    padding: 15px 35px;
    background: linear-gradient(90deg, #facc15, #eab308);
    color: black;
    font-weight: 600;
    border-radius: 999px;
    text-decoration: none;
    transition: 0.3s;
}
.button a:hover {
    transform: scale(1.05);
}

/* Карточки */
.card {
    background: rgba(30, 41, 59, 0.6);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-5px);
}

/* Центр */
.center {
    text-align: center;
}

/* Разделитель */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #334155, transparent);
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
st.markdown('<div class="hero-title">🐝 BeeTraker AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Интеллектуальный анализ активности пчёл в один клик</div>', unsafe_allow_html=True)

st.markdown("""
<div class="button">
<a href="#">🚀 Попробовать</a>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- ДЕМО ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    file = st.file_uploader("Загрузите изображение", type=['jpg','png','jpeg'])

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
    st.markdown('<div class="card center">', unsafe_allow_html=True)
    if file:
        st.metric("🐝 Найдено пчёл", count)
        st.success("Готово")
    else:
        st.info("Загрузите фото")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- БЛОКИ (КАК СТАРТАП) ---
colA, colB, colC = st.columns(3)

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Проблема")
    st.write("Невозможно вручную отслеживать активность пчёл в реальном времени.")
    st.markdown('</div>', unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Решение")
    st.write("ИИ автоматически считает и анализирует пчёл за секунды.")
    st.markdown('</div>', unsafe_allow_html=True)

with colC:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Цель")
    st.write("Предотвратить гибель пчелиных колоний.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- О ПРОЕКТЕ ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## 📖 О проекте")
st.write("""
BeeTraker превращает хаотичное движение пчёл в точные данные.  
Это помогает заранее обнаружить проблемы и спасти ульи.
""")
st.markdown('</div>', unsafe_allow_html=True)

# --- ПРЕИМУЩЕСТВА ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔥 Преимущества")
    st.write("""
    - Мгновенный анализ  
    - Высокая точность  
    - Простота использования  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Будущее")
    st.write("""
    - Распознавание болезней  
    - Мобильное приложение  
    - Онлайн мониторинг  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- ФУТЕР ---
st.markdown("""
<div class="center">
    <div class="divider"></div>
    <p>© 2026 BeeTraker AI • Сделано с ИИ</p>
</div>
""", unsafe_allow_html=True)
