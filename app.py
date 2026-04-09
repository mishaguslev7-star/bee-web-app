```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- НАСТРОЙКИ ---
st.set_page_config(
    page_title="BeeTracker AI",
    page_icon="🐝",
    layout="wide"
)

# --- СТИЛИ ---
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');

html, body {
    font-family: 'Montserrat', sans-serif;
    background: linear-gradient(135deg, #ecfeff, #f0fdfa);
}

/* Фон */
.stApp {
    background: radial-gradient(circle at top, #ccfbf1, #f0fdfa);
}

/* Анимация */
@keyframes fadeUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Заголовок */
h1 {
    font-size: 3.5rem !important;
    font-weight: 800;
    background: linear-gradient(90deg, #14B8A6, #0D9488);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Upload */
div[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(10px);
    border: 2px dashed #14B8A6 !important;
    border-radius: 25px !important;
    transition: 0.3s;
}

div[data-testid="stFileUploadDropzone"]:hover {
    transform: scale(1.02);
}

/* Карточки */
.info-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    padding: 1.5rem;
    border-radius: 20px;
    margin-top: 15px;
    border: 1px solid #ccfbf1;
    animation: fadeUp 0.6s ease;
}

/* Метрика */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.8);
    border-radius: 20px;
    padding: 20px;
}

/* Loader */
.loader {
    text-align: center;
    font-weight: 600;
    color: #0d9488;
    animation: fadeUp 0.5s;
}

</style>
""", unsafe_allow_html=True)

# --- МОДЕЛЬ ---
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        return None
    return YOLO("best.pt")

model = load_model()

# --- ШАПКА ---
st.markdown("## 🐝 BeeTracker AI")
st.markdown("Интеллектуальный анализ пчелиных семей")

st.markdown("""
<div style="background: linear-gradient(135deg,#14B8A6,#0D9488);
padding:15px;border-radius:15px;color:white;margin-bottom:20px;">
🚀 Искусственный интеллект анализирует пчёл за секунды
</div>
""", unsafe_allow_html=True)

# --- ЗАГРУЗКА ---
col1, col2 = st.columns([1.5, 1])

with col1:
    file = st.file_uploader("Загрузите фото", type=["jpg","png","jpeg"])

if file is not None and model is not None:
    image = Image.open(file)

    st.markdown('<p class="loader">🐝 Анализируем...</p>', unsafe_allow_html=True)

    results = model(image)[0]
    annotated = results.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    count = len(results.boxes)

    with col1:
        st.image(annotated, use_container_width=True)

    with col2:
        st.metric("Обнаружено пчёл", count)

        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Вывод")

        if count == 0:
            st.markdown("❌ Пчёлы не обнаружены")
        elif count < 15:
            st.markdown("⚠️ Низкая активность")
        else:
            st.markdown("✅ Всё в норме")

        st.markdown("</div>", unsafe_allow_html=True)

        img_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Скачать результат", img_bytes, "result.jpg")

else:
    st.info("👈 Загрузите изображение")

# --- ИНФО ---
st.markdown("## 🔬 О проекте")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="info-card">🚨 Контроль состояния ульев</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="info-card">⚡ AI на базе YOLO</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="info-card">🚀 Будущее — мобильное приложение</div>', unsafe_allow_html=True)

st.markdown("<center>BeeTracker • 2026</center>", unsafe_allow_html=True)
```
