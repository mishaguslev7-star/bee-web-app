import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Настройка страницы
st.set_page_config(page_title="BeeTraker AI", page_icon="🐝", layout="wide")

# Кастомный дизайн и скрытие лишних элементов GitHub/Deploy
st.markdown("""
    <style>
    /* Скрываем меню Streamlit и кнопку GitHub */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #0f172a; color: white; }
    h1, h2 { color: #facc15 !important; }
    .stMetric { background-color: #1e293b; padding: 20px; border-radius: 15px; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

# Загрузка нейросети
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- ВЕРХНЯЯ ЧАСТЬ: ИНТЕРФЕЙС ---
st.title("🐝 BeeTraker AI")
st.write("Загрузите фото рамки для мгновенного анализа нейросетью.")

col_upload, col_res = st.columns([1.5, 1])

with col_upload:
    file = st.file_uploader("Выберите изображение", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
if file:
    img = Image.open(file)
    with st.spinner('Нейросеть обрабатывает снимок...'):
        results = model(img)[0]
        annotated_img = results.plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        count = len(results.boxes)
    
    with col_upload:
        st.image(annotated_img, use_container_width=True)
    with col_res:
        st.metric("Обнаружено пчел", f"{count} шт.")
        st.success("Анализ завершен успешно!")
else:
    with col_res:
        st.info("Ожидание загрузки фото...")

st.markdown("---")

# --- НИЖНЯЯ ЧАСТЬ: О ПРОЕКТЕ (сразу на главной) ---
st.header("📖 О проекте BeeTraker")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    **Суть решения**
    Этот проект заменяет ручной подсчет пчел автоматизированным анализом. Это позволяет пчеловоду объективно оценивать динамику развития семьи и прогнозировать медосбор.
    """)

with col_info2:
    st.markdown("""
    **Технологии**
    Используется нейросеть **YOLO**, обученная на тысячах снимков. Сайт работает автономно и не требует мощностей вашего телефона или ПК для обработки.
    """)
