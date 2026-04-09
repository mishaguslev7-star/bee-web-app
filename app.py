import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Конфигурация страницы
st.set_page_config(page_title="BeeTraker AI", page_icon="🐝", layout="wide")

# Красивый темный дизайн
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: white; }
    h1, h2 { color: #facc15 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; white-space: pre-wrap; background-color: #1e293b; 
        border-radius: 10px 10px 0 0; color: white;
    }
    .stMetric { background-color: #1e293b; padding: 20px; border-radius: 15px; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

# Загрузка нейросети
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Главное меню
tab1, tab2 = st.tabs(["🚀 Анализатор пчел", "ℹ️ О проекте"])

with tab1:
    st.title("🐝 BeeTraker AI")
    st.write("Загрузите фото рамки для мгновенного анализа нейросетью.")
    
    file = st.file_uploader("Выберите изображение", type=['jpg', 'jpeg', 'png'])
    
    if file:
        img = Image.open(file)
        # Реальная работа нейросети
        with st.spinner('Нейросеть обрабатывает снимок...'):
            results = model(img)[0]
            annotated_img = results.plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            count = len(results.boxes)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(annotated_img, use_container_width=True)
        with col2:
            st.metric("Обнаружено пчел", f"{count} шт.")
            st.success("Анализ завершен успешно!")

with tab2:
    st.title("📖 О проекте")
    st.markdown("""
    ### Суть решения
    Этот проект заменяет ручной, утомительный подсчет пчел на автоматизированный анализ с помощью **компьютерного зрения**.
    
    ### Технологии
    - **YOLOv8/11**: Передовая нейросеть для обнаружения объектов в реальном времени.
    - **Streamlit**: Фреймворк для создания веб-интерфейсов на Python.
    - **Облачные вычисления**: Сайт работает автономно и не требует мощностей вашего устройства.
    
    ### Почему это важно?
    Точный подсчет пчел позволяет пчеловоду объективно оценивать динамику развития семьи, 
    прогнозировать медосбор и вовремя реагировать на изменения в улье.
    """)
