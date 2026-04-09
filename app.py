import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Настройка страницы в стиле "Научная вселенная Первых"
st.set_page_config(
    page_title="BeeTraker AI — Пчелиный учёт",
    page_icon="🐝",
    layout="wide"
)

# --- УЛУЧШЕННЫЙ ДИЗАЙН И АНИМАЦИИ ---
st.markdown("""
    <style>
    /* Скрытие служебных элементов */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Анимация появления */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Основной фон — темно-синий из презентации */
    .stApp { 
        background-color: #0a192f; 
        color: #e6f1ff;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Золотистые заголовки */
    h1, h2, h3 { 
        color: #facc15 !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Стилизация карточек с информацией */
    .info-card {
        background-color: #112240;
        border: 1px solid #233554;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .info-card:hover {
        transform: translateY(-5px);
        border-color: #facc15;
    }

    /* Кнопка загрузки */
    .stFileUploader section {
        background-color: #112240 !important;
        border: 2px dashed #facc15 !important;
        border-radius: 15px !important;
    }

    /* Метрики */
    [data-testid="stMetricValue"] { color: #facc15 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Загрузка модели YOLO11
@st.cache_resource
def load_model():
    try:
        # Загружаем твою модель best.pt
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

model = load_model()

# --- ВЕРХНЯЯ ПАНЕЛЬ (ЛОГО И ЗАГОЛОВОК) ---
st.markdown('<h1 style="text-align: center;">🐝 BeeTraker AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">Наш проект превращает хаотичное движение пчел в ценные данные</p>', unsafe_allow_html=True)

st.markdown("---")

# --- ОСНОВНОЙ БЛОК: АНАЛИЗ ---
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown("### 📸 Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите фото рамки", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    with st.spinner('Нейросеть YOLO11 анализирует кадр...'):
        # Предсказание: отключаем маски (masks=False), чтобы убрать синие пятна
        results = model(img_array)[0]
        annotated_img = results.plot(masks=False, kpts=False, probs=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        bee_count = len(results.boxes)

    with col_left:
        st.image(annotated_img, caption="Результат распознавания", use_container_width=True)

    with col_right:
        st.markdown("### 📊 Статистика")
        st.metric(label="Найдено пчел на рамке", value=f"{bee_count} шт.")
        st.success("Анализ завершен. Данные готовы для мониторинга.")
        
        st.markdown("""
        <div class="info-card">
        <strong>Точность:</strong> Использование YOLO11 обеспечивает высокую скорость и фиксацию объектов, которые трудно заметить человеческим глазом.
        </div>
        """, unsafe_allow_html=True)
else:
    with col_right:
        st.info("Ожидание изображения для начала учёта...")

st.markdown("---")

# --- ИНФОРМАЦИОННЫЙ БЛОК (ИЗ ПРЕЗЕНТАЦИИ) ---
st.markdown("## 🔬 О проекте")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="info-card">
    <h3>📢 Актуальность</h3>
    Владельцам сотен ульев физически невозможно заглядывать в каждый улей ежедневно. 
    BeeTraker позволяет заметить проблему до того, как колония погибнет полностью.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="info-card">
    <h3>💡 Решение</h3>
    Веб-сайт автоматически распознаёт и отслеживает пчёл. Система заменяет многочасовой ручной труд 
    и обеспечивает доступность мониторинга для каждого.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="info-card">
    <h3>🚀 Перспективы</h3>
    Оптимизация нейросети и переход на новые версии YOLO для работы на высоких скоростях. 
    Дообучение модели для распознавания болезней.
    </div>
    """, unsafe_allow_html=True)

# Нижний колонтитул
st.markdown(
    '<p style="text-align: center; color: #8892b0; margin-top: 50px;">Всероссийский фестиваль "Научная вселенная Первых" • Пермский край</p>', 
    unsafe_allow_html=True
)
