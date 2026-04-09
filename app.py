import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Настройка страницы (дизайн)
st.set_page_config(
    page_title="BeeTraker AI: Профессиональный подсчет пчел",
    page_icon="🐝",
    layout="wide"
)

# --- БЛОК АНИМАЦИИ И КАСТОМНОГО ДИЗАЙНА ---
# Мы добавляем CSS-анимацию keyframes и скрываем лишние элементы GitHub
st.markdown("""
    <style>
    /* 1. Скрываем меню Streamlit и кнопку GitHub */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 2. Общая анимация появления для всех элементов */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translate3d(0, -20px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    .stApp { 
        background-color: #0f172a; color: white; 
        animation: fadeInDown 1.2s ease-out; /* Плавное появление всего сайта */
    }
    
    /* 3. Стилизация заголовков и текста */
    h1, h2, h3 { color: #facc15; font-family: 'Inter', sans-serif; font-weight: 800; }
    .stMarkdown p { color: #a1a1aa; font-size: 1.1rem; }
    
    /* 4. Стилизация карточек Проблема/Решение/Цель с тенью и анимацией ховера */
    .css-163rgbv, [data-testid="stVerticalBlock"] > div > div > [data-testid="stMarkdownContainer"] {
        background-color: #1e293b; 
        border-radius: 20px; 
        padding: 25px; 
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    .css-163rgbv:hover {
        transform: translateY(-5px); /* Карточка немного приподнимается */
        box-shadow: 0 10px 20px rgba(250, 204, 21, 0.2); /* Появляется золотистое свечение */
    }
    
    /* 5. Стилизация метрики (числа пчел) */
    [data-testid="stMetricValue"] { color: #facc15; font-size: 3.5rem !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #a1a1aa !important; font-size: 1.1rem; }
    
    /* 6. Плавное появление кнопок и инпутов */
    .stButton, .stFileUploader { animation: fadeInDown 1.5s ease-out; }
    
    </style>
""", unsafe_allow_html=True)

# --- ИНИЦИАЛИЗАЦИЯ (Кешируем модель) ---
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt") # Убедись, что файл best.pt лежит рядом
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели. Убедитесь, что 'best.pt' лежит в папке GitHub. Подробнее: {e}")
        return None

model = load_model()

# --- ВЕРХНЯЯ ЧАСТЬ: ИНТЕРФЕЙС АНАЛИЗА ---
st.title("🐝 BeeTraker AI")
st.write("Профессиональный инструмент для мгновенного и точного подсчета пчел на рамке.")

# Создаем две колонки: слева загрузка, справа результат
col_upload, col_result = st.columns([1.5, 1])

with col_upload:
    st.subheader("📁 Загрузите изображение")
    uploaded_file = st.file_uploader("Выберите фото рамки (JPG, PNG)...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is None:
        st.info("Пожалуйста, загрузите фотографию для начала анализа.")

if uploaded_file is not None and model is not None:
    # 1. Загрузка фото
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # 2. РАБОТА НЕЙРОСЕТИ (реальное предсказание)
    with st.spinner('Нейросеть YOLO анализирует изображение...'):
        results = model(img_array)[0]
        
        # 3. ПОСТ-ОБРАБОТКА (Рисуем рамки)
        annotated_img = results.plot() 
        # Конвертируем цвета (из BGR в RGB)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        # Считаем количество пчел
        bee_count = len(results.boxes)

    # 4. ВЫВОД РЕЗУЛЬТАТОВ (в правую колонку)
    with col_upload:
        st.image(annotated_img, caption=f"Обработано AI. Найдено {bee_count} пчел.", use_container_width=True)

    with col_result:
        st.subheader("📊 Результат анализа")
        # Красивая карточка с числом пчел
        st.metric(label="Обнаружено объектов (пчел)", value=f"{bee_count} шт.")
        st.success(f"Анализ завершен успешно! Нейросеть YOLO {model.ckpt.get('version', '')} выполнила подсчет.")

st.markdown("---") # Разделительная линия

# --- НИЖНЯЯ ЧАСТЬ: ИНФОРМАЦИЯ О ПРОЕКТЕ (сразу на главной) ---
st.header("💡 О проекте BeeTraker")

# Создаем три колонки для красивых карточек
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    ### ⚠️ Проблема
    Ручной подсчет пчел на рамках — это трудоемкий, долгий и неточный процесс. При большом количестве ульев на пасеке пчеловод физически не успевает контролировать силу каждой семьи, что может привести к гибели колоний от болезней или голода.
    """)

with col_info2:
    st.markdown("""
    ### ⚡️ Решение
    BeeTraker AI использует передовую нейросеть **YOLO**, обученную на тысячах снимков пчел. Наша система за секунды анализирует фотографию рамки, рисует рамку вокруг каждой пчелы и выдает точное количество объектов.
    """)

with col_info3:
    st.markdown("""
    ### 🎯 Цель
    Автоматизировать мониторинг активности пчелиных семей. Это позволяет пчеловодам объективно оценивать динамику развития колоний, вовремя выявлять проблемы и принимать меры для предотвращения гибели пчел, повышая эффективность пасеки.
    """)
