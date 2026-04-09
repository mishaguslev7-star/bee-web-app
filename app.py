import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- 1. CONFIG ---
st.set_page_config(
    page_title="BeeTraker AI", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. УСИЛЕННЫЙ CSS (Фикс кнопки + Цвета презентации) ---
st.markdown("""
    <style>
    /* Полная очистка интерфейса Streamlit */
    header, footer, #MainMenu {visibility: hidden !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    
    /* Исправление черных оверлеев и блокировок */
    .stDeployButton {display:none !important;}
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #FFB800 !important;
        background: #FFFFFF !important;
        z-index: 999 !important;
    }

    /* Шрифты и фон */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .stApp { background-color: #F8FAFC; color: #1E293B; }

    /* Заголовки в стиле презентации */
    h1 {
        color: #1E293B !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        border-left: 15px solid #FFB800;
        padding-left: 20px;
        margin-bottom: 0px !important;
    }
    h2, h3 { color: #1E293B !important; font-weight: 700 !important; }

    /* Инфо-карточки */
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        border-top: 6px solid #FFB800;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
        height: 100%;
        margin-bottom: 20px;
    }

    /* Метрики */
    [data-testid="stMetricValue"] {
        color: #FFB800 !important;
        font-size: 4rem !important;
        font-weight: 800 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 4. ВЕРХНЯЯ ПАНЕЛЬ ---
st.markdown("<p style='color: #64748B; font-weight: 600; letter-spacing: 3px; margin-bottom:0;'>НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ • ПЕРМСКИЙ КРАЙ</p>", unsafe_allow_html=True)
c_t, c_l = st.columns([4, 1])
with c_t:
    st.title("ПЧЕЛИНЫЙ УЧЁТ")
    st.markdown("<p style='font-size: 1.3rem; color: #475569;'>Комплексное ИИ-решение для промышленного и частного пчеловодства</p>", unsafe_allow_html=True)
with c_l:
    st.markdown("""
    <div style="background: #1E293B; color: #FFB800; padding: 15px; border-radius: 15px; text-align: center; font-weight: 800; margin-top: 10px;">
        BEE SCAN v11
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 5. ОСНОВНОЙ ФУНКЦИОНАЛ ---
col_left, col_right = st.columns([1.6, 1])

with col_left:
    st.markdown("### 📤 Загрузка данных")
    # Уникальный ключ помогает избежать конфликтов с прошлыми сессиями
    file = st.file_uploader("Загрузить фото", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed", key="uploader_final")

if file:
    img = Image.open(file)
    with st.spinner('Нейросеть YOLO11 анализирует кадр...'):
        results = model(img)[0]
        # Отключаем маски, чтобы убрать синие пятна
        annotated_img = results.plot(masks=False, kpts=False, probs=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        count = len(results.boxes)
    
    with col_left:
        st.image(annotated_img, use_container_width=True, caption="Результат работы детектора")
    
    with col_right:
        st.markdown("### 📊 Аналитика")
        st.metric(label="ОБНАРУЖЕНО ОСОБЕЙ", value=f"{count}")
        
        st.markdown(f"""
        <div class="info-card">
            <strong>Технический отчет:</strong><br>
            • Объект: Apis mellifera<br>
            • Состояние: {'Требует внимания' if count < 10 else 'Стабильное'}<br>
            • Плотность: {round(count/1.2, 1)} ед/см² (условно)
        </div>
        """, unsafe_allow_html=True)
        
        # Кнопка скачивания
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("📥 Скачать результат анализа", res_bytes, "bee_analysis.jpg", "image/jpeg")
else:
    with col_right:
        st.info("Пожалуйста, загрузите фотографию рамки или летка для запуска системы распознавания.")

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. ИНФОРМАЦИОННЫЙ БЛОК (ИЗ ПРЕЗЕНТАЦИИ) ---
st.markdown("## 🔬 О ПЛАТФОРМЕ")

# Ряд 1: Актуальность и Решение
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="info-card">
        <h3>🚨 Почему это важно?</h3>
        Пчела движется со скоростью до <b>30 км/ч</b>. Человек не способен мгновенно посчитать 50-100 особей. 
        Наше решение предотвращает <b>Коллапс пчелиных семей</b>, позволяя заметить снижение популяции на ранних стадиях.
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="info-card">
        <h3>💡 Наше решение</h3>
        Веб-сервис на базе <b>YOLO11</b> автоматически распознаёт и отслеживает пчёл. 
        Это заменяет многочасовой ручной труд и делает мониторинг доступным даже для начинающих пчеловодов.
    </div>
    """, unsafe_allow_html=True)

# Ряд 2: Цели и ЦА
c3, c4 = st.columns(2)
with c3:
    st.markdown("""
    <div class="info-card">
        <h3>👥 Целевая аудитория</h3>
        • Пчеловоды и фермерские хозяйства<br>
        • Учёные-энтомологи и биологи<br>
        • Научно-исследовательские институты (НИИ)<br>
        • Экологические организации
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Будущее проекта</h3>
        Дообучение модели для определения болезней (варроатоз) и 
        запуск мобильного приложения для работы в полевых условиях без доступа к интернету.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #94A3B8;'>BeeTraker AI • Пермь 2026</p>", unsafe_allow_html=True)
