import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Пчелиный учёт | AI", page_icon="🐝", layout="wide")

# --- БИРЮЗОВО-СИНЯЯ ТЕМА ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Бирюзово-синий фон */
    .stApp {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
    }
    
    .main > div {
        padding: 2rem 3rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(20, 184, 166, 0.3);
        box-shadow: 0 20px 40px -12px rgba(0,0,0,0.08);
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    /* Заголовки в бирюзово-синих тонах */
    h1, h2, h3 {
        color: #0F766E !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }
    
    h1 {
        font-size: 3.5rem !important;
        border-left: 8px solid #14B8A6;
        padding-left: 1.5rem;
        animation: float 4s ease-in-out infinite;
        color: #0F766E !important;
    }
    
    h2 {
        color: #0D9488 !important;
        border-bottom: 2px solid #99F6E4;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #115E59 !important;
    }

    /* Метрики */
    .stMetric {
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 24px 28px !important;
        border-radius: 20px !important;
        border: 2px solid #5EEAD4 !important;
        box-shadow: 0 8px 0 #0F766E !important;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        border-color: #14B8A6 !important;
        box-shadow: 0 12px 0 #0D9488 !important;
    }
    
    .stMetric label {
        color: #134E4A !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        color: #0F766E !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
    }

    /* Кнопка Uploader */
    div[data-testid="stFileUploader"] {
        background: rgba(240, 253, 250, 0.8);
        padding: 1.5rem;
        border-radius: 24px;
        border: 2px dashed #2DD4BF;
        transition: border 0.2s;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #0D9488;
        background: rgba(204, 251, 241, 1);
    }

    /* Карточки с информацией */
    .info-card {
        background: rgba(240, 253, 250, 0.7);
        padding: 1.8rem;
        border-radius: 24px;
        border-left: 6px solid #14B8A6;
        backdrop-filter: blur(4px);
        height: 100%;
        transition: all 0.3s;
        color: #042F2E;
    }
    
    .info-card:hover {
        background: rgba(204, 251, 241, 0.9);
        transform: scale(1.01);
    }
    
    .info-card h3 {
        color: #115E59 !important;
    }
    
    .info-card ul li {
        color: #134E4A;
    }

    hr {
        margin: 3rem 0;
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2DD4BF, transparent);
    }
    
    .stSpinner > div {
        border-top-color: #14B8A6 !important;
    }
    
    .stImage {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.08);
        border: 2px solid #99F6E4;
    }
    
    /* Текст */
    p, li, div {
        color: #134E4A;
    }
    
    .stSuccess {
        background-color: #D1FAE5 !important;
        color: #065F46 !important;
        border-left: 4px solid #10B981 !important;
    }
    
    .stWarning {
        background-color: #CCFBF1 !important;
        color: #0F766E !important;
        border-left: 4px solid #14B8A6 !important;
    }
    
    .stInfo {
        background-color: #E0F2FE !important;
        color: #0369A1 !important;
        border-left: 4px solid #0EA5E9 !important;
    }
    
    .stError {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
        border-left: 4px solid #EF4444 !important;
    }
    
    /* Кнопки */
    .stDownloadButton button {
        background: linear-gradient(135deg, #14B8A6, #0D9488) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 0 #0F766E !important;
        transition: all 0.2s !important;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 0 #115E59 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- ЗАГРУЗКА МОДЕЛИ YOLOv11 ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Файл модели '{model_path}' не найден в директории приложения.")
        st.info("💡 Поместите обученную модель YOLOv11 с именем 'best.pt' в корневую папку проекта.")
        return None
    
    if os.path.getsize(model_path) == 0:
        st.error(f"❌ Файл модели '{model_path}' пуст (0 байт).")
        return None
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {str(e)}")
        st.info("💡 Убедитесь, что файл 'best.pt' не поврежден и является корректной моделью YOLOv11.")
        return None

model = load_model()

# --- ШАПКА ---
st.markdown("""
<div style="text-align: left; margin-bottom: 20px;">
    <span style="color: #0F766E; font-weight: 600; letter-spacing: 3px; font-size: 1rem;">🪐 НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ</span>
</div>
""", unsafe_allow_html=True)

col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🐝 ПЧЕЛИНЫЙ УЧЁТ")
    st.markdown("""
    <p style="font-size: 1.4rem; color: #134E4A; margin-top: -15px; margin-bottom: 30px; font-weight: 400;">
    Интеллектуальный мониторинг экосистемы пасеки
    </p>
    """, unsafe_allow_html=True)
with col_logo:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #14B8A6, #0D9488); padding: 12px 18px; border-radius: 60px; text-align: center; box-shadow: 0 6px 0 #0F766E;">
        <span style="color: white; font-weight: 800; font-size: 1.2rem;">AI ТРЕКЕР</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
col_upload, col_res = st.columns([1.5, 1])

with col_upload:
    st.markdown("### 📸 Загрузка снимка")
    file = st.file_uploader("Выберите фото для анализа", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

# Логика обработки
if file is not None and model is not None:
    try:
        img = Image.open(file)
        
        with st.spinner('🧠 Искусственный интеллект анализирует снимок...'):
            results = model(img)[0]
            
            try:
                annotated_img = results.plot()
            except:
                annotated_img = results.plot(conf=True, labels=True, boxes=True)
            
            if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            count = len(results.boxes) if results.boxes is not None else 0
        
        with col_upload:
            file_size = len(file.getvalue()) / 1024
            st.caption(f"📎 {file.name} • {file_size:.1f}KB")
            st.image(annotated_img, use_container_width=True, caption="Результат распознавания YOLOv11")
        
        with col_res:
            st.markdown("### 📊 Результаты мониторинга")
            st.metric(label="🐝 ОБНАРУЖЕНО ПЧЁЛ", value=f"{count} шт.")
            
            st.markdown("""
            <div style="background: #F0FDFA; padding: 20px; border-radius: 16px; margin-top: 20px; border: 1px solid #99F6E4;">
                <h4 style="margin-top: 0; color: #0F766E;">📈 Анализ активности</h4>
            """, unsafe_allow_html=True)
            
            if count == 0:
                st.warning("⚠️ Пчёлы не обнаружены. Проверьте рамку или освещение.")
            elif count < 10:
                st.info("🔍 Низкая активность. Возможно, семья в стадии развития или похолодание.")
            elif count < 30:
                st.success("✅ Нормальная активность для данного участка рамки.")
            else:
                st.success("🚀 Высокая плотность! Отличный признак сильной семьи.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            save_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            is_success, buffer = cv2.imencode(".jpg", save_img)
            if is_success:
                st.download_button(
                    label="📥 Скачать размеченное фото",
                    data=buffer.tobytes(),
                    file_name="bee_analysis.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
                
    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {str(e)}")
        st.info("Попробуйте загрузить другое изображение.")
        
elif model is None:
    with col_res:
        st.markdown("### 📊 Результаты мониторинга")
        st.warning("⚠️ Модель не загружена")
        st.markdown("""
        <div style="background: #CCFBF1; border-radius: 16px; padding: 20px; border-left: 4px solid #14B8A6;">
            <p style="margin: 0; color: #0F766E;"><strong>📁 Что нужно сделать:</strong></p>
            <ol style="margin-top: 10px; color: #134E4A;">
                <li>Убедитесь, что файл <code>best.pt</code> находится в корневой папке приложения</li>
                <li>Файл не должен быть пустым (размер > 0 байт)</li>
                <li>Модель должна быть YOLOv11, обученная для распознавания пчёл</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
else:
    with col_res:
        st.markdown("### 📊 Результаты мониторинга")
        st.info("👈 Загрузите фотографию улья или рамки, чтобы начать автоматический учёт.")
        st.markdown("""
        <div style="background: #F0FDFA; border-radius: 24px; padding: 30px; text-align: center; border: 2px dashed #5EEAD4; margin-top: 20px;">
            <span style="font-size: 48px;">🐝</span>
            <p style="color: #0F766E; margin-top: 10px; font-weight: 500;">Ожидание загрузки данных...</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- АКТУАЛЬНОСТЬ ПРОЕКТА ---
st.header("🎯 АКТУАЛЬНОСТЬ ПРОЕКТА")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card" style="text-align: center;">
        <span style="font-size: 48px;">🏭</span>
        <h3 style="margin-top: 10px;">Масштаб пасек</h3>
        <p style="font-size: 1.1rem; line-height: 1.6;">
        Владельцам сотен ульев физически невозможно заглядывать в каждый улей ежедневно. 
        Без автоматизации они узнают о проблеме когда становится уже слишком поздно.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card" style="text-align: center;">
        <span style="font-size: 48px;">⚡</span>
        <h3 style="margin-top: 10px;">Скорость движения</h3>
        <p style="font-size: 1.1rem; line-height: 1.6;">
        Пчела движется со скоростью до <strong>25-30 км/ч</strong>. 
        Человеческий глаз не способен точно зафиксировать и посчитать 50-100 летающих особей.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card" style="text-align: center;">
        <span style="font-size: 48px;">🛡️</span>
        <h3 style="margin-top: 10px;">Предотвращение коллапса</h3>
        <p style="font-size: 1.1rem; line-height: 1.6;">
        С помощью нейросети можно заметить, что пчелы стали вялыми или их количество резко сократилось. 
        Это позволяет спасти колонию до того, как она погибнет полностью.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- РЕШЕНИЕ (без блока о разработке) ---
st.header("💡 НАШЕ РЕШЕНИЕ")

st.markdown("""
<div class="info-card" style="max-width: 900px; margin: 0 auto;">
    <p style="font-size: 1.2rem; line-height: 1.8; text-align: center;">
    Веб-сайт автоматически распознаёт и отслеживает пчёл на загруженных фотографиях и видео. 
    Система на базе <strong>YOLOv11</strong> сама анализирует изображение и выдает понятный результат, 
    заменяя многочасовой ручной труд и обеспечивая доступность мониторинга без технических знаний.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- ЦЕЛЕВАЯ АУДИТОРИЯ И ПЕРСПЕКТИВЫ ---
col_aud, col_persp = st.columns(2)

with col_aud:
    st.markdown("""
    <div class="info-card">
        <h3 style="text-align: center;">👥 Целевая аудитория</h3>
        <ul style="font-size: 1.1rem; line-height: 2.2;">
            <li>🧑‍🌾 Пчеловоды и владельцы пасек</li>
            <li>🔬 Учёные-биологи и энтомологи</li>
            <li>🏛️ Научно-исследовательские институты</li>
            <li>🌾 Фермерские хозяйства</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_persp:
    st.markdown("""
    <div class="info-card">
        <h3 style="text-align: center;">🚀 Перспективы проекта</h3>
        <ul style="font-size: 1.1rem; line-height: 2.2;">
            <li>📱 Мобильное приложение для смартфонов</li>
            <li>🎯 Распознавание признаков болезней пчёл</li>
            <li>⚡ Трекинг в реальном времени на YOLOv11</li>
            <li>🐝 Определение подвидов пчёл</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- ФУТЕР ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #0F766E; padding: 20px 0;">
    <p style="font-size: 1.2rem; margin-bottom: 5px;">
        <strong style="color: #115E59;">🪐 Научная вселенная Первых</strong>
    </p>
    <p style="margin-top: 0; opacity: 0.8;">
        Всероссийский фестиваль • 2026
    </p>
</div>
""", unsafe_allow_html=True)
