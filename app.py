import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Пчелиный учёт | AI", page_icon="🐝", layout="wide")

# --- СВЕТЛАЯ ТЕМА В СТИЛЕ ПРЕЗЕНТАЦИИ ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Светлый фон */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
    }
    
    .main > div {
        padding: 2rem 3rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(234, 179, 8, 0.3);
        box-shadow: 0 20px 40px -12px rgba(0,0,0,0.1);
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

    /* Заголовки в янтарных тонах */
    h1, h2, h3 {
        color: #B45309 !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }
    
    h1 {
        font-size: 3.5rem !important;
        border-left: 8px solid #EAB308;
        padding-left: 1.5rem;
        animation: float 4s ease-in-out infinite;
        color: #92400E !important;
    }
    
    h2 {
        color: #B45309 !important;
        border-bottom: 2px solid #FDE68A;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #92400E !important;
    }

    /* Метрики */
    .stMetric {
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 24px 28px !important;
        border-radius: 20px !important;
        border: 2px solid #FDE68A !important;
        box-shadow: 0 8px 0 #D97706 !important;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        border-color: #EAB308 !important;
        box-shadow: 0 12px 0 #B45309 !important;
    }
    
    .stMetric label {
        color: #78350F !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        color: #92400E !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
    }

    /* Кнопка Uploader */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 251, 235, 0.8);
        padding: 1.5rem;
        border-radius: 24px;
        border: 2px dashed #FBBF24;
        transition: border 0.2s;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #D97706;
        background: rgba(255, 251, 235, 1);
    }

    /* Карточки с информацией */
    .info-card {
        background: rgba(255, 251, 235, 0.7);
        padding: 1.8rem;
        border-radius: 24px;
        border-left: 6px solid #EAB308;
        backdrop-filter: blur(4px);
        height: 100%;
        transition: all 0.3s;
        color: #451A03;
    }
    
    .info-card:hover {
        background: rgba(254, 243, 199, 0.9);
        transform: scale(1.01);
    }
    
    .info-card h3 {
        color: #92400E !important;
    }
    
    .info-card ul li {
        color: #451A03;
    }

    hr {
        margin: 3rem 0;
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FBBF24, transparent);
    }
    
    .stSpinner > div {
        border-top-color: #EAB308 !important;
    }
    
    .stImage {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.1);
        border: 2px solid #FDE68A;
    }
    
    /* Текст */
    p, li, div {
        color: #451A03;
    }
    
    .stSuccess {
        background-color: #DCFCE7 !important;
        color: #166534 !important;
        border-left: 4px solid #22C55E !important;
    }
    
    .stWarning {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border-left: 4px solid #F59E0B !important;
    }
    
    .stInfo {
        background-color: #EFF6FF !important;
        color: #1E3A8A !important;
        border-left: 4px solid #3B82F6 !important;
    }
    
    .stError {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
        border-left: 4px solid #EF4444 !important;
    }
    
    /* Кнопки */
    .stDownloadButton button {
        background: linear-gradient(135deg, #EAB308, #CA8A04) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 0 #92400E !important;
        transition: all 0.2s !important;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 0 #78350F !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- ЗАГРУЗКА МОДЕЛИ С ПРОВЕРКОЙ ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    
    # Проверяем существование файла
    if not os.path.exists(model_path):
        st.error(f"❌ Файл модели '{model_path}' не найден в директории приложения.")
        st.info("💡 Поместите обученную модель YOLO с именем 'best.pt' в корневую папку проекта.")
        return None
    
    # Проверяем размер файла (не пустой ли)
    if os.path.getsize(model_path) == 0:
        st.error(f"❌ Файл модели '{model_path}' пуст (0 байт).")
        return None
    
    try:
        # Загружаем модель с явным указанием пути
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {str(e)}")
        st.info("💡 Убедитесь, что файл 'best.pt' не поврежден и является корректной моделью YOLO.")
        return None

model = load_model()

# --- ШАПКА ---
st.markdown("""
<div style="text-align: left; margin-bottom: 20px;">
    <span style="color: #78350F; font-weight: 600; letter-spacing: 3px; font-size: 1rem;">🪐 НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ</span>
</div>
""", unsafe_allow_html=True)

col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🐝 ПЧЕЛИНЫЙ УЧЁТ")
    st.markdown("""
    <p style="font-size: 1.4rem; color: #78350F; margin-top: -15px; margin-bottom: 30px; font-weight: 400;">
    Интеллектуальный мониторинг экосистемы пасеки
    </p>
    """, unsafe_allow_html=True)
with col_logo:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #EAB308, #CA8A04); padding: 12px 18px; border-radius: 60px; text-align: center; box-shadow: 0 6px 0 #92400E;">
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
            
            # Безопасный вызов plot()
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
            st.image(annotated_img, use_container_width=True, caption="Результат распознавания")
        
        with col_res:
            st.markdown("### 📊 Результаты мониторинга")
            st.metric(label="🐝 ОБНАРУЖЕНО ПЧЁЛ", value=f"{count} шт.")
            
            st.markdown("""
            <div style="background: #FFFBEB; padding: 20px; border-radius: 16px; margin-top: 20px; border: 1px solid #FDE68A;">
                <h4 style="margin-top: 0; color: #92400E;">📈 Анализ активности</h4>
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
        <div style="background: #FEF3C7; border-radius: 16px; padding: 20px; border-left: 4px solid #EAB308;">
            <p style="margin: 0;"><strong>📁 Что нужно сделать:</strong></p>
            <ol style="margin-top: 10px;">
                <li>Убедитесь, что файл <code>best.pt</code> находится в корневой папке приложения</li>
                <li>Файл не должен быть пустым (размер > 0 байт)</li>
                <li>Модель должна быть обучена для распознавания пчёл</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
else:
    with col_res:
        st.markdown("### 📊 Результаты мониторинга")
        st.info("👈 Загрузите фотографию улья или рамки, чтобы начать автоматический учёт.")
        st.markdown("""
        <div style="background: #FFFBEB; border-radius: 24px; padding: 30px; text-align: center; border: 2px dashed #FBBF24; margin-top: 20px;">
            <span style="font-size: 48px;">🐝</span>
            <p style="color: #78350F; margin-top: 10px; font-weight: 500;">Ожидание загрузки данных...</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- АКТУАЛЬНОСТЬ ПРОЕКТА (ИЗ ПРЕЗЕНТАЦИИ) ---
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

# --- РЕШЕНИЕ И ТЕХНОЛОГИИ ---
st.header("💡 НАШЕ РЕШЕНИЕ")

col_sol1, col_sol2 = st.columns(2)

with col_sol1:
    st.markdown("""
    <div class="info-card">
        <h3>🌐 Веб-платформа с ИИ</h3>
        <p style="font-size: 1.1rem; line-height: 1.6;">
        Создан веб-сайт, который автоматически распознаёт и отслеживает пчёл на загруженных фотографиях и видео. 
        Система сама анализирует изображение и выдает понятный результат.
        </p>
        <p style="margin-top: 15px;">
        ✅ Разработка заменяет собой многочасовой ручной труд<br>
        ✅ Обеспечивает доступность мониторинга для людей без технических знаний<br>
        ✅ Позволяет отслеживать активность пчёл во времени
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_sol2:
    st.markdown("""
    <div class="info-card">
        <h3>👥 Целевая аудитория</h3>
        <ul style="font-size: 1.1rem; line-height: 2;">
            <li>🧑‍🌾 Пчеловоды и фермеры</li>
            <li>🔬 Учёные-биологи и энтомологи</li>
            <li>🏛️ Научно-исследовательские институты</li>
            <li>🌾 Сельскохозяйственные предприятия</li>
        </ul>
        <h3 style="margin-top: 25px;">🚀 Перспективы развития</h3>
        <ul>
            <li>📱 Мобильное приложение для смартфонов</li>
            <li>🎯 Распознавание признаков болезней пчёл</li>
            <li>⚡ Трекинг в реальном времени на YOLOv10</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- ФУТЕР ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #78350F; padding: 20px 0;">
    <p style="font-size: 1.2rem; margin-bottom: 5px;">
        <strong style="color: #92400E;">🪐 Научная вселенная Первых</strong>
    </p>
    <p style="margin-top: 0; opacity: 0.8;">
        Всероссийский фестиваль • 2026
    </p>
</div>
""", unsafe_allow_html=True)
