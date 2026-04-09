import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Пчелиный учёт | AI", page_icon="🐝", layout="wide")

# --- КАСТОМНЫЙ ДИЗАЙН (соответствует стилю презентации "Научная вселенная Первых") ---
st.markdown("""
    <style>
    /* Импорт шрифта */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }

    /* Скрываем меню Streamlit и кнопку GitHub */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Фон и основные цвета в стиле презентации */
    .stApp {
        background: linear-gradient(145deg, #0B1120 0%, #0F172A 100%);
        color: #E2E8F0;
    }
    
    /* Анимированный градиентный бордер для главного контейнера */
    .main > div {
        padding: 2rem 3rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(250, 204, 21, 0.2);
        box-shadow: 0 20px 40px -12px rgba(0,0,0,0.8);
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    /* Анимации */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(250, 204, 21, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(250, 204, 21, 0); }
        100% { box-shadow: 0 0 0 0 rgba(250, 204, 21, 0); }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    /* Заголовки в стиле презентации (желтый + тень) */
    h1, h2, h3 {
        color: #FACC15 !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        text-shadow: 0 4px 12px rgba(250, 204, 21, 0.15);
    }
    
    h1 {
        font-size: 3.5rem !important;
        border-left: 8px solid #FACC15;
        padding-left: 1.5rem;
        animation: float 4s ease-in-out infinite;
    }
    
    /* Метрики */
    .stMetric {
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        padding: 24px 28px !important;
        border-radius: 20px !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 8px 0 #020617 !important;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        border-color: #FACC15 !important;
        box-shadow: 0 12px 0 #020617 !important;
    }
    
    .stMetric label {
        color: #94A3B8 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem !important;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        color: #FACC15 !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
    }

    /* Кнопка Uploader */
    div[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        padding: 1.5rem;
        border-radius: 24px;
        border: 2px dashed #475569;
        transition: border 0.2s;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #FACC15;
    }

    /* Карточки с информацией */
    .info-card {
        background: rgba(30, 41, 59, 0.5);
        padding: 1.8rem;
        border-radius: 24px;
        border-left: 6px solid #FACC15;
        backdrop-filter: blur(4px);
        height: 100%;
        transition: all 0.3s;
    }
    
    .info-card:hover {
        background: rgba(30, 41, 59, 0.8);
        transform: scale(1.01);
    }

    /* Стилизация полосы разделителя */
    hr {
        margin: 3rem 0;
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FACC15, transparent);
    }
    
    /* Анимация для спиннера */
    .stSpinner > div {
        border-top-color: #FACC15 !important;
    }
    
    /* Стили для изображений */
    .stImage {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    try:
        # Убедитесь, что файл best.pt лежит в той же папке, что и скрипт
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

model = load_model()

# --- ШАПКА И ЗАГОЛОВОК ---
st.markdown("""
<div style="text-align: left; margin-bottom: 20px;">
    <span style="color: #94A3B8; font-weight: 600; letter-spacing: 3px; font-size: 1rem;">ВСЕРОССИЙСКИЙ ФЕСТИВАЛЬ • НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ</span>
</div>
""", unsafe_allow_html=True)

col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🐝 ПЧЕЛИНЫЙ УЧЁТ")
    st.markdown("""
    <p style="font-size: 1.4rem; color: #CBD5E1; margin-top: -15px; margin-bottom: 30px; font-weight: 400;">
    Интеллектуальный мониторинг экосистемы пасеки
    </p>
    """, unsafe_allow_html=True)
with col_logo:
    st.markdown("""
    <div style="background: #FACC15; padding: 12px 18px; border-radius: 60px; text-align: center; box-shadow: 0 6px 0 #B45309; margin-top: 20px;">
        <span style="color: #0F172A; font-weight: 800; font-size: 1.2rem;">AI ТРЕКЕР</span>
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
        # Чтение изображения
        img = Image.open(file)
        
        with st.spinner('🧠 Искусственный интеллект анализирует снимок...'):
            # Обработка YOLO
            results = model(img)[0]
            
            # Исправление ошибки: plot() без аргументов или с базовыми параметрами
            # Для старых версий используем просто plot()
            try:
                annotated_img = results.plot()  # Пробуем без аргументов
            except:
                # Если не работает, пробуем с минимальными аргументами
                annotated_img = results.plot(conf=True, labels=True, boxes=True)
            
            # Конвертация BGR (OpenCV) -> RGB (PIL/Streamlit)
            if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Подсчет пчел
            count = len(results.boxes) if results.boxes is not None else 0
        
        with col_upload:
            # Показываем информацию о файле
            file_size = len(file.getvalue()) / 1024  # Размер в KB
            st.caption(f"📎 {file.name} • {file_size:.1f}KB")
            
            # Отображаем размеченное изображение
            st.image(annotated_img, use_container_width=True, caption="Результат распознавания")
        
        with col_res:
            st.markdown("### 📊 Результаты мониторинга")
            
            # Исправлено: передаем значение как строку с единицами измерения
            st.metric(label="🐝 ОБНАРУЖЕНО ПЧЁЛ", value=f"{count} шт.")
            
            # Дополнительная аналитика
            st.markdown("""
            <div style="background: #1e293b; padding: 20px; border-radius: 16px; margin-top: 20px; border: 1px solid #334155;">
                <h4 style="margin-top: 0; color: #FACC15;">Анализ активности</h4>
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
            
            # Кнопка скачивания размеченного фото
            # Конвертируем обратно в BGR для сохранения в JPEG
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
        st.info("Попробуйте загрузить другое изображение или проверьте формат файла.")
        
elif model is None:
    with col_res:
        st.error("❌ Модель не загружена. Проверьте наличие файла 'best.pt' в директории приложения.")
else:
    with col_res:
        st.markdown("### 📊 Результаты мониторинга")
        st.info("👈 Загрузите фотографию улья или рамки, чтобы начать автоматический учёт.")
        # Визуальная заглушка
        st.markdown("""
        <div style="background: #0f172a; border-radius: 24px; padding: 30px; text-align: center; border: 1px dashed #475569; margin-top: 20px;">
            <span style="font-size: 48px;">🐝</span>
            <p style="color: #94a3b8; margin-top: 10px;">Ожидание загрузки данных...</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- НИЖНЯЯ ЧАСТЬ: О ПРОЕКТЕ ---
st.header("📖 О ПРОЕКТЕ «ПЧЕЛИНЫЙ УЧЁТ»")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top: 0;">🎯 Актуальность и решение</h3>
        <p style="font-size: 1.1rem; line-height: 1.6;">
        Владельцам сотен ульев физически невозможно заглядывать в каждый улей ежедневно. 
        Без автоматизации они узнают о проблеме, когда становится уже слишком поздно. 
        Пчела движется со скоростью до <strong>30 км/ч</strong> — человеческий глаз не способен точно зафиксировать и посчитать 50-100 летающих особей.
        </p>
        <p style="margin-top: 15px; background: #2a3a55; padding: 10px; border-radius: 12px;">
        ✅ <strong>Наше решение:</strong> Веб-сайт с нейросетью, который автоматически распознаёт и отслеживает пчёл на фото и видео. 
        Система сама анализирует изображение и выдает понятный результат.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top: 0;">👥 Целевая аудитория</h3>
        <ul style="font-size: 1.1rem; line-height: 2;">
            <li>🧑‍🌾 Пчеловоды и владельцы пасек</li>
            <li>🔬 Учёные-энтомологи и биологи</li>
            <li>🏛️ Научно-исследовательские институты</li>
            <li>🌾 Фермерские хозяйства</li>
        </ul>
        <h3 style="margin-top: 25px;">🚀 Перспективы проекта</h3>
        <ul>
            <li>Переход на YOLOv8/v10 для трекинга в реальном времени</li>
            <li>Дообучение для распознавания болезней (варроатоз, гнилец)</li>
            <li>Разработка мобильного приложения «навел камеру — получил отчёт»</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- ФУТЕР В СТИЛЕ ПРЕЗЕНТАЦИИ ---
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; color: #64748B; padding: 10px 0;">
    <div>
        <strong style="color: #FACC15;">Лебедянцев Лев • Гуслев Михаил</strong><br>
        Пермский край, г. Пермь, СОШ «Лицей Пермского педагогического»
    </div>
    <div style="text-align: right;">
        <span style="font-size: 0.9rem;">Всероссийский фестиваль<br>«Научная вселенная Первых»</span>
    </div>
</div>
""", unsafe_allow_html=True)
