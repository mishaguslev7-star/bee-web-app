import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- 1. НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="BeeTracker AI", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ВСЕ ФУНКЦИИ СКЛОНЕНИЯ (ДОЛЖНЫ БЫТЬ В НАЧАЛЕ) ---
def get_bee_word(n):
    if 11 <= n % 100 <= 19:
        return "особей"
    last_digit = n % 10
    if last_digit == 1:
        return "особь"
    if 2 <= last_digit <= 4:
        return "особи"
    return "особей"

def get_fix_word(n):
    # Если заканчивается на 1 (но не 11), то "зафиксирована"
    if n % 10 == 1 and n % 100 != 11:
        return "зафиксирована"
    return "зафиксировано"

# --- 3. DARK UI/UX DESIGN ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@800&family=Montserrat:wght@300;400;700&display=swap');
    
    header, footer, #MainMenu {visibility: hidden !important;}
    
    .stApp {
        background-color: #0F172A;
    }

    /* КРУПНЫЕ СОТЫ НА ФОНЕ */
    .honeycomb-background {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: url('https://www.transparenttextures.com/patterns/hexellence.png');
        background-repeat: repeat;
        background-size: 300px;
        -webkit-mask-image: radial-gradient(circle at 10% 10%, black 0%, transparent 50%),
                             radial-gradient(circle at 90% 90%, black 0%, transparent 50%);
        mask-image: radial-gradient(circle at 10% 10%, black 0%, transparent 50%),
                    radial-gradient(circle at 90% 90%, black 0%, transparent 50%);
        opacity: 0.25;
        pointer-events: none;
        z-index: -1;
    }

    .main .block-container { 
        max-width: 1200px; 
        padding: 2rem 1rem !important; 
        position: relative;
        z-index: 10;
    }

    /* УВЕЛИЧЕННЫЙ ШРИФТ ОСНОВНОГО ТЕКСТА */
    p { 
        color: #CBD5E1; 
        line-height: 1.7; 
        margin: 0; 
        font-size: 1.1rem !important; /* Увеличено */
    }

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);
        color: #2DD4BF;
        text-transform: uppercase;
        letter-spacing: 4px;
        display: flex;
        align-items: center;
        gap: 20px;
        text-shadow: 0 0 20px rgba(45, 212, 191, 0.3);
    }

    /* АНИМАЦИЯ ПЧЕЛЫ */
    @keyframes bee-swing {
        0% { transform: rotate(-8deg); }
        50% { transform: rotate(8deg); }
        100% { transform: rotate(-8deg); }
    }
    .swinging-bee {
        display: inline-block;
        animation: bee-swing 2.5s ease-in-out infinite;
        transform-origin: center;
    }

    .info-card, .goal-card {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(45, 212, 191, 0.1);
        border-radius: 24px;
        padding: 2rem;
        height: 100%;
        font-size: 1.15rem; /* Текст в карточках чуть крупнее */
    }

    .goal-card {
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.15), rgba(15, 23, 42, 0.9)) !important;
        border: 2px solid #2DD4BF;
        margin-bottom: 2rem;
    }

    .section-header {
        font-family: 'Orbitron', sans-serif;
        color: #5EEAD4;
        letter-spacing: 2px;
        margin: 3.5rem 0 1.5rem 0;
        border-left: 5px solid #2DD4BF;
        padding-left: 15px;
        font-size: 1.6rem;
    }

    h3 { color: #5EEAD4 !important; font-size: 1.3rem !important; margin-bottom: 12px !important; }
    b { color: #5EEAD4; }

    [data-testid="stMetricValue"] { 
        color: #2DD4BF !important; 
        font-family: 'Orbitron', sans-serif; 
        font-size: 2.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="honeycomb-background"></div>', unsafe_allow_html=True)

# --- 4. ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            return YOLO(model_path)
        except Exception:
            return None
    return None

model = load_model()

# --- 5. ШАПКА ---
st.markdown("""
    <div class="main-title">
        <span class="swinging-bee" style="font-size: 1.2em;">🐝</span> BeeTracker
    </div>
    <p style="color: #94A3B8; margin-bottom: 2.5rem; padding-left: 5px; font-size: 1.2rem !important;">Интеллектуальный мониторинг экосистемы пасеки</p>
    
    <div class="goal-card">
        <h3>🎯 Цель проекта</h3>
        <p>Создание системы автоматического контроля за пчелами, которая помогает вовремя заметить экологическую угрозу и спасти пасеку от гибели.</p>
    </div>
""", unsafe_allow_html=True)

# --- 6. АНАЛИЗ ---
col_up, col_res = st.columns([1.4, 1], gap="large")

with col_up:
    st.markdown("### 📥 Анализ изображения")
    file = st.file_uploader("Загрузите фото", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    
    if file:
        img = Image.open(file)
        if model:
            with st.spinner('Нейросеть BeeTracker анализирует кадр...'):
                # Скорректированные параметры для плотных групп пчел
                results = model(img, conf=0.35, iou=0.6)[0] 
                annotated_img = results.plot(labels=True, boxes=True)
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img, use_container_width=True)
                count = len(results.boxes)

with col_res:
    if file and model:
        bee_word = get_bee_word(count)
        fix_word = get_fix_word(count)
        
        st.metric("ОБНАРУЖЕНО", f"{count} {bee_word}")
        
        if count > 0:
            avg_conf = np.mean(results.boxes.conf.cpu().numpy()) * 100
            st.metric("УВЕРЕННОСТЬ AI", f"{avg_conf:.1f}%")
        
        st.markdown('<div style="margin-top: 3.5rem;"></div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-card">
            <h3>📊 Текущий отчёт</h3>
            На снимке <b>{fix_word} {count} {bee_word}</b>.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
        
        res_bytes = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button("💾 Сохранить результат", res_bytes, "bee_report.jpg", use_container_width=True)

# --- 7. О ПРОЕКТЕ ---
st.markdown('<div class="section-header">🔍 О ПРОЕКТЕ</div>', unsafe_allow_html=True)
a1, a2 = st.columns(2, gap="medium")

with a1:
    st.markdown("""
    <div class="info-card">
        <h3>Почему это важно?</h3>
        Пчела летает со скоростью до <b>30 км/ч</b>. Владельцам сотен ульев физически невозможно заглядывать в каждый ежедневно — без автоматизации проблему замечают слишком поздно.
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="info-card">
        <h3>Как мы это решаем?</h3>
        Наш проект превращает хаотичное движение пчел в ценные данные. Используя нейросеть <b>YOLO11</b>, система мгновенно фиксирует активность на летке.
    </div>
    """, unsafe_allow_html=True)

# --- 8. ПЕРСПЕКТИВЫ ---
st.markdown('<div class="section-header">🚀 ПЕРСПЕКТИВЫ РАЗВИТИЯ</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns(3, gap="medium")

with p1:
    st.markdown('<div class="info-card"><h3>Скорость и точность</h3>Оптимизация нейросети для работы на более высоких скоростях.</div>', unsafe_allow_html=True)
with p2:
    st.markdown('<div class="info-card"><h3>Мобильное приложение</h3>Разработка версии для смартфонов для мгновенных отчетов прямо у улья.</div>', unsafe_allow_html=True)
with p3:
    st.markdown('<div class="info-card"><h3>Диагностика</h3>Распознавание подвидов пчел и определение признаков болезней.</div>', unsafe_allow_html=True)

st.markdown("<br><br><p style='text-align: center; color: #475569; font-size: 0.9rem !important;'>НАУЧНАЯ ВСЕЛЕННАЯ ПЕРВЫХ • 2026</p>", unsafe_allow_html=True)
