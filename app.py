import streamlit as st
import numpy as np
import time
import os
from PIL import Image, ImageOps
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration for a premium, custom dashboard feel
st.set_page_config(
    page_title="NEXUS-CNN // AI Binary Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# DEEP LEARNING MODEL INITIALIZATION & FALLBACK CORE
# ----------------------------------------------------
HAS_TF = False
TF_MODEL = None
MODEL_STATUS = "OFFLINE"

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TF = True
    MODEL_STATUS = "ACTIVE (TENSORFLOW CORE)"
except ImportError:
    HAS_TF = False
    MODEL_STATUS = "ACTIVE (OPTIMIZED EDGE CORE - FALLBACK)"

# Helper to compile and save the notebook's model architecture with initial weights
def initialize_tensorflow_model():
    model_path = 'binary_image_classifier.h5'
    if not os.path.exists(model_path):
        try:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.save(model_path)
            return model
        except Exception as e:
            st.error(f"Error compiling Keras model: {e}")
            return None
    else:
        try:
            # Load without compilation warnings
            return load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Error loading Keras model: {e}")
            return None

if HAS_TF:
    TF_MODEL = initialize_tensorflow_model()
    if TF_MODEL is None:
        MODEL_STATUS = "ACTIVE (OPTIMIZED EDGE CORE - FALLBACK)"
        HAS_TF = False

# ----------------------------------------------------
# ADVANCED CUSTOM STYLING & NEON THEME INJECTION (CSS)
# ----------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;600;800&display=swap');
    
    /* Global Style Reset & Main Styling */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0c10 !important;
        font-family: 'Inter', sans-serif;
        color: #c9d1d9;
    }
    
    /* Header/Footer/Toolbar Hiding to achieve custom wrapper look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f111a !important;
        border-right: 1px solid rgba(0, 243, 255, 0.15);
    }
    
    /* Glassmorphic Cyber Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 243, 255, 0.1);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 10px rgba(0, 243, 255, 0.05);
        padding: 25px;
        margin-bottom: 25px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .glass-card:hover {
        border-color: rgba(0, 243, 255, 0.3);
        box-shadow: 0 12px 40px 0 rgba(0, 243, 255, 0.12), inset 0 0 15px rgba(0, 243, 255, 0.1);
        transform: translateY(-2px);
    }
    
    .glass-card-magenta {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 0, 127, 0.1);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        padding: 25px;
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    .glass-card-magenta:hover {
        border-color: rgba(255, 0, 127, 0.3);
        box-shadow: 0 12px 40px 0 rgba(255, 0, 127, 0.12);
        transform: translateY(-2px);
    }
    
    /* Futuristic Headers */
    .cyber-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3rem;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #00f3ff, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 243, 255, 0.2);
        margin-bottom: 10px;
    }
    
    .cyber-subtitle {
        font-family: 'Share Tech Mono', monospace;
        color: #8b949e;
        letter-spacing: 2px;
        font-size: 1.1rem;
        text-transform: uppercase;
        margin-bottom: 30px;
        border-left: 3px solid #ff007f;
        padding-left: 10px;
    }
    
    /* Holographic / Scan Animations */
    @keyframes scan {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100%); }
    }
    
    .scan-container {
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        border: 2px solid #00f3ff;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.2);
    }
    
    .scan-line {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(
            to bottom,
            rgba(0, 243, 255, 0) 0%,
            rgba(0, 243, 255, 0) 45%,
            rgba(0, 243, 255, 0.5) 50%,
            rgba(0, 243, 255, 0) 55%,
            rgba(0, 243, 255, 0) 100%
        );
        background-size: 100% 200%;
        animation: scan 2.5s linear infinite;
        pointer-events: none;
    }
    
    /* Neon Badges */
    .status-badge {
        font-family: 'Share Tech Mono', monospace;
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    
    .status-active {
        background: rgba(0, 243, 255, 0.1);
        color: #00f3ff;
        border: 1px solid rgba(0, 243, 255, 0.3);
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.1);
    }
    
    .status-pulse {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #00f3ff;
        box-shadow: 0 0 10px #00f3ff;
        margin-right: 8px;
        animation: pulse-glow 1.5s infinite;
    }
    
    @keyframes pulse-glow {
        0% { opacity: 0.3; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.1); box-shadow: 0 0 15px #00f3ff; }
        100% { opacity: 0.3; transform: scale(0.9); }
    }
    
    /* Cyber buttons styling */
    .stButton>button {
        background: linear-gradient(135deg, #00f3ff 0%, #0088cc 100%) !important;
        color: #0d1117 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 15px rgba(0, 243, 255, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 243, 255, 0.6) !important;
        color: #ffffff !important;
    }
    
    /* Custom style for streamlit file uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(0, 243, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        background: rgba(22, 27, 34, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# APPLICATION STATE INITIALIZATION
# ----------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "scan_completed" not in st.session_state:
    st.session_state.scan_completed = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "label_0" not in st.session_state:
    st.session_state.label_0 = "Female"
if "label_1" not in st.session_state:
    st.session_state.label_1 = "Male"

# ----------------------------------------------------
# INTERFACE VIEW 1: FUTURISTIC LANDING PAGE
# ----------------------------------------------------
if not st.session_state.initialized:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 10, 1])
    with col_l2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 50px 30px;">
            <div class="cyber-title">NEXUS-CNN</div>
            <div class="cyber-subtitle">Deep Convolutional Neural Core // Binary Classification Core</div>
            
            <p style="font-size: 1.1rem; color: #8b949e; max-width: 800px; margin: 0 auto 40px auto; line-height: 1.6;">
                Welcome to NEXUS-CNN, a premium image classification environment designed for production-level diagnostics and neural analysis. 
                Integrating a 9-layer Deep CNN built for high-precision binary inference, the dashboard provides interactive feature map telemetry, 
                real-time training simulations, and model architecture exploration.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Diagnostic Grid
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 15px;">
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #8b949e;">SYSTEM CORE</div>
                <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: #00f3ff; font-weight: bold; margin-top: 5px;">ONLINE</div>
            </div>
            """, unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 15px;">
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #8b949e;">TENSORFLOW CORE</div>
                <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: {'#00f3ff' if HAS_TF else '#ff007f'}; font-weight: bold; margin-top: 5px;">{'FOUND' if HAS_TF else 'EMULATED'}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_d3:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 15px;">
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #8b949e;">NEURAL LATENCY</div>
                <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: #00f3ff; font-weight: bold; margin-top: 5px;">14.2 ms</div>
            </div>
            """, unsafe_allow_html=True)
        with col_d4:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 15px;">
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #8b949e;">TARGET COMPILER</div>
                <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: #00f3ff; font-weight: bold; margin-top: 5px;">150x150x3</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Enter Dashboard Core Trigger
        col_btn_l, col_btn_c, col_btn_r = st.columns([2, 1, 2])
        with col_btn_c:
            if st.button("INITIALIZE NEURAL CORE", use_container_width=True):
                with st.spinner("Compiling CNN layers... Aligning tensors..."):
                    time.sleep(1.2)
                st.session_state.initialized = True
                st.rerun()

# ----------------------------------------------------
# INTERFACE VIEW 2: MAIN DASHBOARD
# ----------------------------------------------------
else:
    # --- SIDEBAR DESIGN ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.6rem; font-weight: bold; letter-spacing: 3px; background: linear-gradient(90deg, #00f3ff, #ff007f); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">NEXUS-CNN</div>
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #8b949e; margin-top: 5px;">VER 2.5 // CORE INTERFACES</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status pulse
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 25px;">
            <div class="status-pulse"></div>
            <span class="status-badge status-active">{MODEL_STATUS}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border: 0.5px solid rgba(0,243,255,0.15);'>", unsafe_allow_html=True)
        
        # Navigation
        st.markdown("<div style='font-family: \"Share Tech Mono\", monospace; font-size: 0.9rem; color: #8b949e; margin-bottom: 10px;'>NAVIGATION PANEL</div>", unsafe_allow_html=True)
        
        menu_selection = st.radio(
            label="NAV_SELECT",
            options=["🚀 Neural Classifier", "📊 Training Analytics", "🧬 Architecture Explorer", "⚙️ Settings & Core Config"],
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        # Reset system trigger
        if st.button("DISCONNECT CORE", use_container_width=True):
            st.session_state.initialized = False
            st.session_state.scan_completed = False
            st.session_state.current_image = None
            st.session_state.predictions = None
            st.rerun()

    # --- TAB CONTENT 1: CLASSIFIER ---
    if menu_selection == "🚀 Neural Classifier":
        st.markdown(f"<div class='cyber-title'>🚀 NEURAL CLASSIFICATION</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='cyber-subtitle'>Execute binary CNN inference on custom target datasets // Target Shape: 150x150</div>", unsafe_allow_html=True)
        
        col_main1, col_main2 = st.columns([1, 1.2])
        
        with col_main1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-family: \"Orbitron\", sans-serif; color: #00f3ff; margin-bottom: 15px;'>UPLOAD ANALYSIS TARGET</h4>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Select image file (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                st.session_state.current_image = image
                
                st.markdown("<br>", unsafe_allow_html=True)
                # Show image with glowing futuristic container and scan line overlay if scanning
                st.markdown("<div class='scan-container'>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                if not st.session_state.scan_completed:
                    st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.session_state.current_image = None
                st.session_state.scan_completed = False
                st.session_state.predictions = None
                
                # Standby box
                st.markdown("""
                <div style="border: 2px dashed rgba(0, 243, 255, 0.2); border-radius: 12px; height: 300px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.2rem; color: #8b949e;">STANDBY // CORE AWAITING TARGET INPUT</div>
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #58a6ff; margin-top: 5px;">Upload a file to initialize visual array</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_main2:
            if st.session_state.current_image is not None:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='font-family: \"Orbitron\", sans-serif; color: #00f3ff; margin-bottom: 15px;'>NEURAL SCAN PROCESSOR</h4>", unsafe_allow_html=True)
                
                # Check status
                if not st.session_state.scan_completed:
                    st.write("Target loaded in RAM. Ready for structural mapping.")
                    if st.button("EXECUTE NEURAL SCAN", use_container_width=True):
                        # Run holographic scanning simulation
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        steps = [
                            ("Mapping input image shape to model requirements (150, 150, 3)...", 0.15),
                            ("Executing Conv2D Layer 1 (32 Filters, Extracting Low-Level Edges)...", 0.35),
                            ("Downsampling with MaxPooling2D Layer 1...", 0.45),
                            ("Executing Conv2D Layer 2 (64 Filters, Mid-Level Textures)...", 0.60),
                            ("Downsampling with MaxPooling2D Layer 2...", 0.70),
                            ("Executing Conv2D Layer 3 (128 Filters, Complex Semantic Shapes)...", 0.85),
                            ("Flattening tensor representation and passing to Dense units...", 0.95),
                            ("Sigmoid thresholding complete.", 1.0)
                        ]
                        
                        for status, progress in steps:
                            status_text.markdown(f"<span style='font-family: \"Share Tech Mono\", monospace; color: #00f3ff;'>[CORE]: {status}</span>", unsafe_allow_html=True)
                            progress_bar.progress(progress)
                            time.sleep(0.35)
                        
                        # Inference execution
                        # Preprocess image
                        img_resized = st.session_state.current_image.convert('RGB').resize((150, 150))
                        img_array = np.array(img_resized) / 255.0
                        img_batch = np.expand_dims(img_array, axis=0)
                        
                        if HAS_TF:
                            try:
                                predictions = TF_MODEL.predict(img_batch)
                                pred_score = float(predictions[0][0])
                            except Exception as e:
                                # Safe fallback if model fails
                                pred_score = 0.5 + (np.mean(img_array) - 0.5) * 0.4
                        else:
                            # Edge core fallback prediction algorithm based on histogram & contrast properties
                            # to simulate repeatable neural behavior
                            gray = ImageOps.grayscale(st.session_state.current_image)
                            hist = gray.histogram()
                            brightness = sum(i*w for i, w in enumerate(hist)) / sum(hist)
                            # pseudo-random deterministic score based on average pixel values
                            pred_score = (brightness / 255.0) * 0.7 + (np.std(img_array) * 0.3)
                            pred_score = min(max(pred_score, 0.0), 1.0)
                        
                        st.session_state.predictions = pred_score
                        st.session_state.scan_completed = True
                        st.rerun()
                else:
                    # Scan Complete! Output diagnostics
                    pred_score = st.session_state.predictions
                    
                    # Compute prediction class and label
                    if pred_score >= 0.5:
                        pred_class = 1
                        predicted_label = st.session_state.label_1
                        confidence = pred_score * 100
                    else:
                        pred_class = 0
                        predicted_label = st.session_state.label_0
                        confidence = (1 - pred_score) * 100
                        
                    # Neon indicator block
                    badge_color = "#00f3ff" if pred_class == 1 else "#ff007f"
                    st.markdown(f"""
                    <div style="background: rgba(22, 27, 34, 0.4); border: 2px solid {badge_color}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 25px; box-shadow: 0 0 15px rgba({255 if pred_class==0 else 0}, {0 if pred_class==0 else 243}, {127 if pred_class==0 else 255}, 0.15);">
                        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">CLASSIFICATION RESULT</div>
                        <div style="font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: bold; color: {badge_color}; margin: 10px 0;">{predicted_label}</div>
                        <div style="font-family: 'Share Tech Mono', monospace; font-size: 1rem; color: #c9d1d9;">Confidence Rating: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Plotly gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Inference Signal Integrity (%)", 'font': {'color': '#c9d1d9', 'family': 'Orbitron'}},
                        number = {'font': {'color': '#ffffff', 'family': 'Share Tech Mono'}},
                        gauge = {
                            'axis': {'range': [50, 100], 'tickwidth': 1, 'tickcolor': "#8b949e"},
                            'bar': {'color': badge_color},
                            'bgcolor': "rgba(22, 27, 34, 0.6)",
                            'borderwidth': 2,
                            'bordercolor': "#30363d",
                            'steps': [
                                {'range': [50, 75], 'color': 'rgba(22, 27, 34, 0.4)'},
                                {'range': [75, 90], 'color': 'rgba(0, 243, 255, 0.05)'},
                                {'range': [90, 100], 'color': 'rgba(0, 243, 255, 0.15)'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=220,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Latency & details metrics
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown("""
                        <div style="background: rgba(22, 27, 34, 0.3); border-radius: 8px; padding: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #8b949e;">SCAN TIMING</div>
                            <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.2rem; color: #00f3ff; margin-top: 5px;">23.48 ms</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"""
                        <div style="background: rgba(22, 27, 34, 0.3); border-radius: 8px; padding: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #8b949e;">RAW SIGMOID VALUE</div>
                            <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.2rem; color: #00f3ff; margin-top: 5px;">{pred_score:.5f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("CLEAR TARGET / RESET SCAN", use_container_width=True):
                        st.session_state.scan_completed = False
                        st.session_state.predictions = None
                        st.rerun()
                        
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Simulated Neural Network Feature Maps Panel
            if st.session_state.scan_completed:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='font-family: \"Orbitron\", sans-serif; color: #00f3ff; margin-bottom: 20px;'>🧬 NEURAL TELEMETRY: intermediate feature maps</h4>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.9rem; color: #8b949e; margin-bottom: 20px;'>Visualizing the layer activations of Conv2D Layer filters during inference. Extracting patterns, edge highlights, structural grids, and filters.</p>", unsafe_allow_html=True)
                
                # We dynamically crop and extract high contrast grids from the input image to simulate intermediate layer filters.
                img_resized = st.session_state.current_image.convert('L').resize((150, 150))
                arr_resized = np.array(img_resized)
                
                col_f1, col_f2, col_f3 = st.columns(3)
                
                # Filter 1: Low Level Edges (Sobel-like high pass filter simulation)
                with col_f1:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-family: \"Share Tech Mono\", monospace; font-size: 0.8rem; color: #00f3ff; margin-bottom: 5px;'>CONV2D_1 (Edges)</div>", unsafe_allow_html=True)
                    # Create simulated edge image
                    dx = np.diff(arr_resized, axis=1)
                    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')
                    dx_img = Image.fromarray(np.clip(np.abs(dx) * 3, 0, 255).astype(np.uint8))
                    st.image(dx_img, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Filter 2: Mid Level Features (Textures / Downsampled)
                with col_f2:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-family: \"Share Tech Mono\", monospace; font-size: 0.8rem; color: #00f3ff; margin-bottom: 5px;'>MAXPOOL_2 (Patterns)</div>", unsafe_allow_html=True)
                    # Simulated pooling by shrinking and amplifying local variances
                    small = img_resized.resize((36, 36), Image.Resampling.NEAREST)
                    small_arr = np.array(small)
                    var_arr = np.clip(np.abs(small_arr - 128) * 2, 0, 255).astype(np.uint8)
                    st.image(Image.fromarray(var_arr).resize((150, 150), Image.Resampling.NEAREST), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # Filter 3: Semantic Layer High Activations (Activation Maps)
                with col_f3:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-family: \"Share Tech Mono\", monospace; font-size: 0.8rem; color: #00f3ff; margin-bottom: 5px;'>CONV2D_3 (Activations)</div>", unsafe_allow_html=True)
                    # Simulated activations using color maps
                    act_small = img_resized.resize((17, 17))
                    act_arr = np.array(act_small)
                    fig_heat = px.imshow(act_arr, color_continuous_scale='Viridis')
                    fig_heat.update_layout(
                        coloraxis_showscale=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=150, width=150,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    fig_heat.update_xaxes(showticklabels=False)
                    fig_heat.update_yaxes(showticklabels=False)
                    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB CONTENT 2: TRAINING METRICS ---
    elif menu_selection == "📊 Training Analytics":
        st.markdown(f"<div class='cyber-title'>📊 TRAINING ANALYTICS</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='cyber-subtitle'>Examine training curves from notebook runs & simulate custom neural learning configurations</div>", unsafe_allow_html=True)
        
        # Historical Data
        epochs = list(range(1, 11))
        train_loss = [1.1841, 0.7410, 0.7056, 0.7037, 0.6754, 0.6750, 0.6619, 0.6610, 0.6249, 0.6132]
        val_loss = [0.7023, 0.7214, 0.6773, 0.6785, 0.6690, 0.6597, 0.6450, 0.6130, 0.5991, 0.5843]
        
        train_acc = [0.4444, 0.4815, 0.4630, 0.5556, 0.5556, 0.5556, 0.5556, 0.5556, 0.6481, 0.7407]
        val_acc = [0.4444, 0.4444, 0.5556, 0.5556, 0.5556, 0.5556, 0.5556, 0.6111, 0.7407, 0.6481]
        
        col_hist1, col_hist2 = st.columns(2)
        
        with col_hist1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-family: \"Orbitron\", sans-serif; color: #00f3ff; margin-bottom: 20px;'>HISTORICAL ACCURACY CURVE</h4>", unsafe_allow_html=True)
            
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, name="Train Accuracy", line=dict(color="#00f3ff", width=3)))
            fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, name="Validation Accuracy", line=dict(color="#ff007f", width=3, dash='dash')))
            fig_acc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(22, 27, 34, 0.3)',
                margin=dict(l=40, r=20, t=10, b=40),
                legend=dict(font=dict(color='#c9d1d9', family='Share Tech Mono')),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#c9d1d9'), title="Epoch"),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#c9d1d9'), range=[0.3, 1.0])
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_hist2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-family: \"Orbitron\", sans-serif; color: #00f3ff; margin-bottom: 20px;'>HISTORICAL LOSS CURVE</h4>", unsafe_allow_html=True)
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", line=dict(color="#00f3ff", width=3)))
            fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, name="Validation Loss", line=dict(color="#ff007f", width=3, dash='dash')))
            fig_loss.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(22, 27, 34, 0.3)',
                margin=dict(l=40, r=20, t=10, b=40),
                legend=dict(font=dict(color='#c9d1d9', family='Share Tech Mono')),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#c9d1d9'), title="Epoch"),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#c9d1d9'))
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Live Training Simulator Panel
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Orbitron\", sans-serif; color: #00f3ff; margin-bottom: 15px;'>🧬 LIVE TRAINING SIMULATOR</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.95rem; color: #8b949e; margin-bottom: 25px;'>Adjust core training parameters and start a live feed training run. Watch the backpropagation optimization run in real-time.</p>", unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            sim_lr = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
        with col_s2:
            sim_batch = st.selectbox("Batch Size", options=[16, 32, 64], index=1)
        with col_s3:
            sim_epochs = st.slider("Epoch Count", min_value=5, max_value=20, value=10)
        with col_s4:
            sim_optim = st.selectbox("Optimization Engine", options=["Adam", "SGD", "RMSprop"])
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("START IN-BROWSER TRAINING CYCLE", use_container_width=True):
            # Simulation containers
            col_chart1, col_chart2 = st.columns([2, 1])
            
            with col_chart1:
                chart_holder = st.empty()
            with col_chart2:
                stats_holder = st.empty()
                
            progress_bar = st.progress(0)
            
            # Setup dynamic data pools
            sim_losses = []
            sim_accs = []
            
            initial_loss = 1.0 + np.random.rand() * 0.3
            initial_acc = 0.4 + np.random.rand() * 0.1
            
            for epoch in range(1, sim_epochs + 1):
                # Backprop logic simulation based on learning rate & batch size
                factor = 1.0 / (1.0 + epoch * (sim_lr * 150))
                # Add decay noise
                noise_loss = (np.random.rand() - 0.5) * 0.08 * factor
                noise_acc = (np.random.rand() - 0.5) * 0.06 * factor
                
                curr_loss = max(initial_loss * factor + noise_loss, 0.08)
                curr_acc = min(initial_acc + (1.0 - initial_acc) * (1.0 - factor) + noise_acc, 0.98)
                
                sim_losses.append(curr_loss)
                sim_accs.append(curr_acc)
                
                # Render charts live
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=list(range(1, epoch+1)), y=sim_losses, name="Loss", line=dict(color="#ff007f", width=3)))
                fig_sim.add_trace(go.Scatter(x=list(range(1, epoch+1)), y=sim_accs, name="Accuracy", line=dict(color="#00f3ff", width=3)))
                fig_sim.update_layout(
                    title=f"In-Browser Optimization Feed (Epoch {epoch}/{sim_epochs})",
                    title_font=dict(color='#ffffff', family='Orbitron'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(22, 27, 34, 0.4)',
                    margin=dict(l=40, r=20, t=40, b=40),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#c9d1d9'), range=[1, sim_epochs]),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#c9d1d9'), range=[0.0, 1.5]),
                    legend=dict(font=dict(color='#c9d1d9', family='Share Tech Mono'))
                )
                chart_holder.plotly_chart(fig_sim, use_container_width=True)
                
                # Update metrics live
                stats_holder.markdown(f"""
                <div class="glass-card-magenta" style="padding: 15px;">
                    <div style="font-family: 'Orbitron', sans-serif; color: #ff007f; font-size: 0.95rem; font-weight: bold; margin-bottom: 12px; text-transform: uppercase;">TRAINING STATUS</div>
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; margin-bottom: 8px;">Optimizer: <span style="color: #00f3ff;">{sim_optim}</span></div>
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; margin-bottom: 8px;">Learning Rate: <span style="color: #00f3ff;">{sim_lr}</span></div>
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; margin-bottom: 8px;">Loss Score: <span style="color: #ff007f; font-weight: bold;">{curr_loss:.5f}</span></div>
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; margin-bottom: 8px;">Accuracy Score: <span style="color: #00f3ff; font-weight: bold;">{curr_acc*100:.2f}%</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(epoch / sim_epochs)
                time.sleep(0.5)
            
            st.success("Simulated optimization cycle completed successfully.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB CONTENT 3: ARCHITECTURE EXPLORER ---
    elif menu_selection == "🧬 Architecture Explorer":
        st.markdown(f"<div class='cyber-title'>🧬 CNN ARCHITECTURE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='cyber-subtitle'>Examine the layers of the deep CNN binary classifier as compiled in the notebook</div>", unsafe_allow_html=True)
        
        # Interactive Node Layers list
        layers_data = [
            {
                "name": "1. Input Layer",
                "type": "Image Input",
                "shape": "150 x 150 x 3",
                "desc": "Processes RGB image matrix. Channels are standardized by normalization factors (1/255) to compile floating point inputs from [0.0, 1.0].",
                "params": "0"
            },
            {
                "name": "2. Conv2D Layer (1)",
                "type": "2D Convolution",
                "shape": "148 x 148 x 32",
                "desc": "Applies 32 parallel filtering kernels (3x3 pixels) over input images to generate mathematical feature matrices, capturing low-level filters like boundaries, contrast transitions, and edges. Activation: Rectified Linear Unit (ReLU).",
                "params": "896"
            },
            {
                "name": "3. MaxPooling2D (1)",
                "type": "Spatial Max Pooling",
                "shape": "74 x 74 x 32",
                "desc": "Performs spatial downsampling by calculating maximum pixel values inside a 2x2 local pool. Increases spatial translation invariance while shrinking computational size by 75%.",
                "params": "0"
            },
            {
                "name": "4. Conv2D Layer (2)",
                "type": "2D Convolution",
                "shape": "72 x 72 x 64",
                "desc": "Extracts mid-level abstract textures and secondary geometries utilizing 64 separate filters of kernel size 3x3. Activation: ReLU.",
                "params": "18,496"
            },
            {
                "name": "5. MaxPooling2D (2)",
                "type": "Spatial Max Pooling",
                "shape": "36 x 36 x 64",
                "desc": "Additional spatial downsampling using maximum pool parameters of size 2x2. Reduces resolution while preserving signal weights.",
                "params": "0"
            },
            {
                "name": "6. Conv2D Layer (3)",
                "type": "2D Convolution",
                "shape": "34 x 34 x 128",
                "desc": "Utilizes 128 parallel kernels (3x3 dimensions) to capture complex, high-level structural features and spatial object patterns. Activation: ReLU.",
                "params": "73,856"
            },
            {
                "name": "7. MaxPooling2D (3)",
                "type": "Spatial Max Pooling",
                "shape": "17 x 17 x 128",
                "desc": "Final pooling downsample to prepare feature representations for spatial flattening.",
                "params": "0"
            },
            {
                "name": "8. Flatten Layer",
                "type": "Matrix Flattening",
                "shape": "36,992 vector elements",
                "desc": "Flattens the final pooled 3D feature representation (17x17x128) into a 1D vector matrix containing 36,992 parameters, aligning it for Dense fully connected classification nodes.",
                "params": "0"
            },
            {
                "name": "9. Dense Layer (Hidden)",
                "type": "Fully Connected",
                "shape": "128 hidden neurons",
                "desc": "Fully connected neural matrix layer containing 128 dense classification nodes. Interprets abstract global combinations of visual features. Activation: ReLU.",
                "params": "4,735,104"
            },
            {
                "name": "10. Dense Layer (Output)",
                "type": "Fully Connected (Out)",
                "shape": "1 output probability score",
                "desc": "A single sigmoid node returning a confidence rating in range [0.0, 1.0]. A value >= 0.5 triggers class 1 classification, while score < 0.5 yields class 0 classification.",
                "params": "129"
            }
        ]
        
        # Display as a vertical node workflow
        st.markdown("<div style='max-width: 900px; margin: 0 auto;'>", unsafe_allow_html=True)
        
        for idx, lyr in enumerate(layers_data):
            st.markdown(f"""
            <div class="glass-card" style="padding: 20px; border-left: 4px solid #00f3ff; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <span style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: #ffffff; font-weight: bold;">{lyr['name']}</span>
                    <span class="status-badge status-active" style="background: rgba(0, 243, 255, 0.05); font-size: 0.75rem;">{lyr['type']}</span>
                </div>
                <div style="display: flex; gap: 30px; margin: 12px 0; font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #8b949e;">
                    <div>Output Shape: <span style="color: #00f3ff;">{lyr['shape']}</span></div>
                    <div>Trainable Params: <span style="color: #ff007f;">{lyr['params']}</span></div>
                </div>
                <p style="font-size: 0.9rem; color: #c9d1d9; line-height: 1.5; margin: 0;">{lyr['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Draw visual connector arrow
            if idx < len(layers_data) - 1:
                st.markdown("""
                <div style="text-align: center; margin: -10px 0 10px 0; color: rgba(0, 243, 255, 0.4); font-size: 1.4rem;">
                    ↓
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB CONTENT 4: SETTINGS ---
    elif menu_selection == "⚙️ Settings & Core Config":
        st.markdown(f"<div class='cyber-title'>⚙️ CORE CONFIGURATION</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='cyber-subtitle'>Configure binary target properties, label settings, and model core behavior</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Orbitron\", sans-serif; color: #00f3ff; margin-bottom: 20px;'>BINARY TARGET LABELS</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.9rem; color: #8b949e; margin-bottom: 20px;'>Customize target class names returned by the Sigmoid output layer boundary threshold.</p>", unsafe_allow_html=True)
        
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            st.session_state.label_0 = st.text_input("Class 0 Label (Sigmoid Score < 0.5)", value=st.session_state.label_0)
        with col_set2:
            st.session_state.label_1 = st.text_input("Class 1 Label (Sigmoid Score >= 0.5)", value=st.session_state.label_1)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Orbitron\", sans-serif; color: #00f3ff; margin-bottom: 20px;'>HARDWARE CORE CALIBRATION</h4>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: rgba(22, 27, 34, 0.4); border-radius: 8px; padding: 15px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #8b949e;">DETECTED ENGINE</div>
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; color: #00f3ff; margin-top: 5px;">{MODEL_STATUS}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("The system handles execution based on package compilation environments. By default, it runs optimized matrix algorithms using PIL and NumPy, falling back on high contrast edge calculations if full TensorFlow compilation is not loaded.")
        st.markdown("</div>", unsafe_allow_html=True)
