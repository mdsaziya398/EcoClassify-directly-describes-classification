"""
app.py — Smart Waste Classification System
Streamlit web application for AI-powered waste classification.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH   = PROJECT_ROOT / "models" / "waste_classifier.keras"

CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
IMAGE_SIZE = (224, 224)

# ─── Category info ────────────────────────────────────────────────────────────
CATEGORY_INFO = {
    "Cardboard": {
        "icon": "📦",
        "tip": "Generally recyclable when clean and dry. Flatten boxes before disposal.",
        "color": "#D4A017",
    },
    "Glass": {
        "icon": "🍶",
        "tip": "Commonly recyclable. Handle carefully — separate by color if required locally.",
        "color": "#4FC3F7",
    },
    "Metal": {
        "icon": "🔩",
        "tip": "Often highly recyclable and valuable for material recovery. Rinse cans if possible.",
        "color": "#90A4AE",
    },
    "Paper": {
        "icon": "📄",
        "tip": "Generally recyclable when clean and not heavily contaminated with food/grease.",
        "color": "#A5D6A7",
    },
    "Plastic": {
        "icon": "🧴",
        "tip": "Check the resin code. Sort according to your local recycling rules.",
        "color": "#CE93D8",
    },
    "Trash": {
        "icon": "🗑️",
        "tip": "General/non-recyclable waste. Dispose responsibly at designated facilities.",
        "color": "#EF9A9A",
    },
}

DISCLAIMER = (
    "⚠️ Recycling rules vary by location. Always follow your local municipality's guidelines."
)


# ─── TensorFlow / Model Loading ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    """Load the Keras model once and cache it."""
    if not MODEL_PATH.exists():
        return None, "Model not found"
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as exc:
        return None, str(exc)


def preprocess_image(pil_image: Image.Image):
    """Resize and preprocess image for MobileNetV2."""
    try:
        import tensorflow as tf
        img = pil_image.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        return np.expand_dims(arr, axis=0)
    except Exception as exc:
        raise RuntimeError(f"Preprocessing failed: {exc}")


def run_prediction(model, pil_image: Image.Image):
    """Return (predicted_class, confidence, all_probabilities)."""
    input_arr = preprocess_image(pil_image)
    probs = model.predict(input_arr, verbose=0)[0]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]) * 100, probs.tolist()


# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ───────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ──────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Hero Banner ─────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(99,102,241,0.18));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #10b981, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle {
    color: rgba(255,255,255,0.7);
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* ── Card ────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

/* ── Result Badge ────────────────────────── */
.result-badge {
    display: inline-block;
    padding: 0.6rem 1.8rem;
    border-radius: 50px;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 0.5rem 0;
}

/* ── Confidence Ring ─────────────────────── */
.confidence-box {
    text-align: center;
    padding: 1.2rem;
    border-radius: 16px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
}
.confidence-value {
    font-size: 3rem;
    font-weight: 700;
    color: #10b981;
}
.confidence-label { color: rgba(255,255,255,0.6); font-size: 0.9rem; }

/* ── Upload area tweak ───────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(16,185,129,0.5);
    border-radius: 14px;
    padding: 1rem;
}

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.8) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
.sidebar-header {
    color: #10b981;
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}
.cat-chip {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.85rem;
    margin: 0.2rem;
    background: rgba(16,185,129,0.2);
    border: 1px solid rgba(16,185,129,0.4);
    color: #d1fae5;
}

/* ── Progress bar colour ─────────────────── */
.stProgress > div > div > div { background: linear-gradient(90deg, #10b981, #6366f1); }

/* ── Disclaimer ──────────────────────────── */
.disclaimer {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    padding: 0.5rem 1rem;
    border-left: 3px solid rgba(99,102,241,0.5);
    border-radius: 4px;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-header">♻️ Smart Waste Classifier</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📋 About")
    st.markdown(
        "An AI-powered system that identifies waste type from a photo "
        "using **MobileNetV2** deep learning, trained on 2,527 waste images."
    )

    st.markdown("---")
    st.markdown("### 🗂️ Supported Categories")
    for cls in CLASSES:
        info = CATEGORY_INFO[cls]
        st.markdown(
            f'<span class="cat-chip">{info["icon"]} {cls}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.markdown(
        """
| Property | Value |
|---|---|
| Architecture | MobileNetV2 |
| Pre-training | ImageNet |
| Input size | 224 × 224 |
| Classes | 6 |
| Dataset | 2,527 images |
"""
    )

    st.markdown("---")
    st.markdown("### 🔄 How It Works")
    st.markdown(
        """
1. 📤 Upload a waste image
2. 🔍 Image is resized & normalized
3. 🧠 MobileNetV2 extracts features
4. 📊 Softmax gives probabilities
5. ✅ Top class shown as prediction
"""
    )

    st.markdown("---")
    st.markdown(
        '<p class="disclaimer">Recycling practices vary by location. '
        "Always follow local guidelines.</p>",
        unsafe_allow_html=True,
    )


# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1 class="hero-title">♻️ Smart Waste Classification</h1>
  <p class="hero-subtitle">
    AI-powered waste identification for smarter and more sustainable waste management.
  </p>
</div>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
model, model_error = load_model()

if model_error:
    if "not found" in model_error.lower() or model is None:
        st.error(
            "🚫 **Model not found.** Please train the model first using the training script:\n\n"
            "```bash\npython src/train.py\n```"
        )
    else:
        st.error(f"⚠️ **Model loading error:** {model_error}")
    st.stop()

st.success("✅ AI model loaded and ready.")

st.markdown("---")

# ─── Upload Section ───────────────────────────────────────────────────────────
st.markdown("## 📤 Upload a Waste Image")
st.markdown("Drag & drop or click to browse. Supported formats: **JPG, JPEG, PNG**")

uploaded_file = st.file_uploader(
    label="Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key="waste_image_uploader",
)

if uploaded_file is not None:
    # ── Validate / open image ────────────────────────────────────────────
    try:
        pil_img = Image.open(uploaded_file)
        pil_img.verify()          # check for corruption
        pil_img = Image.open(uploaded_file)  # re-open after verify
    except Exception as exc:
        st.error(f"⚠️ **Could not read the uploaded image:** {exc}")
        st.stop()

    st.markdown("---")

    # ── Two-column layout ────────────────────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Uploaded Image")
        st.image(pil_img, use_container_width=True, caption=uploaded_file.name)
        st.markdown(
            f"**Size:** {pil_img.width} × {pil_img.height} px &nbsp;|&nbsp; "
            f"**Format:** {pil_img.format or 'N/A'}",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Running Classification…")

        with st.spinner("Analysing image with AI…"):
            try:
                pred_class, confidence, all_probs = run_prediction(model, pil_img)
            except Exception as exc:
                st.error(f"⚠️ **Prediction failed:** {exc}")
                st.stop()

        # ── Result ──────────────────────────────────────────────────────
        info = CATEGORY_INFO[pred_class]
        badge_color = info["color"]

        st.markdown("### ✅ Prediction Result")
        st.markdown(
            f'<div class="result-badge" style="background:{badge_color}22; '
            f'border:2px solid {badge_color}; color:{badge_color};">'
            f'{info["icon"]} {pred_class}</div>',
            unsafe_allow_html=True,
        )

        # Confidence
        st.markdown(
            f'<div class="confidence-box">'
            f'<div class="confidence-value">{confidence:.1f}%</div>'
            f'<div class="confidence-label">Confidence</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Category tip
        st.info(f"💡 **Disposal Tip:** {info['tip']}")
        st.markdown(
            f'<p class="disclaimer">{DISCLAIMER}</p>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Probability Distribution ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Class Probability Distribution")

    prob_col1, prob_col2 = st.columns([2, 1], gap="large")

    with prob_col1:
        # Bar chart using Matplotlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        colors = [
            CATEGORY_INFO[c]["color"] if c == pred_class else "#ffffff33"
            for c in CLASSES
        ]
        bars = ax.barh(CLASSES, [p * 100 for p in all_probs], color=colors, edgecolor="none", height=0.6)

        for bar, prob in zip(bars, all_probs):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.1f}%",
                va="center",
                color="white",
                fontsize=10,
            )

        ax.set_xlim(0, 115)
        ax.set_xlabel("Probability (%)", color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=11)
        ax.spines[:].set_visible(False)
        ax.xaxis.label.set_color("white")
        ax.set_title("All Class Probabilities", color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with prob_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Scores**")
        for cls, prob in zip(CLASSES, all_probs):
            pct = prob * 100
            icon = CATEGORY_INFO[cls]["icon"]
            is_top = cls == pred_class
            bold = "**" if is_top else ""
            st.markdown(f"{icon} {bold}{cls}{bold}")
            st.progress(float(prob), text=f"{pct:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Category Info Cards ───────────────────────────────────────────────
    st.markdown("### 🗂️ All Category Information")
    st.markdown("*Recycling practices vary by location. Always check local guidelines.*")

    card_cols = st.columns(3)
    for i, cls in enumerate(CLASSES):
        info = CATEGORY_INFO[cls]
        with card_cols[i % 3]:
            highlighted = cls == pred_class
            border_style = f"3px solid {info['color']}" if highlighted else "1px solid rgba(255,255,255,0.12)"
            st.markdown(
                f'<div class="card" style="border:{border_style};">'
                f'<h4 style="color:{info["color"]}; margin:0;">{info["icon"]} {cls}</h4>'
                f'<p style="color:rgba(255,255,255,0.75); font-size:0.88rem; margin-top:0.5rem;">'
                f'{info["tip"]}</p>'
                f'{"<p style=\'color:" + info["color"] + ";font-size:0.75rem;font-weight:600;\'>✦ Predicted Class</p>" if highlighted else ""}'
                f"</div>",
                unsafe_allow_html=True,
            )

else:
    # ── Empty State ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="card" style="text-align:center; padding: 4rem 2rem;">
      <div style="font-size:5rem;">📸</div>
      <h3 style="color:rgba(255,255,255,0.85); margin-top:1rem;">
        Upload a waste image to get started
      </h3>
      <p style="color:rgba(255,255,255,0.5);">
        The AI will identify the waste type and suggest disposal guidance.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Show category overview when idle
    st.markdown("---")
    st.markdown("### 🗂️ Supported Waste Categories")
    idle_cols = st.columns(3)
    for i, cls in enumerate(CLASSES):
        info = CATEGORY_INFO[cls]
        with idle_cols[i % 3]:
            st.markdown(
                f'<div class="card">'
                f'<h4 style="color:{info["color"]}; margin:0 0 0.4rem 0;">{info["icon"]} {cls}</h4>'
                f'<p style="color:rgba(255,255,255,0.7); font-size:0.88rem; margin:0;">'
                f'{info["tip"]}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
