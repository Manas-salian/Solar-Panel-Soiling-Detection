"""
Solar Panel Soiling Detection - Web Dashboard
===============================================
A simple Streamlit app to visualize training results and run live
predictions on new solar panel images using the trained model.

Run from the project root (with your venv active):
    streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from src.config import (
    IMG_SIZE,
    CLASS_NAMES,
    OUTPUT_DIR,
    PREDICTION_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)

st.set_page_config(
    page_title="Solar Panel Soiling Detection",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Panel Soiling Detection Dashboard")
st.caption(
    "A deep learning project that classifies solar panel images as "
    "Clean or Soiled/Dusty using MobileNetV2 transfer learning."
)

tab1, tab2, tab3 = st.tabs(["🔍 Try It", "📊 Training Results", "ℹ️ About"])


# ---------------------------------------------------------------------------
# Tab 1: Live prediction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Upload a solar panel image")
    st.write("Upload a photo of a solar panel to check if it's clean or needs cleaning.")

    model_path = os.path.join(OUTPUT_DIR, "solar_soiling_final_model.keras")

    if not os.path.exists(model_path):
        st.warning(
            f"No trained model found at `{model_path}`. "
            "Run `python main.py` first to train and save a model."
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            image = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.image(image, caption="Uploaded image", use_container_width=True)

            with st.spinner("Loading model and predicting..."):
                model = tf.keras.models.load_model(model_path)

                img_resized = image.resize(IMG_SIZE)
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)  # add batch dim

                prediction = model.predict(img_array, verbose=0)[0][0]
                predicted_class = CLASS_NAMES[int(prediction >= PREDICTION_THRESHOLD)]
                confidence = (
                    prediction if predicted_class == "dusty" else 1 - prediction
                )

            with col2:
                st.markdown("### Prediction")
                if predicted_class == "dusty":
                    st.error(f"🟤 **Dusty** — cleaning recommended")
                else:
                    st.success(f"🟢 **Clean** — no action needed")
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                st.progress(float(confidence))
                if confidence < LOW_CONFIDENCE_THRESHOLD:
                    st.warning(
                        "This is a borderline result. Try a clearer image or use "
                        "the prediction as a manual review prompt."
                    )


# ---------------------------------------------------------------------------
# Tab 2: Training results
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Training & Evaluation Results")

    col1, col2 = st.columns(2)

    initial_curve = os.path.join(OUTPUT_DIR, "training_curves_initial.png")
    finetune_curve = os.path.join(OUTPUT_DIR, "training_curves_finetune.png")
    confusion_matrix_img = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

    with col1:
        st.markdown("**Initial Training (frozen backbone)**")
        if os.path.exists(initial_curve):
            st.image(initial_curve, use_container_width=True)
        else:
            st.info("Not generated yet. Run `python main.py` first.")

        st.markdown("**Confusion Matrix**")
        if os.path.exists(confusion_matrix_img):
            st.image(confusion_matrix_img, use_container_width=True)
        else:
            st.info("Not generated yet. Run `python main.py` first.")

    with col2:
        st.markdown("**Fine-Tuning Stage**")
        if os.path.exists(finetune_curve):
            st.image(finetune_curve, use_container_width=True)
        else:
            st.info("Not generated yet, or fine-tuning was skipped.")

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        
        st.markdown("---")
        st.markdown("### 📈 Evaluation Metrics (Test Set)")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Test Accuracy", f"{metrics_data.get('accuracy', 0) * 100:.1f}%")
        m_col2.metric("Clean F1-Score", f"{metrics_data.get('clean', {}).get('f1-score', 0):.2f}")
        m_col3.metric("Dusty F1-Score", f"{metrics_data.get('dusty', {}).get('f1-score', 0):.2f}")



# ---------------------------------------------------------------------------
# Tab 3: About
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("About this project")
    st.markdown(
        """
        **Solar Panel Soiling Detection** is a binary image classifier that
        identifies whether a solar panel is clean or covered in dust/soil,
        using transfer learning with **MobileNetV2**.

        **Why it matters:** Dust accumulation on solar panels can
        significantly reduce power output. Automating soiling detection
        from photos lets operators schedule cleaning only when needed —
        cutting maintenance costs and recovering lost energy.

        **Model:** MobileNetV2 backbone (pretrained on ImageNet) +
        a small classification head, fine-tuned on a labeled dataset of
        clean and dusty solar panel images.
        """
    )
