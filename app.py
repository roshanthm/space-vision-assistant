import streamlit as st
import cv2
import numpy as np
from detector import detect
from nasa_api import fetch_nasa_image
from explain import explain_result
from heatmap import generate_heatmap
from report import generate_pdf_report
import os

st.set_page_config(
    page_title="Space Vision Assistant",
    layout="wide",
    page_icon="🌌"
)

st.title("🌌 Space Vision Assistant — AI NASA Object Analyzer")

st.sidebar.header("Options")

mode = st.sidebar.selectbox(
    "Choose Input Mode",
    ["Live Camera", "Upload Image", "NASA Image of the Day"]
)

col1, col2 = st.columns(2)


# ================================
# LIVE CAMERA
# ================================
if mode == "Live Camera":
    run = st.checkbox("Start Camera")

    cam = cv2.VideoCapture(0)

    while run:
        ret, frame = cam.read()
        if not ret:
            st.error("Camera not accessible")
            break

        label, conf, annotated = detect(frame)

        with col1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected")

        with col2:
            st.write(f"### Prediction: **{label}**")
            st.write(f"Confidence: **{conf*100:.2f}%**")

        explain_text = explain_result(label)
        st.write("### 🔭 AI Explanation")
        st.write(explain_text)

        if st.button("Stop"):
            break

    cam.release()


# ================================
# UPLOAD IMAGE
# ================================
elif mode == "Upload Image":
    file = st.file_uploader("Upload space image", type=["jpg", "png"])

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        label, conf, annotated = detect(img)

        with col1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected")

        with col2:
            st.write(f"### Prediction: **{label}**")
            st.write(f"Confidence: **{conf*100:.2f}%**")

        explain_text = explain_result(label)
        st.write("### 🔭 AI Explanation")
        st.write(explain_text)

        heatmap_img = generate_heatmap(img)
        st.image(heatmap_img, caption="Attention Heatmap")

        if st.button("Download PDF Report"):
            pdf_path = generate_pdf_report(label, conf, explain_text)
            with open(pdf_path, "rb") as f:
                st.download_button("Download Report", f, file_name="report.pdf")


# ================================
# NASA IMAGE OF THE DAY
# ================================
elif mode == "NASA Image of the Day":
    img = fetch_nasa_image()

    st.image(img, caption="🛰 NASA APOD Image")

    label, conf, annotated = detect(img)

    st.write(f"### Prediction: **{label}**")
    st.write(f"Confidence: **{conf*100:.2f}%**")

    explain_text = explain_result(label)
    st.write("### 🔭 AI Explanation")
    st.write(explain_text)

    heatmap_img = generate_heatmap(img)
    st.image(heatmap_img, caption="Attention Heatmap")
