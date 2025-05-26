from ultralytics import YOLO
import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Deteksi Sel Darah", layout="centered")
st.title("🩸 Deteksi Sel Darah Otomatis dengan YOLOv11")
st.write("Upload gambar mikroskopis untuk mendeteksi sel darah secara otomatis.")

# Load model YOLOv11 kamu (ganti path jika beda)
model = YOLO("runs/detect/train/weights/best.pt")

uploaded_file = st.file_uploader("📤 Upload gambar", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Simpan gambar sementara
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(img_path)
    st.image(image, caption="🖼️ Gambar yang diupload", use_column_width=True)

    st.write("🔍 Mendeteksi...")
    results = model.predict(source=img_path, save=False)

    # Visualisasi hasil
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="✅ Hasil Deteksi", use_column_width=True)

    # Tampilkan label dan confidence score
    st.subheader("📊 Deteksi:")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        st.write(f"- **{label}**: {conf:.2f}")
