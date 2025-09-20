import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import pandas as pd

model = YOLO('runs\\weights\\best.pt')

st.title("Blood Cell Detection App")
st.write("Upload a blood smear image to detect RBCs, WBCs, and Platelets separately.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    line_thickness = st.slider("Bounding Box Thickness", 1, 5, 2)

    results = model.predict(image_np, imgsz=640, conf=conf_threshold)

    def plot_class(result, class_idx, color=(0,255,0)):
        boxes = [box for box in result.boxes if int(box.cls) == class_idx]
        img_copy = result.orig_img.copy()
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = result.names[int(box.cls)]
            conf = float(box.conf.cpu().numpy())
            cv2.rectangle(img_copy, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, line_thickness)
            cv2.putText(img_copy, f"{label} {conf:.2f}", (xyxy[0], xyxy[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), len(boxes)

    platelet_img, platelet_count = plot_class(results[0], 0, color=(0,0,255))
    rbc_img, rbc_count = plot_class(results[0], 1, color=(0,255,0))
    wbc_img, wbc_count = plot_class(results[0], 2, color=(255,0,0))

    tab1, tab2, tab3 = st.tabs(["Platelets", "RBCs", "WBCs"])

    with tab1:
        st.image(platelet_img, use_column_width=True)
        st.write(f"**Platelet Count:** {platelet_count}")

    with tab2:
        st.image(rbc_img, use_column_width=True)
        st.write(f"**RBC Count:** {rbc_count}")

    with tab3:
        st.image(wbc_img, use_column_width=True)
        st.write(f"**WBC Count:** {wbc_count}")

    stats = pd.DataFrame({
        "Cell Type": ["Platelet", "RBC", "WBC"],
        "Count": [platelet_count, rbc_count, wbc_count]
    })
    st.subheader("Overall Cell Counts")
    st.table(stats)
