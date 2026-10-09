import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

MODEL_PATH = "plant_disease_weighted_cnn.keras"

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_healthy",
]


@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.title("🌿 Plant Leaf Disease Detection")
st.write(
    "Upload a leaf image to classify it using the trained CNN model."
)

try:
    model = load_cnn_model()
except Exception as exc:
    st.error(
        f"Could not load the model file: {MODEL_PATH}. "
        "Place plant_disease_weighted_cnn.keras in the same folder as app.py."
    )
    st.exception(exc)
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read this image. Please upload a valid JPG or PNG.")
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(
            image_rgb,
            caption="Uploaded leaf image",
            use_container_width=True
        )

        if st.button("Predict Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                resized_image = cv2.resize(image_rgb, (224, 224))
                image_array = resized_image.astype(np.float32)
                # The trained CNN includes a Rescaling(1./255) layer,
                # so do not normalize the image a second time here.
                image_array = np.expand_dims(image_array, axis=0)

                probabilities = model.predict(image_array, verbose=0)[0]
                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index] * 100)

            st.subheader("Prediction Result")
            st.success(f"Predicted class: {predicted_class}")
            st.metric("Prediction confidence", f"{confidence:.2f}%")

            if predicted_class.endswith("healthy"):
                st.info("The model classified this leaf as healthy.")
            else:
                st.warning(
                    "The model classified this leaf as potentially diseased. "
                    "This is an AI prediction, not a confirmed diagnosis."
                )

            st.caption(
                "This model recognizes only six classes: Pepper Bell bacterial "
                "spot/healthy, Potato early blight/healthy, and Tomato early "
                "blight/healthy. Predictions outside these classes may be wrong."
            )
