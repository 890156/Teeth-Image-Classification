import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

#load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("teeth_classifier_resnet50.keras")
    return model

model = load_model()


class_names= ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Streamlit UI
st.title(" Teeth Image Classification")
st.write("Upload a dental image to classify it")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"]
)



if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize to 256x256 (exact training size)
    img = image.resize((256, 256))
    img_array = np.array(img, dtype=np.float32)

    # 🔹 Apply SAME preprocessing used during training
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # show probabilities
    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

