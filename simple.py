
# Import necessary libraries
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load the model outside the function, so you don't reload it every time
loaded_model = load_model('new_model.h5')

# Preprocess the image
def load_and_preprocess_image(img):
    # Convert PIL image to numpy array
    img_array = image.img_to_array(img)
    # Resize the image to (128, 128)
    img_array = np.resize(img_array, (128, 128, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Assuming you've normalized images by dividing by 255 during training

# Streamlit app
def app():
    st.title("Gender and Age Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')  # Convert to grayscale using .convert('L')
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Predicting...")

        img_array = load_and_preprocess_image(img)
        prediction = loaded_model.predict(img_array)
        pred_gender = gender_dict[round(prediction[0][0][0])]
        pred_age = round(prediction[1][0][0])

        st.write(f"Predicted Gender: {pred_gender}")
        st.write(f"Predicted Age: {pred_age}")

# Assuming gender_dict is defined as:
gender_dict = {0: 'Male', 1: 'Female'}

if __name__ == "__main__":
    app()
