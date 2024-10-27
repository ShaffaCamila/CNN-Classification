# Importing libraries and dependencies for creating the UI and supporting the deep learning model
import streamlit as st  
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Dropout
from tensorflow.keras.activations import swish

# Define FixedDropout layer
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

# Register FixedDropout and swish activation
get_custom_objects().update({'FixedDropout': FixedDropout, 'swish': swish})

# Hide deprecation warnings that don't affect the app's functionality
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Food Classifier",
    page_icon="🍽️",
    initial_sidebar_state='expanded'
)

# Custom CSS to hide Streamlit's menu and footer
hide_streamlit_style = """
	<style>
    #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load your trained CNN model
model = load_model('./model/model.h5', custom_objects={'FixedDropout': FixedDropout, 'swish': swish})

# Class labels for food categories
class_names = {
    0: 'Dessert 🍰',
    1: 'Noodles-Pasta 🍜',
    2: 'Seafood 🍤',
    3: 'Vegetable-Fruit 🥦'
}

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image
    image = np.array(image) / 255.0    # Scale pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Sidebar content with menu selection
with st.sidebar:
    st.title("🍽️ Food Classifier")
    menu = st.radio("Navigation", ["📋 Define Food Classification & How to Use", "📸 Upload Image"])

# Main content based on sidebar selection
if menu == "📋 Define Food Classification & How to Use":
    st.subheader("🍱 Define Food Classification")
    st.write("Our model classifies food into four categories:")
    st.write("- **Dessert 🍰**: Includes cakes, pastries, and other sweets.")
    st.write("- **Pasta 🍝**: Spaghetti, penne, fusilli, and other pasta-based dishes.")
    st.write("- **Seafood 🍤**: Fish, shrimp, and other seafood.")
    st.write("- **Vegetable-Fruit 🥦**: Fresh vegetables and fruits.")

    # Add space before the next section
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.subheader("🛠️ How to Use This App")
    st.write("1. Go to the 'Upload Image' tab in the sidebar.")
    st.write("2. Upload an image of a food item.")
    st.write("3. Click 'Classify Image' to see the classification result and confidence score.")


elif menu == "📸 Upload Image":
    st.subheader("📸 Upload an Image of Food")
    file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        # Display the uploaded image at a smaller size
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", width=200)  # Display smaller image

        # Add a button to process classification
        if st.button("Classify Image", type="primary"):
            # Show loading spinner
            with st.spinner("Classifying... Please wait..."):
                # Process the image and predict
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)

                # Get the predicted class and confidence level
                predicted_index = np.argmax(predictions)
                predicted_food_name = class_names[predicted_index]
                confidence = predictions[0][predicted_index] * 100

            # Add space before the result section
            st.markdown("<br>", unsafe_allow_html=True)

            # Set a confidence threshold (e.g., 70%)
            confidence_threshold = 70.0

            # Show classification result or warning if confidence is low
            st.write("### Result :")
            if confidence >= confidence_threshold:
                # Displaying the classification result with a rectangle border
              st.markdown(
                f"""
                <div style="border: 3px solid #CBDCEB; padding: 10px; border-radius: 5px; background-color: #133E87;">
                    <h4 style="color: #F3F3E0;">🍽️ This image is classified as: <strong>{predicted_food_name}</strong></h4>
                    <p style="color: #F3F3E0;">Confidence Score: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            else:
                st.warning("⚠️ The image might not match any of the defined food categories. Please upload a clearer or more representative image.")

