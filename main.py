import streamlit as st
import tensorflow as tf
import numpy as np

# CSS for background image
page_bg_img = """
<style>
body {
    background-image: url("image.png");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white; /* Ensures text is visible against the background */
}
</style>
"""
# Apply CSS for background
st.markdown(page_bg_img, unsafe_allow_html=True)

def model_prediction(test_image):
    cnn = tf.keras.models.load_model('trained_model.keras')
    image1 = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr1 = tf.keras.preprocessing.image.img_to_array(image1)
    input_arr1 = np.array([input_arr1])  # CONVERT SINGLE IMAGE TO BATCH
    prediction1 = cnn.predict(input_arr1)
    result_index1 = np.argmax(prediction1)
    return result_index1

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Image Recognition"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path, use_column_width=True)
    
    st.markdown("""
    Welcome to the Fruits and Vegetables Recognition System! 🍎🥦🍇

    Our goal is to help you quickly and accurately identify different fruits and vegetables. 
    Simply upload an image, and our system will recognize it, making your experience enjoyable and informative. 
    Let’s celebrate the diversity of produce together and make identification fun and easy!
    """)

    st.markdown("---")
    
    st.markdown("""
    ### How It Works
    1. **Upload Image:** Head over to the **Image Recognition page** and upload an image of a fruit or vegetable.
    2. **Analysis:** Our system will analyze the image using cutting-edge algorithms to identify the type of produce.
    3. **Results:** See the results instantly, along with helpful information about the fruit or vegetable you uploaded.
    """)

    st.markdown("---")

    st.markdown("""
    ### Why Choose Us?
    - **Accuracy:** Utilizing advanced machine learning, our system provides accurate recognition.
    - **User-Friendly:** Designed for a smooth, enjoyable experience for all users.
    - **Fast and Reliable:** Get results in seconds, allowing you to focus on what matters.
    """)

    st.markdown("---")

    st.markdown("""
    ### Get Started
    Click on the **Image Recognition page** in the sidebar to upload an image and discover the power of our Fruits and Vegetables Recognition System!
    """)

    st.markdown("---")

# About Project
elif app_mode == "About":
    st.header("About")

    st.markdown("---")
    
    st.markdown("""
    ### Dataset
    - This dataset is a diverse collection of images representing various fruits and vegetables, designed for image recognition tasks.
    - It includes a broad range of items commonly found in kitchens and markets, creating a valuable resource for developing food-recognition applications.
    - The dataset covers:
      - **Fruits**: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango
      - **Vegetables**: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant
    """)

    st.markdown("---")

    st.markdown("""
    ### Content
    - The dataset is organized into three main folders:
      1. **Train**: Contains 100 images per category, used to train the model on various fruits and vegetables.
      2. **Test**: Contains 10 images per category, designated for evaluating model performance.
      3. **Validation**: Contains 10 images per category, used for fine-tuning and cross-validating model accuracy.
    - Each folder has subdirectories for each type of fruit and vegetable, ensuring an organized and straightforward structure for efficient model training.
    """)

    st.markdown("---")

# Prediction Page
elif app_mode == "Image Recognition":
    st.header("Image Recognition")

    test_image = st.file_uploader("Choose an Image:")
    if test_image and st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)

    if test_image and st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = [
            'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
            'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
            'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'radish', 'soy beans',
            'spinach', 'sweetcorn', 'sweet potato', 'tomato', 'turnip', 'watermelon'
        ]
        
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
