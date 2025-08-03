import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model= tf.keras.models.load_model('trai_model.keras')
    image= tf.keras.preprocessing.image.load_img(test_image,target_size=(64, 64))
    input_arr= tf.keras.preprocessing.image.img_to_array(image)
    input_arr= np.array([input_arr]) #convert single image to a batch
    prediction= model.predict(input_arr)
    result_index= np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Fish Disease Detection"])

#Home Page
if(app_mode=="Home"):
    st.header("FISH DISEASE DETECTION MODEL")
    image_path = "fishes.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
*Welcome to the Fish Disease Recognition System!*

Our mission is to assist in identifying fish diseases efficiently. Upload an image of a fish, and our system will analyze it to detect any potential diseases.

### How It Works
1. *Upload Image:* Go to the *Disease Recognition* page and upload an image of a fish showing signs of disease.
2. *Analysis:* Our system will process the image using advanced algorithms to identify possible diseases.
3. *Results:* View the results and receive recommendations for further action.

### Why Choose Us?
- *Accuracy:* Our system utilizes state-of-the-art machine learning techniques for precise disease detection.
- *User-Friendly:* Simple and intuitive interface for seamless user experience.
- *Fast and Efficient:* Receive results in seconds, allowing for quick decision-making.

### Get Started
Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Fish Disease Recognition System.

### About Us
Learn more about the project, our team, and our goals on the *About* page. 
""")
    
#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    ## About Dataset
### Context :
#### General Introduction
The data was created to build a deep learning based fish skin-image to disease identification model.which can help aquaculture . In the dataset there are total** 7** class

1. Bacterial diseases - Aeromoniasis .There are total 250 image .
2. Bacterial gill disease . total number of image 250
3. Bacterial Red disease . total number of image 250
4. Fungal diseases . Saprolegniasis total number of image 250
5. Healthy Fish . total number of image 250
6. Parasitic diseases . total number of image 250
7. Viral diseases White tail disease . total number of image 250
### Data collection
This is the most common disease in freshwater aquaculture.This is a custom dataset. The fish images have been collected from various sources to validate the suggested approach. For example, some images were obtained from a university agricultural department, while others came from an agricultural farm in ODISHA, INDIA, with the help of expert who can identify fish diseases. Some images are collected from agricultural website portal.            
""")
    
#Prediction Page
elif(app_mode=="Fish Disease Detection"):
    st.header("Fish Disease Detection")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    #predict button
    if(st.button("Predict")):
        with st.spinner("Please Wait..."):
            st.write("Model Prediction")
            result_index = model_prediction(test_image)
            #Define Class
            class_name= ['Bacterial Red disease', 'Bacterial diseases - Aeromoniasis', 'Bacterial gill disease', 'Fungal diseases Saprolegniasis', 'Healthy Fish', 'Parasitic diseases', 'Viral diseases White tail disease']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
