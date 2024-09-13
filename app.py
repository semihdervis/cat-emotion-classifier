import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
from utils import predict

# Set a descriptive title for the app
st.title("Cat Emotion Classifier")

# Image uploader widget
uploaded_file = st.file_uploader("Upload a cat image for emotion classification", type=["jpg", "png"])

# Display the uploaded image and perform classification when button is clicked
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Use two columns for layout: image on the left, results on the right
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', width=300)  # Display uploaded image

    # Classification button
    if st.button('Classify Image'):
        try:
            # Display a spinner during the prediction process
            with st.spinner('Classifying the image...'):
                label_confidence = predict(image)
            
            # Show success message
            st.success('Classification complete!')
            
            # Convert the prediction results into a DataFrame for visualization
            df = pd.DataFrame(list(label_confidence.items()), columns=['Label', 'Confidence'])
            
            # Display the prediction results as a horizontal bar chart using Altair
            with col2:
                chart = alt.Chart(df).mark_bar().encode(
                    x='Confidence:Q',
                    y=alt.Y('Label:N', sort='-x')
                ).properties(
                    width=400,
                    height=300
                )
                st.altair_chart(chart)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Instruction message when no image is uploaded
else:
    st.info("Please upload an image to classify.")
