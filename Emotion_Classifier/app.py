# #https://www.gradio.app/guides/image-classification-in-tensorflow

import gradio as gr
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('Models/EmotionClassifier.h5')

# Define a function to predict the class
def predict_class(image):

    # Preprocess the image (resize and normalize)
    resize = tf.image.resize(image, (150, 150))  # Resize to the input size of the model
    resize = np.array(resize) / 255.0  # Normalize the image to [0,1]

    # Make predictions
    yhat = model.predict(np.expand_dims(resize, 0))  # Predict using the model

    print(yhat)

    # Determine the class based on the threshold
    if yhat > 0.5:
        return "Sad"
    else:
        return "Happy"

# Create the Gradio interface
image_input = gr.Image(type="pil")
output_text = gr.Textbox(label="Predicted Emotion")

gr.Interface(fn=predict_class, inputs=image_input, outputs=output_text, title="Emotion Classifier").launch(share=True)
