# Emotion Classifier: Sad and Happy Face Detection
## Overview of the Analysis
The goal of this project is to develop a machine learning model capable of automatically detecting and classifying facial expressions as either "happy" or "sad" from a dataset containing over 200 images. This project is particularly relevant in fields such as psychology, social sciences, and human-computer interaction, where understanding and interpreting human emotions through facial expressions is crucial.

## Research Questions
1. Classification Accuracy: Can we build a model to accurately classify facial expressions as happy or sad?
2. Performance Across Demographics: Does the model perform equally well across different demographic groups?
3. Detection of Mixed Emotions: Can the model detect mixed emotions (e.g., slightly happy, neutral, slightly sad)?
   
## Data Source
Initially, we obtained images datasets from websites such as Kaggle. However, the quality of the data contains lots of inaccuracies such as sad images in Happy folder, vice versa. We decided the best course of action in order to ensure data quality and integrity is to collate our own data by sourcing from publicly available images, specifically:
* [iStock](https://www.istockphoto.com)
* [Getty Images](https://www.gettyimages.com.au)

This dataset provides a diverse collection of images labeled with sad and happy expressions, suitable for training models to detect and classify emotions based on facial features. The images vary in terms of lighting, angle, and demographic characteristics, ensuring the model's robustness and generalizability.

## Data Preprocessing
- Target Variable: The target variable for the model is the emotional state, either sad or happy, detected from facial images.
- Feature Variables: The feature variables include pixel data extracted from the images.
- Removed Variables: Any irrelevant data, such as image file names, were excluded from the training data.
  
## Model Development
**Neurons, Layers, and Activation Functions:**
- Input Layer: The input layer corresponds to the flattened pixel data from the images.
- Hidden Layers: The model is designed with multiple hidden layers using the ReLU activation function to capture complex patterns in the image data.
- Output Layer: The output layer has one neuron with a sigmoid activation function to classify the images into sad or happy categories.

**Model Performance:**
The model's performance is evaluated using standard metrics such as accuracy, precision, and recall on a test dataset. The initial results indicate that the model achieves reasonable accuracy but requires further tuning to reach optimal performance.

## Steps Taken to Increase Model Performance
**Attempt 1:**
The first model utilized a basic architecture with 2 hidden layers. Despite its simplicity, it provided a baseline for further improvements.

**Attempt 2:**
In the second model, additional hidden layers were introduced, and the number of neurons was increased to improve the model’s ability to capture more complex patterns.

**Attempt 3:**
The final model employed hyperparameter tuning using Keras Tuner to optimize the number of neurons and layers, as well as the choice of activation functions. This approach systematically explored different architectures to identify the optimal configuration.

## Deployment
The model is designed to be deployed as a platform where users can upload photos, and the model automatically classifies them into happy or sad categories.

## Summary
The emotion detection model developed in this project provides a foundational approach to identifying emotional states from facial images. While the model demonstrated moderate success, there is potential for further improvement by exploring additional machine learning techniques and refining the model architecture.

## Recommendations:
- Ensemble Methods: Consider exploring ensemble methods such as Random Forest or Gradient Boosting for potentially better performance.
- Feature Engineering: Invest in feature engineering techniques to extract more informative features from the image data.
- Hyperparameter Tuning: Continue experimenting with different hyperparameters and model architectures to enhance performance.


