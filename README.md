# Emotion Classifier: Sad and Happy Face Detection

## Overview of the Analysis

The goal of this project is to develop a machine learning model capable of automatically detecting and classifying facial expressions as either "happy" or "sad" from a dataset containing over 200 images. This project is particularly relevant in fields such as psychology, social sciences, and human-computer interaction, where understanding and interpreting human emotions through facial expressions is crucial.

## Research Questions

1. Classification Accuracy: Can we build a model to accurately classify facial expressions as happy or sad?
2. Performance Across Demographics: Does the model perform equally well across different demographic groups?
3. Detection of Mixed Emotions: Can the model detect mixed emotions (e.g., slightly happy, neutral, slightly sad)?

## Data Source

Initially, we obtained images datasets from websites such as Kaggle. However, the quality of the data contains lots of inaccuracies such as sad images in Happy folder, vice versa. We decided the best course of action in order to ensure data quality and integrity is to collate our own data by sourcing from publicly available images, specifically:

- [iStock](https://www.istockphoto.com)
- [Getty Images](https://www.gettyimages.com.au)

This dataset provides a diverse collection of images labeled with sad and happy expressions, suitable for training models to detect and classify emotions based on facial features. The images vary in terms of lighting, angle, and demographic characteristics, ensuring the model's robustness and generalizability.

## Data Preprocessing

- Target Variable: The target variable for the model is the emotional state, either sad or happy, detected from facial images.
- Feature Variables: The feature variables include pixel data extracted from the images.

## Model Development

**Neurons, Layers, and Activation Functions:**

- Input Layer: The input layer is designed to handle the flattened pixel data from the images, with a shape of (150, 150, 3).
- Hidden Layers: The model incorporates multiple hidden layers using the ReLU activation function. These layers are designed to capture complex patterns within the image data. MaxPooling layers are included after each convolutional layer to reduce the spatial dimensions of the feature maps, helping the model generalize better and prevent overfitting.
- Output Layer: The output layer consists of a single neuron with a sigmoid activation function. This configuration is ideal for binary classification tasks, such as distinguishing between happy and sad faces.

**Model Performance:**

The model's performance was evaluated using standard metrics, including accuracy and loss, on a test dataset. Initial results showed a moderate accuracy, indicating the potential for further tuning to achieve better performance.

## Steps Taken to Increase Model Performance

### Initial Model:

The initial model utilized a straightforward architecture with two hidden layers:

```
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

**Performance:**

- Accuracy: 0.6716
- Loss: 1.7206

**Description:**
The initial model served as a baseline, employing a basic architecture with a minimal number of convolutional layers and filters. This setup provided a starting point to understand how the model performs with the given data. The relatively high loss indicates that the model struggled with generalizing the data, which is reflected in the moderate accuracy.

### Attempt 1:

In the first attempt to improve the model, additional convolutional layers with increased filters were introduced:

```
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

**Performance:**

- Accuracy: 0.6866
- Loss: 1.6995

**Description:**
The decision to increase the number of filters was aimed at capturing more intricate features from the images. The slight improvement in accuracy suggests that the model was better at identifying patterns in the data. However, the high loss remained a concern, indicating that while the model became slightly better at classifying images, it still overfitted or failed to generalize well.

### Attempt 2:

The second attempt increased the complexity of the model by further increasing the number of filters in the convolutional layers:

```
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

**Performance:**

- Accuracy: 0.7015
- Loss: 1.8275

**Description:**
This attempt aimed to further improve the model's ability to capture complex patterns by increasing the number of filters in the convolutional layers. While the accuracy increased slightly, the loss also increased, indicating that the model was starting to overfit. This suggests that while the model became more complex, it did not necessarily generalize better to unseen data.

### Attempt 3:

In the third attempt, an additional convolutional layer with 32 filters was introduced:

```
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

**Performance:**

- Accuracy: 0.7612
- Loss: 1.3509

**Description:**
Adding an additional convolutional layer aimed to provide the model with more depth to capture even more intricate patterns. This attempt yielded the best accuracy so far, along with a significant reduction in loss. The improvement in performance suggests that the model benefitted from the additional complexity, allowing it to better generalize to the test data.

### Attempt 4:

The final attempt used hyperparameter tuning with Keras Tuner to optimize the model:

```
from keras_tuner import RandomSearch

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Int('conv_1_units', min_value=16, max_value=64, step=16),
                                     kernel_size=(3, 3), activation='relu', input_shape=(150,150,3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(hp.Int('conv_2_units', min_value=16, max_value=64, step=16),
                                     kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=10)
tuner.search(X_train_normalized, y_train_encoded, epochs=15, validation_data=(X_test_normalized, y_test_encoded))
best_model = tuner.get_best_models()[0]
```

**Performance:**

- Best Accuracy: 0.7612
- Loss: 0.5453

**Description:**
This attempt focused on finding the optimal combination of layers, neurons, and learning rates by using a systematic search approach with Keras Tuner. The model achieved the same accuracy as the third attempt but with a significantly lower loss. This reduction in loss indicates better generalization and suggests that the model is more robust in making predictions on unseen data.

### Attempt 5:

In the fifth attempt, the hyperparameters identified in Attempt 4 were used, but batch normalization layers were added to the model to improve training stability and potentially increase accuracy:

```
model.add(tf.keras.layers.Conv2D(64, (3,3), 1, activation='relu', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
tf.keras.layers.BatchNormalization()

model.add(tf.keras.layers.Conv2D(48, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'))
tf.keras.layers.BatchNormalization()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

- .h5 file for this model is 192.90 MB which exceeds Git hub limit of 100 MB.
- Please find it hear - `https://drive.google.com/file/d/17id3sxNQJVtLt7OKcPuzT00eW4HXR0R9/view?usp=drive_link`

**Performance:**

- Accuracy: 0.7761
- Loss: 0.6248

**Description:**
The introduction of batch normalization in Attempt 5 led to a further improvement in accuracy while maintaining a low loss. Batch normalization helps by normalizing the inputs to each layer, which can speed up training and lead to better generalization. This resulted in the best performance observed across all attempts, with the highest accuracy and a competitive loss.

## Deployment

The model is deployed using Gradio where users can upload photos, and the model classifies them into happy or sad categories. The easiet way to do this is to git clone the repository into local drive. Ensure all relevant libraries are installed then activate the specific conda. Navigate to the appropriate directory that contains `app.py`. Execute it by entering `python app.py`. Click on either the specific local or global link that is displayed.

<img src="https://github.com/TruptiRadadiya/Emotion_Classifier/blob/5e0e317f23bdbc9f531c927acc253d3683992625/Emotion_Classifier/Resources/Emotion%20Classifier.png" width="400">

## Summary

The emotion detection model developed in this project provides a foundational approach to identifying emotional states from facial images. While the model demonstrated moderate success, there is potential for further improvement by exploring additional machine learning techniques and refining the model architecture.

## Recommendations:

- Ensemble Methods: Consider exploring ensemble methods such as Random Forest or Gradient Boosting for potentially better performance.
- Feature Engineering: Invest in feature engineering techniques to extract more informative features from the image data.
- Hyperparameter Tuning: Continue experimenting with different hyperparameters and model architectures to enhance performance.

## Observation:

**Classification Accuracy: Can we build a model to accurately classify facial expressions as happy or sad?**
Yes, through the iterations described, we can build a model to classify facial expressions as happy or sad with reasonable accuracy. The final model achieved an accuracy of 0.7612, which indicates that the model is quite effective at distinguishing between these two emotions. However, further improvements could be explored, such as increasing the dataset size or experimenting with more advanced architectures, to push this accuracy even higher.

**Performance Across Demographics: Does the model perform equally well across different demographic groups?**
The current evaluation did not specifically measure performance across different demographic groups, such as age, gender, or ethnicity. To ensure that the model performs equally well across various demographics, it would be essential to include diverse data and conduct fairness testing to detect any potential biases. Such testing is crucial for deploying the model in real-world applications, where equitable performance across all groups is necessary.

**Detection of Mixed Emotions: Can the model detect mixed emotions (e.g., slightly happy, neutral, slightly sad)?**
The model, in its current form, is a binary classifier that distinguishes between happy and sad expressions. It does not detect mixed emotions or provide nuanced classifications like "slightly happy" or "neutral." Detecting mixed emotions would require a more sophisticated model, possibly using a multi-class classification approach or even a regression model to predict the degree of happiness or sadness. This would involve collecting and labeling a dataset with these nuanced emotional categories.
