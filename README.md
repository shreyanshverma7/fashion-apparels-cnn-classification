# fashion-apparels-cnn-classification
This project is about implementing Convolutional Neural Networks (CNNs) for fashion apparels classification using the collected dataset.

Dataset Link: 'https://www.kaggle.com/datasets/shreyanshverma27/new-data-fashion/data'

## CNN Classification of Fashion Apparels

### 1. Introduction

This report documents the process of developing a Convolutional Neural Network (CNN) model to classify fashion apparel images. The data was sourced from Kaggle datasets, and the project includes data collection, model design, training, evaluation, and conclusions. The objective is to classify images into categories such as 'black', 'blue', 'dress', 'pants', 'shirt', 'shoes', and 'shorts'.

### 2. Data Collection

The dataset was collected from Kaggle. The images were organized into categories based on the type of apparel and color. The dataset contains images with labels that were binarized for multi-label classification.

### 3. Model Design

The model is a Convolutional Neural Network (CNN) designed using the Keras library. The architecture includes multiple convolutional layers followed by pooling layers, batch normalization, and dropout layers to prevent overfitting.

**Code Snippet: Model Design**
```python
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout

model = Sequential([
    Conv2D(32, (3, 3), input_shape=(96, 96, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),

    Dense(len(categories), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 4. Training Process

The data was split into training, validation, and test sets. An `ImageDataGenerator` was used for data augmentation to improve model generalization. The model was trained using the training set and validated on the validation set.


### 5. Evaluation Metrics

The model's performance was evaluated using accuracy and confusion matrices. These metrics help in understanding the classification performance across different apparel categories.


### 6. Challenges and Solutions

1. **Overfitting**: Initial models overfitted to the training data. Dropout layers and batch normalization were used to improve generalization.
2. **Model Complexity**: Increasing the depth of the network improved accuracy but also increased training time and computational requirements. A balance was found by optimizing the number of layers and units.

### 7. Conclusions

The CNN model achieved satisfactory performance in classifying fashion apparels. Data augmentation and regularization techniques were crucial in enhancing model generalization. Further improvements can be made by using more sophisticated architectures or transfer learning from pre-trained models.
