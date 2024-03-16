# Plant Disease Classification with EfficientNetB5
This repository contains code for training a convolutional neural network (CNN) to classify plant diseases using transfer learning with EfficientNetB5.

## Project Overview
This project aims to develop a deep learning model for identifying different plant diseases based on image data. The model leverages transfer learning from the pre-trained EfficientNetB5 model, fine-tuning its final layers for the specific task of plant disease classification.

## Dependencies
This code requires the following Python libraries:

* pandas
* numpy
* os
* matplotlib.pyplot
* Pillow (PIL Fork)
* tensorflow
* keras

## Dataset
The code expects a dataset of plant images organized into subfolders representing different disease classes. The script assumes the following structure:

dataset/
  train/
    Corn_Rust/
      image1.jpg
      image2.jpg
      ...
    Potato_Early_Blight/
      image3.jpg
      image4.jpg
      ...
    ...
  valid/
    ... (similar structure for validation data)
  test/
    ... (similar structure for test data)
    
## Running the Script
* Clone this repository.
* Install the required libraries (pip install -r requirements.txt).
Place your plant disease image dataset in the dataset folder.
Run the script: python train.py

## Explanation of the Code
The train.py script performs the following steps:

### Data Loading and Preprocessing:
Loads image paths and labels from the dataset.
Splits the data into training, validation, and test sets.
Applies data augmentation techniques (optional).
Creates data generators for efficient data loading during training and evaluation.
### Model Building:
Loads the pre-trained EfficientNetB5 model with frozen weights.
Defines a new model architecture using the pre-trained model as a feature extractor.
Adds new layers for classification specific to plant diseases.
### Model Training:
Compiles the model with an optimizer, loss function, and metrics.
Trains the model using the training data generator and monitors performance on the validation data.
Implements early stopping and learning rate reduction techniques to prevent overfitting.
### Fine-tuning:
Unfreezes a small number of final layers in the pre-trained model for further training.
Fine-tunes the model with a lower learning rate to adapt to the plant disease classification task.
### Model Evaluation:
Evaluates the model's performance on the unseen test data using metrics like accuracy and F1-score.
Generates a classification report for detailed class-wise performance analysis.
Creates a confusion matrix to visualize model predictions.
Optional Analysis
The script includes commented-out sections for visualizing data distribution, sample training images, and misclassified images. These sections can be uncommented for further analysis.

## Further Improvements
This code provides a foundation for plant disease classification using transfer learning.  Here are some potential improvements to consider:

Experiment with different data augmentation techniques.
Optimize hyperparameters like learning rate and number of epochs.
Explore different pre-trained models or architectures.
Implement class imbalance handling techniques (if applicable).

## Contributing
We welcome contributions to this project! Please create a pull request with your improvements.
