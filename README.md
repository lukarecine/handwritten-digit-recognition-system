# Handwritten Digit Recognition System Using CNN, DNN, and DBN

## Overview

This project demonstrates a system for handwritten digit recognition using three different machine learning models: **CNN (Convolutional Neural Network)**, **DNN (Deep Neural Network)**, and **DBN (Deep Belief Network)**. The project processes images from the MNIST dataset and evaluates the performance of each model. A real-time recognition system allows for live testing with digit correction and feedback capabilities for future model retraining.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Real-Time Recognition System](#real-time-recognition-system)
- [Performance Evaluation](#performance-evaluation)
- [Usage](#usage)
- [License](#license)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/lukarecine/handwritten-digit-recognition-system.git
   cd handwritten-digit-recognition-system
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the MNIST dataset (optional):
   The models are pre-trained on the MNIST dataset, but if you want to retrain, ensure the dataset is available or downloaded from:
   - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## Project Structure


├── CNN_Model.ipynb          # Implementation of the CNN model
├── DNN_Model.ipynb          # Implementation of the DNN model
├── DBN_Model.ipynb          # Implementation of the DBN model with RBM pretraining
├── real_time_recognition.py  # Real-time digit recognition using Tkinter and OpenCV
├── requirements.txt         # List of required dependencies
├── README.md                # Project documentation

## Models Used

### CNN (Convolutional Neural Network)
- **Description**: CNN is used to extract spatial features from images using convolutional layers. It employs elastic distortion and data augmentation to improve generalization.
- **Architecture**: Consists of convolutional, max-pooling, and dense layers with ReLU activation. Dropout regularization is used to prevent overfitting.

### DNN (Deep Neural Network)
- **Description**: DNN consists of fully connected layers. Unlike CNN, DNN lacks spatial awareness but compensates with depth and regularization techniques such as dropout.
- **Architecture**: Multiple fully connected layers, each with ReLU activation, batch normalization, and dropout.
- 
### DBN (Deep Belief Network)
- **Description**: DBN is a generative model pre-trained layer by layer using Restricted Boltzmann Machines (RBMs). Fine-tuning is done with backpropagation.
- **Architecture**: Consists of stacked RBM layers followed by fine-tuning with dense layers.


## Real-Time Recognition System

A Tkinter-based user interface is used for live handwritten digit recognition:
- **Live Video Feed**: Uses OpenCV to capture video from a webcam and extract digit areas.
- **Digit Prediction**: Uses the loaded model (CNN, DNN, or DBN) to predict the digit in real-time.
- **Correction Mode**: Users can manually correct predictions, and the corrected data is saved for model retraining.
- **Graph Visualization**: Displays the prediction confidence for each digit in real-time.

## Performance Evaluation

Each model's performance is evaluated based on:
- **Accuracy**: Average accuracy over several folds of cross-validation.
- **Precision, Recall, F1-Score**: These metrics are calculated for each digit and displayed in a bar chart and confusion matrix for better comparison.

## Usage

1. **Run Models**:
   You can run any of the models by opening the corresponding `.ipynb` files in Google Colab or Jupyter Notebook:
   ```bash
   jupyter notebook CNN_Model.ipynb
   ```

2. **Real-Time Recognition System**:
   The real-time digit recognition system can be started by running the `real_time_recognition.py` script:
   ```bash
   python digit_recognition_system.py
   ```

3. **Correction Mode**:
   Enable correction mode from the UI to correct any misclassified digits. The corrected data is saved for further model improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
