# Lab5 - Medical image segmentation with U-net

## Files included
- [attention_u_model.py](attention_u_model.py)
- [modified_u_model.py](modified_u_model.py)

## 1. attention_u_model.py

This file contains the implementation of the attention U-net model. The attention U-net model is a modified version of the U-net model that includes attention gates. The attention gates are used to improve the performance of the model by focusing on the most relevant parts of the input image.

### Key Features:
- **Attention U-net Model**: Incorporates attention gates to focus on the most relevant parts of the input image.
- **Data Loading**: Loads images and masks from specified directories and normalizes them.
- **Model Training**: Trains the attention U-net model using the loaded data.
- **Model Evaluation**: Evaluates the model on the test set and computes metrics such as pixel accuracy, IoU, and Dice coefficient.

### Usage:
1. **Load Dataset**: The script loads images and masks from specified directories and splits them into training and testing sets.
2. **Model Training**: Trains the attention U-net model using the loaded data.
3. **Model Evaluation**: Evaluates the model on the test set and computes metrics such as pixel accuracy, IoU, and Dice coefficient.

### Example:
To run the script, ensure you have the required directories set in your environment variables and execute the script in your Python environment.

## 2. modified_u_model.py

This file contains an enhanced version of the attention U-net model with data augmentation. The data augmentation techniques include rotation, width shift, height shift, shear, zoom, and horizontal flip. These augmentations help improve the model's robustness and generalization by artificially increasing the diversity of the training dataset. The model is trained using these augmented datasets and includes callbacks for model checkpointing, early stopping, and learning rate reduction.

### Key Features:
- **Attention U-net Model**: Incorporates attention gates to focus on the most relevant parts of the input image.
- **Data Augmentation**: Utilizes various augmentation techniques to enhance the training dataset.
- **Callbacks**: Implements model checkpointing, early stopping, and learning rate reduction to optimize training.

### Usage:
1. **Load Dataset**: The script loads images and masks from specified directories and splits them into training and testing sets.
2. **Data Augmentation**: Applies augmentation techniques to the training data.
3. **Model Training**: Trains the attention U-net model using the augmented data and specified callbacks.
4. **Model Evaluation**: Evaluates the model on the test set and computes metrics such as pixel accuracy, IoU, and Dice coefficient.

### Example:
To run the script, ensure you have the required directories set in your environment variables and execute the script in your Python environment.