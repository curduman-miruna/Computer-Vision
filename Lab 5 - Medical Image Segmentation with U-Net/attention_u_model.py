import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dotenv import load_dotenv

load_dotenv()

# Data loading function
def load_images_and_masks(image_dir, mask_dir, img_size=(128, 128)):
    images, masks = [], []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            img = load_img(img_path, target_size=img_size)
            mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
            images.append(img_to_array(img) / 255.0)  # Normalize images
            masks.append(img_to_array(mask) / 255.0)  # Normalize masks
    return np.array(images), np.array(masks)


# Attention Gate Layer
def attention_gate(x, g, inter_channels):
    # Convolution for gating and input feature map
    theta_x = layers.Conv2D(inter_channels, (1, 1), padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding='same')(g)

    # Add the two feature maps
    add_x_g = layers.Add()([theta_x, phi_g])

    # Apply a ReLU activation and a convolution for refining the attention map
    add_x_g_relu = layers.ReLU()(add_x_g)
    psi = layers.Conv2D(1, (1, 1), padding='same')(add_x_g_relu)
    psi = layers.Activation('sigmoid')(psi)

    return layers.Multiply()([x, psi])


# Attention U-Net model definition
def attention_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder with Attention Gates
    u1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    a1 = attention_gate(c3, u1, 128)  # Attention Gate between encoder and decoder
    u1 = layers.concatenate([u1, a1])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    a2 = attention_gate(c2, u2, 64)
    u2 = layers.concatenate([u2, a2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    a3 = attention_gate(c1, u3, 32)
    u3 = layers.concatenate([u3, a3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    return Model(inputs, outputs)


# Load dataset
images, masks = load_images_and_masks(
    os.getenv("IMAGES_DIRECTORY"),  os.getenv("MASKS_DIRECTORY"),
    img_size=(128, 128)
)

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.3, random_state=42)
model = attention_unet(input_shape=(128, 128, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

model.save("attention_unet_model.keras")
model.summary()

predictions = model.predict(X_test)
predicted_masks = (predictions > 0.5).astype(int)  # Apply threshold to get binary masks


# Define evaluation metrics
def pixel_accuracy(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total
def iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union
def dice_coefficient(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return 2 * intersection / (y_true.sum() + y_pred.sum())


# Compute metrics on the test set
pixel_accuracies = []
ious = []
dice_coeffs = []

for true_mask, pred_mask in zip(y_test, predicted_masks):
    pixel_accuracies.append(pixel_accuracy(true_mask, pred_mask))
    ious.append(iou(true_mask, pred_mask))
    dice_coeffs.append(dice_coefficient(true_mask, pred_mask))

# Mean Metrics
mean_pixel_accuracy = np.mean(pixel_accuracies)
mean_iou = np.mean(ious)
mean_dice = np.mean(dice_coeffs)

print(f"Mean Pixel Accuracy: {mean_pixel_accuracy}")
print(f"Mean IoU (Jaccard's Index): {mean_iou}")
print(f"Mean Dice Coefficient: {mean_dice}")
