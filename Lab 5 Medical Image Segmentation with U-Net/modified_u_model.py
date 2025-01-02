import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from dotenv import load_dotenv

load_dotenv()
def load_images_and_masks(image_dir, mask_dir, img_size=(128, 128)):
    images, masks = [], []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):  # Ensure corresponding mask exists
            img = load_img(img_path, target_size=img_size)
            mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
            images.append(img_to_array(img) / 255.0)  # Normalize images
            masks.append(img_to_array(mask) / 255.0)  # Normalize masks
    return np.array(images), np.array(masks)

# Attention Gate Layer
def attention_gate(x, g, inter_channels):
    theta_x = layers.Conv2D(inter_channels, (1, 1), padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding='same')(g)
    add_x_g = layers.Add()([theta_x, phi_g])
    add_x_g_relu = layers.ReLU()(add_x_g)
    psi = layers.Conv2D(1, (1, 1), padding='same')(add_x_g_relu)
    psi = layers.Activation('sigmoid')(psi)
    return layers.Multiply()([x, psi])

# Attention U-Net model definition
def attention_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)

    #Encoder
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
    u1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    a1 = attention_gate(c3, u1, 128)

    # Decoder with Attention Gates
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

# Data augmentation
data_gen_args = dict(rotation_range=10,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Fit the data generators
image_datagen.fit(X_train, augment=True, seed=42)
mask_datagen.fit(y_train, augment=True, seed=42)

# Create data generators
image_generator = image_datagen.flow(X_train, batch_size=16, seed=42)
mask_generator = mask_datagen.flow(y_train, batch_size=16, seed=42)

train_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_generator(lambda: image_generator, output_signature=tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32)),
                                     tf.data.Dataset.from_generator(lambda: mask_generator, output_signature=tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32))))

model = attention_unet(input_shape=(128, 128, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks_list = [
    callbacks.ModelCheckpoint("best_attention_unet_model.keras", save_best_only=True, monitor='val_loss'),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]

# Train the model
steps_per_epoch = len(X_train) // 16  # Assuming batch size is 16

history = model.fit(train_dataset, validation_data=(X_test, y_test), epochs=50, callbacks=callbacks_list, steps_per_epoch=steps_per_epoch)
model.save("attention_unet_model_mod_2.keras")
model.summary()

# Evaluate model
predictions = model.predict(X_test)
predicted_masks = (predictions > 0.5).astype(int)

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