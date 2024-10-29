# flowers_classification.py

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Parameters
IMG_HEIGHT, IMG_WIDTH = 28, 28
BATCH_SIZE = 32
MODEL_PATH = "model/flowers_classification_model.keras"

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load Dataset
def load_data(data_dir):
    common_args = {
        "seed": 123,
        "image_size": (IMG_HEIGHT, IMG_WIDTH),
        "batch_size": BATCH_SIZE,
        "validation_split": 0.2
    }
    train_dataset = image_dataset_from_directory(data_dir, subset="training", **common_args)
    validation_dataset = image_dataset_from_directory(data_dir, subset="validation", **common_args)
    return train_dataset, validation_dataset

# Build Model
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Save and Load Model
def save_model(model):
    model.save(MODEL_PATH)

def load_model():
    return models.load_model(MODEL_PATH)

# Train Model
def train_model(data_dir, num_classes):
    train_dataset, validation_dataset = load_data(data_dir)
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    model = build_model(input_shape, num_classes)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_dataset, validation_data=validation_dataset, epochs=20, callbacks=[early_stopping])
    save_model(model)
    return model


# Predict for All Images in Subdirectories and Calculate Success Rate
def predict_for_all_images(model, data_dir, label_map):
    total_images = 0
    correct_predictions = 0

    for subdir, _, files in os.walk(data_dir):
        subdirname = os.path.basename(subdir)
        if subdirname not in label_map.values():
            continue  # Skip directories not in label map (e.g., non-class directories)

        for file_name in files:
            file_path = os.path.join(subdir, file_name)
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            predicted_label = label_map[tf.argmax(predictions[0]).numpy()]

            # Check if the prediction is correct
            is_correct = (predicted_label == subdirname)
            print(f"{subdirname} = {predicted_label} - {'Correct' if is_correct else 'Incorrect'}")

            # Update counts
            total_images += 1
            if is_correct:
                correct_predictions += 1

    # Calculate and print success rate
    success_rate = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print(f"\nTotal images: {total_images}, Correct predictions: {correct_predictions}")
    print(f"Success rate: {success_rate:.2f}%")

# Main Function
def main(data_dir):
    label_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {i: label for i, label in enumerate(label_names)}
    num_classes = len(label_names)

    if os.path.exists(MODEL_PATH):
        print("Model found, loading existing model...")
        model = load_model()
    else:
        print("No model found, building a new model...")
        model = train_model(data_dir, num_classes)

    print("Predicting for all images in all subdirectories...")
    predict_for_all_images(model, data_dir, label_map)

# Specify the path to your dataset directory
if __name__ == '__main__':
    data_dir = sys.argv[1]
    main(data_dir)
