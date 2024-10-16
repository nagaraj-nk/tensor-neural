import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import os

# Parameters
img_height = 28
img_width = 28
batch_size = 32
model_path = "my_image_classification_model.keras"

# Function to load dataset
def load_data(data_dir):
    train_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    validation_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return train_dataset, validation_dataset

# Model building function
def build_model(input_shape):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Assuming 10 classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Save and load model
def save_model(model):
    model.save(model_path)

def load_model():
    return models.load_model(model_path)

# Main code
def main(data_dir):
    # Check if a model already exists locally
    if os.path.exists(model_path):
        print("Model found, loading existing model...")
        model = load_model()
    else:
        print("No model found, building a new model...")
        # Load data
        train_dataset, validation_dataset = load_data(data_dir)

        # Build the model
        input_shape = (img_height, img_width, 3)  # For RGB images
        model = build_model(input_shape)

        # Train the model
        model.fit(train_dataset, validation_data=validation_dataset, epochs=5)

        # Save the model locally
        save_model(model)

    # Test the model with a sample image
    print("Testing with a sample image...")

    import sys
    test_image_path = sys.argv[1]  # Replace with actual test image path
    img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    print(f"Predictions: {predictions}")
    print(f"Predicted class: {tf.argmax(predictions[0])}")

# Specify the path to your dataset directory
data_dir = 'newshapes'

# Run the main code
if __name__ == '__main__':
    main(data_dir)
