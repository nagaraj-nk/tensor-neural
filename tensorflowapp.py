import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, callbacks
import os
import sys

# Parameters
img_height = 28
img_width = 28
batch_size = 32
model_path = "model\\flowers_classification_model.keras"

os.makedirs(os.path.dirname(model_path), exist_ok=True)

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
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', strides=1),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', strides=1),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', strides=1),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='sgd',
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
    # Create a mapping from subdirectory names to labels
    label_names = sorted([dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))])
    label_map = {index: label for index, label in enumerate(label_names)}
    
    # Count the number of classes dynamically
    num_classes = len(label_names)

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
        model = build_model(input_shape, num_classes)

        # Train the model
                
        # Define early stopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',   # Monitor validation loss
            patience=3,           # Stop if val_loss doesn't improve for 3 consecutive epochs
            restore_best_weights=True  # Restore the model weights from the epoch with the best val_loss
        )

        # Train the model with early stopping
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=100,            # Set a high maximum number of epochs
            callbacks=[early_stopping]  # Pass the callback here
        )

        # Save the model locally
        save_model(model)

    # Test the model with a sample image
    print("Testing with a sample image...")
    test_image_path = sys.argv[2]  # Replace with actual test image path
    img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    
    # Map the predicted index back to the label (subdirectory name)
    predicted_label = label_map[tf.argmax(predictions[0]).numpy()]
    print(f"Predicted class: {predicted_label}")

# Specify the path to your dataset directory
data_dir = sys.argv[1]

# Run the main code
if __name__ == '__main__':
    main(data_dir)
