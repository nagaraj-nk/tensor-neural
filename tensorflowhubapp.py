import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

# Load a pre-trained model from TensorFlow Hub
model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")

# Define a function for preprocessing the input image
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image by resizing it to the target size, normalizing pixel values, 
    and converting to a tensor format for model input.
    """
    # Load the image from file
    image = Image.open(image_path)

    # Resize the image to the target size (224x224 for MobileNet)
    image = image.resize(target_size)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize pixel values to the range [0, 1]
    image_array = image_array / 255.0

    # Add a batch dimension to the image tensor (required by Keras models)
    image_tensor = tf.expand_dims(image_array, 0)

    return image_tensor

# Load the ImageNet labels
with open('imagenet_class_index.json', 'r') as f:
    class_labels = json.load(f)

def decode_predictions(predictions, top_k=1):
    """
    Decodes the model prediction to human-readable class labels.
    """
    predicted_classes = np.argsort(predictions[0])[::-1][:top_k]  # Get the top K predictions
    decoded_predictions = [(class_labels[str(pred_class)][1], predictions[0][pred_class]) 
                           for pred_class in predicted_classes]
    return decoded_predictions

# Load and preprocess the input image
import sys
image_path = sys.argv[1]  # Replace with the actual path to your image
preprocessed_image = preprocess_image(image_path)

# Make predictions using the model
predictions = model(preprocessed_image).numpy()

# Decode and print the top K predictions
top_k_predictions = decode_predictions(predictions, top_k=5)
for i, (label, score) in enumerate(top_k_predictions):
    print(f"Prediction {i + 1}: {label} (score: {score:.4f})")

# Optionally, visualize the input image
plt.imshow(np.array(Image.open(image_path)))
plt.axis('off')
plt.show()
