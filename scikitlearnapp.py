import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Parameters
img_height = 28
img_width = 28
model_path = "model\\flowers_classification_model.pkl"

# Function to load dataset
def load_data(data_dir):
    X = []
    y = []
    label_names = sorted([dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))])
    label_map = {label: index for index, label in enumerate(label_names)}

    for label in label_names:
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize((img_width, img_height))
                img_array = np.asarray(img).flatten()  # Flatten image into a vector
                X.append(img_array)
                y.append(label_map[label])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    
    return X, y, label_names  # Ensure that label_names is returned

# Model building function (Random Forest in this case)
def build_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

# Save and load model
def save_model(model):
    dump(model, model_path)

def load_model():
    return load(model_path)

# Main code
def main(data_dir):
    # Check if a model already exists locally
    if os.path.exists(model_path):
        print("Model found, loading existing model...")
        model = load_model()
        _, _, label_names = load_data(data_dir)  # Ensure label_names is available for prediction
    else:
        print("No model found, building a new model...")
        # Load data
        X, y, label_names = load_data(data_dir)

        # Split data into train and test sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the model
        model = build_model()

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation accuracy: {accuracy * 100:.2f}%")

        # Save the model locally
        save_model(model)

    # Test the model with a sample image
    print("Testing with a sample image...")
    test_image_path = sys.argv[2]  # Replace with actual test image path
    img = Image.open(test_image_path).convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = np.asarray(img).flatten().reshape(1, -1)  # Flatten and reshape for prediction

    prediction = model.predict(img_array)
    
    # Map the predicted index back to the label
    predicted_label = label_names[prediction[0]]  # Ensure label_names is used here
    print(f"Predicted class: {predicted_label}")

# Specify the path to your dataset directory
data_dir = sys.argv[1]

# Run the main code
if __name__ == '__main__':
    main(data_dir)
