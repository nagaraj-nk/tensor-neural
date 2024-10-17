import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
import sys

# Parameters
img_height = 28
img_width = 28
batch_size = 32
model_path = "flowers_classification_model.pth"

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

# Function to load dataset
def load_data(data_dir):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, dataset.classes

# Model building function
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (img_height // 8) * (img_width // 8), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * (img_height // 8) * (img_width // 8))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Save and load model
def save_model(model):
    torch.save(model.state_dict(), model_path)

def load_model(num_classes):
    model = CNNModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

# Main code
def main(data_dir):
    # Create a mapping from subdirectory names to labels
    label_names = sorted([dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))])
    
    # Count the number of classes dynamically
    num_classes = len(label_names)

    # Check if a model already exists locally
    if os.path.exists(model_path):
        print("Model found, loading existing model...")
        model = load_model(num_classes)
    else:
        print("No model found, building a new model...")

        # Load data
        train_loader, val_loader, _ = load_data(data_dir)

        # Build the model
        model = CNNModel(num_classes)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(5):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        # Save the model locally
        save_model(model)

    # Test the model with a sample image
    print("Testing with a sample image...")
    test_image_path = sys.argv[2]  # Replace with actual test image path
    img = Image.open(test_image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        predictions = model(img)
        predicted_label_idx = predictions.argmax(dim=1).item()

    # Map the predicted index back to the label (subdirectory name)
    predicted_label = label_names[predicted_label_idx]
    print(f"Predicted class: {predicted_label}")

# Specify the path to your dataset directory
data_dir = sys.argv[1]

# Run the main code
if __name__ == '__main__':
    main(data_dir)
