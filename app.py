# -*- coding: utf-8 -*-
# Import necessary libraries
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# --- Try connecting to Google Drive ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    # *** Adjust these paths to match your Google Drive structure ***
    gdrive_base_dir = '/content/drive/My Drive/dl-udemy'
    model_path = os.path.join(gdrive_base_dir, 'pytorch_models', 'malaria_detector_pytorch_best.pth')
    test_image_dir = os.path.join(gdrive_base_dir, 'cell_images', 'test')
    print("Google Drive mounted successfully.")
except ModuleNotFoundError:
    print("Not running in Google Colab or Drive mount failed.")
    print("Please ensure the script can access the model and image files.")
    # Set local paths if not in Colab (adjust as needed)
    model_path = './pytorch_models/malaria_detector_pytorch_best.pth' # Example local path
    test_image_dir = './cell_images/test' # Example local path
    # exit() # Optional: exit if not in Colab and paths aren't set
# remove this following line to use the complete inference, next line is colab right after traning only
model_path = 'malaria_detector_pytorch_best.pth' # Example local path
# --- Configuration ---
IMAGE_HEIGHT = 130
IMAGE_WIDTH = 130
# Define the transformations used during testing/evaluation (must match training)
# Typically includes Resize and ToTensor (which also scales to [0, 1])
inference_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    # Add normalization here ONLY if it was used during training the saved model
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Example normalization
])

# --- Define the Model Architecture (Must be EXACTLY the same as during training) ---
class SimpleCNN(nn.Module):
    def __init__(self, input_shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH)): # Use default shape
        super(SimpleCNN, self).__init__()
        C, H, W = input_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Calculate flattened size dynamically (important for loading weights)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            self.flattened_size = int(torch.flatten(dummy_output, 1).shape[1])

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout is automatically disabled in model.eval()
            nn.Linear(128, 1) # Output is 1 logit for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model ---
print(f"Loading model from: {model_path}")
# Instantiate the model structure
model = SimpleCNN(input_shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH))

# Load the saved weights (state_dict)
try:
    model.load_state_dict(torch.load(model_path, map_location=device)) # map_location ensures correct device loading
    model.to(device) # Move model to the selected device
    model.eval() # Set model to evaluation mode (important!)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Cannot perform inference.")
    exit()
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    print("Ensure the SimpleCNN class definition matches the architecture of the saved model.")
    exit()

# --- Define Class Names (Based on folder structure or training setup) ---
# IMPORTANT: Ensure this matches the indices used during training
# Usually ImageFolder assigns alphabetical order: {'parasitized': 0, 'uninfected': 1} or vice versa
# Check your training output or dataset.class_to_idx if unsure.
class_indices = {'parasitized': 0, 'uninfected': 1} # Adjust if your indices were different
index_to_class = {v: k for k, v in class_indices.items()} # For easy lookup
positive_class_index = 1 # Assuming 'uninfected' is the positive class (output > 0.5). Adjust if 'parasitized' is positive.


# --- Image Selection ---
parasitized_dir = os.path.join(test_image_dir, 'parasitized')
uninfected_dir = os.path.join(test_image_dir, 'uninfected')

try:
    parasitized_images = [os.path.join(parasitized_dir, f) for f in os.listdir(parasitized_dir) if os.path.isfile(os.path.join(parasitized_dir, f))]
    uninfected_images = [os.path.join(uninfected_dir, f) for f in os.listdir(uninfected_dir) if os.path.isfile(os.path.join(uninfected_dir, f))]

    if not parasitized_images or not uninfected_images:
        print("Error: Could not find images in 'parasitized' or 'uninfected' test directories.")
        exit()

    selected_parasitized_img_path = random.choice(parasitized_images)
    selected_uninfected_img_path = random.choice(uninfected_images)

    images_to_predict = {
        "Parasitized": selected_parasitized_img_path,
        "Uninfected": selected_uninfected_img_path
    }
    print("\nSelected images for prediction:")
    print(f"- Parasitized: {os.path.basename(selected_parasitized_img_path)}")
    print(f"- Uninfected: {os.path.basename(selected_uninfected_img_path)}")

except FileNotFoundError:
    print(f"Error: Test image directory or subdirectories not found at {test_image_dir}")
    exit()
except Exception as e:
    print(f"Error listing or selecting images: {e}")
    exit()

# --- Prediction and Display Function ---
def predict_and_display(image_path, actual_class_name, model, transform, device, index_to_class_map, positive_class_idx):
    """Loads, predicts, and displays a single image with annotations."""
    try:
        # Load and transform the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dim and send to device

        # Perform prediction
        with torch.no_grad(): # Disable gradient calculation for inference
            output_logits = model(img_tensor)
            # Apply sigmoid to get probability for the positive class
            probability = torch.sigmoid(output_logits).item()

        # Determine predicted class
        # Assumes output > 0.5 corresponds to the positive class index
        if probability > 0.5:
             predicted_class_index = positive_class_idx
        else:
             predicted_class_index = 1 - positive_class_idx # The other class index

        predicted_class_name = index_to_class_map.get(predicted_class_index, "Unknown")

        # Display results
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Actual Class: {actual_class_name}\n"
                  f"Predicted: {predicted_class_name}\n"
                  f"Probability (Positive Class): {probability:.4f}",
                  fontsize=10)
        plt.axis('off')
        plt.show()

        print(f"\n--- Image: {os.path.basename(image_path)} ---")
        print(f"Actual Class:      {actual_class_name}")
        print(f"Predicted Class:   {predicted_class_name}")
        print(f"Predicted Prob:    {probability:.4f}") # Probability of the positive class

    except FileNotFoundError:
        print(f"Error: Cannot find image file at {image_path}")
    except Exception as e:
        print(f"Error processing or predicting image {os.path.basename(image_path)}: {e}")

# --- Perform Inference on Selected Images ---
print("\n--- Performing Inference ---")
for actual_class, img_path in images_to_predict.items():
     predict_and_display(
         image_path=img_path,
         actual_class_name=actual_class,
         model=model,
         transform=inference_transforms,
         device=device,
         index_to_class_map=index_to_class,
         positive_class_idx=positive_class_index # Make sure this matches your model's output interpretation
    )

print("\n--- Inference Script Finished ---")