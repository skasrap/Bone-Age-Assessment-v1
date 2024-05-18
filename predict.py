# scripts/predict.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.io import imread
from data.data_loader import BoneAgeDataset
from models.gender_model import Gender_Model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
test_path = "/kaggle/input/rsna-bone-age/boneage-test-dataset"
model_path = "best_model.pth"

# Load test data
test_df = pd.read_csv(os.path.join(test_path, "boneage-test-dataset.csv"))
test_df['path'] = test_df['Case ID'].map(lambda x: os.path.join(test_path, f"{x}.png"))
test_df['exists'] = test_df['path'].map(os.path.exists)
test_df["male"] = test_df.Sex.map({'M': 1, 'F': 0})

print(f"Total test samples: {len(test_df)}")

# Define transformations
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=None),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Load test dataset and dataloader
test_data = BoneAgeDataset(test_df, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=100)

# Initialize model
model = Gender_Model(dense_layer1_size=128, dense_layer2_size=128, p_dropout=0.15, regressor_layer_size=256)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Make predictions
predictions_list = []
for imgs, males in test_loader:
    imgs, males = imgs.to(DEVICE), males.to(DEVICE)
    with torch.no_grad():
        predictions = model(imgs, males).cpu().numpy()
        predictions_list.append(predictions)

# Save predictions
predictions_df = pd.DataFrame(np.concatenate(predictions_list), columns=['predictions'])
predictions_df['Case ID'] = test_df['Case ID']
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
