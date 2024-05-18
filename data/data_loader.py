# data/data_loader.py
# Keep in mind that hyperparameters (e.g., batch size) are hard coded here.
import os
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

train_path = "/kaggle/input/rsna-bone-age/boneage-training-dataset"
train_df = pd.read_csv(os.path.join("/kaggle/input/rsna-bone-age/boneage-training-dataset.csv"))

train_df['path'] = train_df['id'].map(lambda x: os.path.join(train_path, f"{x}.png"))
train_df['exists'] = train_df['path'].map(os.path.exists)  # Check for missing files
train_df.dropna(inplace=True)

print(f"Total training samples: {len(train_df)}")
print(f"Existing training samples: {sum(train_df.exists)}")

train_df['boneage'].hist(figsize=(5, 3))
plt.show()
train_df['male'].astype(int).hist(figsize=(5, 3))
plt.show()
plt.imshow(imread(train_df['path'][0]), cmap="gray")
plt.title(f"Training sample {train_df.id[0]}, bone age: {train_df.boneage[0]}, male: {train_df.male[0]}")
plt.show()

stratified_df = train_df.copy()
stratified_df['boneage_category'] = pd.cut(stratified_df['boneage'], 10)
stratified_train_df, stratified_valid_df = train_test_split(
    stratified_df, test_size=0.25, random_state=2024, stratify=stratified_df['boneage_category']
)
print(f"No. of training samples: {len(stratified_train_df)}. \nNo. of validation samples: {len(stratified_valid_df)}.")

class BoneAgeDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = imread(row['path'])
        male = row['male'].astype(float)
        male = torch.tensor(male, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        if 'boneage' in row.keys():
            label = row['boneage'].astype(float)
            label = torch.tensor(label, dtype=torch.float32)
            return img, male, label
        else:
            return img, male

    def __len__(self):
        return len(self.data)

def get_dataloaders(train_data, valid_data, batch_size=128, transform=None):
    train_dataset = BoneAgeDataset(train_data, transform=transform)
    valid_dataset = BoneAgeDataset(valid_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
