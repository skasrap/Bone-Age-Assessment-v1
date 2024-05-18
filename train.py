# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.gender_model.py import Gender_Model
from data.data_loader import get_dataloaders, stratified_train_df, stratified_valid_df

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(train_loader, model, loss, metric, optimizer, scheduler):
    model.train()
    epoch_loss = 0.0
    epoch_metric_val = 0.0
    for imgs, genders, labels in train_loader:
        imgs, genders, labels = imgs.to(DEVICE), genders.to(DEVICE), labels.view(-1, 1).to(DEVICE)
        optimizer.zero_grad()
        predictions = model(imgs, genders)
        running_loss = loss(predictions, labels)
        running_metric_val = metric(predictions, labels)
        running_loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += running_loss.item()
        epoch_metric_val += running_metric_val.item()
    return epoch_loss / len(train_loader), epoch_metric_val / len(train_loader)

def validate_one_epoch(val_loader, model, loss, metric):
    model.eval()
    epoch_loss = 0.0
    epoch_metric_val = 0.0
    with torch.no_grad():
        for imgs, genders, labels in val_loader:
            imgs, genders, labels = imgs.to(DEVICE), genders.to(DEVICE), labels.view(-1, 1).to(DEVICE)
            predictions = model(imgs, genders)
            running_loss = loss(predictions, labels)
            running_metric_val = metric(predictions, labels)
            epoch_loss += running_loss.item()
            epoch_metric_val += running_metric_val.item()
    return epoch_loss / len(val_loader), epoch_metric_val / len(val_loader)

def main():
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=None),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=None),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Hardcoded Hyperparameters (Based on the best validation result on wandb sweep with 5 tries).
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.001
    step_size = 10
    gamma = 0.9
    dense_layer1_size = 128
    dense_layer2_size = 128
    p_dropout = 0.15
    regressor_layer_size = 256

    # Load data
    train_loader, val_loader = get_dataloaders(stratified_train_df, stratified_valid_df, batch_size, train_transform)

    # Initialize model
    model = Gender_Model(dense_layer1_size, dense_layer2_size, p_dropout, regressor_layer_size)
    model.to(DEVICE)

    # Define loss function, metric, optimizer, and scheduler
    loss_f = nn.MSELoss()
    metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    best_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss, train_L1 = train_one_epoch(train_loader, model, loss_f, metric, optimizer, scheduler)
        val_loss, val_L1 = validate_one_epoch(val_loader, model, loss_f, metric)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {train_loss:.4f} - Training MAE: {train_L1:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {val_loss:.4f} - Validation MAE: {val_L1:.4f}")

if __name__ == "__main__":
    main()
