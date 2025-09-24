


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from dataset import KinshipDataset
from model import ViTKinshipModel

# Config
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DATA_DIR = '../data/images'  # Update as needed
CSV_FILE = '../data/train_pairs.csv'  # Update as needed

# Data split
df = pd.read_csv(CSV_FILE)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('../data/train_split.csv', index=False)
val_df.to_csv('../data/val_split.csv', index=False)

# Transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = KinshipDataset('../data/train_split.csv', DATA_DIR, transform=data_transforms)
val_dataset = KinshipDataset('../data/val_split.csv', DATA_DIR, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTKinshipModel().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for (img1, img2), labels in train_loader:
        img1, img2 = img1.to(device), img2.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img1.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss:.4f}')

    # TODO: Add validation loop and metrics

# TODO: Save best model based on validation
