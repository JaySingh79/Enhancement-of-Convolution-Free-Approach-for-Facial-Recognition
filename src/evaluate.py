import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import KinshipDataset
from model import ViTKinshipModel
import pandas as pd
import os

# Config
BATCH_SIZE = 16
DATA_DIR = '../data/images'  # Update as needed
CSV_FILE = '../data/val_split.csv'  # Update as needed
MODEL_PATH = '../models/vit_kinship.pth'  # Update as needed

# Transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = KinshipDataset(CSV_FILE, DATA_DIR, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTKinshipModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for (img1, img2), labels in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        outputs = model(img1, img2)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# TODO: Calculate and print ROC AUC, accuracy, and confusion matrix
