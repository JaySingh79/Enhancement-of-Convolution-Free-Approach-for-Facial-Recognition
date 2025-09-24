import torch
from torchvision import transforms
from PIL import Image
from model import ViTKinshipModel
import sys
import os

def predict(img1_path, img2_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTKinshipModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img1, img2)
        prob = torch.sigmoid(output).item()
    return prob

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python src/predict.py <img1_path> <img2_path> <model_path>')
        sys.exit(1)
    img1_path, img2_path, model_path = sys.argv[1:4]
    prob = predict(img1_path, img2_path, model_path)
    print(f'Kinship probability: {prob:.4f}')
