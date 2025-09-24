import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class KinshipDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.pairs_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs_frame)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.root_dir, self.pairs_frame.iloc[idx, 0])
        img2_name = os.path.join(self.root_dir, self.pairs_frame.iloc[idx, 1])
        image1 = Image.open(img1_name).convert('RGB')
        image2 = Image.open(img2_name).convert('RGB')
        label = self.pairs_frame.iloc[idx, 2]
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return (image1, image2), label
