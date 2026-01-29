import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms

class ImageData(Dataset):
    def __init__(self, data_path, secret_size=100, num_samples=None, split='train'):
        self.data_path = data_path
        self.secret_size = secret_size
        self.split = split
        
        metadata_file = os.path.join(data_path, 'metadata_generated.jsonl')
        self.samples = []
        
        with open(metadata_file, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        
        if num_samples is not None:
            self.samples = self.samples[:num_samples]
        
        self.transform = transforms.Compose([
            transforms.Resize(512), 
            transforms.CenterCrop(512), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.data_path, sample['file_name'])
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        secret = torch.randint(0, 2, (self.secret_size,)).float()
        
        return image, secret