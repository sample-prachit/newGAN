import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Sampler

def copy_G_params(model):
    flatten = deepcopy_params(model)
    lst = [flatten[f] for f in flatten]
    return lst

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def get_dir(args):
    task_name = args.name
    saved_model_folder = os.path.join("train_results", task_name, "models")
    saved_image_folder = os.path.join("train_results", task_name, "images")

    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    return saved_model_folder, saved_image_folder

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.images = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(path, name))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

class InfiniteSamplerWrapper(Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        while True:
            order = torch.randperm(self.num_samples)
            for i in range(self.num_samples):
                yield order[i].item()
            
    def __len__(self):
        return None

def deepcopy_params(model):
    flatten = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            flatten[name] = param.data.clone().detach()
    return flatten
