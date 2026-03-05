# Aria, Tori, Parisa

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.io import read_image

import matplotlib.pyplot as plt
import torchvision.models as models

import torch.nn as nn


class PreProcessing (ImageFolder):
    # data augmentation
    # resize and normalize
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ensure classes are sorted numerically
    def find_classes(self, directory):
        classes = sorted(os.listdir(directory), key=lambda x: int(x))
        class_to_idx = {class_name: int(class_name) for class_name in classes}
        return classes, class_to_idx


full_train = PreProcessing(
    "/kaggle/input/ucsc-cse-144-winter-2026-final-project/train",
    transform=PreProcessing.transform
)

# get data split
total = len(full_train)
val_size = int(0.1 * total)
train_size = total - val_size  

train_dataset, val_set = torch.utils.data.random_split(full_train, [train_size, val_size]) # update later

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)


print("Number of training images:", len(train_dataset))
print("Classes:", full_train.classes[:10])

# --MODEL-------

# load pretrained resnet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print("Model loaded successfully")

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 100 classes
model.fc = nn.Linear(model.fc.in_features, 100)


# show dataset files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))


#TO DO (aria push test)
# Create dataloader
# Modify resnet layer to 100 classes
# start training
# train on 20 epochs