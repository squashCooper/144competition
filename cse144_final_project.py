# Aria, Tori, Parisa

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.io import read_image

import matplotlib.pyplot as plt
import torchvision.models as models


# resize images
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# ensure classes are sorted numerically
class SortedImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = sorted(os.listdir(directory), key=lambda x: int(x))
        class_to_idx = {class_name: int(class_name) for class_name in classes}
        return classes, class_to_idx


train_dataset = SortedImageFolder(
    "/kaggle/input/ucsc-cse-144-winter-2026-final-project/train",
    transform=transform
)
train_dataset, val_set = torch.utils.data.random_split(full_train, [55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)


print("Number of training images:", len(train_dataset))
print("Classes:", train_dataset.classes[:10])


# load pretrained resnet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

print("Model loaded successfully")

# show dataset files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))


#TO DO (aria push test)
# Create dataloader
# Modify resnet layer to 100 classes
# start training
# train on 20 epochs