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


# -- TRAIN ONE EPOCH ------
def train_one_epoch(model, train_loader):    
    model.train()
    correct = 0
    total = 0
    cumulative_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        cumulative_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    avg_loss = cumulative_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# -- EVALUATE ------ 
def evaluate(model, loader):
    model.eval()
    cumulative_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            cumulative_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = cumulative_loss / total
    accuracy = correct / total 

    return avg_loss, accuracy

# -- PREPROCESSING -------


class PreProcessing(ImageFolder):
    # ensure classes are sorted numerically
    def find_classes(self, directory):
        classes = sorted(os.listdir(directory), key=lambda x: int(x))
        class_to_idx = {class_name: int(class_name) for class_name in classes}
        return classes, class_to_idx

    pass


# -- TRANSFORMS ----------
train_transform = transforms.Compose([
    # data augmentation
    # resize, randomly rotate, random brightness, normalize
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --DATALOADER----------
data_root = "/kaggle/input/ucsc-cse-144-winter-2026-final-project/train"
full_train = PreProcessing(data_root, transform=train_transform)
total = len(full_train)
val_size = int(0.1 * total)
train_size = total - val_size  

generator = torch.Generator().manual_seed(42)
indices = torch.randperm(total, generator=generator).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(full_train, train_indices)
val_set = torch.utils.data.Subset(
    PreProcessing(data_root, transform=val_transform), val_indices
)  # update later

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)
dv_set = val_loader


print("Number of training images:", len(full_train))
print("Classes:", full_train.classes[:10])

# --MODEL-------

# load pretrained resnet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
print("Model loaded successfully")

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# fine tune the last block 
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, 100)
)

device = torch.device("cuda")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=2e-4, 
    weight_decay=1e-2
)

#!!! from here we can implement training loop !!!!

# show dataset files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))

# ---------------------- Training ----------------------
n_epochs = 10
n_epochs = 10
log_every = 10
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_epoch = -1
ckpt_path = "best_model.pth"
print("Starting training")
print(f"Total epochs: {n_epochs} | device: {device} | optimizer: {optimizer} | loss: CrossEntropyLoss")

for epoch in range(n_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch}, ckpt_path)

    # already returned by train_one_epoch; added val_loss and val_acc to output
    print(f"Epoch {epoch + 1}/{n_epochs} complete -> loss: {train_loss:.4f}, acc: {train_acc:.4f} | val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

print("\nTraining complete")

model.eval()  # set model to evaluation mode

total_loss = 0.0

with torch.no_grad():
    for x, y in dv_set:  # iterate through the validation dataloader
        x, y = x.to(device), y.to(device)  # move data to device
        pred = model(x)                    # forward pass
        loss = criterion(pred, y)          # compute loss
        total_loss += loss.item() * len(x) # accumulate over samples

avg_loss = total_loss / len(dv_set.dataset)  # averaged by total samples
print(f"Validation Loss: {avg_loss:.4f}")