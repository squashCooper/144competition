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

# -- PREPROCESSING -------
class PreProcessing(ImageFolder):
    # data augmentation
    # resize, randomly rotate, random brightness, normalize
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ensure classes are sorted numerically
    def find_classes(self, directory):
        classes = sorted(os.listdir(directory), key=lambda x: int(x))
        class_to_idx = {class_name: int(class_name) for class_name in classes}
        return classes, class_to_idx

# --DATALOADER----------
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
dv_set = val_loader


print("Number of training images:", len(full_train))
print("Classes:", full_train.classes[:10])

# --MODEL-------

# load pretrained resnet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print("Model loaded successfully")

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# fine tune the last block 
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

# ---------------------- Training ----------------------
n_epochs = 10
log_every = 10

print("Starting training")
print(f"Total epochs: {n_epochs} | device: {device} | optimizer: Adam | loss: CrossEntropyLoss")

for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss, correct, seen = 0.0, 0, 0
    print(f"\nEpoch {epoch}/{n_epochs}")

    for batch_idx, (x, y) in enumerate(train_loader, start=1):
        # forward + backward
        x = x.to(device)
        y = y.to(device).long()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        seen += batch_size

        if batch_idx % log_every == 0:
            avg_loss = running_loss / seen
            avg_acc = correct / seen
            print(
                f"  Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"AvgLoss: {avg_loss:.4f} | "
                f"AvgAcc: {avg_acc:.4f}"
            )

    epoch_loss = running_loss / seen
    epoch_acc = correct / seen
    print(f"Epoch {epoch}/{n_epochs} complete -> train_loss: {epoch_loss:.4f}, train_acc: {epoch_acc:.4f}")

    # Validate at end of each epoch
    model.eval()
    val_loss_total, val_correct, val_seen = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dv_set:
            x, y = x.to(device), y.to(device).long()
            pred = model(x)
            loss = criterion(pred, y)

            batch_size = x.size(0)
            val_loss_total += loss.item() * batch_size
            val_correct += (torch.argmax(pred, dim=1) == y).sum().item()
            val_seen += batch_size

    val_loss = val_loss_total / val_seen if val_seen > 0 else 0.0
    val_acc = val_correct / val_seen if val_seen > 0 else 0.0
    print(f"Epoch {epoch}/{n_epochs} validation -> val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    model.train()

print("\nTraining complete")



#!!! from here we can implement training loop !!!!

# show dataset files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))


#TO DO (aria push test)
# Create dataloader
# Modify resnet layer to 100 classes
# start training
# train on 20 epochs
