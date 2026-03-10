import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
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

train_dataset, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)
dv_set = val_loader

print("Number of training images:", len(full_train))
print("Classes:", full_train.classes[:10])

# --MODEL-------------

# load pretrained resnet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
print("Model loaded successfully")

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# fine tune the last block 
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 100)

device = torch.device("cuda")
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=2e-4, 
    weight_decay=1e-2
)

# show dataset files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))

# test training
n_epochs = 15
log_every = 10
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_epoch = -1
ckpt_path = "best_model.pth"
print("Starting training")
print(f"Total epochs: {n_epochs} | device: {device} | optimizer: Adam | loss: CrossEntropyLoss")

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

    print(f"Epoch {epoch + 1}/{n_epochs} complete -> loss: {train_loss:.4f}, acc: {train_acc:.4f} | val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

print("\nTraining complete")

# validation loss
model.eval()

total_loss = 0.0

with torch.no_grad():
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * len(x)

avg_loss = total_loss / len(dv_set.dataset)
print(f"Validation Loss: {avg_loss:.4f}")

# get tests and csv file

# test transform 
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CSVFile(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = sorted(
            [os.path.join(root, f)
             for root, _, files in os.walk(img_dir)
             for f in files if f.endswith('.jpg')],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

# load best model checkpoint
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_dataset = CSVFile(
    "/kaggle/input/ucsc-cse-144-winter-2026-final-project/test",
    transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# run inference
image_ids = []
predicted_labels = []

with torch.no_grad():
    for imgs, paths in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        for path, pred in zip(paths, preds.cpu().numpy()):
            image_ids.append(os.path.basename(path))  # e.g. "192.jpg"
            predicted_labels.append(int(pred))

# save csv
submission = pd.DataFrame({
    "ID": image_ids,
    "Label": predicted_labels
})
submission.to_csv("submission.csv", index=False)
print(submission.head(10))
print(f"Saved {len(submission)} predictions to submission.csv")