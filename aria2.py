SEED = 42
torch.manual_seed(SEED)

# -- PREPROCESSING -------
class PreProcessing(ImageFolder):
    # data augmentation
    #transform training
    # resize, randomly rotate, random brightness, normalize
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #transform validation
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
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
    transform=PreProcessing.train_transform
)

full_val = PreProcessing(
    "/kaggle/input/ucsc-cse-144-winter-2026-final-project/train",
    transform=PreProcessing.val_transform
)


# get data split
total = len(full_train)
val_size = int(0.1 * total)
train_size = total - val_size  

indices = torch.randperm(total, generator=torch.Generator().manual_seed(SEED))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(full_train, train_indices)
val_dataset = torch.utils.data.Subset(full_val, val_indices)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
dv_set = val_loader


print("Number of training images:", len(full_train))
print("Classes:", full_train.classes[:10])