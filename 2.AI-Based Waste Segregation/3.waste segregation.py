import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.metrics import accuracy_score, classification_report
import cv2

# -------------------- UTILITIES ---------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- DATASET -----------------------

def build_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2):

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(
        data_dir,
        transform=train_transform
    )

    num_total = len(full_dataset)
    num_val = int(num_total * val_split)
    num_train = num_total - num_val

    train_ds, val_ds = random_split(
        full_dataset,
        [num_train, num_val]
    )

    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, full_dataset.classes

# -------------------- MODEL -------------------------

def create_model(num_classes, backbone="resnet50"):

    if backbone == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model

    elif backbone == "efficientnet":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
        return model

    else:
        raise ValueError("Unsupported model")

# -------------------- TRAINING -----------------------

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0
    preds_all = []
    labels_all = []

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        out = model(imgs)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(out, 1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    avg_loss = running_loss / len(loader.dataset)

    return avg_loss, acc

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            loss = criterion(out, labels)

            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(out, 1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    avg_loss = running_loss / len(loader.dataset)

    return avg_loss, acc, preds_all, labels_all

# -------------------- TRAIN LOOP ---------------------

def train_model(data_dir, backbone="resnet50", epochs=10, batch=32):

    set_seed()
    device = get_device()

    train_loader, val_loader, classes = build_loaders(
        data_dir,
        batch_size=batch
    )

    model = create_model(
        len(classes),
        backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    best_acc = 0

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc, preds, labels = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model")

    print("\nClassification Report:")
    print(classification_report(
        labels,
        preds,
        target_names=classes
    ))

    return model, classes

# -------------------- INFERENCE ----------------------

def predict_image(model, img_path, classes, img_size=224):

    device = get_device()

    model = model.to(device)
    model.eval()

    img = cv2.imread(img_path)

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img_resized = cv2.resize(
        img_rgb,
        (img_size, img_size)
    )

    img_tensor = transforms.ToTensor()(img_resized)

    img_tensor = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )(img_tensor)

    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, 1).cpu().numpy()[0]

    idx = np.argmax(probs)

    return classes[idx], float(probs[idx])

# -------------------- USAGE EXAMPLE ------------------

if __name__ == "__main__":

    DATASET_DIR = "dataset/"

    model, classes = train_model(
        DATASET_DIR,
        backbone="efficientnet",
        epochs=8,
        batch=16
    )

    img = "test.jpg"

    label, prob = predict_image(
        model,
        img,
        classes
    )

    print(f"Prediction: {label} ({prob:.2f})")