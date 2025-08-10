# train.py
import os, time, glob, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import get_model
from utils.dataset import ImageFolderDataset
from utils.preprocess import get_transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_items_from_folder(base_dir):
    items = []
    for label_name, label_id in [("non-medical", 0), ("medical",1)]:
        d = os.path.join(base_dir, label_name)
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*")):
            items.append((p, label_id))
    return items

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_items_from_folder(args.data_dir)
    if len(items) < 10:
        raise ValueError("Not enough images. Put images inside sample_data/medical and sample_data/non-medical")
    labels = [l for _, l in items]
    train_items, val_items = train_test_split(items, test_size=0.15, random_state=42, stratify=labels)

    train_ds = ImageFolderDataset(train_items, transform=get_transforms(img_size=args.img_size, is_train=True))
    val_ds = ImageFolderDataset(val_items, transform=get_transforms(img_size=args.img_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = get_model(num_classes=2, base_model=args.backbone, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} val_acc={acc:.4f} time={time.time()-t0:.1f}s")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.save_path)
            print("Saved best model.")
    print("Training complete. Best val acc:", best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="sample_data", help="folder with 'medical' and 'non-medical' children")
    parser.add_argument("--save_path", default="best_model.pth")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--backbone", default="mobilenet_v2", choices=["mobilenet_v2","resnet18"])
    args = parser.parse_args()
    train(args)
