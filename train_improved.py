import os
import torch
import timm
import numpy as np
from PIL import Image
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from facenet_pytorch import MTCNN
from sklearn.metrics import classification_report

# ---------------- CONFIG ----------------
DATASET_PATH = "dataset"  # Change to "/kaggle/input/dfdc-dataset/Dataset" for Kaggle
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-5

# Automatically select device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ---------------- FACE DETECTOR ----------------
# NOTE: We use device="cpu" here to avoid CUDA multiprocessing errors in DataLoader workers.
# We set post_process=False so it returns a PIL Image instead of a pre-normalized tensor.
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device="cpu")

# ---------------- TRANSFORMS & AUGMENTATIONS ----------------
# We add richer augmentations specifically helpful for deepfake detection
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    # You can add GaussianBlur or random JPEG compression simulation if packages are available:
    # transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------- CUSTOM DATASET ----------------
class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Path not found: {root}. Check your Dataset folder.")
        self.dataset = datasets.ImageFolder(root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        
        try:
            # Extract face. Because post_process=False, it returns a PIL Image
            face = mtcnn(img)
            if face is not None:
                img = face
        except Exception as e:
            # Fallback to full image if MTCNN fails
            pass 
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# Initialize dataset
try:
    full_dataset = DeepfakeDataset(DATASET_PATH, transform=None)
    print(f"Successfully loaded {len(full_dataset)} images from {DATASET_PATH}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    full_dataset = None

# ---------------- SPLITS & DATALOADERS ----------------
if full_dataset is not None:
    # 80% train, 20% validation/test
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Assign correct transforms to each split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ---------------- MODEL ----------------
    # Tweak model backbone here if you want to experiment (e.g., swin_base_patch4_window7_224 or convnext_base)
    BACKBONE = "vit_base_patch16_224"
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------------- TRAINING WITH MONITORING ----------------
    print(f"Starting training on {DEVICE}...")
    best_val_acc = 0.0
    save_path = "best_deepfake_model.pth"  # For Kaggle use: "/kaggle/working/best_deepfake_model.pth"

    for epoch in range(EPOCHS):
        # --- Train Phase ---
        model.train()
        total_loss, correct = 0, 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
        
        scheduler.step()
        train_acc = correct / len(train_dataset)
        train_loss = total_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, 1)
                val_correct += (preds == labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_acc = val_correct / len(val_dataset)
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
        
        # Save checkpoint if it's the best one so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model checkpoint with Val Acc: {val_acc:.4f}")

    # ---------------- FINAL EVALUATION ----------------
    # Load the best performing checkpoint
    if os.path.exists(save_path):
        print(f"\nLoading best checkpoint for evaluation...")
        model.load_state_dict(torch.load(save_path))
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n===== FINAL CLASSIFICATION REPORT =====")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
else:
    print("Dataset not loaded. Please double check dataset directory paths.")
