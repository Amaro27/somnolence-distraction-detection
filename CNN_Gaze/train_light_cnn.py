import os
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. LIGHTWEIGHT CNN ARCHITECTURE
# ==========================================
class LightGazeCNN(nn.Module):
    """
    Ultra-lightweight Convolutional Neural Network designed for fast 
    binary gaze classification (looking_road vs not_looking_road) on GPU.
    """
    def __init__(self, num_classes=2):
        super(LightGazeCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: Input (3, 112, 112) -> Output (32, 56, 56)
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Input (32, 56, 56) -> Output (64, 28, 28)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Input (64, 28, 28) -> Output (128, 14, 14)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: Input (128, 14, 14) -> Output (256, 7, 7)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global Average Pooling -> Output (256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. CUSTOM DATASET CLASS
# ==========================================
class GazeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            # Fallback for corrupt image files if any
            image = Image.new('RGB', (112, 112), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 3. DATASET SCANNER & INDEXER
# ==========================================
def scan_dataset(base_dir):
    print(f"[*] Scanning dataset directory: {base_dir} ...", flush=True)
    looking_road_paths = []
    not_looking_road_paths = []

    # Fast traversal directly targeting gaze_on_road folders
    for dist_entry in os.scandir(base_dir):
        if dist_entry.is_dir():
            rgb_path = os.path.join(dist_entry.path, 'dmd_rgb')
            if os.path.isdir(rgb_path):
                for session_entry in os.scandir(rgb_path):
                    if session_entry.is_dir():
                        gaze_path = os.path.join(session_entry.path, 'gaze_on_road')
                        if os.path.isdir(gaze_path):
                            # looking_road
                            lr_dir = os.path.join(gaze_path, 'looking_road')
                            if os.path.isdir(lr_dir):
                                for f in os.scandir(lr_dir):
                                    if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                        looking_road_paths.append(f.path)
                            # not_looking_road
                            nlr_dir = os.path.join(gaze_path, 'not_looking_road')
                            if os.path.isdir(nlr_dir):
                                for f in os.scandir(nlr_dir):
                                    if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                        not_looking_road_paths.append(f.path)

    print(f"[+] Found {len(looking_road_paths)} images for 'looking_road' (Class 1)", flush=True)
    print(f"[+] Found {len(not_looking_road_paths)} images for 'not_looking_road' (Class 0)", flush=True)
    
    file_paths = not_looking_road_paths + looking_road_paths
    labels = [0] * len(not_looking_road_paths) + [1] * len(looking_road_paths)
    
    return file_paths, labels

# ==========================================
# 4. TRAINING & EVALUATION LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train Light CNN on DMD Gaze Dataset")
    parser.add_argument('--dataset_path', type=str, default='Database-PEF/Datos_Entrenamiento_DMD', help='Path to Datos_Entrenamiento_DMD')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=112, help='Image resolution (width/height)')
    parser.add_argument('--max_samples', type=int, default=0, help='Limit max samples for quick testing (0 = use all)')
    parser.add_argument('--model_output', type=str, default='light_cnn_gaze_model.pth', help='Path to save best model')
    args = parser.parse_args()

    # Hardware check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================", flush=True)
    print(f"[*] Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[*] Memory Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB", flush=True)
    print(f"==================================================", flush=True)

    # Scan dataset
    file_paths, labels = scan_dataset(args.dataset_path)
    
    if len(file_paths) == 0:
        raise RuntimeError("No images found! Check the dataset path.")

    # Optional subsampling if max_samples is set
    if args.max_samples > 0 and args.max_samples < len(file_paths):
        print(f"[!] Subsampling dataset to max {args.max_samples} images for fast trial...", flush=True)
        indices = np.random.choice(len(file_paths), size=args.max_samples, replace=False)
        file_paths = [file_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    # Train / Validation Split (80% / 20%)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"[*] Train set: {len(train_paths)} samples | Validation set: {len(val_paths)} samples", flush=True)

    # Calculate class weights for weighted loss (handles ~80/20 class imbalance)
    class_counts = np.bincount(train_labels)
    total_train = len(train_labels)
    class_weights = total_train / (2.0 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"[*] Class distribution in train: Class 0 (not_looking): {class_counts[0]}, Class 1 (looking): {class_counts[1]}", flush=True)
    print(f"[*] Class Weights: {class_weights}", flush=True)

    # Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets & DataLoaders
    train_dataset = GazeDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = GazeDataset(val_paths, val_labels, transform=val_transform)

    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize Model
    model = LightGazeCNN(num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] LightGazeCNN model initialized with {num_params:,} trainable parameters.")

    # Criterion, Optimizer, Scaler (AMP)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_acc = 0.0
    start_time = time.time()

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == targets).sum().item()
            total_train_samples += images.size(0)

        train_loss = running_loss / total_train_samples
        train_acc = (correct_train / total_train_samples) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val_samples += images.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = val_loss / total_val_samples
        val_acc = (correct_val / total_val_samples) * 100.0
        epoch_sec = time.time() - epoch_start

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%", flush=True)

        scheduler.step(val_acc)

        # Save Best Model Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': ['not_looking_road', 'looking_road']
            }, args.model_output)
            print(f"  [+] Saved new best model checkpoint to {args.model_output} (Val Acc: {val_acc:.2f}%)", flush=True)

    total_time = time.time() - start_time
    print(f"\n==================================================", flush=True)
    print(f"[+] Training completed in {total_time/60:.2f} minutes.", flush=True)
    print(f"[+] Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"==================================================", flush=True)

    # Print Final Classification Report
    print("\n[*] Final Validation Classification Report:", flush=True)
    target_names = ['not_looking_road', 'looking_road']
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4), flush=True)

if __name__ == '__main__':
    main()
