import os
import sys

# Ensure PyTorch CUDA cu121 is loaded from user site-packages
user_site = r'C:\Users\genyd\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages'
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)

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
            nn.Dropout(0.1),
            
            # Block 2: Input (32, 56, 56) -> Output (64, 28, 28)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            
            # Block 3: Input (64, 28, 28) -> Output (128, 14, 14)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            
            # Block 4: Input (128, 14, 14) -> Output (256, 7, 7)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            
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
        except Exception:
            image = Image.new('RGB', (112, 112), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 3. SESSION-BASED DATASET SCANNER
# ==========================================
def scan_dataset_by_session(base_dir):
    print(f"[*] Scanning dataset directory by session: {base_dir} ...", flush=True)
    file_paths = []
    labels = []
    session_ids = []

    for dist_entry in os.scandir(base_dir):
        if dist_entry.is_dir():
            dist_name = dist_entry.name
            rgb_path = os.path.join(dist_entry.path, 'dmd_rgb')
            if os.path.isdir(rgb_path):
                for session_entry in os.scandir(rgb_path):
                    if session_entry.is_dir():
                        sess_name = session_entry.name
                        unique_session_id = f"{dist_name}_{sess_name}"
                        gaze_path = os.path.join(session_entry.path, 'gaze_on_road')
                        if os.path.isdir(gaze_path):
                            # looking_road (Class 1)
                            lr_dir = os.path.join(gaze_path, 'looking_road')
                            if os.path.isdir(lr_dir):
                                for f in os.scandir(lr_dir):
                                    if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                        file_paths.append(f.path)
                                        labels.append(1)
                                        session_ids.append(unique_session_id)
                            # not_looking_road (Class 0)
                            nlr_dir = os.path.join(gaze_path, 'not_looking_road')
                            if os.path.isdir(nlr_dir):
                                for f in os.scandir(nlr_dir):
                                    if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                        file_paths.append(f.path)
                                        labels.append(0)
                                        session_ids.append(unique_session_id)

    print(f"[+] Total images scanned: {len(file_paths)}", flush=True)
    print(f"[+] Unique sessions identified: {len(set(session_ids))}", flush=True)
    return file_paths, labels, session_ids

# ==========================================
# 4. PLOTTING HELPERS
# ==========================================
def generate_and_save_plots(history, y_true, y_probs, y_preds, prefix="session_split"):
    """
    Generates and saves Loss/Accuracy curves, Confusion Matrix, and ROC Curve.
    """
    # 1. Loss & Accuracy Curves
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Plot
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 's-', label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True)

    # Accuracy Plot
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy', color='blue', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 's-', label='Validation Accuracy', color='green', linewidth=2)
    ax2.set_title('Training vs Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True)

    plt.tight_layout()
    loss_acc_file = f"{prefix}_loss_acc.png"
    plt.savefig(loss_acc_file, dpi=300)
    plt.close()
    print(f"[+] Saved Loss & Accuracy plot to {loss_acc_file}", flush=True)

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['not_looking_road', 'looking_road']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Validation Confusion Matrix (Session Split)',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    cm_file = f"{prefix}_confusion_matrix.png"
    plt.savefig(cm_file, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix plot to {cm_file}", flush=True)

    # 3. ROC Curve & AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_file = f"{prefix}_roc_curve.png"
    plt.savefig(roc_file, dpi=300)
    plt.close()
    print(f"[+] Saved ROC Curve plot to {roc_file} (AUC: {roc_auc:.4f})", flush=True)

# ==========================================
# 5. MAIN TRAINING FUNCTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train Session-Split Light CNN on DMD Gaze Dataset")
    parser.add_argument('--dataset_path', type=str, default='Database-PEF/Datos_Entrenamiento_DMD', help='Path to Datos_Entrenamiento_DMD')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=112, help='Image resolution')
    parser.add_argument('--max_samples', type=int, default=0, help='Limit max samples for fast testing (0 = use all)')
    parser.add_argument('--experiment_name', type=str, default='session_split', help='Prefix name for output plots and logs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================", flush=True)
    print(f"[*] Experiment: {args.experiment_name}", flush=True)
    print(f"[*] Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"==================================================", flush=True)

    # 1. Scan Dataset by Session
    file_paths, labels, session_ids = scan_dataset_by_session(args.dataset_path)

    if len(file_paths) == 0:
        raise RuntimeError("No images found!")

    # Subsampling option if specified
    if args.max_samples > 0 and args.max_samples < len(file_paths):
        print(f"[!] Subsampling dataset to max {args.max_samples} images while maintaining session groups...", flush=True)
        np.random.seed(42)
        indices = np.random.choice(len(file_paths), size=args.max_samples, replace=False)
        file_paths = [file_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        session_ids = [session_ids[i] for i in indices]

    # 2. Session-Based Train-Test Split (GroupShuffleSplit)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(file_paths, labels, groups=session_ids))

    train_paths = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_sessions = sorted(list(set([session_ids[i] for i in train_idx])))

    val_paths = [file_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_sessions = sorted(list(set([session_ids[i] for i in val_idx])))

    print(f"\n[+] SESSION SPLIT CONFIRMATION:")
    print(f"    - Training Sessions ({len(train_sessions)}): {train_sessions}")
    print(f"    - Validation Sessions ({len(val_sessions)}): {val_sessions}")
    
    # Check session intersection (MUST BE ZERO)
    intersection = set(train_sessions).intersection(set(val_sessions))
    print(f"    - Overlapping Sessions count: {len(intersection)}")
    assert len(intersection) == 0, "ERROR: Session overlap detected!"

    print(f"[*] Train set: {len(train_paths)} samples | Validation set: {len(val_paths)} samples", flush=True)

    # Class Weights for Imbalance
    class_counts = np.bincount(train_labels)
    total_train = len(train_labels)
    class_weights = total_train / (2.0 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"[*] Class Counts in Train: Class 0 (not_looking): {class_counts[0]}, Class 1 (looking): {class_counts[1]}", flush=True)
    print(f"[*] Class Weights: {class_weights}", flush=True)

    # 3. Data Transforms & Dataloaders
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

    train_dataset = GazeDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = GazeDataset(val_paths, val_labels, transform=val_transform)

    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    # 4. Model Initialization
    model = LightGazeCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    model_save_path = f"light_cnn_{args.experiment_name}_model.pth"

    # Start Timer
    start_time = time.time()

    # 5. Training Loop
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

        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val_samples = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    probs = torch.softmax(outputs, dim=1)[:, 1]

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val_samples += images.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = val_loss / total_val_samples
        val_acc = (correct_val / total_val_samples) * 100.0
        epoch_sec = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%", flush=True)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_sessions': train_sessions,
                'val_sessions': val_sessions
            }, model_save_path)
            print(f"  [+] Saved new best model to {model_save_path} (Val Acc: {val_acc:.2f}%)", flush=True)

    total_training_time_sec = time.time() - start_time
    total_training_time_min = total_training_time_sec / 60.0

    print(f"\n==================================================", flush=True)
    print(f"[+] Experiment '{args.experiment_name}' Training Completed!", flush=True)
    print(f"[+] Total Training Time: {total_training_time_min:.2f} minutes ({total_training_time_sec:.1f} seconds)", flush=True)
    print(f"[+] Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"==================================================", flush=True)

    # 6. Generate Plots and Reports
    generate_and_save_plots(history, all_targets, all_probs, all_preds, prefix=args.experiment_name)

    print("\n[*] Final Validation Classification Report (Session Split):", flush=True)
    target_names = ['not_looking_road', 'looking_road']
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4), flush=True)

if __name__ == '__main__':
    main()
