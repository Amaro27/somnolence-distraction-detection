import os
import sys

# Ensure PyTorch CUDA cu121 is loaded from user site-packages if present
user_site = r'C:\Users\genyd\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages'
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import time
import argparse
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# ==========================================
# 1. MODEL BUILDER
# ==========================================
def load_transfer_model(model_name, model_path, device, num_classes=2):
    """
    Builds model architecture and loads trained checkpoint weights.
    """
    print(f"[*] Initializing {model_name} architecture...", flush=True)
    if model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = model.to(device)

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"[+] Loaded trained model weights from '{model_path}'", flush=True)
        except Exception as e:
            print(f"[!] Error loading checkpoint '{model_path}': {e}", flush=True)
    else:
        print(f"[!] Warning: Model file '{model_path}' not found. Operating with uninitialized weights.", flush=True)

    model.eval()
    return model

# ==========================================
# 2. REAL-TIME WEBCAM DETECTION & SMOOTHING
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Real-Time Transfer Learning Gaze Classifier with Mode Temporal Smoothing")
    parser.add_argument('--model_name', type=str, default='mobilenet_v3_small', choices=['mobilenet_v3_small', 'resnet18'], help='Model architecture')
    parser.add_argument('--model_path', type=str, default='transfer_mobilenet_v3_small_full_model.pth', help='Path to saved model weights')
    parser.add_argument('--window_size', type=int, default=15, help='Number of consecutive frames for mode temporal smoothing')
    parser.add_argument('--img_size', type=int, default=224, help='Input image resolution (224 for ImageNet models)')
    parser.add_argument('--camera_idx', type=int, default=0, help='Webcam device index')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================", flush=True)
    print(f"[*] Running Real-Time Detection on device: {device}", flush=True)
    print(f"[*] Architecture: {args.model_name}", flush=True)
    print(f"[*] Temporal Smoothing Window (Mode): {args.window_size} frames", flush=True)
    print(f"==================================================", flush=True)

    model = load_transfer_model(args.model_name, args.model_path, device)

    # Preprocessing transforms (ImageNet standard)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = ['not_looking_road', 'looking_road']

    # Initialize sliding buffers for Mode smoothing
    prediction_buffer = deque(maxlen=args.window_size)
    confidence_buffer = deque(maxlen=args.window_size)

    cap = cv2.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera (index {args.camera_idx}).", flush=True)
        return

    print("[*] Starting video stream... Press 'q' to exit.", flush=True)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame from webcam.", flush=True)
            break

        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = curr_time

        # Convert BGR (OpenCV) -> RGB (PIL Image)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Single frame inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            conf_tensor, pred_tensor = torch.max(probabilities, 1)

        raw_pred_idx = pred_tensor.item()
        raw_conf = conf_tensor.item() * 100.0

        # Push prediction and confidence into sliding buffer
        prediction_buffer.append(raw_pred_idx)
        confidence_buffer.append(raw_conf)

        # Calculate Mode over N frames using Counter
        smoothed_pred_idx = Counter(prediction_buffer).most_common(1)[0][0]
        smoothed_conf = np.mean(confidence_buffer)

        smoothed_label = class_names[smoothed_pred_idx]
        raw_label = class_names[raw_pred_idx]

        # UI Styling (Green = looking_road, Red = not_looking_road)
        color = (0, 255, 0) if smoothed_label == 'looking_road' else (0, 0, 255)
        
        # Display Text Overlays
        fps_text = f"FPS: {fps:.1f}"
        mode_text = f"MODE ({len(prediction_buffer)}/{args.window_size}f): {smoothed_label} ({smoothed_conf:.1f}%)"
        raw_text = f"Raw Frame: {raw_label} ({raw_conf:.1f}%)"

        cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, mode_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(frame, raw_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        cv2.imshow('Real-Time Transfer Learning Gaze Detection (Mode Smoothed)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
