import os
import sys

user_site = r'C:\Users\genyd\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages'
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import time
import argparse
from collections import deque
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# ==========================================
# 1. CLAHE TRANSFORM
# ==========================================
class CLAHETransform:
    def __init__(self, clip_limit=2.5, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        np_img = np.array(img.convert('L'))
        enhanced = self.clahe.apply(np_img)
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb_enhanced)

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class GazeResNetLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, num_classes=2):
        super(GazeResNetLSTM, self).__init__()
        
        resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(64, num_classes)
        )

    def extract_single_frame_feature(self, img_tensor):
        # img_tensor: (1, C, H, W)
        with torch.no_grad():
            feat = self.feature_extractor(img_tensor)  # (1, 512, 1, 1)
        return feat.view(1, self.feature_dim)  # (1, 512)

    def forward_sequence_features(self, feat_sequence):
        # feat_sequence: (1, Seq_Len, 512)
        lstm_out, _ = self.lstm(feat_sequence)
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step)

def load_checkpoint_robustly(model, model_path, device):
    if not os.path.exists(model_path):
        print(f"[!] Warning: Model path '{model_path}' not found.", flush=True)
        return

    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

        # Check if this is a unified checkpoint containing feature_extractor
        if any(k.startswith('feature_extractor.') for k in state_dict.keys()):
            model.load_state_dict(state_dict, strict=True)
            print(f"[+] Loaded unified CNN-LSTM model weights from '{model_path}'", flush=True)
        else:
            # Checkpoint only has LSTM and Classifier weights
            # 1. Load LSTM & Classifier
            model.load_state_dict(state_dict, strict=False)
            print(f"[+] Loaded LSTM & Classifier weights from '{model_path}'", flush=True)
            
            # 2. Automatically load ResNet18 backbone weights from trained CLAHE model
            backbone_path = 'resnet18_preprocess_clahe_full_model.pth'
            if os.path.exists(backbone_path):
                bb_state = torch.load(backbone_path, map_location=device, weights_only=False)
                unified_bb = {}
                for k, v in bb_state.items():
                    if not k.startswith('fc.'):
                        if k.startswith('conv1.'):
                            new_k = k.replace('conv1.', 'feature_extractor.0.')
                        elif k.startswith('bn1.'):
                            new_k = k.replace('bn1.', 'feature_extractor.1.')
                        elif k.startswith('layer1.'):
                            new_k = k.replace('layer1.', 'feature_extractor.4.')
                        elif k.startswith('layer2.'):
                            new_k = k.replace('layer2.', 'feature_extractor.5.')
                        elif k.startswith('layer3.'):
                            new_k = k.replace('layer3.', 'feature_extractor.6.')
                        elif k.startswith('layer4.'):
                            new_k = k.replace('layer4.', 'feature_extractor.7.')
                        else:
                            new_k = f'feature_extractor.{k}'
                        unified_bb[new_k] = v
                model.load_state_dict(unified_bb, strict=False)
                print(f"[+] Loaded pretrained ResNet18 backbone weights from '{backbone_path}'", flush=True)
            else:
                print(f"[!] Warning: '{backbone_path}' not found; backbone will use default ImageNet weights.", flush=True)
    except Exception as e:
        print(f"[!] Error loading checkpoint: {e}", flush=True)

# ==========================================
# 3. REAL-TIME RUNNER
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Real-Time Temporal CNN-LSTM Driver Gaze Classifier")
    parser.add_argument('--model_path', type=str, default='resnet18_lstm_unified.pth', help='Path to trained LSTM model weights')
    parser.add_argument('--seq_len', type=int, default=20, help='Temporal window length (frames)')
    parser.add_argument('--camera_idx', type=int, default=0, help='Webcam index')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================", flush=True)
    print(f"[*] Real-Time Temporal CNN-LSTM Gaze Detection on {device}", flush=True)
    print(f"[*] Temporal Window: {args.seq_len} consecutive frames", flush=True)
    print(f"==================================================", flush=True)

    model = GazeResNetLSTM(hidden_dim=128, num_layers=2, num_classes=2).to(device)
    load_checkpoint_robustly(model, args.model_path, device)
    model.eval()

    clahe = CLAHETransform(clip_limit=2.5, tile_grid_size=(8, 8))
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        clahe,
        transforms.ToTensor(),
        norm
    ])

    class_names = ['not_looking_road', 'looking_road']

    # FIFO queue of pre-extracted 512-dim features
    feature_buffer = deque(maxlen=args.seq_len)
    inference_times = []

    cap = cv2.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {args.camera_idx}", flush=True)
        return

    print("[*] Webcam stream active. Press 'q' to quit.", flush=True)
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        diff = curr_time - prev_time
        if diff > 0:
            fps = 1.0 / diff
        prev_time = curr_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Benchmark complete CNN-LSTM inference latency (Spatial Feature Extraction + Temporal LSTM)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_infer_start = time.perf_counter()

        # 1. Extract 512-dim feature for current frame
        feat = model.extract_single_frame_feature(input_tensor)
        feature_buffer.append(feat)

        # 2. Pass temporal sequence through LSTM
        stacked_features = torch.cat(list(feature_buffer), dim=0).unsqueeze(0)  # (1, curr_len, 512)
        
        with torch.no_grad():
            outputs = model.forward_sequence_features(stacked_features)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        infer_time_ms = (time.perf_counter() - t_infer_start) * 1000.0
        inference_times.append(infer_time_ms)

        label = class_names[pred_idx.item()]
        conf_score = confidence.item() * 100.0

        color = (0, 255, 0) if label == 'looking_road' else (0, 0, 255)
        
        # Display overlays
        fps_text = f"FPS: {fps:.1f} | Latency: {infer_time_ms:.1f} ms"
        status_text = f"LSTM ({len(feature_buffer)}/{args.seq_len}f): {label} ({conf_score:.1f}%)"

        cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, status_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        cv2.imshow('Real-Time Temporal CNN-LSTM Driver Gaze Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if inference_times:
        avg_infer = np.mean(inference_times)
        med_infer = np.median(inference_times)
        min_infer = np.min(inference_times)
        max_infer = np.max(inference_times)
        p95_infer = np.percentile(inference_times, 95)
        fps_cap = 1000.0 / avg_infer if avg_infer > 0 else 0.0
        print("\n" + "=" * 55, flush=True)
        print("[*] INFERENCE BENCHMARK REPORT (Optimized CNN-LSTM)", flush=True)
        print(f"[*] Device:                 {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
        print(f"[*] Total Frames Processed: {len(inference_times):,}", flush=True)
        print(f"[*] Avg Inference Latency:  {avg_infer:.2f} ms ({fps_cap:.1f} FPS theoretical cap)", flush=True)
        print(f"[*] Median Latency:         {med_infer:.2f} ms", flush=True)
        print(f"[*] 95th Percentile (p95):  {p95_infer:.2f} ms", flush=True)
        print(f"[*] Min / Max Latency:      {min_infer:.2f} ms / {max_infer:.2f} ms", flush=True)
        print("=" * 55 + "\n", flush=True)

if __name__ == '__main__':
    main()
