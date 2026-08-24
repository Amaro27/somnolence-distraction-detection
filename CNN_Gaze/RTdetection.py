import time
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ==========================================
# 1. LIGHTWEIGHT CNN ARCHITECTURE
# (Must match the trained model definition)
# ==========================================
class LightGazeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LightGazeCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
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
# 2. LOAD MODEL & INFERENCE SETUP
# ==========================================
def main():
    MODEL_PATH = 'C:\\Users\\genyd\\OneDrive\\Documentos\\Trabajos\\PEF\\light_cnn_gaze_model.pth'  # Path to your saved weights
    IMG_SIZE = 112                          # Match your training resolution
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running inference on device: {device}")

    # Initialize model architecture and load weights
    model = LightGazeCNN(num_classes=2).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # Handle full checkpoint dict or raw state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[+] Successfully loaded model weights from '{MODEL_PATH}'")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    model.eval()

    # Preprocessing transforms (Must match validation/evaluation transforms)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = ['not_looking_road', 'looking_road']

    # Initialize Webcam (0 is usually the default laptop camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open camera.")
        return

    print("[*] Press 'q' to quit the webcam feed.")

    # FPS Calculation variables
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = curr_time

        # Convert OpenCV BGR frame -> PIL Image (RGB) for PyTorch transforms
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply preprocessing & add batch dimension: Shape (1, 3, H, W)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Model Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        label = class_names[predicted_idx.item()]
        conf_score = confidence.item() * 100

        # UI Styling (Green if looking on road, Red if distracted)
        color = (0, 255, 0) if label == 'looking_road' else (0, 0, 255)
        display_text = f"{label} ({conf_score:.1f}%)"

        # Overlay Info on Video Frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, display_text, (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show Window
        cv2.imshow('Real-Time Gaze Classification', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()