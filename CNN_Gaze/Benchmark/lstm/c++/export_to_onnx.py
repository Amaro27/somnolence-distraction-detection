"""
PyTorch (.pth) to ONNX (.onnx) Conversion & Parity Verification for GazeResNetLSTM.

Splits and converts the trained CNN-LSTM gaze checkpoint (`resnet18_lstm_unified.pth`) into
two ONNX sub-networks compatible with OpenCV DNN (`cv::dnn::readNetFromONNX`) in C++:
    1. Spatial CNN Backbone (`resnet18_lstm_backbone.onnx`):
       Input:  [1, 3, 224, 224] (float32, CLAHE + ImageNet normalized)
       Output: [1, 512]         (float32 spatial feature vector)
    2. Temporal LSTM + Classifier Head (`resnet18_lstm_head.onnx`):
       Input:  [1, seq_len, 512] (float32 stacked temporal feature sequence, default seq_len=20)
       Output: [1, 2]            (float32 class logits: [not_looking_road, looking_road])

Usage:
    # Export default resnet18_lstm_unified.pth to ONNX and verify with OpenCV DNN:
    python export_to_onnx.py

    # Custom checkpoint or sequence length:
    python export_to_onnx.py --pth path/to/resnet18_lstm_unified.pth --seq-len 20
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_BENCH_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELOS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "Modelos"))

if LSTM_BENCH_DIR not in sys.path:
    sys.path.insert(0, LSTM_BENCH_DIR)

from benchmark_RTdetection_lstm import GazeResNetLSTM, load_checkpoint_robustly


class ResNet18FeatureExtractorONNX(nn.Module):
    """Wraps GazeResNetLSTM.feature_extractor to output a flat (B, 512) feature vector."""

    def __init__(self, base_model: GazeResNetLSTM):
        super().__init__()
        self.feature_extractor = base_model.feature_extractor
        self.feature_dim = base_model.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        return feat.view(x.size(0), self.feature_dim)


class LSTMSequenceClassifierONNX(nn.Module):
    """Wraps GazeResNetLSTM.lstm + classifier to process (1, seq_len, 512) -> (1, 2)."""

    def __init__(self, base_model: GazeResNetLSTM):
        super().__init__()
        self.lstm = base_model.lstm
        self.classifier = base_model.classifier

    def forward(self, feat_sequence: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(feat_sequence)
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step)


def export_lstm_checkpoint_to_onnx(
    pth_path: str = None,
    backbone_onnx_path: str = None,
    head_onnx_path: str = None,
    img_size: int = 224,
    seq_len: int = 20,
    opset: int = 13,
    verify: bool = True,
) -> tuple[str, str]:
    device = torch.device("cpu")
    model = GazeResNetLSTM(hidden_dim=128, num_layers=2, num_classes=2).to(device)
    load_checkpoint_robustly(model, pth_path, device)
    model.eval()

    if backbone_onnx_path is None:
        backbone_onnx_path = os.path.join(MODELOS_DIR, "resnet18_lstm_backbone.onnx")
    if head_onnx_path is None:
        head_onnx_path = os.path.join(MODELOS_DIR, "resnet18_lstm_head.onnx")

    os.makedirs(os.path.dirname(os.path.abspath(backbone_onnx_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(head_onnx_path)), exist_ok=True)

    backbone_module = ResNet18FeatureExtractorONNX(model).eval()
    head_module = LSTMSequenceClassifierONNX(model).eval()

    # 1. Export ResNet18 spatial backbone
    dummy_img = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    print(f"\n[*] Exporting Spatial CNN Backbone (1x3x{img_size}x{img_size} -> 1x512): {backbone_onnx_path}")
    torch.onnx.export(
        backbone_module,
        dummy_img,
        backbone_onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["features"],
    )
    print(f"[+] Backbone ONNX export complete ({os.path.getsize(backbone_onnx_path) / (1024 * 1024):.2f} MB)")

    # 2. Export Temporal LSTM + Classifier head (static sequence shape for OpenCV DNN compatibility)
    dummy_seq = torch.randn(1, seq_len, 512, dtype=torch.float32)
    print(f"\n[*] Exporting Temporal LSTM Head (1x{seq_len}x512 -> 1x2): {head_onnx_path}")
    torch.onnx.export(
        head_module,
        dummy_seq,
        head_onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["sequence"],
        output_names=["logits"],
    )
    print(f"[+] LSTM Head ONNX export complete ({os.path.getsize(head_onnx_path) / (1024 * 1024):.2f} MB)")

    # 3. Verify numerical parity with OpenCV cv::dnn
    if verify:
        print("\n[*] Verifying Numerical Parity (PyTorch vs OpenCV cv::dnn)...")
        with torch.no_grad():
            pt_feat = backbone_module(dummy_img).numpy()
            pt_logits = head_module(dummy_seq).numpy()

        net_bb = cv2.dnn.readNetFromONNX(backbone_onnx_path)
        net_bb.setInput(dummy_img.numpy())
        cv_feat = net_bb.forward()

        net_head = cv2.dnn.readNetFromONNX(head_onnx_path)
        net_head.setInput(dummy_seq.numpy())
        cv_logits = net_head.forward()

        bb_diff = float(np.max(np.abs(pt_feat - cv_feat)))
        head_diff = float(np.max(np.abs(pt_logits - cv_logits)))

        print(f"    Backbone max |diff| : {bb_diff:.2e}")
        print(f"    PyTorch Head logits : {pt_logits.flatten().tolist()}")
        print(f"    cv::dnn Head logits : {cv_logits.flatten().tolist()}")
        print(f"    LSTM Head max |diff|: {head_diff:.2e}")

        if bb_diff > 1e-4 or head_diff > 1e-4:
            print("[!] WARNING: Numerical difference exceeds 1e-4 threshold!")
        else:
            print("[+] Numerical parity verified for both CNN backbone and LSTM head.")

    return backbone_onnx_path, head_onnx_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert GazeResNetLSTM (.pth) to ONNX backbone + LSTM head for C++ OpenCV DNN."
    )
    parser.add_argument(
        "--pth",
        type=str,
        default=os.path.join(MODELOS_DIR, "resnet18_lstm_unified.pth"),
        help="Path to input .pth file (default: CNN_Gaze/Modelos/resnet18_lstm_unified.pth).",
    )
    parser.add_argument(
        "--backbone-onnx",
        type=str,
        default=None,
        help="Output path for ResNet18 backbone ONNX (default: CNN_Gaze/Modelos/resnet18_lstm_backbone.onnx).",
    )
    parser.add_argument(
        "--head-onnx",
        type=str,
        default=None,
        help="Output path for LSTM head ONNX (default: CNN_Gaze/Modelos/resnet18_lstm_head.onnx).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image spatial resolution (default: 224).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=20,
        help="Temporal sequence window length (default: 20).",
    )
    args = parser.parse_args()

    export_lstm_checkpoint_to_onnx(
        pth_path=args.pth,
        backbone_onnx_path=args.backbone_onnx,
        head_onnx_path=args.head_onnx,
        img_size=args.img_size,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
