"""
PyTorch (.pth) to ONNX (.onnx) Model Conversion & Parity Verification Utility.

Converts trained gaze classification checkpoints from PyTorch `.pth` format into
`.onnx` format compatible with OpenCV DNN (`cv::dnn::readNetFromONNX`) in C++.

Supported architectures in `CNN_Gaze/Modelos`:
    1. LightGazeCNN (`light_cnn_session_full_model.pth`) -> 1x3x112x112
    2. MobileNetV3-Small (`transfer_mobilenet_v3_small_full_model.pth`) -> 1x3x224x224
    3. ResNet18 (`transfer_resnet18_full_model.pth`) -> 1x3x224x224
    4. ResNet18 CLAHE (`resnet18_preprocess_clahe_full_model.pth`) -> 1x3x224x224
    5. GazeResNetLSTM (`resnet18_lstm_unified.pth`) -> feature extractor + sequence head

Usage:
    # Convert the default LightGazeCNN model for benchmark_RTdetection:
    python export_to_onnx.py

    # Convert a specific .pth file:
    python export_to_onnx.py --pth path/to/model.pth --arch light_cnn

    # Convert all standard CNN models in CNN_Gaze/Modelos:
    python export_to_onnx.py --all
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_BENCH_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELOS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "Modelos"))

if CNN_BENCH_DIR not in sys.path:
    sys.path.insert(0, CNN_BENCH_DIR)

from benchmark_RTdetection import LightGazeCNN


def build_architecture(arch_name: str, num_classes: int = 2) -> tuple[nn.Module, int]:
    """Returns (model, default_img_size) for the given architecture identifier."""
    if arch_name == "light_cnn":
        return LightGazeCNN(num_classes=num_classes), 112
    elif arch_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model, 224
    elif arch_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model, 224
    elif arch_name == "resnet18_clahe":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
        return model, 224
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")


def infer_arch_from_state_dict(state_dict: dict) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("features.0.0.") for k in keys):
        return "mobilenet_v3_small"
    if any(k.startswith("fc.1.") for k in keys):
        return "resnet18_clahe"
    if any(k.startswith("conv1.") for k in keys):
        return "resnet18"
    if any(k.startswith("features.0.") for k in keys):
        return "light_cnn"
    raise ValueError("Could not automatically infer model architecture from checkpoint keys.")


def export_checkpoint_to_onnx(
    pth_path: str,
    onnx_path: str = None,
    arch: str = "auto",
    img_size: int = None,
    opset: int = 13,
    verify: bool = True,
) -> str:
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    if onnx_path is None:
        base, _ = os.path.splitext(pth_path)
        onnx_path = base + ".onnx"

    print(f"\n[*] Loading PyTorch checkpoint: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    resolved_arch = infer_arch_from_state_dict(state_dict) if arch == "auto" else arch
    model, default_size = build_architecture(resolved_arch)
    resolved_size = img_size or default_size

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, resolved_size, resolved_size, dtype=torch.float32)

    print(f"[*] Exporting '{resolved_arch}' (input 1x3x{resolved_size}x{resolved_size}) -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"[+] ONNX export complete ({os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB)")

    if verify:
        with torch.no_grad():
            pt_logits = model(dummy_input).numpy()

        net = cv2.dnn.readNetFromONNX(onnx_path)
        net.setInput(dummy_input.numpy())
        cv_logits = net.forward()

        max_abs_diff = float(np.max(np.abs(pt_logits - cv_logits)))
        print(f"[*] Parity Check (PyTorch vs OpenCV cv::dnn):")
        print(f"    PyTorch logits : {pt_logits.flatten().tolist()}")
        print(f"    cv::dnn logits : {cv_logits.flatten().tolist()}")
        print(f"    Max |diff|     : {max_abs_diff:.2e}")
        if max_abs_diff > 1e-4:
            print("[!] WARNING: Numerical difference exceeds 1e-4 threshold!")
        else:
            print("[+] Numerical parity verified (within float32 tolerance).")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch .pth gaze models to ONNX for C++ OpenCV DNN.")
    parser.add_argument(
        "--pth",
        type=str,
        default=os.path.join(MODELOS_DIR, "light_cnn_session_full_model.pth"),
        help="Path to input .pth file (default: light_cnn_session_full_model.pth).",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to output .onnx file (default: same path/name with .onnx extension).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "light_cnn", "mobilenet_v3_small", "resnet18", "resnet18_clahe"],
        help="Model architecture (default: auto-detect from state_dict).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override input image resolution (default: 112 for light_cnn, 224 for ImageNet models).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all standard CNN .pth checkpoints in CNN_Gaze/Modelos.",
    )
    args = parser.parse_args()

    if args.all:
        candidates = [
            "light_cnn_session_full_model.pth",
            "transfer_mobilenet_v3_small_full_model.pth",
            "transfer_resnet18_full_model.pth",
            "resnet18_preprocess_clahe_full_model.pth",
        ]
        for fname in candidates:
            fpath = os.path.join(MODELOS_DIR, fname)
            if os.path.isfile(fpath):
                export_checkpoint_to_onnx(fpath)
    else:
        export_checkpoint_to_onnx(
            pth_path=args.pth,
            onnx_path=args.onnx,
            arch=args.arch,
            img_size=args.img_size,
        )


if __name__ == "__main__":
    main()
