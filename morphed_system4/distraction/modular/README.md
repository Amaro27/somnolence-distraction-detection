# Modular Driver Distraction Detection System

A production-grade, modular C++ implementation of the real-time spatio-temporal driver gaze classifier (**GazeResNetLSTM**), located at `morphed_system4/distraction/modular/`.

---

## 1. Key Architectural Features

- **Multithreaded Camera Capture**: Camera/video ingestion runs on a dedicated background thread decoupled from the inference pipeline.
- **Latest-Frame Mailbox (Drop-Oldest)**: Thread-safe single-slot mailbox (`ThreadSafeMailbox<cv::Mat>`). The processing pipeline always fetches the freshest available frame with zero lag or latency accumulation.
- **Decoupled Alert Engine (`AlertManager`)**:
  - Implements a **2.0-second sliding window mode** of detections.
  - Defaults to **Interpretation A (Binarize Threshold)**: $P(\text{looking\_road}) \ge \text{threshold} \implies 1$ (`LookingRoad`), otherwise $0$ (`NotLookingRoad`).
  - Readily supports switching to **Interpretation B (Filter Uncertain)** via CLI flag `--policy filter`.
  - Emits placeholder print statements on alert triggers and recoveries.
  - Dispatches rate-controlled, non-blocking auditory tone pulses.
  - Includes dedicated extension hooks for future multimodal somnolence integration (immediate strong cues vs. 2 light cues in 2 minutes).
- **Interactive Console Controls**: Non-blocking keyboard controller for live demos:
  - `m` / `M`: Toggle vehicle movement state (`Stationary` <-> `Moving`). Pauses/resumes detections.
  - `c` / `C`: Trigger calibration baseline placeholder.
  - `q` / `Q`: Graceful shutdown.
- **Strict Type Safety**: Focused strong wrapper types (`ConfidenceThreshold`, `DetectionWindowDuration`, `SequenceLengthInFrames`, `ImageDimension2D`) and explicit `enum class` states (`GazeClass`, `VehicleMovementState`, `DriverState`, `RiskLevel`) prevent parameter mix-up.
- **Zero Benchmarking & Zero Visualization Overhead**: All profiling scopes, CSV loggers, and GUI windows (`cv::imshow`, `cv::waitKey`) have been removed for minimal compute overhead.

---

## 2. Directory Layout

```text
morphed_system4/distraction/modular/
├── CMakeLists.txt              # CMake build configuration (C++17, AVX2, OpenCV)
├── CMakePresets.json           # Presets for GCC (UCRT64) and Native builds
├── README.md                   # System documentation
├── include/
│   ├── AlertManager.hpp        # Alert state machine, mode evaluator & audio dispatcher
│   ├── CameraCapture.hpp       # Dedicated background camera capture worker thread
│   ├── ConsoleController.hpp   # Non-blocking interactive keyboard listener
│   ├── GazeDetector.hpp        # 2-stage ONNX inference engine (ResNet18 + LSTM)
│   ├── Preprocessor.hpp        # CLAHE, resizing & NCHW ImageNet affine normalization
│   ├── ThreadSafeMailbox.hpp   # Single-slot drop-oldest thread-safe frame mailbox
│   └── Types.hpp               # Strongly-typed wrappers, configs, and domain enums
└── src/
    ├── AlertManager.cpp
    ├── CameraCapture.cpp
    ├── ConsoleController.cpp
    ├── GazeDetector.cpp
    ├── Preprocessor.cpp
    └── main.cpp                # Pipeline orchestrator and application lifecycle
```

---

## 3. Building

Using CMake and MSYS2 UCRT64 (or MSVC / GCC):

```powershell
cd morphed_system4/distraction/modular
cmake --preset gcc-win-release
cmake --build --preset gcc-win-release
```

The compiled binary will be placed at:
`out/build/gcc-win-release/modular_distraction.exe`

---

## 4. Usage & Command-Line Arguments

The system directly looks for models in `input/`:
- `input/resnet18_lstm_backbone.onnx`
- `input/resnet18_lstm_head.onnx`

```text
Usage: modular_distraction [OPTIONS]

Options:
  --source <0|path>          Webcam index (default: 0) or path to video file
  --backbone-path <path>     Path to resnet18_lstm_backbone.onnx (default: input/resnet18_lstm_backbone.onnx)
  --head-path <path>         Path to resnet18_lstm_head.onnx (default: input/resnet18_lstm_head.onnx)
  --conf-threshold <0.70>    Confidence threshold for gaze assessment (default: 0.70)
  --window-ms <2000>         Sliding mode evaluation duration in ms (default: 2000)
  --seq-len <10>             LSTM sequence memory window in frames (default: 10)
  --policy <binarize|filter> Threshold evaluation policy (default: binarize)
  --stationary               Start in stationary state (detections paused)
  --help, -h                 Display help message
```

### Live Demo Keyboard Commands:
- Press **`m`** to toggle between **MOVING** (active detections) and **STATIONARY** (detections paused).
- Press **`c`** to trigger the calibration baseline placeholder.
- Press **`q`** or **Ctrl+C** to stop all worker threads and exit cleanly.
