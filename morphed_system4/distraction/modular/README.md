# Modular Driver Distraction & Somnolence Detection System

A high-performance, production-grade C++ implementation combining real-time spatio-temporal driver gaze classification (**GazeResNetLSTM**) with physiological fatigue and postural monitoring (**Dlib 68-Landmark, EAR, Head Pitch Droop, and MAR Yawn**), located at `morphed_system4/distraction/modular/`.

---

## 1. Key Architectural Features

- **Decoupled Parallel Pipeline Architecture**:
  - **Camera Capture Thread**: Dedicated background worker reading camera or video frames with video FPS pacing, broadcasting fresh frames to consumer mailboxes.
  - **Worker 1 (Gaze Distraction Pipeline, ~45 ms)**:
    - Preprocessing: CLAHE contrast enhancement, 224x224 bilinear resizing, and ImageNet normalization.
    - Two-Stage Inference: ResNet-18 spatial feature backbone followed by temporal LSTM recurrence (sequence length = 20 frames).
  - **Worker 2 (Somnolence & Postural Pipeline, ~50 ms)**:
    - **FaceLandmarkDetector**: Single-pass 68-point Dlib facial landmark extraction on every frame (**zero bounding box caching** to eliminate phantom face hallucinations). Includes a 3.0-second lost-face recovery tracker.
    - **EarDetector**: 6-point Eye Aspect Ratio (EAR), two-phase physiological calibration, cascaded adaptive filtering (median filter + moving average baseline drift), blink duration & micro-sleep detection (>500 ms), and rolling 30-second PERCLOS tracking.
    - **HeadOrientationDetector**: Pitch droop detection using roll-invariant vertical facial proportion ($\rho$). Roll tracking and buffers have been removed. Time-based droop ($\le -20^\circ$ for $\ge 1.5\text{ s}$) emits a Light fatigue cue with single-event latching.
    - **MarDetector**: 20-point inner-lip Mouth Aspect Ratio (MAR) tracking mouth opening independently of calibration. Evaluates sustained yawn events ($\text{MAR} > 0.55$ for $\ge 4.0\text{ s}$) with a 2.0-second re-arm cooldown.

- **Thread-Safe Mailboxes (`ThreadSafeMailbox<cv::Mat>`)**:
  - Independent single-slot mailboxes with drop-oldest policy (`mailbox_gaze` and `mailbox_somno`) guarantee that both workers always process the freshest frame with zero latency accumulation or cross-worker lock contention.

- **Unified Multimodal Alert Coordinator (`AlertManager`)**:
  - Internal mutex protection for thread safety across concurrent gaze and somnolence workers.
  - **Distraction Window**: 2.0-second sliding mode evaluation with configurable confidence threshold policies (`BinarizeThreshold` or `FilterUncertain`).
  - **Somnolence Fusion**:
    - **Strong Cues** (immediate critical alarm): Micro-sleep ($>500\text{ ms}$) or High PERCLOS ($\ge 15.0\%$).
    - **Light Cues** (accumulate over a 2-minute sliding window): Medium PERCLOS ($7.5\% - 15.0\%$), Head Droop ($\le -20^\circ$ for $\ge 1.5\text{ s}$), and Sustained Yawn ($>4.0\text{ s}$).
    - **Trigger**: 1 Strong cue OR 2 Light cues within 2 minutes elevates driver state to `Drowsy` with `RiskLevel::High`.

- **Coordinated Physiological Calibration State Machine**:
  - **Stationary Gate**: Calibration can **only** be initiated while `vehicle_state == Stationary`. Presses during movement are rejected.
  - **In-Flight Lockout**: Repeated `'c'` keypresses while calibration is active are ignored.
  - **Movement Interruption Safeguard**: If the vehicle starts `Moving` while calibration is in progress, the FSM immediately aborts calibration and reverts state to `Pending`.
  - **Lost-Face Safeguard**: If the driver's face is lost during calibration, the FSM aborts and reverts to `Pending`.
  - **Lifecycle**:
    - `Pending`: Awaiting stationary user trigger.
    - `CalibOpen` (~4s): Driver looks forward at the road with eyes open; accumulates open-eye EAR and neutral pitch ratio $\rho$.
    - `Transition` (~2s): Prompt displayed to close eyes; head pose baseline $\rho$ is finalized.
    - `CalibClosed` (~4s): Driver keeps eyes gently closed; accumulates closed-eye floor and calculates dynamic descent/closure thresholds.
    - `Completed`: Both physiological modules transition to active monitoring.

- **Interactive Console Controls**: Non-blocking keyboard controller for live testing:
  - `m` / `M`: Toggle vehicle movement state (`Stationary` <-> `Moving`).
  - `c` / `C`: Trigger physiological calibration (stationary only).
  - `q` / `Q`: Graceful multi-worker shutdown.

---

## 2. Directory Layout

```text
morphed_system4/distraction/modular/
├── CMakeLists.txt                    # CMake build configuration (C++17, AVX2, OpenCV, Dlib)
├── CMakePresets.json                 # Presets for GCC (UCRT64) and Native builds
├── README.md                         # System documentation
├── model_path_resolution.patch       # Local patch storing multi-directory path resolution logic
├── input/                            # Required model assets
│   ├── resnet18_lstm_backbone.onnx   # Spatial feature extractor ONNX model
│   ├── resnet18_lstm_head.onnx       # Temporal LSTM classifier ONNX model
│   └── shape_predictor_68_face_landmarks.dat # Dlib 68-point facial landmark model
├── include/
│   ├── AlertManager.hpp              # Thread-safe alert coordinator & multimodal cue fusion
│   ├── CameraCapture.hpp             # Background camera worker with multi-mailbox broadcast
│   ├── ConsoleController.hpp         # Non-blocking interactive keyboard listener
│   ├── EarDetector.hpp               # Eye Aspect Ratio, two-phase calibration, blinks & PERCLOS
│   ├── FaceLandmarkDetector.hpp      # Single-pass Dlib 68-point landmark extractor
│   ├── GazeDetector.hpp              # 2-stage ONNX inference engine (ResNet18 + LSTM)
│   ├── HeadOrientationDetector.hpp   # Pitch droop estimation invariant to roll
│   ├── MarDetector.hpp               # Mouth Aspect Ratio & sustained yawn detector
│   ├── Preprocessor.hpp              # CLAHE, resizing & NCHW ImageNet affine normalization
│   ├── ThreadSafeMailbox.hpp         # Single-slot drop-oldest thread-safe frame mailbox
│   └── Types.hpp                     # Strongly-typed wrappers, configs, DTOs, and domain enums
└── src/
    ├── AlertManager.cpp
    ├── CameraCapture.cpp
    ├── ConsoleController.cpp
    ├── EarDetector.cpp
    ├── FaceLandmarkDetector.cpp
    ├── GazeDetector.cpp
    ├── HeadOrientationDetector.cpp
    ├── MarDetector.cpp
    ├── Preprocessor.cpp
    └── main.cpp                      # Dual-worker orchestrator and application lifecycle
```

---

## 3. Building

The project builds via CMake and MSYS2 UCRT64 GCC (or compatible C++17 compilers with OpenCV and Dlib):

```powershell
cd morphed_system4/distraction/modular
cmake --preset gcc-win-release
cmake --build --preset gcc-win-release
```

The compiled binary will be placed at:
`out/build/gcc-win-release/modular_distraction.exe`

---

## 4. Usage & Command-Line Arguments

The application expects all three models in `input/`:
- `input/resnet18_lstm_backbone.onnx`
- `input/resnet18_lstm_head.onnx`
- `input/shape_predictor_68_face_landmarks.dat`

```text
Usage: modular_distraction [OPTIONS]

Options:
  --source <0|path>          Webcam index (default: 0) or path to video file
  --backbone-path <path>     Path to resnet18_lstm_backbone.onnx
  --head-path <path>         Path to resnet18_lstm_head.onnx
  --predictor-path <path>    Path to shape_predictor_68_face_landmarks.dat
  --conf-threshold <0.70>    Confidence threshold for gaze assessment (default: 0.70)
  --window-ms <2000>         Sliding mode evaluation duration in ms (default: 2000)
  --seq-len <20>             LSTM sequence memory window in frames (default: 20)
  --policy <binarize|filter> Threshold evaluation policy (default: binarize)
  --stationary               Start in stationary state (detections paused)
  --help, -h                 Display help message
```

### Live Demo Keyboard Commands:
- Press **`m`** to toggle between **MOVING** (active detections) and **STATIONARY** (detections paused).
- Press **`c`** while **STATIONARY** to trigger the coordinated 2-phase physiological calibration.
- Press **`q`** or **Ctrl+C** to stop all worker threads and exit cleanly.

---

## 5. Future Roadmap: Calibration Acoustic Guidance

To facilitate eyes-free in-cabin calibration while the vehicle is stationary, dedicated **acoustic guidance tones** are planned.

> [!NOTE]
> **Separation of Concerns**: Calibration acoustic guidance tones are strictly instructional and feedback-oriented. They are completely separate and distinct from the **auditory alerts** dispatched by `AlertManager` (which communicate distraction and drowsiness risk levels).

### Planned Two-Sound Acoustic Pattern

The calibration feedback uses two primary acoustic elements:
1. **Instruction Tone (~1.0s)**: A sustained, gentle tone indicating that a new posture/stage should be assumed.
2. **Cadence Beeps (0.5s interval)**: A rhythmic metronome-like pulse confirming that calibration data is actively and stably being accumulated.

### Two-Phase Calibration Timeline

```
[Start Calibration ('c')]
        │
        ▼
 Phase 1: Neutral Head Pose & Open Eyes (~4 seconds)
  ├── 1.0s Instruction Tone ("Look straight ahead with eyes open")
  ├── 1.0 - 2.0s Brief Silence / Stabilization window
  └── Active Measurement: 8 Beeps over 4.0 seconds (1 beep every 0.5s)
        │
        ▼
 Transition (~2 seconds)
  └── Silence / Transition prompt ("Close your eyes now and hold")
        │
        ▼
 Phase 2: Closed Eyes Floor Baseline (~4 seconds)
  ├── 1.0s Instruction Tone ("Keep eyes gently closed")
  ├── 1.0 - 2.0s Brief Silence / Stabilization window
  └── Active Measurement: 8 Beeps over 4.0 seconds (1 beep every 0.5s)
        │
        ▼
 Completion
  └── Distinct Harmonious Success Chime -> Transitions system to Completed (Active Monitoring)
```

*(Note: Specific sound frequencies, durations, and volume profiles may be calibrated during in-vehicle acoustic testing).*
