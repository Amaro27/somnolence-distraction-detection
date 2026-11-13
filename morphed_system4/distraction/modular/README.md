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
├── input/                            # Required model & acoustic assets
│   ├── resnet18_lstm_backbone.onnx   # Spatial feature extractor ONNX model
│   ├── resnet18_lstm_head.onnx       # Temporal LSTM classifier ONNX model
│   ├── shape_predictor_68_face_landmarks.dat # Dlib 68-point facial landmark model
│   └── sounds/                       # WAV audio files for calibration guidance
├── include/
│   ├── AlertManager.hpp              # Thread-safe alert coordinator & multimodal cue fusion
│   ├── CalibrationAudioGuide.hpp     # Multi-stage asynchronous calibration audio sequence manager
│   ├── CameraCapture.hpp             # Background camera worker with multi-mailbox broadcast
│   ├── ConsoleController.hpp         # Non-blocking interactive keyboard listener
│   ├── EarDetector.hpp               # Eye Aspect Ratio, two-phase calibration, blinks & PERCLOS
│   ├── FaceLandmarkDetector.hpp      # Single-pass Dlib 68-point landmark extractor
│   ├── GazeDetector.hpp              # 2-stage ONNX inference engine (ResNet18 + LSTM)
│   ├── HeadOrientationDetector.hpp   # Pitch droop estimation invariant to roll
│   ├── MarDetector.hpp               # Mouth Aspect Ratio & sustained yawn detector
│   ├── Preprocessor.hpp              # CLAHE, resizing & NCHW ImageNet affine normalization
│   ├── SoundPlayer.hpp               # OS-guarded low-latency audio playback (WinMM / POSIX)
│   ├── ThreadSafeMailbox.hpp         # Single-slot drop-oldest thread-safe frame mailbox
│   └── Types.hpp                     # Strongly-typed wrappers, configs, DTOs, and domain enums
└── src/
    ├── AlertManager.cpp
    ├── CalibrationAudioGuide.cpp
    ├── CameraCapture.cpp
    ├── ConsoleController.cpp
    ├── EarDetector.cpp
    ├── FaceLandmarkDetector.cpp
    ├── GazeDetector.cpp
    ├── HeadOrientationDetector.cpp
    ├── MarDetector.cpp
    ├── Preprocessor.cpp
    ├── SoundPlayer.cpp
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

## 5. Calibration Acoustic Guidance

The system provides fully automated, eyes-free **acoustic guidance feedback** during physiological calibration when the vehicle is stationary.

> [!NOTE]
> **Separation of Concerns**: Calibration acoustic guidance tones are strictly instructional and feedback-oriented (`SoundPlayer` / `CalibrationAudioGuide`). They are completely separate and decoupled from the **auditory alerts** dispatched by `AlertManager` (which communicate distraction and drowsiness risk levels).

### 5.1 Where to Change Sound Paths & Settings

All sound paths and timing parameters are defined in [`include/Types.hpp`](file:///c:/Users/joabl/OneDrive/Documentos/GitHub/somnolence-distraction-detection-merges/morphed_system4/distraction/modular/include/Types.hpp) inside `struct CalibrationAudioConfig`:

```cpp
struct CalibrationAudioConfig {
    // Modify any audio file path here:
    std::string phase1_instructions_path{"input/sounds/calib_phase_1.wav"};
    std::string phase1_tone_path        {"input/sounds/sine-415-Hz-1-second-padded.wav"};
    std::string cadence_beep_path       {"input/sounds/single-beep-350-hz.wav"};
    std::string phase2_instructions_path{"input/sounds/calib_phase_2.wav"};
    std::string success_sound_path      {"input/sounds/tada-LOUD.wav"};

    // Sequence timings:
    std::chrono::milliseconds silence_gap{1000};          // Silence between speech/tone and beeps
    std::chrono::milliseconds beep_interval{500};         // 500 ms interval between cadence beeps
    size_t num_beeps{8};                                  // 8 beeps total (8 * 500ms = 4.0s active measurement)
    std::chrono::milliseconds transition_silence{1000};   // Silence during transition
};
```

You can either:
1. **Swap/Replace files directly** inside `morphed_system4/distraction/modular/input/sounds/`.
2. **Change default paths** in [`include/Types.hpp`](file:///c:/Users/joabl/OneDrive/Documentos/GitHub/somnolence-distraction-detection-merges/morphed_system4/distraction/modular/include/Types.hpp) under `CalibrationAudioConfig`.
3. **Override paths programmatically** in `main.cpp` by modifying `ear_config.audio_config.*` before instantiating `EarDetector`.

`SoundPlayer` automatically searches and resolves relative paths across the build directory (`input/sounds/...`), project directory (`morphed_system4/distraction/modular/input/sounds/...`), and repository root.

### 5.2 OS-Guarded Sound Reproduction

Sound playback is handled by [`SoundPlayer`](file:///c:/Users/joabl/OneDrive/Documentos/GitHub/somnolence-distraction-detection-merges/morphed_system4/distraction/modular/include/SoundPlayer.hpp) with compile-time OS guards:
- **Windows (`_WIN32`)**: Native Win32 Multimedia API (`PlaySoundA` linking `-lwinmm`) using `SND_FILENAME | SND_SYNC` for speech/tones, `SND_FILENAME | SND_ASYNC` for rapid beeps, and `PlaySoundA(NULL, NULL, 0)` for instant non-blocking cancellation.
- **Linux / macOS (`__linux__` / `__APPLE__`)**: POSIX subcommands (`aplay -q` / `paplay`) with killall termination.
- **Other OS**: Clean no-op fallback.

### 5.3 Active Calibration Acoustic Sequence

```
[User presses 'c' while Stationary]
        │
        ▼
 Phase 1: Neutral Head Pose & Open Eyes
  ├── 1. Guided Instructions: "calib_phase_1.wav" (~7.7s)
  ├── 2. Silence Gap (1.0s)
  ├── 3. Posture Preparation Tone: "sine-415-Hz-1-second-padded.wav" (~1.4s)
  ├── 4. Silence Gap (1.0s)
  └── 5. Active Measurement Window (4.0 seconds):
         8 cadence beeps ("single-beep-350-hz.wav"), 1 beep every 500 ms.
         *EAR open eye aperture & Head neutral pitch (rho) are actively accumulated.*
        │
        ▼
 Transition (~1.0s)
  ├── Head orientation neutral baseline (rho) finalized.
  └── Brief Silence (no transition tone).
        │
        ▼
 Phase 2: Closed Eyes Floor Baseline
  ├── 1. Guided Instructions: "calib_phase_2.wav" (~7.6s) [replaces 1s tone]
  ├── 2. Silence Gap (1.0s)
  └── 3. Active Measurement Window (4.0 seconds):
         8 cadence beeps ("single-beep-350-hz.wav"), 1 beep every 500 ms.
         *EAR closed eye floor is actively accumulated.*
        │
        ▼
 Completion
  ├── Compute calibrated descent (70%) and closure (20%) thresholds.
  ├── Success Chime: "tada-LOUD.wav" (~1.6s).
  └── Transition to Completed (Active Driver Monitoring Enabled).
```

If the vehicle begins moving, the driver's face is lost, or the user quits at **any** point during calibration, audio playback is stopped immediately and state is cleanly reverted to `Pending`.
