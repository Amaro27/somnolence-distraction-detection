# Modular Driver Distraction & Somnolence Detection System

A high-performance, production-grade C++ implementation combining real-time spatio-temporal driver gaze classification (**GazeResNetLSTM**, 10-frame pre-optimized default or 20-frame unified) with physiological fatigue and postural monitoring (**CNN Yawning, Dlib 68-Landmark, EAR, Head Pitch Droop**), hardware **MPU-6050 IMU** motion gating (Linux/Raspberry Pi), **Telegram** alerts, and live MP4 debug recording, located at `morphed_system6/distraction/modular/`.

---

## 1. Key Architectural Features

- **Decoupled Parallel Pipeline Architecture**:
  - **Camera Capture Thread**: Dedicated background worker reading camera or video frames with video FPS pacing, broadcasting fresh frames to consumer mailboxes.
  - **Worker 1 (Gaze Distraction Pipeline)**:
    - Preprocessing: CLAHE contrast enhancement, 224x224 bilinear resizing, and ImageNet normalization.
    - Two-Stage Inference: ResNet-18 spatial feature backbone followed by temporal LSTM recurrence (`preopt` 1-layer, sequence length = 10 frames by default; `--model-type unified` selects 2-layer 20 frames).
  - **Worker 2 (Somnolence & Postural Pipeline)**:
    - **YawnDetector**: Full-frame CNN yawning ONNX classifier (`cnn_yawning.onnx`, 112x112 RGB, probability >= 0.80 sustained >= 1.0 s).
    - **FaceLandmarkDetector**: Single-pass 68-point Dlib facial landmark extraction on every frame (**zero bounding box caching**) with 500 ms lost-face probe throttling.
    - **EarDetector**: 6-point Eye Aspect Ratio (EAR), two-phase physiological calibration, cascaded adaptive filtering, blink duration & micro-sleep detection (>500 ms), and rolling 30-second PERCLOS tracking (>= 15.0% Strong cue).
    - **HeadOrientationDetector**: Pitch droop detection using roll-invariant vertical facial proportion ($\rho$). Time-based droop ($\le -25.5^\circ$ for $\ge 1.5\text{ s}$) emits a Light fatigue cue.
  - **Hardware IMU Worker (`MPU6050`)**: Optional Linux I2C motion detector (`/dev/i2c-1`, `0x68`) that toggles Stationary <-> Moving automatically (graceful no-op stub on non-Linux hosts).

- **Thread-Safe Mailboxes (`ThreadSafeMailbox<cv::Mat>`)**:
  - Independent single-slot mailboxes with drop-oldest policy (`mailbox_gaze` and `mailbox_somno`) guarantee that both workers always process the freshest frame with zero latency accumulation or cross-worker lock contention.

- **Unified Multimodal Alert Coordinator (`AlertManager`)**:
  - Internal mutex protection for thread safety across concurrent gaze and somnolence workers.
  - **Distraction Window**: Two-stage temporal filter combining a sliding mode window (`mode_frames` matching LSTM `seq_len` by default: 10 frames for `preopt`, 20 for `unified`; confidence `0.54`, `BinarizeThreshold`, `FilterUncertain`, or `Argmax`) and a 2.0-second sustained `NotLookingRoad` duration requirement (NHTSA rule).
  - **Somnolence Fusion**:
    - **Strong Cues** (immediate critical alarm): Micro-sleep ($>500\text{ ms}$) or High PERCLOS ($\ge 15.0\%$).
    - **Light Cues** (accumulate over a 2-minute sliding window): Head Droop ($\le -25.5^\circ$ for $\ge 1.5\text{ s}$) and Sustained Yawn ($\ge 1.0\text{ s}$).
    - **Trigger**: 1 Strong cue OR 2 Light cues within 2 minutes elevates driver state to `Drowsy`, escalating at 63.5 BPM across `{-21.0, -16.5, -12.0, -7.5, -3.0} dB` and dispatching async **Telegram** notifications.

- **Sequential Physiological Calibration Function (`run_calibration`)**:
  - **Default Stationary Startup & Calibration Gate**: The system starts `Stationary` by default. Gaze distraction, EAR, head droop, and yawn (MAR) detections all remain paused until calibration has succeeded (`system_calibrated == true`) and vehicle state is `Moving`.
  - **Stationary Gate**: Calibration can **only** be initiated while `vehicle_state == Stationary`. Presses during movement are rejected.
  - **In-Flight Lockout & Non-Aborting Sequence**: Repeated `'c'` and movement `'m'` keypresses while calibration is active are ignored/rejected. Missing-face frames are skipped, and any measurement window with insufficient face samples repeats its cadence beeps until `min_samples_per_window` (default 15) samples are collected. Only `'q'` / Ctrl+C exits early.
  - **Single-Thread Execution**: Runs top-to-bottom on `somnolence_worker` (no separate calibration thread), returning an all-or-nothing `CalibrationResult` applied cleanly to `EarDetector` and `HeadOrientationDetector`.

- **Unified Audio-Only `AudioManager`**:
  - Centralizes all sound playback across `AlertManager` (distraction & drowsiness alerts) and `run_calibration()` (voice instructions, posture tone, cadence beeps, success chime) atop OS-guarded `SoundPlayer`.

- **Interactive Console Controls**: Non-blocking keyboard controller for live testing:
  - `m` / `M`: Toggle vehicle movement state (`Stationary` <-> `Moving`).
  - `c` / `C`: Trigger physiological calibration (stationary only).
  - `r` / `R`: Toggle MP4 HUD recording (`DEBUG_VIEWS` builds).
  - `q` / `Q`: Graceful multi-worker shutdown.

---

## 2. Directory Layout

```text
morphed_system6/distraction/modular/
├── CMakeLists.txt                    # CMake build configuration (C++17, AVX2/NEON, OpenCV, Dlib, libcurl)
├── CMakePresets.json                 # Presets for GCC (UCRT64) and Linux / Raspberry Pi builds
├── README.md                         # System documentation
├── input/                            # Required model & acoustic assets
│   ├── resnet18_lstm_preopt_backbone.onnx # Default 10-frame spatial backbone
│   ├── resnet18_lstm_preopt_head.onnx     # Default 10-frame 1-layer LSTM head
│   ├── resnet18_lstm_backbone.onnx        # Unified 20-frame spatial backbone
│   ├── resnet18_lstm_head.onnx            # Unified 20-frame 2-layer LSTM head
│   ├── cnn_yawning.onnx                   # Full-frame CNN yawning ONNX model
│   ├── shape_predictor_68_face_landmarks.dat # Dlib 68-point facial landmark model
│   └── sounds/                       # WAV audio files for calibration & alerts
├── include/
│   ├── AlertLogic.hpp
│   ├── AlertManager.hpp              # Thread-safe alert coordinator & multimodal cue fusion
│   ├── AudioManager.hpp              # Unified audio-only playback & alert sound dispatch
│   ├── Calibration.hpp               # Sequential free-function calibration (run_calibration)
│   ├── CameraCapture.hpp             # Background camera worker with multi-mailbox broadcast
│   ├── ConsoleController.hpp         # Non-blocking interactive keyboard listener
│   ├── DebugVisualizer.hpp           # Debug HUD overlay & MP4 recorder (no-op in Release)
│   ├── EarDetector.hpp               # Eye Aspect Ratio, blinks & PERCLOS
│   ├── FaceGeometry.hpp
│   ├── FaceLandmarkDetector.hpp      # Single-pass Dlib 68-point landmark extractor
│   ├── GazeDetector.hpp              # 2-stage ONNX inference engine (ResNet18 + LSTM)
│   ├── HeadOrientationDetector.hpp   # Pitch droop estimation invariant to roll
│   ├── MarDetector.hpp               # Legacy Mouth Aspect Ratio detector
│   ├── MathUtils.hpp
│   ├── MPU6050.hpp                   # Linux I2C MPU-6050 IMU driver (stub on non-Linux)
│   ├── PathUtils.hpp
│   ├── PipelineCoordinator.hpp
│   ├── Preprocessor.hpp              # CLAHE, resizing & NCHW ImageNet affine normalization
│   ├── SoundPlayer.hpp               # 48 kHz streaming ring-buffer audio engine
│   ├── TelegramNotifier.hpp          # Async libcurl Telegram alert helper
│   ├── ThreadSafeMailbox.hpp         # Single-slot drop-oldest thread-safe frame mailbox
│   ├── Types.hpp                     # Strongly-typed wrappers, configs, DTOs, and domain enums
│   └── YawnDetector.hpp              # Full-frame ONNX CNN yawning detector
```

---

## 3. Building

The project builds via CMake and MSYS2 UCRT64 GCC (or compatible C++17 compilers with OpenCV, Dlib, and libcurl):

```powershell
cd morphed_system6/distraction/modular
cmake --preset gcc-win-release
cmake --build --preset gcc-win-release
```

The compiled binary will be placed at:
`out/build/gcc-win-release/modular_distraction.exe`

---

## 4. Usage & Command-Line Arguments

```text
Usage: modular_distraction [OPTIONS]

Options:
  --source <0|path>          Webcam index (default: 0) or path to video file
  --model-type <preopt|unified> LSTM preset (default: preopt: 1-layer, 10f)
  --backbone-path <path>     Custom path to ResNet-18 backbone ONNX
  --head-path <path>         Custom path to LSTM head ONNX
  --predictor-path <path>    Path to shape_predictor_68_face_landmarks.dat
  --yawn-model-path <path>   Path to cnn_yawning.onnx
  --conf-threshold <0.54>    Confidence threshold for gaze assessment (default: 0.54)
  --mode-frames <N>          Sliding mode window size in frames (default: 10 preopt / 20 unified)
  --window-ms <2000>         Sustained distraction duration in ms (default: 2000)
  --seq-len <N>              LSTM sequence memory window in frames (default: 10 preopt / 20 unified)
  --policy <binarize|filter|argmax> Threshold evaluation policy (default: binarize)
  --record <path.mp4>        Record live HUD & video to MP4 (debug builds)
  --moving                   Start in moving state (default: stationary)
  --no-imu                   Disable hardware IMU (manual 'm' toggle)
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
> **Unified Audio Architecture**: All audio reproduction across the system routes through [`AudioManager`](file:///c:/Users/joabl/OneDrive/Documentos/GitHub/somnolence-distraction-detection-merges/morphed_system4/distraction/modular/include/AudioManager.hpp) and OS-guarded [`SoundPlayer`](file:///c:/Users/joabl/OneDrive/Documentos/GitHub/somnolence-distraction-detection-merges/morphed_system4/distraction/modular/include/SoundPlayer.hpp). `AudioManager` stays purely audio-focused, while [`run_calibration()`](file:///c:/Users/joabl/OneDrive/Documentos/GitHub/somnolence-distraction-detection-merges/morphed_system4/distraction/modular/include/Calibration.hpp) sequences the calibration steps on `somnolence_worker` without spawning extra threads.

### 5.1 Where to Change Sound Paths & Settings

All sound paths and timing parameters are defined in [`include/Types.hpp`](file:///c:/Users/joabl/OneDrive/Documentos/GitHub/somnolence-distraction-detection-merges/morphed_system4/distraction/modular/include/Types.hpp) inside `struct CalibrationAudioConfig` (and alert paths inside `struct AuditoryAlertConfig`):

```cpp
struct CalibrationAudioConfig {
    // Modify any audio file path here:
    std::string phase1_instructions_path{"input/sounds/calib_phase_1.wav"};
    std::string phase1_tone_path        {"input/sounds/sine-415-Hz-1-second-padded.wav"};
    std::string cadence_beep_path       {"input/sounds/single-beep-350-hz.wav"};
    std::string phase2_instructions_path{"input/sounds/calib_phase_2.wav"};
    std::string success_sound_path      {"input/sounds/tada-LOUD.wav"};

    // Sequence timings & sample minimum:
    std::chrono::milliseconds silence_gap{1000};          // Silence between speech/tone and beeps
    std::chrono::milliseconds beep_interval{500};         // 500 ms interval between cadence beeps
    size_t num_beeps{8};                                  // 8 beeps total (8 * 500ms = 4.0s active measurement)
    std::chrono::milliseconds transition_silence{1000};   // Silence during transition
    size_t min_samples_per_window{15};                    // Minimum valid landmark samples per window
};
```

You can either:
1. **Swap/Replace files directly** inside `morphed_system6/distraction/modular/input/sounds/`.
2. **Change default paths** in `include/Types.hpp` under `CalibrationAudioConfig` / `AuditoryAlertConfig`.
3. **Override paths programmatically** in `main.cpp` by modifying `calib_audio_config.*` or `alert_config.audio_config.*` before execution.

`SoundPlayer` automatically searches and resolves relative paths across the build directory (`input/sounds/...`), project directory (`morphed_system6/distraction/modular/input/sounds/...`), and repository root.

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

If the user quits (`'q'` / Ctrl+C) at any point during calibration, audio playback is stopped immediately via `AudioManager::stop()` and `run_calibration()` exits cleanly. Missing-face frames during measurement are skipped, and cadence beeps repeat automatically until at least `min_samples_per_window` valid samples are collected.
