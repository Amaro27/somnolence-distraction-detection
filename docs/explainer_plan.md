# Real-Time Architecture Explainer Plan (`morphed_system6/distraction/modular/`)

> **Deliverable Scope**: This document is the **implementation blueprint** for a lightweight, interactive, browser-based set of system visualizers stored under `docs/explainer/`. All 5 chapters and verification audits reflect **Morphed System 6 (`morphed_system6/distraction/modular/`)**.

---

## 1. Purpose, Audience & Hard Constraints

### Target System & Anchor
- **Target Runtime**: `morphed_system6/distraction/modular/` — the unified C++17 multi-threaded driver distraction and somnolence detection system combining MS4's full algorithmic/alerting stack (2-stage distraction filter, CNN yawning, `-25.5°` head droop, Telegram alerts, updated dB gains, MPU-6050 IMU auto-gating) with MS5's embedded updates (10-frame pre-optimized LSTM default, OS-guarded IMU stub for cross-platform builds, and live MP4 HUD recording).
- **Explainer Location**: Repository root `docs/explainer_plan.md` and `docs/explainer/`.
- **Python Prototype Anchor**: At the start of Chapter 1, orient engineers by briefly mapping their familiar Python modules in this repo to where each algorithm now runs in C++:
  - Gaze distraction (`distraction/` & `morphed_system2/models_distraction.py`) -> **Gaze Worker**
  - Eye blink & PERCLOS (`ear_mar/` & `morphed_system2/estimation_ear_mar.py`) -> **Fatigue Worker** (Eye monitor)
  - Head pitch droop (`head_orientation/` & `morphed_system2/estimation_head_pose.py`) -> **Fatigue Worker** (Posture monitor)
  - Calibration & cue fusion (`morphed_system2/calibration.py`, `morphed_system2/fusion.py`) -> **Sequential Calibration Runner** & **Alert Coordinator**
  - Post-prototype C++ additions: full-frame **Yawn CNN**, **MPU-6050 IMU** motion worker, **Telegram** alerts, and **MP4 Debug Recorder**.

### Primary Audience
Human ML/CV engineers who wrote and validated the detection algorithms in Python, and now need an accurate, low-friction mental model of the LLM-assisted C++ real-time infrastructure before drafting the academic paper.

### Hard Constraints
1. **Medium**: Self-contained interactive HTML + inline SVG progressive walkthroughs. Must open directly via `file://` in any modern browser with **no local server, no bundler, and no npm/build step**.
2. **Low Token Cost**: One tiny reusable HTML/CSS/JS stepper shell paired with compact declarative per-chapter JS data files.
3. **Minimal Cognitive Noise**: Reveal one idea at a time on a fixed diagram canvas while dimming inactive components.
4. **Plain-English First**: Never open an explanation with C++ variable names, class/method identifiers, file paths, or memory-ordering jargon. Explain behavior using human role names and plain language; tuck exact C++ symbols behind opt-in disclosures.
5. **Code-Faithful Honesty**: Surface real-time trade-offs, doc-vs-code mismatches, and subtle concurrency caveats via opt-in collapsed badges — never hide them, and never clutter the first-pass narrative.
6. **Language & Paper Reuse**: English explainers, paired with Section 5's Paper Mapping & Doc-vs-Code Sync Checklist and clean SVG export.

---

## 2. Anti-Cognitive-Noise Pedagogical Laws

1. **Fixed Canvas, Spotlight Reveal**: One stable `960x540` SVG diagram per chapter; inactive nodes dim to `0.22`; each step spotlights 1–2 concepts.
2. **Strict 3-Tier Progressive Disclosure per Step**:
   - **Tier 1 — Surface Concept (Always Visible)**: `<= 3` short plain-English sentences using consistent human role names (*Camera Reader*, *Gaze Worker*, *Fatigue Worker*, *IMU Motion Worker*, *Main Supervisor*, *Keyboard Listener*, *Alert Coordinator*, *Audio Mixer*, *Telegram Messenger*, *Debug HUD & MP4*). **Zero C++ identifiers, zero file paths, zero memory-ordering jargon.**
   - **Tier 2 — Implementation Note (1-Click Collapsed Badge)**: Plain-English engineering trade-offs, doc-vs-code drift, or concurrency caveats.
   - **Tier 3 — C++ Source Reference (2nd Opt-In Collapsed Toggle)**: Exact `morphed_system6/distraction/modular/` file paths, line ranges, and symbols.
3. **Familiar Algorithm Anchors First**: Chapter 1 anchors the familiar Python algorithms before introducing threads, mailboxes, or locks.
4. **One Consistent Visual Grammar Across All Chapters**:
   - **Rounded Swimlane**: Persistent OS thread
   - **Double-Walled Tray**: Single-slot drop-oldest mailbox
   - **Padlock / Amber Badge**: Mutex-protected shared region
   - **Rounded Green Pill**: Atomic cross-thread flag or state variable
   - **Solid Arrow**: Synchronous call or direct frame handoff
   - **Dashed Purple Arrow / Badge**: Detached background thread (`ASYNC`) or compile-time optional view (`DEBUG`)
5. **Concrete Micro-Traces Over Abstract Concurrency Rules**: Chapters 2–5 each include a 4–6 step concrete Scenario Trace reusing the same stepper engine.

---

## 3. Token-Frugal Reusable Architecture

```text
docs/explainer/
├── index.html                        # Chapter tabs, mode tabs, SVG stage, 3-tier card, Export SVG
├── style.css                         # Calm light palette, dim/spotlight transitions, flow animation
├── engine.js                         # Pure vanilla SVG renderer + stepper + SVG export
└── chapters/
    ├── ch1_thread_map.js             # Chapter 1: Thread Map
    ├── ch2_mailboxes.js              # Chapter 2: Drop-Oldest Mailboxes
    ├── ch3_lifecycle_calibration.js  # Chapter 3: 5-State Lifecycle, IMU & Calibration
    ├── ch4_alerts_locking.js         # Chapter 4: 2-Stage Distraction, Cue Fusion & Priority
    └── ch5_low_level_runtime.js      # Chapter 5: Streaming Audio, Throttling & Edge Presets
```

---

## 4. Detailed 5-Chapter Progressive Curriculum (`morphed_system6/distraction/modular/`)

### Chapter 1: Python-to-C++ Anchor & Thread Map

- **Learning Goal**: Map the single-loop Python prototypes onto MS6's persistent worker threads, hardware IMU thread, and detached alert/debug outputs.
- **Walkthrough (6 Steps — Tier-1 Plain English)**:
  1. **Where Your Python Algorithms Live**: Familiar gaze, eye-closure, and head-pitch detectors move from one sequential Python loop into parallel C++ workers, joined by a full-frame yawning neural net.
  2. **Two Parallel Analysis Workers**: *Gaze Worker* runs a default 10-frame pre-optimized sequence model (`preopt`, or 20-frame `unified` via flag), while *Fatigue Worker* runs the full-frame Yawn CNN independently alongside 68-landmark eye closure and `-25.5°` head-pitch droop.
  3. **Decoupled Camera Reader**: Dedicated *Camera Reader* continuously pulls webcam/video frames so slow neural or landmark passes never stall capture.
  4. **Main Supervisor, Keyboard Listener & IMU Motion Worker**: *Main Supervisor* owns lifecycle and shutdown; *Keyboard Listener* polls `m`/`c`/`r`/`q`; *IMU Motion Worker* reads the MPU-6050 at 20 Hz on Linux (or exits gracefully on non-Linux / `--no-imu`).
  5. **Shared Alert Coordinator & Audio Mixer**: Both analysis workers report into one mutex-guarded *Alert Coordinator*, which dispatches sounds onto an always-on 48 kHz *Audio Mixer* thread.
  6. **Detached Telegram Alerts & Optional Debug HUD**: Alarms fire short-lived detached *Telegram Messenger* HTTP threads, while debug builds add a live OpenCV HUD and MP4 recorder (`'r'` / `--record`) that compile to zero-cost stubs in Release.
- **Tier-3 Refs**: `src/PipelineCoordinator.cpp` (L68–103, L232–316, L399–689), `src/CameraCapture.cpp` (L45, L71–114), `src/ConsoleController.cpp` (L47, L63–104), `include/MPU6050.hpp` (L51–205), `src/YawnDetector.cpp` (L112–171), `src/AlertManager.cpp` (L83–204, L272–339), `src/SoundPlayer.cpp` (L342–388), `src/TelegramNotifier.cpp` (L36–71), `src/DebugVisualizer.cpp` (L176–207, L563–598).

### Chapter 2: Frame Handoff & Real-Time Freshness (Drop-Oldest Mailboxes)

- **Learning Goal**: Explain why FIFO queues lag and how single-slot drop-oldest mailboxes + per-consumer deep clones keep workers on the newest frame.
- **Walkthrough (5 Steps) + Scenario Trace (`"Trace: Fast Camera, Busy Worker"`, 5 Steps)**:
  - Unchanged core design from MS4/MS5 (`mailbox_gaze_`, `mailbox_somno_`, optional `debug_visualizer_.mailbox()`, 100 ms `get_latest` wait, 30 FPS fallback on video files).
- **Tier-3 Refs**: `include/ThreadSafeMailbox.hpp` (L30–69), `src/CameraCapture.cpp` (L71–114), `src/PipelineCoordinator.cpp` (L115–124, L416, L507).

### Chapter 3: The 5-State Lifecycle, Hardware IMU & Calibration Takeover

- **Learning Goal**: Show the 5-state safety FSM driven by both manual keypresses (`'m'`/`'c'`) and 20 Hz MPU-6050 IMU hysteresis, the in-place calibration takeover on *Fatigue Worker*, and mutex-free gaze reset vs. the `clear_perclos` race caveat.
- **Walkthrough (6 Steps) + Scenario Trace (`"Trace: Startup -> Calibrate -> Drive -> Park"`, 6 Steps)**:
  - Hardware IMU hysteresis (`> 0.80 m/s^2` for 4 samples -> Moving; `< 0.35 m/s^2` for 15 samples -> Parked; ignored during `Calibrating`; Linux I2C `/dev/i2c-1` `0x68` with non-Linux stub returning `false`).
  - In-place 2-phase acoustic calibration (`+6 dB` spoken instructions, 1 s preparation tone in both phases, `+3 dB` 8x500 ms cadence beeps until `>=15` samples/phase).
  - Idle work asymmetry (`Gaze Worker` skips inference; `Fatigue Worker` still runs Yawn CNN + landmarks for telemetry while suppressing alerts/PERCLOS).
  - Tier-2 Concurrency Caveat: `ear_detector_.clear_perclos()` is called unsynchronized from both *Main Supervisor* (`L340`) and *IMU Motion Worker* (`L668`), whereas `GazeDetector` uses `gaze_reset_requested_`.
- **Tier-3 Refs**: `include/Types.hpp` (L66–72), `include/MPU6050.hpp` (L51–205), `src/PipelineCoordinator.cpp` (L319–397, L411–431, L480–523, L589–689), `src/Calibration.cpp` (L25–220).

### Chapter 4: Multimodal Alert Fusion, Locks & Priority Preemption

- **Learning Goal**: Show clearly that *Gaze Worker* emits single per-frame predictions while the **2-stage distraction filter** (`10f` sliding mode + `2.0 s` continuous timer) and drowsiness cue fusion live **inside *Alert Coordinator*'s mutex**, followed by outside-lock audio + Telegram dispatch.
- **Walkthrough (6 Steps) + Scenario Trace (`"Trace: Drowsiness Preempts Distraction"`, 6 Steps)**:
  - *Gaze Worker* produces 1 vote/frame (`conf = 0.54`, `BinarizeThreshold` default; `FilterUncertain` and `Argmax` supported).
  - Inside `Alert Coordinator` (`MUTEX`): Stage 1 = `10`-frame sliding mode window (matching 10f `preopt` LSTM, or `20` for `unified`) + Stage 2 = `2000 ms` (`2.0 s`) continuous off-road `distracted_since_` timer.
  - Drowsiness Fusion: 1 Strong cue (`>500 ms` micro-sleep or `>=15%` PERCLOS) OR 2 Light cues in 2 minutes (`<= -25.5°` droop for `1.5 s` or `>= 1.0 s` Yawn CNN `>= 0.80`); `>= 0.2 s` yawning suppresses `BlinkDuration`, `Perclos`, and `HeadDroop` cues.
  - Act outside lock: `+6.0 dB` blocking distraction audio on *Gaze Worker* (then clears `gaze_window_` and resets `distracted_since_`); non-blocking 63.5 BPM escalating drowsiness pulses (`-21.0, -16.5, -12.0, -7.5, -3.0 dB`) immediately cutting off distraction; detached `libcurl` Telegram threads (`3 s` connect / `5 s` total timeout).
- **Tier-3 Refs**: `include/AlertManager.hpp` (L86–99), `src/AlertManager.cpp` (L83–204, L206–339, L360–388, L408–421), `src/AlertLogic.cpp` (L5–59), `src/GazeDetector.cpp` (L159–182), `src/PipelineCoordinator.cpp` (L526–559), `src/TelegramNotifier.cpp` (L36–71), `include/Types.hpp` (L149–255).

### Chapter 5: Low-Level Real-Time Mechanics (Audio Mixer, Throttling, Edge Tuning & Build Modes)

- **Learning Goal**: Explain the 48 kHz streaming audio mixer, generation-ticket cancellation, 500 ms missing-face Dlib throttle (while fast full-frame Yawn CNN keeps running), zero-allocation buffers + 10-frame `preopt` LSTM + OS-guarded IMU + SIMD, and `DEBUG_VIEWS` HUD/MP4 recording vs. zero-cost Release stubs.
- **Walkthrough (5 Steps) + Scenario Trace (`"Trace: 10 ms Ring-Buffer Preemption"`, 4 Steps)**.
- **Tier-3 Refs**: `src/SoundPlayer.cpp` (L36–40, L226–494, L503–519), `src/AudioManager.cpp` (L20–30), `src/PipelineCoordinator.cpp` (L218–229, L255–260, L298–309, L526–555), `include/GazeDetector.hpp` (L70–76), `src/GazeDetector.cpp` (L60–65, L99–119), `include/YawnDetector.hpp` (L58–61), `include/Preprocessor.hpp` (L46–51), `include/MPU6050.hpp` (L51–205), `include/DebugVisualizer.hpp` (L17–179), `src/DebugVisualizer.cpp` (L176–207, L563–598), `CMakeLists.txt` (L8–40, L66–70), `CMakePresets.json` (L31–52).

---

## 5. Lightweight Paper Mapping & Doc-vs-Code Sync Checklist (MS6)

| Explainer Chapter | Candidate Paper Section | Candidate Paper Figure |
| :--- | :--- | :--- |
| **Ch. 1: Thread Map** | System Architecture / Real-Time Pipeline | **Fig. 1**: Multi-Threaded System & Sensor Block Diagram |
| **Ch. 2: Drop-Oldest Mailboxes** | Real-Time Frame Synchronization | **Fig. 2**: Zero-Lag Single-Slot Mailbox Pattern |
| **Ch. 3: Lifecycle, IMU & Calibration** | Operational Protocol & Baseline Calibration | **Fig. 3**: 5-State Safety FSM, IMU Hysteresis & 2-Phase Calibration |
| **Ch. 4: Alert Fusion & Priority** | Multimodal Fusion & Warning Arbitration | **Fig. 4**: 2-Stage Gaze Filter, Cue Fusion Tree & Priority Preemption |
| **Ch. 5: Low-Level Mechanics** | Implementation Details & Embedded Optimization | *Prose / Table* (48 kHz mixer, 500 ms Dlib throttle, 10f LSTM, SIMD) |

### MS6 Doc-vs-Code Nuances to Keep Accurate in the Paper
1. **PERCLOS Cue Rules**: Only **High PERCLOS** (`>= 15.0%` over 30 s, Strong cue) is emitted; **Medium PERCLOS** (`7.5%–15.0%`, Light cue) is disabled in `EarDetector` even though `README.md` still mentions it.
2. **Calibration Phase 2 Tone**: `Calibration.cpp` plays the 1-second preparation tone in **both** Phase 1 and Phase 2 (`README.md` implies Phase 2 omits it).
3. **Audio Engine**: Windows & Linux/Raspberry Pi run the custom **48 kHz stereo 6x10 ms ring-buffer** `AudioStreamEngine` (`waveOut` / `aplay` raw pipe), not `PlaySoundA` or per-clip `aplay` (`README.md` Sec. 5.2 is legacy). Two alert clips (`distraction_warning.wav`, `single-beep-1850-Hz-975-Hz.wav`) decode lazily on first play rather than in `warmup()`.
4. **Unused `tick_drowsiness_if_active()`**: Exists in `AlertManager`, but `PipelineCoordinator` calls `update_somnolence()` every frame using `on_face_lost()` during missing-face intervals.
5. **Screen Recording Requires `DEBUG_VIEWS`**: `'r'` and `--record` encode MP4 at 15 FPS only when `DEBUG_VIEWS` is enabled; Release builds compile `DebugVisualizer` to empty stubs with a 50 ms main loop sleep.
6. **Concurrency Caveat**: `ear_detector_.clear_perclos()` is called from both the Main thread (`PipelineCoordinator.cpp` L340) and IMU thread (`L668`) without a mutex against `somnolence_worker_`; guard via an atomic flag (like `gaze_reset_requested_`) before claiming strict lock-freedom on transitions.

---

## 6. Build Roadmap & Verification Status

- **Phases 1–4**: Complete for **Morphed System 6** (`docs/explainer/index.html`, `style.css`, `engine.js`, `chapters/ch1`–`ch5`).
- **Automated Checks**: Verified `node --check`, `index.html` asset links, `960x540` bounds, zero sibling overlaps, Law 2 plain-English `concept` strings (`<= 3` sentences, zero C++ identifiers/paths), and zero stale `morphed_system3/4/5` references in `ch1`–`ch5`.
