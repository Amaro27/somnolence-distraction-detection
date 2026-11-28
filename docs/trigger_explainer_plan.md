# Trigger Estimation & FSM Flowchart Explainer Plan (`morphed_system6/distraction/modular/`)

> **Deliverable Scope**: This document is the **implementation blueprint** for a companion interactive flowchart explainer stored under `docs/trigger_explainer/`. **Phase 1** specifies the standalone **Trigger & FSM Cut-Off Flowcharts** (Sections 1–6, 16 sub-views) with paired *Fused Algorithm* and *System & FSM Cut-Off* views. **Phase 2** (Section 7) sketches two candidate **General Subroutine Overview** designs to finalize after Phase 1 is built and reviewed. All logic, constants, and `file:line` citations are verified against **Morphed System 6 (`morphed_system6/distraction/modular/`)**.

---

## 1. Purpose, Companion Positioning & Target Architecture

### Companion Positioning vs. `docs/explainer/`
- **Target Runtime**: `morphed_system6/distraction/modular/` (C++17 real-time driver distraction and somnolence detection system).
- **Role Split**:
  - `docs/explainer/` (**Bird's-Eye Runtime Explainer**, existing): Explains OS threads, 1-slot drop-oldest mailboxes, mutex boundaries, 48 kHz ring-buffer audio mixing, and release/debug build mechanics.
  - `docs/trigger_explainer/` (**Trigger & FSM Flowchart Explainer**, planned): Explains **step-by-step algorithmic estimation** of each safety trigger (*Head Drooping*, *Blink Duration*, *PERCLOS*, *Yawning*, *Gaze Distraction*) and **how external FSMs, calibration baselines, cross-feature suppression, and alarm arbitration gate, reset, and fuse them**—without collapsing the entire codebase into one unreadable flowchart.
- **Cross-Linking**: Both apps open directly via `file://` in any modern browser (**zero server, zero bundler, zero npm/build step**) and include a subtle cross-link in their top navigation bars (`../explainer/index.html` <-> `../trigger_explainer/index.html`).

### Primary Audience
Human ML/CV engineers who designed the detection algorithms and now need exact, verifiable control-flow diagrams—both **pure algorithmic flowcharts** (for Methodology sections) and **FSM/cut-off runtime flowcharts** (for Real-Time System Integration sections and engineering verification)—with clean vector **SVG export** for paper figures.

### Planned File Layout (`docs/trigger_explainer/`)

```text
docs/trigger_explainer/
├── index.html                        # Section tabs, sub-view tabs, branch trace selector, "<- Back" portal breadcrumb, SVG stage, 3-tier side panel, Export SVG, link to ../explainer/index.html
├── style.css                         # Clean light paper-ready palette, color-coded flowchart shapes, active Yes/No branch spotlighting
├── flowchart_engine.js               # Pure vanilla SVG flowchart renderer (process rects, diamonds, green RETURN pills, coral DROWSY/DISTRACTED pills, clickable GOTO/FROM portals + history stack, path trace stepper, clean SVG export)
└── sections/
    ├── sec1_vehicle_calib_fsm.js     # Section 1: Vehicle & Calibration FSM (3 sub-views: 1A, 1B, 1C)
    ├── sec2_head_droop.js            # Section 2: Head Drooping (2 sub-views: 2A Fused, 2B Cut-Off)
    ├── sec3_blink_perclos.js         # Section 3: Blinking & PERCLOS (4 sub-views: 3A Blink Fused, 3B PERCLOS Fused, 3C Blink Cut-Off, 3D PERCLOS Cut-Off)
    ├── sec4_yawning.js               # Section 4: Yawning (2 sub-views: 4A Fused, 4B Cut-Off)
    ├── sec5_distraction.js           # Section 5: Distraction (2 sub-views: 5A Fused, 5B Cut-Off)
    ├── sec6_alarms_dispatcher.js     # Section 6: Alert Fusion & Alarm Dispatcher (3 sub-views: 6A Gaze Filter, 6B Somnolence Fusion, 6C Outside-Lock Dispatcher)
    └── sec7_subroutine_overview.js   # Phase 2 (Deferred until Phase 1 review): General Subroutine Overview
```

---

## 2. Visual Grammar, UX Laws & 3-Tier Disclosure Policy

1. **Two Paired Perspectives per Trigger**:
   - **Sub-view A — Fused Algorithm View**: Self-contained end-to-end algorithmic flowchart from input frame or landmarks -> math & feature extraction -> threshold/timer/asymmetric decay -> inline cue rule (including the `>= 2 Light cues in 120 s` check for Light cues and the `10f mode + 2.0 s` filter for Distraction) -> ending directly in terminal pills (`RETURN` vs `DROWSY` / `DISTRACTED`). **Zero thread, mutex, `SystemState`, or portal noise.**
   - **Sub-view B — System & FSM Cut-Off View**: Shows the exact C++ runtime reality—`SystemState` movement/calibration gates, calibration baseline injection, atomic/direct state resets, the `0.2 s` yawn cross-suppression wire, and strict C++ class/mutex cut-off boundaries (`GOTO ...` / `FROM ...`) connecting to the dedicated FSM and Alarm Dispatcher sections.
2. **Full Flowchart Upfront + Branch Path Traces**:
   - Unlike `docs/explainer/` (which hides unvisited elements), every flowchart renders **100% visible upfront** so the reader always sees the complete decision tree.
   - Each sub-view includes **Branch Path Traces** (e.g., *"Micro-sleep >= 500 ms vs. Normal Blink"*, *"Yawn >= 0.2 s Suppresses Droop"*, *"Face Lost Mid-Droop"*, *"Drowsiness Preempts Distraction"*). Stepping through a trace spotlights the active nodes and `YES`/`NO` edges while dimming untaken branches to `0.22` opacity.
   - Clicking any individual node on the canvas also inspects that node's 3-tier details in the side panel without requiring a trace.
3. **Clickable `GOTO` / `FROM` Off-Page Portals with Return Breadcrumb**:
   - Every `GOTO` and `FROM` portal node stores `{ sectionId, subviewId, highlightNodeId }`.
   - Clicking a portal jumps straight to the target flowchart, highlights the corresponding entry/target node, and pushes the caller onto a navigation history stack—surfacing a `"<- Back to [Caller Section / Sub-view]"` pill in the stage bar.
4. **Clean Human-Readable Nodes & Color-Coded Flowchart Grammar (Light Paper-Ready Theme, English)**:
   - **Process Box** (`rect`, white fill `#ffffff`, slate/blue stroke `#2563eb` or `#475569`): Clean human-readable step title + optional short summary with zero C++ identifiers, underscores, or raw expressions on-canvas (exact math/code lives inside the **`Code:`** side-panel toggle).
   - **Decision Diamond** (`polygon`, indigo stroke `#4f46e5`, light tint `#eef2ff`): Clear domain question with exact physical thresholds (`Pitch <= -25.5 deg?`, `Looking-Road Prob >= 54%?`, `Moving & Calibrated?`) and crisp `YES` / `NO` edge labels.
   - **Green Terminal Pill** (`RETURN` / `SKIP FRAME`, `#16a34a` border, `#dcfce7` fill, `#14532d` text): Clean cycle exit.
   - **Pink/Coral Terminal Pill** (`DROWSY` / `DISTRACTED`, `#e11d48` border, `#ffe4e6` fill, `#881337` text): Fused-view alarm verdict.
   - **Blue Outgoing Cut-Off Portal** (`GOTO ...` / `SIGNAL ...`, `#0284c7` border, `#e0f2fe` fill, `#0c4a6e` text, right-chevron/tag accent): Exit across a C++ class, thread, or mutex boundary.
   - **Amber Incoming / FSM / Lock Badge** (`FROM ...` / `READ FSM` / `ATOMIC` / `MUTEX`, `#d97706` border, `#fef3c7` fill, `#78350f` text): External state read, baseline injection, atomic reset consume, or lock boundary.
5. **4-Layer Side Panel Disclosures**:
   - **Layer 1 — General Description (Always Visible)**: Human-readable, symbol-free explanation of the overview, selected node, or active trace step.
   - **Layer 2 — `Code:` (Collapsed Bold Toggle)**: Exact C++ expressions, equations, and control-flow snippets implementing that block or step.
   - **Layer 3 — Implementation & Edge-Case Note (Collapsed)**: Clamped time-step rules, hysteresis/decay asymmetries, doc-vs-code nuances, and concurrency caveats.
   - **Layer 4 — C++ Source Reference (Collapsed)**: Exact `morphed_system6/distraction/modular/` file paths, class/function symbols, and verified line ranges.

---

## 3. Phase-1 Flowchart Specifications (Sections 1–6, 16 Sub-views)

### Section 1: Vehicle & Calibration FSM (`sec1_vehicle_calib_fsm.js` — 3 Sub-views)

#### Sub-view 1A — 5-State Movement & Calibration Lifecycle FSM
- **Goal**: Map every transition across the 5 `SystemState` values, both manual keyboard/GUI triggers and Linux 20 Hz MPU-6050 IMU hysteresis, plus all reset/takeover side effects emitted when changing states.
- **Flowchart / State Diagram Structure**:
  - **Entry / Startup**: `Startup Options (--start-moving?)` -> `NO`: `StationaryUncalibrated` (default) / `YES`: `MovingUncalibrated`.
  - **5 State Nodes**: `StationaryUncalibrated`, `MovingUncalibrated`, `Calibrating`, `StationaryCalibrated`, `MovingCalibrated`.
  - **Manual Input Gate (`ConsoleController` / `DebugVisualizer` on Main Thread)**:
    - Key `'c'` (`ConsoleCommand::Calibrate`): `<SystemState is StationaryUncalibrated or StationaryCalibrated?>` -> `YES`: store `Calibrating` -> **`GOTO 1B: RUN_CALIBRATION`**; `NO` (`Moving*` or already `Calibrating`): ignore & log warning.
    - Key `'m'` (`ConsoleCommand::ToggleMovement`): `<SystemState == Calibrating?>` -> `YES`: ignore (locked out during calibration); `NO`: flips `Stationary* <-> Moving*`.
  - **Hardware IMU Path (`imu_worker_loop`, 20 Hz Linux I2C `/dev/i2c-1` `0x68`)**:
    - 5-sample circular magnitude buffer -> `smoothed_mag = mean(mag_buffer[0..4])` -> `dynamic_motion = |smoothed_mag - 9.80665|`.
    - `<dynamic_motion > 0.80 m/s^2?>` -> `count_motion++`, `count_still = 0`; `<count_motion >= 4 AND !is_moving?>` -> `is_moving = true`, `vehicle_moving_ = true`; transitions `StationaryCalibrated -> MovingCalibrated` or `StationaryUncalibrated -> MovingUncalibrated` (implicitly ignoring `Calibrating`).
    - `<dynamic_motion < 0.35 m/s^2?>` -> `count_still++`, `count_motion = 0`; `<count_still >= 15 AND is_moving?>` -> `is_moving = false`, `vehicle_moving_ = false`; transitions `Moving* -> Stationary*`.
  - **Parking Cleanup Portals (`Moving* -> Stationary*` via `'m'` or IMU)**:
    - Always runs: **`GOTO 6C: ALERT_RESET`** (`alert_manager_.reset()`), **`SET ATOMIC -> gaze_reset_requested_ = true`** (consumed by `5B`), `debug_visualizer_.reset()`.
    - Only when leaving `MovingCalibrated`: **`GOTO 3D: CLEAR_PERCLOS`** (`ear_detector_.clear_perclos()`).
- **Branch Path Traces**:
  1. *Safe Boot -> Manual Calibrate ('c') -> Drive ('m')*
  2. *IMU Hysteresis Auto-Move (4 samples > 0.80) & Auto-Park (15 samples < 0.35)*
  3. *Calibration Lockout: 'm' & IMU Blocked Until Calibration Finishes*
- **Tier-2 Notes**:
  - **Non-Linux / `--no-imu` Guard**: `MPU6050::init()` returns `false` on Windows/macOS, exiting `imu_worker_loop` cleanly at startup while keeping manual `'m'` active.
  - **Cross-Thread `clear_perclos()` Caveat**: Both Main (`PipelineCoordinator.cpp` L340) and IMU (`L668`) call `ear_detector_.clear_perclos()` directly on `MovingCalibrated -> StationaryCalibrated`, racing with `somnolence_worker_loop` (which also performs a fallback `ear_detector_.clear_perclos()` on `L520`). `GazeDetector` avoids this race using `gaze_reset_requested_`.
- **Tier-3 C++ Refs**: `include/Types.hpp` (L66–72), `include/MPU6050.hpp` (L38–205), `src/PipelineCoordinator.cpp` (L130–131, L240–242, L319–384, L515–522, L589–689).

#### Sub-view 1B — 2-Phase Acoustic Calibration Routine
- **Goal**: Detail the blocking top-to-bottom calibration takeover that replaces normal monitoring inside `somnolence_worker_loop` whenever `SystemState == Calibrating`.
- **Flowchart Structure**:
  - **Entry**: `FROM 1A: SystemState == Calibrating` (checked at top of `somnolence_worker_loop` before `mailbox_somno_` normal read).
  - **Phase 1 (Open Eyes & Neutral Posture)**:
    - Play `calib_phase_1.wav` at `+6.0 dB` (`play_blocking`) -> `1.0 s` interruptible silence -> play `sine-415-Hz-1-second-padded.wav` (`0 dB`, `play_blocking`) -> `1.0 s` silence.
    - Measurement Window Loop (`measure_window`): plays `8` cadence beeps (`single-beep-350-hz.wav` at `+3.0 dB`, `500 ms` apart = `4.0 s`); between beeps, if `remaining > 30 ms`, fetches latest frame (`wait <= 50 ms`) -> **`GOTO 1C: DLIB_FACE_GATE (Direct Detect)`** -> if landmarks found, computes `EAR` (`FaceGeometry::compute_ear`) and `rho` (`FaceGeometry::compute_rho`), pushing `> 0` values to `open_samples` and `rho_samples`.
    - `<open_samples.size() < 15?>` -> `YES`: repeat 4.0 s beep window / `NO`: proceed.
  - **Transition**: `1.0 s` silence (`"CLOSE EYES NOW AND HOLD"`).
  - **Phase 2 (Closed Eyes Floor)**:
    - Play `calib_phase_2.wav` at `+6.0 dB` (`play_blocking`) -> `1.0 s` silence -> play `sine-415-Hz-1-second-padded.wav` (`play_blocking`) -> `1.0 s` silence.
    - Measurement Window Loop (`8 x 500 ms` beeps at `+3.0 dB`): collects `EAR` into `closed_samples` until `closed_samples.size() >= 15`.
  - **Completion & Baseline Injection**:
    - Play `tada-LOUD.wav` (non-blocking) -> compute `ear_open = mean(open_samples)`, `ear_closed = mean(closed_samples)`, `neutral_rho = median(rho_samples)`.
    - Inject baselines & reset detectors: **`GOTO 3C: APPLY_EAR_BASELINE`**, **`GOTO 2B: APPLY_HEAD_BASELINE`**, **`GOTO 4B: YAWN_RESET`**, **`GOTO 1C: FACE_RESET`**.
    - Store `was_calibrated = true`, `SystemState = StationaryCalibrated` -> `[RETURN TO SOMNOLENCE LOOP]`.
    - **Abort / Shutdown Path**: if `shutdown_requested_` fires during any voice/silence/window, restore `was_calibrated ? StationaryCalibrated : StationaryUncalibrated` and return.
- **Branch Path Traces**:
  1. *Happy Path: >= 15 Samples in Both Phases -> Baselines Applied*
  2. *Retry Loop: Face Occluded in Phase 1 (< 15 Samples -> Repeat 4 s Beeps)*
  3. *Shutdown Abort Mid-Calibration -> Restore Prior Stationary State*
- **Tier-2 Notes**:
  - **Doc-vs-Code Sync**: `Calibration.cpp` L168 plays the 1 s 415 Hz preparation tone in **both** Phase 1 and Phase 2 (whereas legacy `README.md` implies Phase 2 omits it).
  - **30 ms Deadline Guard**: `measure_window` refuses to launch ~21 ms Dlib face detection when `<= 30 ms` remain before the next 500 ms beep deadline so audio cadence stays crisp.
- **Tier-3 C++ Refs**: `src/PipelineCoordinator.cpp` (L470–505), `src/Calibration.cpp` (L14–220), `include/Types.hpp` (L204–217).

#### Sub-view 1C — Shared Dlib 68-Landmark & 500 ms Missing-Face Gate
- **Goal**: Show how `somnolence_worker_loop` produces either `FacialLandmarks68` or `Face Lost` while protecting CPU throughput for `GazeWorker` via 500 ms missing-face throttling.
- **Flowchart Structure**:
  - **Entry**: Raw `cv::Mat` BGR frame inside `somnolence_worker_loop` (after `YawnDetector::process` has already run on the full frame) OR direct probe from `Calibration` (`1B`).
  - **Throttle Check (Normal Loop)**: `<face_missing_ == true AND (now - last_face_probe_time) < 500 ms?>` (`PipelineCoordinator.cpp` L535–538):
    - `YES`: skip Dlib completely -> output **`Face Lost`** -> **`GOTO 2B (Head on_face_lost)`** & **`GOTO 3C/3D (EAR on_face_lost)`**.
    - `NO`: update `last_face_probe_time = now` -> proceed to `FaceLandmarkDetector::detect`.
  - **Dlib Pipeline (`FaceLandmarkDetector.cpp` L69–115)**:
    - Convert 3-ch BGR to Grayscale (`gray_buf_`) -> run Dlib HOG `frontal_face_detector(cimg, 0)`.
    - `<faces.empty()?>`:
      - `YES`: if `!face_missing_`, set `face_missing_ = true`, `face_lost_start_ = now`, `warned_3s_ = false`; else if `now - face_lost_start_ >= 3.0 s && !warned_3s_`, log 3 s warning (`warned_3s_ = true`) -> output **`Face Lost`** -> **`GOTO 2B`** & **`GOTO 3C/3D`**.
      - `NO`: set `face_missing_ = false`, `warned_3s_ = false` -> run `shape_predictor` on `faces[0]` -> extract 68 `(x, y)` coordinates -> output **`FacialLandmarks68`** -> **`GOTO 2B (Head process)`** & **`GOTO 3C (EAR process)`**.
- **Branch Path Traces**:
  1. *Normal Frame -> 68 Landmarks Extracted*
  2. *Face Lost -> 3 s Warning -> 500 ms CPU-Saving Throttle*
- **Tier-2 Notes**:
  - **No Ghost Landmarks**: `shape_predictor` is never run unless `frontal_face_detector` confirms at least one face (`faces[0]`), eliminating hallucinated facial contours on empty seats.
- **Tier-3 C++ Refs**: `src/PipelineCoordinator.cpp` (L494, L533–555), `src/FaceLandmarkDetector.cpp` (L56–121), `include/Types.hpp` (L184–188).

---

### Section 2: Head Drooping (`sec2_head_droop.js` — 2 Sub-views)

#### Sub-view 2A — Head Drooping: Fused Algorithm View
- **Goal**: Present the self-contained anthropometric head-pitch droop estimation and inline Light-cue drowsiness verdict with zero thread/FSM/portal clutter.
- **Flowchart Structure**:
  - **Input**: `68 Facial Landmarks` + neutral baseline `rho_0` (from calibration).
  - **Roll-Invariant Geometry (`FaceGeometry::compute_rho`)**:
    - Compute eye midpoint `e_mid = (p36 + p45) / 2` and eye vector `v_eyes = p45 - p36`.
    - Compute downward face axis `u_y = (-u_x.y, u_x.x)` (orthogonal to `u_x = v_eyes / ||v_eyes||`, fallback `(0, 1)` if `||v_eyes|| < 1e-6`).
    - Project nose tip (`p33`) and chin (`p8`): `d_top = (p33 - e_mid) · u_y`, `d_bot = (p8 - p33) · u_y`, `h_face = max(d_top + d_bot, 1e-6)`, `rho = d_top / h_face`.
  - **Pitch Estimation**: `delta_rho = rho - rho_0`; `pitch_deg = atan2(-delta_rho, 0.18) * (180 / pi)`.
  - **Clamped Time Delta**: `dt = compute_clamped_dt(now)` (`raw_dt > 0.15 s` -> `0.033 s`).
  - **Droop Threshold Decision**: `<pitch_deg <= -25.5 deg?>`:
    - `YES`: `in_droop = true`; `droop_accum = min(2.0 s, droop_accum + dt)` -> `<droop_accum >= 1.5 s AND !droop_cue_emitted?>`:
      - `NO`: **`[RETURN]`** (green).
      - `YES`: `droop_cue_emitted = true`; record `HeadDroop` Light cue in `120 s` sliding window -> `<>= 2 Light cues in 120 s?>`:
        - `YES`: **`[DROWSY]`** (pink/coral).
        - `NO`: **`[RETURN]`** (green).
    - `NO`: asymmetric fast recovery `droop_accum = max(0.0, droop_accum - 2.0 * dt)` -> `<droop_accum <= 0.0?>`:
      - `YES`: `in_droop = false`, `droop_cue_emitted = false` -> **`[RETURN]`** (green).
      - `NO`: **`[RETURN]`** (green).
- **Branch Path Traces**:
  1. *Sustained Nod (`<= -25.5 deg` for `>= 1.5 s`) -> Light Cue (+ 2-in-120s -> DROWSY)*
  2. *Brief Head Bob (`-2.0 * dt` Fast Recovery)*
- **Tier-2 Notes**:
  - **Asymmetric Decay**: Recovery subtracts `2.0 * dt` (twice real-time speed) so raising the head briefly resets short nods quickly while still holding through single-frame landmark noise; accumulator caps at `1.5 + 0.5 = 2.0 s`.
  - **Doc-vs-Code Sync**: `Types.hpp` L195 sets `pitch_droop_thresh = -25.5` (calibrated for BU HT dataset 5.48 deg MAE), whereas legacy comments in `HeadOrientationDetector.cpp` L59 and `Types.hpp` L106 still mention `-20 deg`.
- **Tier-3 C++ Refs**: `src/FaceGeometry.cpp` (L22–49), `src/HeadOrientationDetector.cpp` (L42–94), `include/MathUtils.hpp` (L61–82), `include/Types.hpp` (L193–197).

#### Sub-view 2B — Head Drooping: System & FSM Cut-Off View
- **Goal**: Show how `HeadOrientationDetector` is gated by Dlib face presence/loss, calibration baseline readiness, `0.2 s` yawn suppression, and `MovingCalibrated` before forwarding `HeadOrientationAssessment` to `AlertManager`.
- **Flowchart Structure**:
  - **Incoming Portals**:
    - **`FROM 1C: DLIB_FACE_GATE`** (`68 Landmarks` vs `Face Lost`).
    - **`FROM 1B: APPLY_HEAD_BASELINE`** (sets `baseline_rho_ = neutral_rho`, fallback `0.50`, `calibrated_ = true`, resets accumulators).
    - **`FROM 4B: YAWN_SUPPRESSOR`** (`suppress_by_yawn = (yawn_duration_sec >= 0.2 s)`).
  - **Face Lost Path (`HeadOrientationDetector::on_face_lost`)**:
    - `dt = compute_clamped_dt(now)`; if `droop_accum_sec_ > 0`: `droop_accum_sec_ = max(0.0, droop_accum_sec_ - 1.0 * dt)` (clears `in_droop_` & `droop_cue_emitted_` if `<= 0`). Preserves `last_pitch_deg_` and `last_rho_`; emits `cue = nullopt` -> jumps to Exit FSM Gate.
  - **Face Found Path (`HeadOrientationDetector::process`)**:
    - Compute `dt`, `rho`, `last_rho_ = rho`.
    - `<calibrated_ == true?>`:
      - `NO`: `pitch_deg = 0`, `last_pitch_deg_ = 0`, reset `in_droop_`, `droop_cue_emitted_`, `droop_accum_sec_ = 0` -> Exit FSM Gate.
      - `YES`: compute `pitch_deg` -> **`<!suppress_by_yawn AND pitch_deg <= -25.5 deg?>`**:
        - `YES`: `in_droop_ = true`, `droop_accum_sec_ = min(2.0, droop_accum_sec_ + dt)` -> `<droop_accum_sec_ >= 1.5 s AND !droop_cue_emitted_?>` -> attach `CueEvent(HeadDroop, Light)` -> Exit FSM Gate.
        - `NO` (either head upright OR **`suppress_by_yawn == true`**): `droop_accum_sec_ = max(0.0, droop_accum_sec_ - 2.0 * dt)` (clear flags at `<= 0`) -> Exit FSM Gate.
  - **Exit FSM Gate (`PipelineCoordinator.cpp` L557)**:
    - **`READ 1A: SystemState == MovingCalibrated?`**:
      - `YES`: **`GOTO 6B: SOMNOLENCE_CUE_FUSION`** (sends `HeadOrientationAssessment`).
      - `NO`: **`[RETURN (HUD Telemetry Only)]`** (updates `DebugVisualizer`, no alert emitted).
- **Branch Path Traces**:
  1. *Normal MovingCalibrated Droop -> `GOTO 6B`*
  2. *Yawn >= 0.2 s Forces `-2.0 * dt` Decay (Suppresses Tilted-Up/Down Yawn Artifact)*
  3. *Face Lost Mid-Droop (`-1.0 * dt` Gradual Decay)*
  4. *Parked (Calibrated): Live Pitch Telemetry Without Triggering Alerts*
- **Tier-2 Notes**:
  - **Parked Latch Edge Case**: Because `HeadOrientationDetector::process` runs even while `StationaryCalibrated` (`PipelineCoordinator.cpp` L551), if a parked driver droops for `>= 1.5 s`, `droop_cue_emitted_` sets to `true` and its single `cue` is discarded by `if (is_moving_calibrated)`; the driver would need to lift their head (`droop_accum_sec_ -> 0`) before a fresh `HeadDroop` cue could emit after moving.
- **Tier-3 C++ Refs**: `src/PipelineCoordinator.cpp` (L492, L528–560), `src/HeadOrientationDetector.cpp` (L13–114).

---

### Section 3: Blinking & PERCLOS (`sec3_blink_perclos.js` — 4 Sub-views)

#### Sub-view 3A — Blink Duration: Fused Algorithm View
- **Goal**: Present geometric 6-point EAR computation, the cascaded adaptive open-eye filter (`5`-median -> `40`-MA), dynamic `70%`/`20%` thresholds, and the `>= 500 ms` micro-sleep detector.
- **Flowchart Structure**:
  - **Input**: `68 Facial Landmarks` + calibrated `EAR_closed`, `floor_close`.
  - **Geometric EAR (`FaceGeometry::compute_ear`)**:
    - Right eye (`36..41`): `EAR_R = (||p37-p41|| + ||p38-p40||) / (2 · ||p36-p39||)` (0 if horizontal `< 1e-6`).
    - Left eye (`42..47`): `EAR_L = (||p43-p47|| + ||p44-p46||) / (2 · ||p42-p45||)`.
    - `EAR = (EAR_R + EAR_L) * 0.5`; `dt = compute_clamped_dt(now)`.
  - **Cascaded Adaptive Baseline (`EarDetector.cpp` L99–126)**:
    - `<!in_blink_cycle AND EAR > T_desc?>`:
      - `YES`: push `EAR` into `5`-sample `buffer_raw_` -> when full (`size == 5`), compute `EAR_med = median(buffer_raw_)` -> push into `40`-sample `buffer_median_` -> if `buffer_median_.size() >= 15`: `EAR_open = mean(buffer_median_)`, `range = max(0.05, EAR_open - EAR_closed)`, `T_desc = EAR_closed + 0.70 * range`, `T_close = max(floor_close, EAR_closed + 0.20 * range)`.
      - `NO`: freeze thresholds during blink cycle so closing eyes never pull the threshold down.
  - **Blink Cycle & Micro-Sleep Timer (`L137–182`)**:
    - `<EAR < T_desc?>`:
      - `YES`: if `!in_blink_cycle`, set `in_blink_cycle = true`, `reached_closure = false`, `closure_accum = 0`.
        - `<EAR <= T_close?>`:
          - `YES`: `closure_accum = (!reached_closure) ? dt : (closure_accum + dt)`; `reached_closure = true` -> `<closure_accum >= 0.500 s AND !microsleep_active?>`:
            - `YES`: `microsleep_active = true` -> **`[DROWSY]`** (pink/coral — Strong cue triggers immediate alarm).
            - `NO`: **`[RETURN]`** (green).
          - `NO` (`T_close < EAR < T_desc`, partial blink / lid hover): `closure_accum = max(0.0, closure_accum - 1.5 * dt)` -> **`[RETURN]`** (green).
      - `NO` (`EAR >= T_desc`, eye open): if `in_blink_cycle`, record `last_blink_ms = closure_accum * 1000` (if `reached_closure`), reset `in_blink_cycle = false`, `reached_closure = false`, `microsleep_active = false`, `closure_accum = 0` -> **`[RETURN]`** (green).
- **Branch Path Traces**:
  1. *Open Eye Drift -> Adaptive Threshold Update (5-Med -> 40-MA)*
  2. *Normal Blink (< 500 ms) -> `last_blink_ms` Recorded -> RETURN*
  3. *Micro-Sleep (>= 500 ms) -> Immediate DROWSY*
  4. *Partial Lid Hover (`T_close < EAR < T_desc`) -> `-1.5 * dt` Decay*
- **Tier-2 Notes**:
  - **Safety Floor**: `T_close` can adapt upward if a driver opens their eyes wider than calibration, or drift mildly downward with fatigue, but is clamped by `floor_closure_` (`>= calibrated T_close`).
  - **Boundary Detail**: Code checks `closure_accum_sec_ >= 0.500` (`EarDetector.cpp` L153), whereas docstrings say `>500ms`.
- **Tier-3 C++ Refs**: `src/FaceGeometry.cpp` (L7–20), `src/EarDetector.cpp` (L85–182), `include/Types.hpp` (L222–233).

#### Sub-view 3B — PERCLOS: Fused Algorithm View
- **Goal**: Present the O(1) 30-second rolling closure-percentage algorithm cleanly on its own canvas.
- **Flowchart Structure**:
  - **Inputs**: `EAR`, adaptive `T_close (20%)`, `in_blink_cycle`, timestamp `now`.
  - **Sample Classification & Push**: `is_closed = (EAR <= T_close) ? 1 : 0`; push `(now, is_closed)` to `perclos_buffer_`; if `is_closed == 1`, `closed_count++`.
  - **30 s Sliding Window Prune**: while `!perclos_buffer_.empty() && front.time < now - 30.0 s`, decrement `closed_count` if `front.second == 1`, pop front.
  - **Warmup Clock**: if `!has_perclos_start_`, set `perclos_start_time_ = now`, `has_perclos_start_ = true`; `elapsed = now - perclos_start_time_`; `warmed_up = (elapsed >= 30.0 s)`.
  - **Compute Percentage**: `PERCLOS = (closed_count / perclos_buffer_.size()) * 100.0%`.
  - **High PERCLOS Trigger Gate (`EarDetector.cpp` L223–235)**:
    - `<warmed_up AND in_blink_cycle AND (now - last_perclos_cue) >= 5.0 s AND PERCLOS >= 15.0%?>`:
      - `YES`: `last_perclos_cue = now`; **`clear_perclos()`** (wipes buffer, resets `closed_count = 0` & `has_perclos_start_ = false`, restarting 30 s warmup!) -> **`[DROWSY]`** (pink/coral — Strong cue).
      - `NO`: **`[RETURN]`** (green).
- **Branch Path Traces**:
  1. *Initial 30 s Warmup Accumulation -> Normal Return*
  2. *Warmed Up + Closed During Blink + PERCLOS >= 15% -> DROWSY + Buffer Wipe*
- **Tier-2 Notes**:
  - **`in_blink_cycle` Guard on Emission**: High PERCLOS only fires while `in_blink_cycle` is currently true (`L223`), ensuring the alarm coincides with an active eye descent/closure rather than lagging after the eyes are wide open.
  - **Self-Wipe Behavior**: Triggering High PERCLOS immediately calls `clear_perclos()`, forcing a fresh `30 s` warmup before another PERCLOS cue can fire (separate from the `5 s` cooldown guard).
- **Tier-3 C++ Refs**: `src/EarDetector.cpp` (L17–22, L184–247), `include/Types.hpp` (L230–232).

#### Sub-view 3C — Blink Duration: System & FSM Cut-Off View
- **Goal**: Show how `EarDetector` blink/micro-sleep processing receives calibration baselines, handles `Face Lost`, bows to `0.2 s` yawn suppression, hands off to PERCLOS (`3D`), and gates alerts on `MovingCalibrated`.
- **Flowchart Structure**:
  - **Incoming Portals**:
    - **`FROM 1C: DLIB_FACE_GATE`** (`68 Landmarks` vs `Face Lost`).
    - **`FROM 1B: APPLY_EAR_BASELINE`** (`ear_open`, `ear_closed` -> sets `T_desc`, `T_close`, `floor_close`, pre-fills `buffer_raw_[5]` and `buffer_median_[40]` with `ear_open`, sets `calibrated_ = true`).
    - **`FROM 4B: YAWN_SUPPRESSOR`** (`suppress_by_yawn >= 0.2 s`).
  - **Face Lost Path (`EarDetector::on_face_lost`, L266–274)**:
    - Immediately clears `in_blink_cycle_ = false`, `reached_effective_closure_ = false`, `microsleep_alarm_active_ = false`, `closure_accum_sec_ = 0.0`, `has_last_time_ = false` (prevents missing-face gap from bridging into a false micro-sleep!) -> **`TO 3D: PERCLOS_FACE_LOST_PRUNE`**.
  - **Face Found Path (`EarDetector::process`, L91–182)**:
    - Compute `EAR` + `dt` + Cascaded Adaptive Filter (`L99–126`, runs even when parked!).
    - **`<suppress_by_yawn == true?>`** (`L132–136`):
      - `YES`: wipes `in_blink_cycle_ = false`, `reached_effective_closure_ = false`, `microsleep_alarm_active_ = false`, `closure_accum_sec_ = 0.0` -> **`TO 3D: ROLLING_PERCLOS`**.
      - `NO`: run Blink & Micro-sleep state machine -> when `closure_accum_sec_ >= 500 ms && !microsleep_alarm_active_`: **`<calibrated_ == true?>`** -> `YES`: attach `CueEvent(BlinkDuration, Strong)` -> **`TO 3D: ROLLING_PERCLOS`**.
  - **Exit FSM Gate (`PipelineCoordinator.cpp` L557)**:
    - **`READ 1A: SystemState == MovingCalibrated?`**:
      - `YES`: **`GOTO 6B: SOMNOLENCE_CUE_FUSION`** (passes `EarAssessment`).
      - `NO`: **`[RETURN (HUD Telemetry Only)]`**.
- **Branch Path Traces**:
  1. *Calibrated Moving Micro-Sleep -> `TO 3D` -> `GOTO 6B`*
  2. *Yawn >= 0.2 s Mutes Blink/Squint Accumulator*
  3. *Face Lost Clears Unconfirmed Closure Streak*
- **Tier-2 Notes**:
  - **Telemetry When Parked**: While `StationaryCalibrated`, `EarDetector::process` still updates `EAR`, adaptive open-eye baseline, and `last_blink_duration_ms_` for the HUD, while `microsleep` cues are never forwarded to `AlertManager` (`if (is_moving_calibrated)`).
- **Tier-3 C++ Refs**: `src/PipelineCoordinator.cpp` (L491, L528–560), `src/EarDetector.cpp` (L42–182, L266–274).

#### Sub-view 3D — PERCLOS: System & FSM Cut-Off View
- **Goal**: Detail PERCLOS's strict `calibrated_ && MovingCalibrated` gate, missing-face pruning without false sample insertion, parking reset (`clear_perclos()`), and disabled Medium PERCLOS tier.
- **Flowchart Structure**:
  - **Incoming Portals**:
    - **`FROM 3C: BLINK_TO_PERCLOS`** (`Face Found` path carrying `EAR, T_close, in_blink_cycle, suppress_by_yawn` OR `Face Lost` path).
    - **`FROM 1A: CLEAR_PERCLOS`** (called on `MovingCalibrated -> StationaryCalibrated` from Main `'m'` or IMU park, plus fallback in `somnolence_worker_loop` L520).
    - **`FROM 4B: YAWN_SUPPRESSOR`** (`suppress_by_yawn`).
  - **Face Lost Path (`EarDetector::on_face_lost`, L275–330)**:
    - `<calibrated_ == false?>` -> `YES`: `clear_perclos()`, return `0%` / `30 s` warmup.
    - `NO`: prune entries `< now - 30 s` **without** pushing new samples (`perclos_buffer_` shrinks naturally); if empty, set `has_perclos_start_ = false`; return current `perclos_pct_` with zero cues.
  - **Face Found Path (`EarDetector::process`, L186–263)**:
    - **`READ 1A & Calib: <calibrated_ AND SystemState == MovingCalibrated?>`**:
      - `NO`: call **`clear_perclos()`** every frame -> return `PERCLOS = 0.0%`, `warmed_up = false`, `warmup_remaining = 30.0 s` -> **`[RETURN (PERCLOS Disabled While Parked)]`**.
      - `YES`: `is_closed = (!suppress_by_yawn && EAR <= T_close) ? 1 : 0` -> push & prune `30 s` window -> `<warmed_up && !suppress_by_yawn && in_blink_cycle && cooldown >= 5 s && PERCLOS >= 15.0%?>`:
        - `YES`: emit `CueEvent(Perclos, Strong)`, call **`clear_perclos()`** -> **`GOTO 6B: SOMNOLENCE_CUE_FUSION`**.
        - `NO`: **`GOTO 6B: SOMNOLENCE_CUE_FUSION`** (carries `EarAssessment` including any BlinkDuration cue from `3C`).
- **Branch Path Traces**:
  1. *MovingCalibrated Active PERCLOS >= 15% -> `GOTO 6B`*
  2. *Vehicle Parks (`1A`) -> Immediate `clear_perclos()` & 30 s Reset*
  3. *Face Lost While Driving -> Prune Old Samples Without Adding Closed Frames*
- **Tier-2 Notes**:
  - **Disabled Medium PERCLOS (`7.5%–15.0%`)**: `EarConfig::perclos_medium_thresh` (`7.5`, `Types.hpp` L231) and `EarDetector.hpp` L19 comment note Medium PERCLOS is intentionally disabled; only High PERCLOS (`>= 15.0%`, Strong cue) is emitted in MS6 (`README.md` is stale here).
  - **Concurrency Caveat**: `FROM 1A: CLEAR_PERCLOS` mutates `perclos_buffer_` directly from Main/IMU threads without locking against `somnolence_worker_loop`.
- **Tier-3 C++ Refs**: `src/PipelineCoordinator.cpp` (L340, L515–522, L550, L668), `src/EarDetector.cpp` (L17–22, L184–331), `include/Types.hpp` (L230–232).

---

### Section 4: Yawning (`sec4_yawning.js` — 2 Sub-views)

#### Sub-view 4A — Yawning: Fused Algorithm View
- **Goal**: Present the full-frame ONNX Yawn CNN pipeline, probability threshold, asymmetric linear decay, cooldown, and 2-in-120s Light-cue drowsiness verdict.
- **Flowchart Structure**:
  - **Input**: Full `BGR Frame` (no landmarks required).
  - **Preprocessing (`YawnDetector::pack_nhwc_blob`)**: BGR -> RGB -> `INTER_AREA` resize to `112x112` -> scale `[0, 1]` (`/ 255.0f`) -> NHWC blob (`1 x 112 x 112 x 3`).
  - **Inference**: `cnn_yawning.onnx` (`cv::dnn` CPU) -> scalar `yawn_prob`; `dt = compute_clamped_dt(now)`.
  - **Probability Gate**: `<yawn_prob >= 0.80?>`:
    - `YES`: `in_yawn = true`; `yawn_accum = min(1.0 s, yawn_accum + dt)` -> `<yawn_accum >= 1.0 s AND !yawn_cue_emitted AND (now - last_yawn) >= 2.0 s?>`:
      - `YES`: `yawn_cue_emitted = true`, `last_yawn = now`; record `Yawn` Light cue in `120 s` sliding window -> `<>= 2 Light cues in 120 s?>`:
        - `YES`: **`[DROWSY]`** (pink/coral).
        - `NO`: **`[RETURN]`** (green).
      - `NO`: **`[RETURN]`** (green).
    - `NO`: linear decay `yawn_accum = max(0.0, yawn_accum - 1.0 * dt)` -> `<yawn_accum <= 0.0?>`:
      - `YES`: `yawn_accum = 0`, `in_yawn = false`, `yawn_cue_emitted = false` -> **`[RETURN]`** (green).
      - `NO`: **`[RETURN]`** (green).
- **Branch Path Traces**:
  1. *Sustained Yawn (`prob >= 0.80` for `>= 1.0 s`) -> Light Cue (+ 2-in-120s -> DROWSY)*
  2. *Brief Speech / Mouth Open (`< 1.0 s`) -> Linear `-1.0 * dt` Decay -> RETURN*
- **Tier-2 Notes**:
  - **Full-Frame CNN vs Legacy MAR**: MS6 supersedes landmark-based `MarDetector` (`MAR > 0.55` for `4 s`) with full-frame `YawnDetector` (`prob >= 0.80` for `1.0 s`) because extreme jaw opening distorts Dlib 68-point facial geometry.
- **Tier-3 C++ Refs**: `src/YawnDetector.cpp` (L89–171), `include/Types.hpp` (L248–255).

#### Sub-view 4B — Yawning: System & FSM Cut-Off View
- **Goal**: Highlight `YawnDetector`'s independence from calibration and Dlib face detection, plus its dual outgoing role: early `0.2 s` cross-estimator suppression signal and `MovingCalibrated` Light-cue forwarding.
- **Flowchart Structure**:
  - **Lifecycle Context**:
    - **`READ 1A: SystemState == Calibrating?`** -> `YES`: `somnolence_worker_loop` hijacked by **`GOTO 1B: RUN_CALIBRATION`** (Yawn CNN paused during calibration).
    - **`FROM 1B: YAWN_RESET`** (`yawn_detector_.reset()` called upon calibration completion).
  - **Unconditional Processing (`PipelineCoordinator.cpp` L527)**:
    - Runs on every frame outside `Calibrating`—**zero calibration prerequisite** and **independent of `1C` Dlib face detection** (continues running even while Dlib is throttled to `500 ms` during `Face Lost`).
    - Preprocesses `112x112` NHWC -> `cnn_yawning.onnx` -> updates `yawn_accum_sec_` (`>= 0.80` -> `+dt`, `< 0.80` -> `-1.0 * dt`, `>= 1.0 s` + `2 s` cooldown -> `CueEvent(Yawn, Light)`).
  - **Outgoing Cut-Off 1 — Early Cross-Estimator Suppressor (`L528–531`)**:
    - `<yawn_accum_sec_ >= 0.2 s?>` -> `YES`: **`SIGNAL -> SUPPRESS_BY_YAWN = true`** -> fans out to **`GOTO 2B` (Head Droop)**, **`GOTO 3C/3D` (Blink & PERCLOS)**, and **`GOTO 6B` (Alert Fusion)**.
  - **Outgoing Cut-Off 2 — Exit FSM Gate (`L557`)**:
    - **`READ 1A: SystemState == MovingCalibrated?`**:
      - `YES`: **`GOTO 6B: SOMNOLENCE_CUE_FUSION`** (passes `YawnAssessment`).
      - `NO`: **`[RETURN (HUD Telemetry Only)]`**.
- **Branch Path Traces**:
  1. *0.2 s Early Suppressor Fires Before 1.0 s Light Cue -> `GOTO 2B / 3C / 6B`*
  2. *Face Lost in Dlib (`1C` Throttled) While Yawn CNN Keeps Tracking*
  3. *Parked (Calibrated): Yawn CNN Telemetry Active, Alerts Gated*
- **Tier-2 Notes**:
  - **Legacy `MarDetector` Dead Code**: `src/MarDetector.cpp` and `MarConfig` (`Types.hpp` L238–243) remain compiled in the repo but are never instantiated in `PipelineCoordinator`.
  - **Why `0.2 s` Suppressor Exists**: Yawning naturally squints eyes (dropping EAR) and tilts the head back/forward (altering `rho`), so `0.2 s` of confirmed yawn mutes `BlinkDuration`, `Perclos`, and `HeadDroop` before their `0.5 s`/`1.5 s` timers trip.
- **Tier-3 C++ Refs**: `src/PipelineCoordinator.cpp` (L480–505, L526–560), `src/YawnDetector.cpp` (L80–171), `src/MarDetector.cpp` (L1–110, unused).

---

### Section 5: Distraction (`sec5_distraction.js` — 2 Sub-views)

#### Sub-view 5A — Distraction: Fused Algorithm View
- **Goal**: Present the complete algorithmic path from camera frame -> CLAHE/ImageNet preprocessing -> ResNet-18 + LSTM -> softmax & confidence policy -> 2-stage temporal filter (`10f` sliding mode + `2.0 s` continuous timer) -> `DISTRACTED` vs `RETURN`.
- **Flowchart Structure**:
  - **Input**: `BGR Frame`.
  - **Preprocessing (`Preprocessor::process`)**: BGR -> RGB -> bilinear resize `224x224` -> RGB -> Grayscale -> CLAHE (`clipLimit = 2.5`, `8x8` tiles) -> Grayscale -> RGB -> ImageNet affine `(pixel/255 - mean) / std` NCHW (`1 x 3 x 224 x 224`).
  - **Spatial Backbone (`resnet18_lstm_preopt_backbone.onnx`)**: extracts `512`-d embedding -> pushes into `10`-frame ring buffer (`20` if `--unified`); if warmup (`buf_len < seq_len`), left-pads missing slots with earliest embedding (`GazeDetector::pack_sequence_blob`).
  - **Temporal Head (`resnet18_lstm_preopt_head.onnx`)**: outputs `[logit_not_looking, logit_looking]` -> stable Softmax `P(not_looking)`, `P(looking)`.
  - **Per-Frame Policy (`BinarizeThreshold`, default)**: `<P(looking) >= 0.54?>` -> `YES`: `LookingRoad` vote / `NO`: `NotLookingRoad` vote.
  - **Inline 2-Stage Temporal Filter**:
    - Push vote into `10`-frame sliding `gaze_window_` (`20` if `--unified`); `<gaze_window_.size() < 10?>` -> `YES`: **`[RETURN (Warmup)]`**.
    - **Stage 1 (Majority Mode)**: `<not_looking_votes > looking_votes?>`:
      - `NO`: reset `distracted_since` -> **`[RETURN]`** (green).
      - `YES`: set `distracted_since = now` (if unset) -> **Stage 2 (Sustained Timer)**: `<now - distracted_since >= 2.0 s?>`:
        - `NO`: **`[RETURN]`** (green).
        - `YES`: **`[DISTRACTED]`** (pink/coral; alarm clears window & timer after audio playback).
- **Branch Path Traces**:
  1. *Cold-Start Ring-Buffer Padding -> 10f Mode Warmup -> Normal Road Attention*
  2. *Sustained Off-Road Gaze (10f Majority Mode Held >= 2.0 s) -> DISTRACTED*
  3. *Brief Mirror Glance (< 2.0 s) -> Timer Resets -> RETURN*
- **Tier-2 Notes**:
  - **Grayscale-Replicated CLAHE**: `Preprocessor.cpp` converts resized RGB to 1-channel grayscale, runs CLAHE, and converts back to 3-channel RGB before per-channel ImageNet normalization—ensuring 1:1 parity with Python training preprocessing.
  - **Alternative Policies**: Besides default `BinarizeThreshold` (`0.54`), CLI flags support `Argmax` (`P(looking) >= P(not_looking)`) and `FilterUncertain` (drops ambiguous votes in `(0.46, 0.54)` during Stage 1).
- **Tier-3 C++ Refs**: `src/Preprocessor.cpp` (L33–84), `src/GazeDetector.cpp` (L99–190), `src/AlertLogic.cpp` (L5–49), `include/Types.hpp` (L57–61, L149–155, L172–179).

#### Sub-view 5B — Distraction: System & FSM Cut-Off View (Strict C++ Class/Mutex Boundary)
- **Goal**: Match the exact C++ worker/class boundary where `GazeWorker` (`GazeDetector`) handles atomic resets and `MovingCalibrated` gating, emits **only a single per-frame vote**, and cuts off cleanly into `AlertManager`.
- **Flowchart Structure**:
  - **Thread Context**: `GazeWorker` (`PipelineCoordinator::gaze_worker_loop`, L399–457).
  - **Entry Atomic Reset Gate (`L412`, `L424`)**: **`CONSUME ATOMIC (FROM 1A): gaze_reset_requested_.exchange(false)?`** (checked both before and immediately after `mailbox_gaze_->get_latest`) -> `YES`: `gaze_detector_.reset()` (`ring_head = 0, ring_count = 0`).
  - **Entry FSM Gate (`L429`)**: **`READ 1A: SystemState == MovingCalibrated?`**:
    - `NO` (Parked, Uncalibrated, or Calibrating): **`[SKIP FRAME (Zero Neural Compute)]`** (green — unlike `FatigueWorker`, `GazeWorker` runs zero inference when not actively driving calibrated!).
    - `YES`: proceed to neural pipeline.
  - **Neural Pipeline (`Preprocessor` + `GazeDetector`)**:
    - `Preprocessor::process` -> `ResNet-18` backbone -> `10f` (`20f`) flat ring buffer + left-padding -> `LSTM` head -> Softmax -> Confidence Policy (`BinarizeThreshold >= 0.54` default; `FilterUncertain` / `Argmax` optional) -> builds `DistractionDetectionResult{gaze_class, P(looking), timestamp}`.
  - **Strict C++ Class / Mutex Cut-Off (`L443`)**:
    - **`GOTO 6A: ALERT_MANAGER_GAZE_FILTER`** (`alert_manager_.update_gaze(result)` — crosses from `GazeWorker` into `AlertManager::mutex_`).
- **Branch Path Traces**:
  1. *MovingCalibrated Frame -> ResNet-18 + LSTM -> Single Vote -> `GOTO 6A`*
  2. *Parked / Uncalibrated / Calibrating -> Immediate Frame Skip (Zero CPU/ONNX Load)*
  3. *Vehicle Parks -> Atomic `gaze_reset_requested_` Self-Wipe on Next Wakeup*
- **Tier-2 Notes**:
  - **Mutex-Free Ring-Buffer Reset**: Main and IMU threads never call `gaze_detector_.reset()` directly (avoiding a data race with `GazeWorker`); they set `gaze_reset_requested_` (`atomic<bool>`) so `GazeWorker` wipes its own circular buffer on its own thread.
  - **Warmup Padding**: During the first `seq_len - 1` frames after startup or park-reset, `GazeDetector` still emits classifications using earliest-feature front-padding; `AlertManager` separately withholds distraction checks until its own `mode_window_frames` buffer fills.
- **Tier-3 C++ Refs**: `src/PipelineCoordinator.cpp` (L69–108, L331, L341, L399–457, L669, L677), `src/GazeDetector.cpp` (L94–190).

---

### Section 6: Alert Fusion & Alarm Dispatcher (`sec6_alarms_dispatcher.js` — 3 Sub-views)

#### Sub-view 6A — Gaze 2-Stage Filter (`AlertManager::update_gaze`, Inside Mutex)
- **Goal**: Serve as the exact landing target for `5B`, showing everything `AlertManager` computes inside `mutex_` for distraction—warmup, Drowsy preemption gate, `10f` sliding mode, `2.0 s` continuous timer, and post-alarm window wipe.
- **Flowchart Structure**:
  - **Entry**: **`FROM 5B: PER_FRAME_GAZE_VOTE`** -> **Acquire `AlertManager::mutex_`**.
  - **Window Fill**: push `detection` to `gaze_window_` (pop oldest while `size > mode_window_frames`, default `10` or `20`); `<gaze_window_.size() < mode_window_frames?>` -> `YES`: unlock & **`[RETURN (Warmup)]`**.
  - **Drowsiness Priority Gate (`AlertManager.cpp` L112)**: `<is_drowsiness_alert_reproducing_locked() OR current_state_ == Drowsy?>` -> `YES`: `distracted_since_.reset()`, unlock & **`[RETURN (Preempted by Drowsy)]`**.
  - **Stage 1 (`AlertLogic::evaluate_distraction_mode`)**: compute `not_looking_votes`, `looking_votes`, `total_votes`.
    - `<total_votes == 0 OR not_looking <= looking?>`:
      - `YES` (Attentive or all frames uncertain): `distracted_since_.reset()` -> `<current_state_ == Distracted?>`:
        - `YES`: `current_state_ = Safe`, unlock -> **`GOTO 6C: SILENCE_DISTRACTION`**.
        - `NO`: unlock & **`[RETURN (Safe)]`**.
      - `NO` (Off-road mode majority): if `!distracted_since_`, `distracted_since_ = now`.
        - **Stage 2 (`L133–137`)**: `<now - distracted_since_ >= 2000 ms?>`:
          - `NO`: unlock & **`[RETURN (Timer Counting)]`**.
          - `YES`: `current_state_ = Distracted`, unlock -> **`GOTO 6C: DISPATCH_DISTRACTION_ALARM`** -> upon return from blocking audio (`L187–192`), re-lock `mutex_`, `gaze_window_.clear()`, `distracted_since_.reset()`, and if `current_state_ == Distracted`, revert `current_state_ = Safe` -> **`[RETURN]`**.
- **Branch Path Traces**:
  1. *10f Mode + 2.0 s Timer Met -> `GOTO 6C` Blocking Audio -> Post-Alarm Wipe*
  2. *Preempted by Active Drowsiness (`L112`) -> Immediate Abort*
  3. *Driver Looks Back at Road -> Timer Clears (or Silences Alarm)*
- **Tier-2 Notes**:
  - **Post-Alarm Refill Penalty**: Because `gaze_window_.clear()` wipes all 10 votes after a distraction alarm finishes (`L188`), the driver gets a fresh 10-frame + 2.0 s grace period before a second distraction alarm can trigger.
- **Tier-3 C++ Refs**: `src/AlertManager.cpp` (L83–204, L413–421), `src/AlertLogic.cpp` (L5–49).

#### Sub-view 6B — Somnolence Cue Fusion & Recovery (`AlertManager::update_somnolence`, Inside Mutex)
- **Goal**: Serve as the landing target for `2B`, `3C/3D`, and `4B`, showing inside-mutex fusion of Strong cues vs 2-in-120s Light cues, physiological recovery clearing, and 63.5 BPM gain-ladder stepping.
- **Flowchart Structure**:
  - **Entry Portals**: **`FROM 2B (HeadAssessment)`**, **`FROM 3C/3D (EarAssessment)`**, **`FROM 4B (YawnAssessment)`** (invoked only when `SystemState == MovingCalibrated`) -> **Acquire `AlertManager::mutex_`**.
  - **Defense-in-Depth Yawn Suppression (`L287–292`)**: `is_yawning_suppression = (yawn.yawn_duration_sec >= 0.2)`; filter out `BlinkDuration` and `Perclos` cues from `ear.cues` if active.
  - **Cue Registration Loop (`register_cue_locked`, L206–240)** across surviving `ear.cues`, `head.cue`, `yawn.cue`:
    - `<cue.strength == Strong?>` (`BlinkDuration` or `Perclos`): if `current_state_ != Drowsy`, set `current_state_ = Drowsy`, `drowsiness_step_idx_ = 0`, rewind `last_alert_sound_time_`.
    - `<cue.strength == Light?>` (`HeadDroop` or `Yawn`): push to `cue_window_`, prune entries `< cue.timestamp - 120 s`; `<cue_window_.size() >= 2?>` (`evaluate_somnolence_cues`) -> if `current_state_ != Drowsy`, set `current_state_ = Drowsy`, `drowsiness_step_idx_ = 0`, rewind cooldown.
  - **Active Drowsiness Check (`L302–314`)**: `<current_state_ == Drowsy?>`:
    - `NO`: unlock & **`[RETURN (Safe / Watch)]`**.
    - `YES`: `blink_active = !is_yawning_suppression && ear.in_blink_cycle`; `features_normal = !blink_active && !head.is_drooping && !yawn.is_yawning`.
      - `<features_normal AND no cue triggered this frame?>`:
        - `YES` (Physiological recovery): `current_state_ = Safe`, `drowsiness_step_idx_ = 0`, `cue_window_.clear()`, rewind cooldown, unlock -> **`GOTO 6C: SILENCE_DROWSINESS`**.
        - `NO` (Still drowsy): `<63.5 BPM (~944.9 ms) cooldown elapsed?>` -> `YES`: `gain_db = {-21.0, -16.5, -12.0, -7.5, -3.0}[min(step_idx++, 4)]`, unlock -> **`GOTO 6C: DISPATCH_DROWSINESS_PULSE`** (with `newly_triggered` flag for Telegram) / `NO`: unlock & **`[RETURN (Wait Next Beat)]`**.
- **Branch Path Traces**:
  1. *Single Strong Cue (`BlinkDuration` / `Perclos`) -> Immediate Drowsy + Step 0 (`-21.0 dB`)*
  2. *Two Light Cues Within 120 s (e.g., Yawn + Head Droop) -> Drowsy*
  3. *Eyes Open + Upright Head + Mouth Closed -> Immediate Recovery Silence (`GOTO 6C`)*
- **Tier-2 Notes**:
  - **2-Minute Window Clears on Recovery**: When drowsiness clears (`L308`), `cue_window_.clear()` prevents an old Light cue from immediately pairing with a minor posture shift moments later.
  - **Hardcoded vs Configurable `0.2 s`**: `AlertManager.cpp` L287 uses literal `0.2`, whereas `PipelineCoordinator.cpp` L528 reads `yawn_detector_.config().time_for_ear_suppression_by_yawn` (both `0.2 s` by default).
- **Tier-3 C++ Refs**: `src/AlertManager.cpp` (L206–339, L360–373), `src/AlertLogic.cpp` (L51–59).

#### Sub-view 6C — Outside-Lock Alarm Dispatcher (Audio Preemption & Telegram)
- **Goal**: Show all I/O executed **outside** `AlertManager::mutex_`—blocking distraction audio, priority preemption via `audio_.stop()`, escalating 63.5 BPM pulses on the 48 kHz mixer, and detached `libcurl` Telegram alerts.
- **Flowchart Structure**:
  - **Path 1 — Distraction Dispatch (`FROM 6A: DISPATCH_DISTRACTION_ALARM`)**:
    - Brief lock re-check (`L161`): `<is_drowsiness_alert_reproducing_locked()?>` -> `YES`: abort immediately (`[RETURN TO 6A]`); `NO`: `drowsiness_audio_active_ = false`, unlock.
    - Spawn detached `std::thread` -> `send_telegram_alert` (HTTP POST via `libcurl`, `3 s` connect / `5 s` total timeout).
    - Call `audio_.play_blocking(distraction_warning.wav, +6.0 dB)` -> **blocks `GazeWorker` on `state_cv_` until WAV ends OR `audio_.stop()` increments `playback_id_`** -> **`RETURN TO 6A (Post-Alarm Window Wipe)`**.
  - **Path 2 — Drowsiness Escalation & Preemption (`FROM 6B: DISPATCH_DROWSINESS_PULSE`)**:
    - If newly triggered (`strong_cue` or `2 Light cues`): spawn detached `std::thread` -> `send_telegram_alert` (`3 s` / `5 s` timeout).
    - If `emit_audio`: briefly lock to set `drowsiness_audio_active_ = true` -> unlock -> **`audio_.stop()`** (**immediately preempts any playing distraction clip** and wakes `GazeWorker`!) -> `audio_.play(single-beep-1850-Hz-975-Hz.wav, gain_db)` (non-blocking 48 kHz ring-buffer pulse at `-21.0, -16.5, -12.0, -7.5, -3.0 dB`).
  - **Path 3 — Silence & FSM Reset (`FROM 6A: SILENCE_DISTRACTION`, `FROM 6B: SILENCE_DROWSINESS`, `FROM 1A: ALERT_RESET`)**:
    - From `6A` (attentive): lock check `!is_drowsiness_alert_reproducing_locked()` -> `audio_.stop()`.
    - From `6B` (recovery): lock set `drowsiness_audio_active_ = false` -> `audio_.stop()`.
    - From `1A` (`AlertManager::reset()`): lock clears `gaze_window_`, `cue_window_`, `distracted_since_`, sets `DriverState = Safe`, `drowsiness_step_idx_ = 0`, `drowsiness_audio_active_ = false` -> `audio_.stop()`.
- **Branch Path Traces**:
  1. *Distraction Blocking Audio + Detached Telegram*
  2. *Drowsiness Preemption Cuts Off Distraction Mid-Clip (`playback_id_++`)*
  3. *Park Reset (`1A`) Silences Audio & Clears All Windows*
- **Tier-2 Notes**:
  - **Why Lock Is Released Before Audio/Network**: `FatigueWorker` can still acquire `AlertManager::mutex_` instantaneously and call `audio_.stop()` while `GazeWorker` is blocked inside `play_blocking()`—guaranteeing drowsiness preemption never deadlocks or waits for distraction audio to finish.
  - **48 kHz Mixer & Lazy Decode**: `SoundPlayer` runs a persistent 48 kHz stereo `6x10 ms` ring-buffer thread (`waveOut` on Windows, `aplay` raw pipe on Linux). Note that `distraction_warning.wav` and `single-beep-1850-Hz-975-Hz.wav` decode lazily on first trigger rather than in `warmup()`.
- **Tier-3 C++ Refs**: `src/AlertManager.cpp` (L159–203, L320–338, L375–406), `src/AudioManager.cpp` (L12–30), `src/SoundPlayer.cpp` (L253–295, L503–563), `src/TelegramNotifier.cpp` (L36–71).

---

## 4. Portal Cross-Reference Registry (Sections 1–6)

Every `GOTO`, `FROM`, `TO`, `SIGNAL`, `READ`, and `ATOMIC` portal across Phase 1 is listed below to guarantee 100% bi-directional consistency and zero dangling click targets:

| Source (Sub-view) | Portal Label on Canvas | Target (Sub-view & Node) | Underlying C++ Mechanism |
| :--- | :--- | :--- | :--- |
| `1A` | `GOTO 1B: RUN_CALIBRATION` | `1B` (`entry_calib`) | `system_state_.store(Calibrating)` polled by `somnolence_worker_loop` (`L480`) |
| `1A` | `GOTO 6C: ALERT_RESET` | `6C` (`path_reset`) | Direct call `alert_manager_.reset()` (`L330`, `L339`, `L667`, `L676`) |
| `1A` | `GOTO 3D: CLEAR_PERCLOS` | `3D` (`from_park_clear`) | Direct call `ear_detector_.clear_perclos()` (`L340`, `L668`, fallback `L520`) |
| `1A` | `SET ATOMIC: gaze_reset_requested_` | `5B` (`from_atomic_reset`) | `gaze_reset_requested_.store(true, release)` (`L331`, `L341`, `L669`, `L677`) |
| `1B` | `GOTO 1C: DLIB_DIRECT_DETECT` | `1C` (`dlib_hog`) | `ctx.face.detect(frame, now)` inside `measure_window` (`Calibration.cpp` L63) |
| `1B` | `GOTO 3C: APPLY_EAR_BASELINE` | `3C` (`from_calib_ear`) | `ear_detector_.apply_baseline(open, closed, now)` (`L491`) |
| `1B` | `GOTO 2B: APPLY_HEAD_BASELINE` | `2B` (`from_calib_rho`) | `head_detector_.apply_baseline(neutral_rho)` (`L492`) |
| `1B` | `GOTO 4B: YAWN_RESET` | `4B` (`from_calib_yawn_reset`) | `yawn_detector_.reset()` (`L493`) |
| `1B` | `GOTO 1C: FACE_RESET` | `1C` (`from_calib_face_reset`) | `face_detector_.reset()` (`L494`) |
| `1C` | `GOTO 2B: HEAD (Landmarks / Lost)` | `2B` (`from_face_gate`) | `head_detector_.process(...)` / `on_face_lost(...)` (`L551`, `L554`) |
| `1C` | `GOTO 3C: BLINK (Landmarks / Lost)` | `3C` (`from_face_gate`) | `ear_detector_.process(...)` / `on_face_lost(...)` (`L550`, `L553`) |
| `2B` | `READ 1A: MovingCalibrated?` | `1A` (`st_mov_cal`) | `is_moving_calibrated` local snapshot of `system_state_.load()` (`L515`, `L557`) |
| `2B` | `GOTO 6B: SOMNOLENCE_CUE_FUSION` | `6B` (`from_detectors`) | `alert_manager_.update_somnolence(ear, head, yawn)` (`L558`) |
| `3C` | `TO 3D: ROLLING_PERCLOS` | `3D` (`from_blink_perclos`) | Sequential Section 5 inside `EarDetector::process` / `on_face_lost` |
| `3C` | `GOTO 6B: SOMNOLENCE_CUE_FUSION` | `6B` (`from_detectors`) | `alert_manager_.update_somnolence(ear, head, yawn)` (`L558`) |
| `3D` | `READ 1A: MovingCalibrated?` | `1A` (`st_mov_cal`) | `record_perclos = is_moving_calibrated` (`L186`, `L550`) |
| `3D` | `GOTO 6B: SOMNOLENCE_CUE_FUSION` | `6B` (`from_detectors`) | `alert_manager_.update_somnolence(ear, head, yawn)` (`L558`) |
| `4B` | `READ 1A: Calibrating?` | `1A` (`st_calib`) | `system_state_.load() == SystemState::Calibrating` (`L480`) |
| `4B` | `SIGNAL -> SUPPRESS_BY_YAWN` | `2B`, `3C`, `3D`, `6B` (`from_yawn_sup`) | `suppress_by_yawn = (yawn_res.yawn_duration_sec >= 0.2)` (`L528–531`) |
| `4B` | `GOTO 6B: SOMNOLENCE_CUE_FUSION` | `6B` (`from_detectors`) | `alert_manager_.update_somnolence(ear, head, yawn)` (`L558`) |
| `5B` | `CONSUME ATOMIC: gaze_reset` | `1A` (`park_cleanup`) | `gaze_reset_requested_.exchange(false, acq_rel)` (`L412`, `L424`) |
| `5B` | `READ 1A: MovingCalibrated?` | `1A` (`st_mov_cal`) | `system_state_.load() != SystemState::MovingCalibrated` (`L429`) |
| `5B` | `GOTO 6A: ALERT_MANAGER_GAZE_FILTER` | `6A` (`from_gaze_vote`) | `alert_manager_.update_gaze(result)` (`L443`) |
| `6A` | `GOTO 6C: DISPATCH_DISTRACTION` | `6C` (`path_distraction`) | Outside-lock block inside `AlertManager::update_gaze` (`L159–192`) |
| `6A` | `GOTO 6C: SILENCE_DISTRACTION` | `6C` (`path_reset`) | Outside-lock `audio_.stop()` on `Distracted -> Safe` (`L193–203`) |
| `6B` | `GOTO 6C: DISPATCH_DROWSINESS` | `6C` (`path_drowsiness`) | Outside-lock Telegram + `dispatch_auditory_alert(...)` (`L330–338`) |
| `6B` | `GOTO 6C: SILENCE_DROWSINESS` | `6C` (`path_reset`) | Outside-lock `audio_.stop()` on physiological recovery (`L320–329`) |
| `6C` | `RETURN TO 6A: POST_ALARM_WIPE` | `6A` (`post_alarm_wipe`) | Re-lock & clear `gaze_window_` / `distracted_since_` after `play_blocking` (`L187–192`) |

---

## 5. Phase-2 Proposal: General Subroutine Overview (`sec7_subroutine_overview.js`)

> **Status**: Sketched below for review; **to be finalized and built only after Phase 1 (`sec1`–`sec6`) is implemented and visually validated**.

Once the reader has inspected each cut-off trigger and FSM diagram individually, Section 7 simplifies the entire system into **double-walled clickable `[SUBROUTINE]` blocks** on one clean `960x540` canvas—avoiding a giant spaghetti flowchart while preserving exact control and gating semantics. Two architectural options are proposed:

### Candidate A (Recommended): Two-Loop Subroutine Pipeline Framed by the Two FSMs
- **Layout**:
  - **Top Band (Lifecycle FSM)**: `[SUBROUTINE 1A: 5-State Vehicle & Calibration FSM]` publishing `SystemState` (`Stationary*`, `Calibrating`, `MovingCalibrated`) and emitting parking resets.
  - **Middle Left Lane (Gaze Loop)**: `<MovingCalibrated?>` -> `NO: Skip` / `YES` -> `[SUBROUTINE 5B: Gaze Net (10f LSTM)]` -> `[SUBROUTINE 6A: 2-Stage Gaze Filter (10f + 2.0 s)]`.
  - **Middle Right Lane (Somnolence Loop)**: `<Calibrating?>` (`YES` -> `[SUBROUTINE 1B: 2-Phase Calibration]`) -> `[SUBROUTINE 4B: Yawn CNN]` (with one crisp cross-wire `yawn >= 0.2 s` suppress) -> `[SUBROUTINE 1C: 500 ms Face Gate]` -> parallel `[SUBROUTINE 3C/3D: Blink & PERCLOS]` + `[SUBROUTINE 2B: Head Droop]` -> `<MovingCalibrated?>` -> `[SUBROUTINE 6B: Cue Fusion]`.
  - **Bottom Band (Alarm Arbitration & I/O)**: `[SUBROUTINE 6C: Outside-Lock Alarm Dispatcher]` (`DriverState`: `Safe` / `Distracted` / `Drowsy`, audio preemption & Telegram).
- **Why Recommended**:
  - Fits in ~10 clean subroutine nodes with zero crossing wires.
  - Explains **why** `Gaze` skips all neural inference when parked while `Somnolence` continues running for HUD telemetry, and **how** calibration hijacks only the somnolence loop.
  - Every `[SUBROUTINE]` block links straight to its corresponding Phase-1 cut-off flowchart (`1A`–`6C`) with `"<- Back to General Overview"`.

### Candidate B: Thread-Agnostic Functional Subroutine DAG
- **Layout**: Left-to-right pure dataflow (`Camera + IMU + Calibration` -> 5 parallel `[SUBROUTINE]` feature estimators `Head Droop`, `Blink Duration`, `PERCLOS`, `Yawn CNN`, `Gaze Distraction` -> `[SUBROUTINE: Multimodal Arbitration & Priority]` -> `Auditory / Telegram Alarms`).
- **Trade-off**: Visually simplest for readers who only care about signal fusion, because it completely hides frame loops, `Calibrating` takeover, and parked telemetry asymmetry (deferring those to `docs/explainer/`). Less faithful to MS6's actual execution gates than Candidate A.

---

## 6. Paper Figure Mapping, MS6 Doc-vs-Code Sync Checklist & Build Roadmap

### Candidate Paper Figure Mapping
| Explainer View | Paper Role / Section | Suggested Figure Usage |
| :--- | :--- | :--- |
| **2A, 3A, 3B, 4A, 5A (Fused Views)** | *Methodology / Detection Algorithms* | Individual algorithm flowcharts (clean English SVGs with `RETURN` / `DROWSY` / `DISTRACTED` terminals) |
| **1A + 1B** | *Operational Protocol & Personal Calibration* | 5-State Movement/IMU FSM & 2-Phase Acoustic Calibration sequence |
| **6A + 6B + 6C** | *Multimodal Fusion & Warning Arbitration* | 2-Stage Gaze Filter, Strong/Light Cue Fusion, & Drowsiness Preemption |
| **Phase 2: Sec 7 (Candidate A)** | *System Integration Overview* | High-level subroutine architecture linking all detectors & FSMs |

### MS6 Doc-vs-Code Sync Nuances Recorded in Blueprint
1. **Head Droop Threshold (`-25.5 deg`)**: `HeadOrientationConfig::pitch_droop_thresh` (`Types.hpp` L195) is `-25.5 deg` (matching BU HT dataset 5.48 deg MAE); legacy `-20 deg` comments in `HeadOrientationDetector.cpp` L59, `HeadOrientationDetector.hpp` L14, and `Types.hpp` L106 are stale.
2. **Medium PERCLOS Disabled**: `EarConfig::perclos_medium_thresh` (`7.5%`) is defined in `Types.hpp` L231, but `EarDetector.cpp` only emits **High PERCLOS** (`>= 15.0%` over `30 s`, Strong cue).
3. **Micro-Sleep Equality**: Code checks `closure_accum_sec_ >= 0.500` (`EarDetector.cpp` L153), whereas docstrings write `>500ms`.
4. **Grayscale-Replicated CLAHE**: `Preprocessor.cpp` L40–43 converts `224x224` RGB -> Grayscale -> CLAHE (`2.5`, `8x8`) -> 3-channel Grayscale-RGB before ImageNet normalization.
5. **Calibration Phase 2 Prep Tone**: `Calibration.cpp` L168 plays the 1 s 415 Hz preparation tone in **both** Phase 1 and Phase 2.
6. **Clamped Delta Time (`dt`)**: `MathUtils::compute_clamped_dt` clamps frame gaps `> 0.15 s` down to `0.033 s` (`~30 FPS` equivalent) so face-loss pauses or stalls never spike droop/blink/yawn timers.
7. **Unused Legacy Symbols**: `MarDetector` (`src/MarDetector.cpp`) and `AlertManager::tick_drowsiness_if_active()` (`src/AlertManager.cpp` L341–358) are compiled in MS6 but unused at runtime.
8. **Cross-Thread `clear_perclos()` Caveat**: Main (`L340`) and IMU (`L668`) call `ear_detector_.clear_perclos()` directly on stop without a mutex or atomic handoff against `somnolence_worker_loop`.

### Build & Verification Status
- **Phase 1 Build — Complete (`docs/trigger_explainer/`)**:
  1. Built `docs/trigger_explainer/index.html`, `style.css`, and `flowchart_engine.js` (process boxes, decision diamonds, green `RETURN`/`SKIP` pills, coral `DROWSY`/`DISTRACTED` pills, clickable `GOTO`/`FROM` portals with `"<- Back"` navigation history stack, full-flowchart upfront display + branch trace spotlighting, 4-layer side-panel disclosures including the collapsible bold **`Code:`** toggle, and clean SVG export). Added cross-navigation links between `docs/explainer/index.html` and `docs/trigger_explainer/index.html` and added explicit `[hidden] { display: none !important; }` guards.
  2. Authored `sections/sec1_vehicle_calib_fsm.js` through `sections/sec6_alarms_dispatcher.js` (all 6 sections / 16 sub-views: `1A`–`1C`, `2A`–`2B`, `3A`–`3D`, `4A`–`4B`, `5A`–`5B`, `6A`–`6C`) using clean human-readable labels and symbol-free general descriptions, moving all C++ expressions and formulas into each item's **`Code:`** block.
  3. **Automated Verification Passed (0 Errors)**: Verified `node --check` across all 8 JS files, HTML asset & cross-link existence, CSS `[hidden]` override rules, `viewBox` bounds and `>= 6 px` non-overlap geometry across all 16 sub-views, zero edge-through-node interior crossings, text-inside-shape character/line fit, portal & trace referential integrity, strict no-code-in-nodes/prose linting, non-empty **`Code:`** coverage across all overviews/nodes/traces, and a DOM-mock smoke test exercising all section/sub-view/trace/portal/`Code:`/export flows.
- **Phase 2 Build — Pending User Review (`sec7_subroutine_overview.js`)**:
  1. Review Phase 1 flowcharts and confirm **Candidate A** vs. **Candidate B** for `sec7_subroutine_overview.js`.
  2. Implement Section 7 with clickable drill-down links into Phase-1 cut-off views and re-verify geometry & SVG export.
