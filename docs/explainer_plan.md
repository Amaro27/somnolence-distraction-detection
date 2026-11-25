# Real-Time Architecture Explainer Plan (`morphed_system4/distraction/modular/`)

> **Deliverable Scope**: This document is the **implementation blueprint** for a lightweight, interactive, browser-based set of system visualizers. No HTML, CSS, JS, or C++ explainer files are created during planning; future sessions can implement them incrementally using the token-frugal phases in Section 6.

---

## 1. Purpose, Audience & Hard Constraints

### Target System & Anchor
- **Target Runtime**: `morphed_system4/distraction/modular/` — the C++17 multi-threaded driver distraction and somnolence detection system.
- **Python Prototype Anchor**: At the start of Chapter 1, orient engineers by briefly mapping their familiar Python research modules to where each algorithm now runs in C++:
  - Gaze distraction (`distraction/` & `morphed_system3/standalone_scripts/RTdetection_lstm.py`) -> **Gaze Worker**
  - Eye blink & PERCLOS (`ear_mar/` & `morphed_system2/estimation_ear_mar.py` / `morphed_system3/standalone_scripts/AF_test.py`) -> **Fatigue Worker** (Eye monitor)
  - Head pitch droop (`head_orientation/` & `morphed_system2/estimation_head_pose.py` / `morphed_system3/standalone_scripts/head_pose_simple.py`) -> **Fatigue Worker** (Posture monitor)
  - Yawn detection (`morphed_system3/standalone_scripts/base_yawn_detector.py`) -> **Fatigue Worker** (Yawn CNN)
  - Calibration & multimodal fusion (`morphed_system2/calibration.py`, `morphed_system2/fusion.py`) -> **Sequential Calibration Runner** & **Alert Coordinator**

### Primary Audience
Human ML/CV engineers who wrote and validated the detection algorithms in Python, and now need an accurate, low-friction mental model of the LLM-assisted C++ real-time infrastructure (threads, frame handoffs, state gating, mutexes, audio mixing, and throttling) before drafting the academic paper.

### Hard Constraints
1. **Medium**: Self-contained interactive HTML + inline SVG progressive walkthroughs. Must open directly via `file://` in any modern browser with **no local server, no bundler, and no npm/build step**.
2. **Low Token Cost**: Use one small reusable HTML/CSS/JS stepper shell paired with compact declarative per-chapter JS data files, built across small independent phases.
3. **Minimal Cognitive Noise**: Reveal one idea at a time on a fixed diagram canvas while dimming inactive components.
4. **Plain-English First**: Never open an explanation with C++ variable names, class/method identifiers, file paths, or memory-ordering jargon. Explain behavior using human role names and plain language; tuck exact C++ symbols behind an opt-in reference toggle.
5. **Code-Faithful Honesty**: Surface real-time design trade-offs, stale-documentation mismatches, and subtle concurrency quirks via opt-in collapsed badges — never hide them, and never clutter the first-pass narrative with them.
6. **Language & Paper Reuse**: English explainers, accompanied by a lightweight Paper Mapping appendix (Section 5) linking each explainer chapter to candidate paper figures and an optional clean SVG export.

---

## 2. Anti-Cognitive-Noise Pedagogical Laws

Future sessions implementing the explainers **must** enforce these five rules on every chapter and every step:

1. **Fixed Canvas, Spotlight Reveal**
   - Each chapter renders **one stable SVG diagram** (`viewBox`) across all steps — nodes never jump or reflow between steps.
   - Inactive components and edges dim to `~15–20%` opacity (`opacity: 0.18`).
   - Each step introduces or spotlights at most **1–2 new elements** and the active data/control arrows connecting them.

2. **Strict 3-Tier Progressive Disclosure per Step**
   - **Tier 1 — Surface Concept (Always Visible)**: 1–3 crisp plain-English sentences using consistent human role names:
     - *Camera Reader* (frame capture thread)
     - *Gaze Worker* (distraction inference thread)
     - *Fatigue Worker* (yawning, landmark, eye-closure, and head-droop thread)
     - *Main Supervisor* (lifecycle & UI loop thread)
     - *Keyboard Listener* (non-blocking console key poller thread)
     - *Alert Coordinator* (shared thread-safe rule evaluator & state holder)
     - *Audio Mixer* (continuous low-latency audio streaming thread)
     - *Telegram Messenger* (short-lived detached HTTP sender threads)
     - **Strict rule**: Zero C++ identifiers (`snake_case` / `CamelCase`), zero file paths, and zero `std::memory_order` jargon in Tier 1.
   - **Tier 2 — Implementation Note / Caveat (1-Click Collapsed Badge)**: Collapsed by default behind a subtle `"Implementation Note"` or `"Concurrency Caveat"` toggle. Written in plain human prose explaining real-time engineering trade-offs, doc-vs-code drift, or subtle synchronization edge cases.
   - **Tier 3 — Exact C++ Reference (2nd Opt-In Collapsed Toggle)**: Collapsed by default behind `"C++ Source Reference"`. Only here do exact file paths, line ranges, class/method names, and config constants appear.

3. **Familiar Algorithm Anchors First**
   - Chapter 1 places the four familiar Python algorithms onto the canvas *before* introducing threads, drop-oldest mailboxes, or mutex locks.

4. **One Consistent Visual Grammar Across All Chapters**
   - Reuse the exact same small visual legend in every chapter:
     - **Rounded Swimlane / Box**: Persistent OS thread
     - **Double-Walled Tray**: Single-slot drop-oldest mailbox
     - **Padlock Badge**: Mutex-protected shared region
     - **Diamond / Pill**: Atomic cross-thread flag or state variable
     - **Solid Arrow**: Synchronous call or direct frame handoff
     - **Dashed Arrow**: Fire-and-forget detached background thread or async notification

5. **Concrete Micro-Traces Over Abstract Concurrency Rules**
   - Chapters covering concurrency, state transitions, locking, or audio buffering (Chapters 2–5) pair the structural **Walkthrough** with a 4–6 step **Scenario Trace** (e.g., *"Fast Camera, Busy Worker"*, *"Drowsiness Preempts Distraction"*) running inside the exact same stepper UI.

---

## 3. Token-Frugal Reusable Architecture

### Planned Directory Layout (For Future Implementation)

```text
morphed_system4/distraction/modular/docs/explainer/
├── index.html                        # ~70 lines: chapter tabs, Walkthrough/Trace mode toggle, SVG stage, 3-tier caption card, Export SVG button
├── style.css                         # ~110 lines: calm high-contrast light palette, dim/spotlight transitions, shared visual grammar styles
├── engine.js                         # ~150 lines: pure vanilla SVG renderer, Prev/Next + keyboard navigation, 3-tier toggles, clean SVG export
└── chapters/
    ├── ch1_thread_map.js             # ~110 lines: Chapter 1 data
    ├── ch2_mailboxes.js              # ~120 lines: Chapter 2 data
    ├── ch3_lifecycle_calibration.js  # ~140 lines: Chapter 3 data
    ├── ch4_alerts_locking.js         # ~145 lines: Chapter 4 data
    └── ch5_low_level_runtime.js      # ~130 lines: Chapter 5 data
```

### Why `<script src="...">` Over `fetch(*.json)`
- Opening `index.html` directly via `file://` in Chrome/Edge/Firefox triggers CORS restrictions on `fetch()`.
- Having each `chapters/chX_*.js` file assign its declarative object to `window.EXPLAINER_CHAPTERS.push(...)` guarantees **zero CORS errors**, zero local server setup, and instant offline viewing by simply double-clicking `index.html`.

### Declarative Chapter Schema

Each chapter file defines a pure data object following one compact schema:

```js
window.EXPLAINER_CHAPTERS = window.EXPLAINER_CHAPTERS || [];
window.EXPLAINER_CHAPTERS.push({
  id: "ch1",
  title: "1. Thread Map",
  subtitle: "From one Python loop to decoupled real-time workers",
  viewBox: "0 0 960 540",
  legend: ["thread", "mailbox", "mutex", "atomic", "async"],
  elements: [
    // Static SVG primitives positioned once on the canvas:
    // { id, kind: "lane"|"box"|"mailbox"|"pill"|"edge"|"badge", x, y, w, h, label, sublabel, icon, from, to, dashed, points }
  ],
  walkthrough: [
    {
      title: "Step title in plain English",
      show: ["el_1", "el_2"],       // Elements visible (all previously shown stay softly visible unless omitted)
      highlight: ["el_2"],          // Elements at 100% spotlight; unhighlighted shown elements dim to ~0.18–0.35
      edgeFlow: ["edge_1"],         // Animated or accented arrows active on this step
      concept: "Tier 1: 1-3 plain-English sentences. Zero C++ identifiers or file paths.",
      note: "Tier 2 (optional): Collapsed Implementation Note / Caveat in plain English.",
      codeRef: "Tier 3 (optional): Exact file paths, line numbers, and C++ symbol names."
    }
  ],
  traces: [
    {
      id: "trace_id",
      label: "Scenario Trace: ...",
      steps: [ /* Same step object shape as walkthrough */ ]
    }
  ]
});
```

### Line & Token Budgets
- **Shared Shell** (`index.html` + `style.css` + `engine.js`): **≤ 350 lines total**.
- **Each Chapter Data File**: **≤ 150 lines** (keep element coordinates clean on a 10px/20px grid; keep `concept` under ~45 words per step).
- **Entire 5-Chapter Suite**: **~1,000 lines total**, easily implemented across 3–4 small, low-cost turns.

---

## 4. Detailed 5-Chapter Progressive Curriculum

> All paths below are relative to `morphed_system4/distraction/modular/` unless prefixed with a repository-root folder (`distraction/`, `ear_mar/`, etc.).

---

### Chapter 1: Python-to-C++ Anchor & Thread Map

- **Learning Goal**: Bridge the engineers' single-loop Python mental model to the 5 persistent threads (plus short-lived alert threads) in the C++ runtime without diving into locks or buffers yet.
- **Visual Diagram Layout** (`960 x 540`):
  - Left column: *Camera Reader* thread + *Keyboard Listener* thread.
  - Center-top lane: *Gaze Worker* thread (containing *Image Normalizer*, *Spatial Backbone*, *Temporal Sequence Memory*).
  - Center-bottom lane: *Fatigue Worker* thread (containing *Full-Frame Yawn Detector*, *68-Point Face Finder*, *Eye Closure & PERCLOS*, *Head Pitch Droop*).
  - Top/Right supervisory rail: *Main Supervisor* thread, *Alert Coordinator*, *Audio Mixer* thread, and dashed *Telegram Messenger* bursts.
- **Step-by-Step Walkthrough (6 Steps — Tier-1 Plain English)**:
  1. **Where Your Python Algorithms Live**: In the original Python prototypes, one loop read a frame and ran every detector sequentially. In C++, those exact four algorithms still exist, split into two independent analysis groups: road-gaze distraction vs. physiological fatigue.
  2. **Two Parallel Worker Threads**: Instead of waiting for eye and posture analysis before checking road gaze, the system runs two dedicated worker threads side-by-side — the *Gaze Worker* (ResNet-18 + 20-frame LSTM) and the *Fatigue Worker* (Yawn neural net, 68-landmark face finder, eye closure/PERCLOS, and head droop).
  3. **Decoupled Camera Reader**: A dedicated *Camera Reader* thread continuously pulls frames from the webcam or video file so slow neural or landmark passes never stall frame capture.
  4. **Main Supervisor & Keyboard Listener**: A *Main Supervisor* thread coordinates startup, shutdown, and optional debug display, while a tiny *Keyboard Listener* thread watches for movement (`m`), calibration (`c`), and quit (`q`) commands without blocking video processing.
  5. **Shared Alert Coordinator & Audio Mixer**: Both analysis workers send their frame results to a shared *Alert Coordinator*, which dispatches warning sounds to a dedicated always-on *Audio Mixer* thread.
  6. **Fire-and-Forget Remote Notifications**: When a distraction or drowsiness alert triggers, the coordinator spawns a short-lived *Telegram Messenger* background thread to send the HTTP alert and detach immediately, keeping real-time detection moving.
- **Scenario Trace**: *None needed* — Chapter 1 establishes the static map; dynamic traces start in Chapter 2.
- **Tier-2 Opt-In Implementation Notes / Caveats**:
  - *Yawn Detector Shift*: Older project documentation describes a geometric lip-ratio yawn detector, and that geometric module is still compiled into the project. However, the active runtime runs a dedicated full-frame neural network for yawning independently of face-landmark detection.
  - *Startup Thread Count*: In normal operation, 5 long-lived threads run continuously (*Main Supervisor*, *Camera Reader*, *Gaze Worker*, *Fatigue Worker*, *Keyboard Listener*), plus the internal *Audio Mixer* thread initialized on startup.
- **Tier-3 Opt-In C++ Source References**:
  - Thread orchestration: `src/PipelineCoordinator.cpp` (`run()`, L210–284; `gaze_worker_loop()`, L363–429; `somnolence_worker_loop()`, L431–551).
  - Camera capture thread: `src/CameraCapture.cpp` (`start()`, L45; `capture_loop()`, L71–114).
  - Console polling thread: `src/ConsoleController.cpp` (`start()`, L47; `listener_loop()`, L63–101).
  - Audio mixer thread: `src/SoundPlayer.cpp` (`AudioStreamEngine::start()`, L342–388).
  - Detached Telegram threads: `src/AlertManager.cpp` (L177–179, L263–265, L331–333).
  - Active yawn CNN vs. legacy MAR: `include/YawnDetector.hpp`, `CMakeLists.txt` (L56–57).

---

### Chapter 2: Frame Handoff & Real-Time Freshness (Drop-Oldest Mailboxes)

- **Learning Goal**: Explain why standard FIFO frame queues create dangerous lag in driver monitoring, and how single-slot drop-oldest mailboxes keep every worker on the freshest possible frame.
- **Visual Diagram Layout** (`960 x 540`):
  - Left: *Camera Reader* (with frame pacing badge).
  - Center: 2 (or 3 in Debug mode) independent **Double-Walled Mailbox Trays** — one for *Gaze Worker*, one for *Fatigue Worker* (and an optional third tray for the *Debug Window*).
  - Right: *Gaze Worker* and *Fatigue Worker* consuming independently at their own speeds.
- **Step-by-Step Walkthrough (5 Steps — Tier-1 Plain English)**:
  1. **Why Queues Fail for Safety**: If a camera delivers 30 frames per second while a detector takes 50 milliseconds per frame, a traditional waiting queue grows longer every second — meaning alarms eventually fire on stale video from seconds ago.
  2. **The Drop-Oldest Mailbox**: Between the camera and each worker sits a single-slot mailbox tray. When a new frame arrives before the worker picks up the old one, the new frame simply overwrites it.
  3. **Independent Deep Copies per Consumer**: Because image matrices share underlying pixel memory by default, the *Camera Reader* creates an isolated copy for each mailbox so one worker's preprocessing can never corrupt another worker's frame.
  4. **Sleep Until Fresh (With Shutdown Escape)**: Workers sleep efficiently when no new frame is waiting, waking instantly when a frame arrives or up to every 100 milliseconds to check whether the program is shutting down.
  5. **Live Webcam vs. Video File Pacing**: For a live webcam, the *Camera Reader* runs at the camera's natural cadence; for recorded video files, it measures elapsed processing time and sleeps between reads to replay at the video's native frame rate (defaulting to 30 FPS if metadata is invalid).
- **Scenario Trace (5 Steps): `"Fast Camera, Busy Worker"`**:
  1. Camera deposits **Frame #1** into both mailboxes and wakes both workers.
  2. *Gaze Worker* grabs **Frame #1** and begins a ~45 ms neural pass.
  3. While *Gaze Worker* is still busy, Camera deposits **Frame #2** into the gaze mailbox (slot now holds #2).
  4. Next camera tick arrives: **Frame #3** overwrites **Frame #2** in the slot (#2 is cleanly discarded without blocking the camera).
  5. *Gaze Worker* finishes #1, immediately pulls **Frame #3**, and stays synchronized with live reality with zero queue lag.
- **Tier-2 Opt-In Implementation Notes / Caveats**:
  - *Memory Bandwidth Trade-off*: Cloning every frame once per consumer (2 full-frame copies in Release builds; 3 copies when the Debug View is compiled in) spends a small amount of memory copy time to guarantee zero cross-thread pixel data races.
  - *Different Worker Cadences*: Because each worker owns its own mailbox, a temporary slowdown in face-landmark detection never slows down road-gaze inference, and vice versa.
- **Tier-3 Opt-In C++ Source References**:
  - Single-slot drop-oldest container: `include/ThreadSafeMailbox.hpp` (`put()`, L30–37; `get_latest()`, L45–58; `stop()`, L63–69).
  - Deep clone broadcast & video FPS pacing: `src/CameraCapture.cpp` (`capture_loop()`, L72–114).
  - Mailbox wiring: `src/PipelineCoordinator.cpp` (L114–123, L380, L471).

---

### Chapter 3: The 5-State Lifecycle & Calibration Takeover

- **Learning Goal**: Clarify how the system gates detections safely across driving vs. parked states, how calibration hijacks the *Fatigue Worker* in place without spawning a new thread, and how resets avoid race conditions.
- **Visual Diagram Layout** (`960 x 540`):
  - Top: 5-node state machine pill diagram (*Stationary Uncalibrated*, *Moving Uncalibrated*, *Calibrating*, *Stationary Calibrated*, *Moving Calibrated*).
  - Center: *Main Supervisor* updating the shared atomic state flag.
  - Bottom: *Gaze Worker* (gated on Moving + Calibrated) and *Fatigue Worker* (running either active monitoring OR sequential calibration takeover).
- **Step-by-Step Walkthrough (6 Steps — Tier-1 Plain English)**:
  1. **Safe Stationary Startup**: By default, the system boots parked and uncalibrated with all alerts disabled, so no alarms fire before the driver calibrates their normal open-eye and closed-eye baselines.
  2. **Moving Without Calibration**: Switching to moving mode before calibrating still keeps alerts suppressed and warns the driver to stop and calibrate first (a startup flag can launch directly into this moving-uncalibrated state).
  3. **Calibrating Gate & Lockout**: Calibration can only start while stationary. Pressing calibrate while moving—or pressing it again while calibration is already running—is ignored. Movement toggles are also locked out during calibration.
  4. **In-Place Calibration Takeover**: Rather than spawning a separate calibration thread, the *Fatigue Worker* itself temporarily pauses normal monitoring and runs the entire 2-phase audio-guided sequence top-to-bottom: Phase 1 (neutral posture, eyes open) and Phase 2 (eyes gently closed), repeating 4-second beep windows until at least 15 valid face samples are collected per phase, then applying the new eye and head-pitch baselines atomically before returning to stationary-calibrated mode.
  5. **Gaze Worker Gating**: Whenever the shared state is anything other than *Moving & Calibrated*, the *Gaze Worker* pulls frames to keep its mailbox drained but skips all neural inference.
  6. **Mutex-Free Reset on Stopping**: When the driver stops the car, the *Main Supervisor* cannot safely wipe the *Gaze Worker*'s sequence memory directly while it might be mid-inference. Instead, it flips a lightweight atomic "reset yourself" flag; the *Gaze Worker* clears its own sequence buffer on its very next loop pass.
- **Scenario Trace (6 Steps): `"Driver Session: Startup -> Calibrate -> Drive -> Stop"`**:
  1. System starts in **Stationary (Uncalibrated)**; both workers skip alerting.
  2. User triggers Calibrate (`c`) -> shared state becomes **Calibrating**; *Fatigue Worker* enters the guided Phase 1 & Phase 2 audio loops while *Gaze Worker* discards incoming frames.
  3. Calibration collects ≥15 samples per phase -> applies open/closed eye thresholds and neutral head-pitch baseline -> state becomes **Stationary (Calibrated)**.
  4. User toggles Movement (`m`) -> state becomes **Moving (Calibrated)** -> *Gaze Worker* fills its 20-frame sequence buffer and *Fatigue Worker* enables PERCLOS and alert reporting.
  5. User toggles Movement (`m`) to park -> state returns to **Stationary (Calibrated)**; *Main Supervisor* resets the *Alert Coordinator* and raises the atomic gaze-reset flag.
  6. *Gaze Worker* sees the flag on its next wake cycle, resets its sequence ring buffer cleanly on its own thread, and resumes idle waiting.
- **Tier-2 Opt-In Implementation Notes / Caveats**:
  - *Subtle Concurrency Caveat (`clear_perclos`)*: While the *Gaze Worker* resets cleanly via an atomic handoff flag, stopping the car from *Moving Calibrated* calls the eye-closure buffer reset directly from the *Main Supervisor* thread without a mutex, creating a potential race if the *Fatigue Worker* is mid-frame. This is worth verifying or wrapping in an atomic flag like gaze reset before claiming zero data races in the paper.
  - *Idle Work Asymmetry*: When stationary or uncalibrated, the *Gaze Worker* skips all computation, whereas the *Fatigue Worker* still runs the yawn neural net and face landmark detector on incoming frames (only withholding alert updates and clearing PERCLOS) so debug telemetry stays visible.
  - *Calibration Audio Sequence Detail*: Phase 2 plays both the Phase 2 spoken instructions and the 1-second preparation tone before starting its measurement beeps (even though older README notes say the voice instruction replaced the tone).
- **Tier-3 Opt-In C++ Source References**:
  - 5-state enum: `include/Types.hpp` (`SystemState`, L64–70).
  - Console command state transitions & reset flag: `src/PipelineCoordinator.cpp` (`handle_console_command()`, L286–361; `gaze_reset_requested_` check, L376–395).
  - Calibration takeover & baseline application: `src/PipelineCoordinator.cpp` (L444–486), `src/Calibration.cpp` (`measure_window()`, L25–76; `run_calibration()`, L80–220).
  - Unsynchronized cross-thread call: `src/PipelineCoordinator.cpp` (L307 calling `ear_detector_.clear_perclos()`).

---

### Chapter 4: Multimodal Alert Fusion, Locks & Priority Preemption

- **Learning Goal**: Show how two asynchronous workers safely share one decision engine via short mutex critical sections, how distraction vs. drowsiness rules work, and why drowsiness always wins.
- **Visual Diagram Layout** (`960 x 540`):
  - Left: Two independent input paths (*Gaze Worker* -> 20-frame vote window + 2.0 s timer; *Fatigue Worker* -> Strong cues + Light cues + 0.2 s yawn suppression filter).
  - Center: *Alert Coordinator* wrapped in a **Padlock Boundary** (separating fast inside-lock evaluation from slow outside-lock side effects).
  - Right: *Audio Mixer* (blocking distraction warning vs. non-blocking escalating drowsiness pulses) and detached *Telegram Messenger* threads.
- **Step-by-Step Walkthrough (6 Steps — Tier-1 Plain English)**:
  1. **One Shared Turnstile Lock**: Both workers report into the same *Alert Coordinator*, which uses a single mutex lock so simultaneous gaze and fatigue updates never corrupt shared driver state or cue histories.
  2. **Distraction Rule (Spatial-Temporal + Duration)**: Inside the lock, gaze predictions fill a 20-frame sliding majority-vote window (filtering or binarizing by confidence threshold). If the majority vote is "not looking at the road" continuously for at least 2.0 seconds, driver state becomes *Distracted*.
  3. **Drowsiness Cue Fusion & Yawn Suppression**: On the fatigue path, either **1 Strong cue** (micro-sleep > 500 ms or High PERCLOS ≥ 15%) OR **2 Light cues within a 2-minute window** (head droop ≤ −25.5° for ≥ 1.5 s, or sustained yawn ≥ 1.0 s) immediately elevates state to *Drowsy*. If the driver has been yawning for ≥ 0.2 s, eye-closure and head-droop cues are suppressed so squinting during a yawn does not trigger false micro-sleep or droop alarms.
  4. **Compute Inside the Lock, Act Outside the Lock**: To prevent one worker from freezing the other during slow I/O, the *Alert Coordinator* computes state changes and copies warning metadata under the lock, **releases the lock**, and only then logs, spawns Telegram alerts, or triggers audio.
  5. **Drowsiness Priority & Escalating Cadence**: Drowsiness strictly overrides distraction. While *Drowsy* (or while drowsiness audio is still playing), gaze distraction evaluation is ignored, any active distraction sound is cut off immediately, and drowsiness pulses play non-blockingly at ~63.5 beats per minute across 5 escalating loudness steps (−21 dB up to −3 dB) until eyes, head pitch, and yawning return to normal.
  6. **Non-Blocking Remote Alerts**: Telegram notifications run on detached threads with short network timeouts (3 s connect, 5 s total) so an offline cellular connection never stalls audio or video workers.
- **Scenario Trace (6 Steps): `"Drowsiness Preempts an Active Distraction Warning"`**:
  1. *Gaze Worker* reports sustained off-road gaze (≥ 2.0 s); acquires lock, sets state to *Distracted*, releases lock, and starts blocking playback of the distraction warning.
  2. Mid-playback, *Fatigue Worker* detects a 550 ms eye closure (Strong micro-sleep cue) and acquires the coordinator lock.
  3. Coordinator elevates state to *Drowsy* and releases lock.
  4. Coordinator immediately stops active audio (cutting off the distraction clip) and triggers the first −21 dB drowsiness beep.
  5. Stopping audio immediately unblocks the *Gaze Worker*, which clears its 20-frame gaze window and skips further distraction checks while *Drowsy* remains active.
  6. Once the driver's eyes reopen and posture recovers, the next fatigue update clears *Drowsy* back to *Safe* and stops audio.
- **Tier-2 Opt-In Implementation Notes / Caveats**:
  - *Intentional Blocking Trade-off on Distraction Audio*: Distraction alerts play synchronously on the *Gaze Worker* thread, pausing new gaze inference for the duration of the warning clip (unless preempted by drowsiness), then wiping the 20-frame window so evaluation starts fresh afterward.
  - *Disabled Medium PERCLOS Light Cue*: Although older documentation lists Medium PERCLOS (7.5%–15.0%) as a Light cue, it is intentionally disabled in the eye-closure detector; only High PERCLOS (≥ 15.0%) emits a Strong cue.
  - *Unused Lost-Face Drowsiness Ticker*: The *Alert Coordinator* contains a helper method meant to keep ticking drowsiness beeps if face landmarks disappear, but it is unused in the main loop because the *Fatigue Worker* calls `update_somnolence` on every frame (even during face loss, using gradual decay).
- **Tier-3 Opt-In C++ Source References**:
  - Mutex & state evaluation: `include/AlertManager.hpp` (L88–99), `src/AlertManager.cpp` (`update_gaze()`, L83–204; `register_cue_locked()`, L206–240; `update_somnolence()`, L272–339; `try_consume_drowsiness_cooldown()`, L360–373; `dispatch_auditory_alert()`, L375–388; `is_drowsiness_alert_reproducing_locked()`, L413–421).
  - Stateless evaluation logic: `include/AlertLogic.hpp`, `src/AlertLogic.cpp` (`evaluate_distraction_mode()`, L5–49; `evaluate_somnolence_cues()`, L51–59).
  - Yawn suppression & fatigue update wiring: `src/PipelineCoordinator.cpp` (L490–523).
  - Default alert/cue thresholds: `include/Types.hpp` (`AlertConfig` L170–177, `AuditoryAlertConfig` L158–165, `HeadOrientationConfig` L191–195, `EarConfig` L220–231, `YawnConfig` L246–253).
  - Telegram timeouts & async thread: `src/TelegramNotifier.cpp` (L59–60), `src/AlertManager.cpp` (L177–179, L331–333).

---

### Chapter 5: Low-Level Real-Time Mechanics (Audio Mixer, Throttling, Memory & Build Modes)

- **Learning Goal**: Demystify the low-level systems engineering choices LLMs introduced to keep CPU usage bounded, eliminate audio startup glitches, avoid memory jitter, and strip debug overhead in production.
- **Visual Diagram Layout** (`960 x 540`):
  - Top-left: **Pre-cached PCM Map -> 6x10 ms Ring Buffer -> High-Priority Mixer Thread** (Audio Engine).
  - Top-right: **Face-Loss CPU Throttle** (500 ms face probe gate vs. continuous full-frame Yawn CNN).
  - Bottom-left: **Pre-allocated Hot-Path Buffers** (reusable image blobs, flat LSTM circular buffer).
  - Bottom-right: **Compile-Time Split (`DEBUG_VIEWS` ON vs. OFF)** showing how the 3rd mailbox, HUD rendering, and latency stats vanish in Release.
- **Step-by-Step Walkthrough (5 Steps — Tier-1 Plain English)**:
  1. **Always-On Streaming Audio Mixer**: Instead of opening the OS audio device from scratch on every alert, startup pre-decodes all WAV clips to 48 kHz stereo memory and launches a time-critical mixer thread feeding a 60 ms ring buffer (six 10 ms blocks) into Windows `waveOut` or a persistent Linux `aplay` stream. Playing a beep is just swapping a memory pointer — giving sub-10 ms response and zero clicks.
  2. **Instant Generation-Ticket Cancellation**: Every sound playback increments an integer playback counter. Stopping or preempting an alert clears the active sample pointer, increments the counter, and wakes any waiting thread — guaranteeing instant interruption without holding the mixer lock during waits.
  3. **500 ms Face-Loss CPU Throttling**: Scanning an entire image for a face is expensive. Once the face disappears, the *Fatigue Worker* throttles full-frame face searches to once every 500 milliseconds to leave CPU headroom for the *Gaze Worker* (while the fast full-frame Yawn CNN continues running on every frame).
  4. **Zero-Allocation Hot Paths**: Both neural detectors, the image normalizer, and the face grayscale converter pre-allocate their working tensors and flat circular sequence buffers during initialization so steady-state frame loops avoid repeated heap allocations.
  5. **Debug Telemetry vs. Zero-Cost Release**: Compiling with debug views enabled adds a 3rd mailbox feeding an OpenCV HUD window (drawn on the *Main Supervisor* thread) plus console logs and end-of-run latency percentiles. In Release builds, all visualizer hooks compile away into empty no-ops and the main loop sleeps 50 ms per check.
- **Scenario Trace (4 Steps): `"Audio Preemption Inside the 10 ms Ring Buffer"`**:
  1. Distraction alert activates: active pointer points to pre-cached distraction clip with ticket `#4`, and *Gaze Worker* waits on the condition variable for ticket `#4` to finish.
  2. Every 10 ms, the high-priority *Audio Mixer* copies the next 480 stereo frames with gain scaling into the ring buffer.
  3. Drowsiness preempts: `stop` bumps the ticket to `#5` and clears the pointer; `play` immediately sets the pointer to the drowsiness beep with ticket `#6`.
  4. *Gaze Worker* wakes immediately because ticket `!= 4`, while the very next 10 ms audio block streams the drowsiness beep seamlessly without reopening the audio device.
- **Tier-2 Opt-In Implementation Notes / Caveats**:
  - *Audio Implementation Drift*: Older README notes mention Windows `PlaySoundA` and spawning per-clip `aplay` commands, whereas the current runtime implements a custom 48 kHz continuous ring-buffer streaming engine on Windows and Linux (macOS still falls back to `afplay`). Also note that two alert sound paths (`distraction_warning.wav` and `single-beep-1850-Hz-975-Hz.wav`) decode and cache lazily on first play rather than in the initial pre-warm list.
  - *Headless Release Timing*: In Release builds (`DEBUG_VIEWS` disabled), the visualizer stub sleeps for 50 ms on each supervisory loop tick, capping console-command polling responsiveness at ~20 Hz while consuming virtually zero CPU.
- **Tier-3 Opt-In C++ Source References**:
  - Continuous 48 kHz ring-buffer mixer & pre-warm cache: `src/SoundPlayer.cpp` (constants L36–40; `AudioStreamEngine`, L226–494; `warmup()`, L503–519).
  - Non-locking blocking wrapper: `src/AudioManager.cpp` (`play_blocking()`, L20–26; `stop()`, L28–30).
  - Throttled lost-face probing: `include/FaceLandmarkDetector.hpp` (L62, L65), `src/PipelineCoordinator.cpp` (L497–508), `include/Types.hpp` (L185).
  - Pre-allocated buffers: `include/GazeDetector.hpp` (L70–76), `include/YawnDetector.hpp` (L58–61), `include/Preprocessor.hpp` (L46–51).
  - Compile-time `DEBUG_VIEWS` toggle & Release 50 ms sleep: `CMakeLists.txt` (L27–33, L68–70), `include/DebugVisualizer.hpp` (L15–157), `src/PipelineCoordinator.cpp` (L196–207, L265–276).

---

## 5. Lightweight Paper Mapping & Doc-vs-Code Sync Checklist

### Explainer-to-Paper Figure Mapping
| Explainer Chapter | Candidate Paper Section | Candidate Paper Figure | Simplification Notes (Using Clean SVG Export) |
| :--- | :--- | :--- | :--- |
| **Ch. 1: Thread Map** | System Architecture / Real-Time Pipeline | **Fig. 1**: Multi-Threaded System Block Diagram | Export Step 6 SVG; remove interactive badges, keep 5 persistent lanes + Alert/Audio sinks in monochrome/gray-scale-safe strokes. |
| **Ch. 2: Drop-Oldest Mailboxes** | Real-Time Frame Synchronization | **Fig. 2**: Zero-Lag Single-Slot Mailbox Pattern | Export Trace Step 4 or Walkthrough Step 2; highlight overwrite/drop-oldest semantics vs. FIFO lag. |
| **Ch. 3: Lifecycle & Calibration** | Operational Protocol & Baseline Calibration | **Fig. 3**: 5-State Safety FSM & 2-Phase Calibration Timeline | Export state-machine subset or convert calibration phases into a compact horizontal timeline with timing bars. |
| **Ch. 4: Alert Fusion & Priority** | Multimodal Fusion & Warning Arbitration | **Fig. 4**: Cue Fusion Tree & Priority Preemption Flow | Export Walkthrough Step 5; clearly show Strong vs. 2-Light-in-120s fusion, 0.2 s yawn suppression, and Drowsy > Distracted preemption. |
| **Ch. 5: Low-Level Mechanics** | Implementation Details & Latency Optimization | *Usually prose / table* (optional **Fig. 5** if space permits) | Cite exact 48 kHz/10 ms audio ring buffer, 500 ms face-loss throttle, and pre-allocated buffers in Implementation Details. |

> **SVG Export Note**: Including a lightweight **"Export Clean SVG"** button in `engine.js` lets paper authors download the current diagram state as a crisp vector `.svg` file (stripping UI controls/hover chrome) for direct cleanup in Inkscape/Illustrator/LaTeX.

### Doc-vs-Code Sync Checklist (Critical for Paper Accuracy)
Before writing numbers or architectural claims into the paper, authors should note where `README.md` or legacy headers diverge from the compiled C++ implementation:

1. **Yawning Algorithm**: Active runtime uses **CNN yawning inference** (`YawnDetector`, ONNX `112x112` NHWC RGB, `prob_threshold = 0.80`, `sustained_duration = 1.0 s`), **not** geometric MAR (`MarDetector`, 0.55 / 4.0 s), even though `MarDetector.cpp` remains in `CMakeLists.txt`.
2. **Head Droop Threshold**: Actual default in `Types.hpp` (`HeadOrientationConfig::pitch_droop_thresh`) is **`-25.5` degrees** (calibrated for the 5.48° BU-HT MAE) sustained for **`1.5 s`**, whereas `README.md` and older code comments mention `-20°`.
3. **Gaze Confidence Default**: Actual C++ default is **`0.54`** (`PipelineCoordinator::Options::conf_threshold` and `AlertConfig::min_confidence`), whereas `README.md` and `print_help()` still display `0.70`.
4. **PERCLOS Cue Rules**: Only **High PERCLOS** (`>= 15.0%` over 30 s, Strong cue) is active; **Medium PERCLOS** (`7.5%–15.0%`, Light cue) is explicitly disabled in `EarDetector` even though `README.md` still lists it.
5. **Calibration Phase 2 Tone**: `Calibration.cpp` plays the 1-second preparation tone in **both** Phase 1 and Phase 2 (whereas `README.md` states Phase 2 omits the tone).
6. **Audio Engine**: Windows and Linux run a custom continuous **48 kHz stereo 6x10 ms ring-buffer** `AudioStreamEngine` (`waveOut` on Windows, `aplay` raw pipe on Linux), not `PlaySoundA` or per-clip `aplay` invocations as `README.md` describes.
7. **Concurrency Caveat to Audit**: `PipelineCoordinator::handle_console_command` invokes `ear_detector_.clear_perclos()` from the main thread on `MovingCalibrated -> StationaryCalibrated` without synchronization against `somnolence_worker_`; either fix via an in-band atomic flag (matching `gaze_reset_requested_`) or avoid over-claiming lock-freedom on that transition.

---

## 6. Token-Frugal Incremental Build Roadmap

Future implementation sessions should build the visualizer suite in small, verifiable increments to avoid large monolithic generations:

- **Phase 1 — Reusable Shell + Chapter 1**:
  - Create `index.html`, `style.css`, `engine.js`, and `chapters/ch1_thread_map.js`.
  - Verify step navigation, dim/spotlight transitions, and 3-tier collapse toggles via `file://`.
- **Phase 2 — Frame & Lifecycle Chapters**:
  - Create `chapters/ch2_mailboxes.js` and `chapters/ch3_lifecycle_calibration.js` (including their Scenario Traces).
- **Phase 3 — Concurrency & Low-Level Chapters**:
  - Create `chapters/ch4_alerts_locking.js` and `chapters/ch5_low_level_runtime.js` (including their Scenario Traces).
- **Phase 4 — Polish & Paper Export (Optional)**:
  - Add clean SVG export serialization in `engine.js` and audit every Tier-1 `concept` string to verify zero leaked C++ symbols or file paths.

---

## 7. Verification Criteria (For Future Implementation Sessions)

1. **Zero-Setup Execution**: Double-clicking `morphed_system4/distraction/modular/docs/explainer/index.html` opens cleanly over `file://` with zero browser console errors.
2. **Low-Cognitive-Noise Audit**: Every step's always-visible Tier-1 `concept` text contains **zero C++ identifiers, zero header/source filenames, and ≤ 3 short sentences**.
3. **Traceability Audit**: Every Tier-3 reference corresponds to verified symbols and lines in `morphed_system4/distraction/modular/`.
4. **Navigation Completeness**: Both Walkthrough steps and Scenario Traces advance/rewind smoothly via UI buttons and Left/Right arrow keys, and resetting or switching chapters restores step 0 cleanly.
