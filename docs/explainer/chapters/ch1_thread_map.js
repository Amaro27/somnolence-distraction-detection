window.EXPLAINER_CHAPTERS = window.EXPLAINER_CHAPTERS || [];
window.EXPLAINER_CHAPTERS.push({
  id: "ch1",
  title: "1. Thread Map",
  subtitle: "From one Python loop to parallel C++ workers, IMU & alerts (MS6)",
  viewBox: "0 0 960 540",
  legend: ["thread", "mutex", "sync", "async", "debug"],
  elements: [
    // Left column: Keyboard, IMU, Camera Reader, Python Reference Anchor
    { id: "kbd_lane", kind: "lane", x: 20, y: 18, w: 180, h: 68, label: "Keyboard Listener", sublabel: "m / c / r / q", tag: "THREAD" },
    { id: "imu_lane", kind: "lane", x: 20, y: 100, w: 180, h: 86, label: "IMU Motion Worker", sublabel: "MPU-6050 20 Hz (Linux)", tag: "THREAD" },
    { id: "cam_lane", kind: "lane", x: 20, y: 202, w: 180, h: 132, label: "Camera Reader", sublabel: "Webcam or video file", tag: "THREAD" },
    { id: "py_loop", kind: "box", x: 20, y: 352, w: 180, h: 168, label: "Python Prototype Loop", sublabel: "1 frame at a time:\nGaze -> Landmarks ->\nEyes -> Head -> Alerts", pythonOrigin: "morphed_system2" },

    // Top bar: Main Supervisor
    { id: "main_lane", kind: "lane", x: 230, y: 18, w: 696, h: 60, label: "Main Supervisor", sublabel: "Startup, driving/parked mode switches, optional debug HUD & MP4 recording", tag: "THREAD" },

    // Center-top: Gaze Worker + 3 stages
    { id: "gaze_lane", kind: "lane", x: 230, y: 100, w: 440, h: 162, label: "Gaze Worker", sublabel: "Road-gaze pipeline (10f pre-opt / 20f unified)", tag: "THREAD" },
    { id: "norm_box", kind: "box", x: 244, y: 150, w: 124, h: 96, label: "Image Prep", sublabel: "Contrast + 224x224" },
    { id: "backbone_box", kind: "box", x: 388, y: 150, w: 124, h: 96, label: "ResNet-18", sublabel: "Spatial features", pythonOrigin: "distraction/" },
    { id: "lstm_box", kind: "box", x: 532, y: 150, w: 124, h: 96, label: "10-Frame LSTM", sublabel: "1-layer pre-opt (or 20f)", pythonOrigin: "distraction/" },

    // Center-bottom: Fatigue Worker + 4 stages
    { id: "fatigue_lane", kind: "lane", x: 230, y: 288, w: 440, h: 232, label: "Fatigue Worker", sublabel: "CNN yawn, 68-pt landmarks, eyes, head & calibration", tag: "THREAD" },
    { id: "yawn_box", kind: "box", x: 246, y: 338, w: 196, h: 76, label: "Full-Frame Yawn CNN", sublabel: "112x112 (>=0.80, 1.0s)" },
    { id: "face_box", kind: "box", x: 458, y: 338, w: 196, h: 76, label: "68-Point Face Finder", sublabel: "Single-pass landmarks", pythonOrigin: "Dlib 68-pt" },
    { id: "ear_box", kind: "box", x: 246, y: 428, w: 196, h: 76, label: "Eye Closure & PERCLOS", sublabel: "Blinks + 30s closure %", pythonOrigin: "ear_mar/" },
    { id: "head_box", kind: "box", x: 458, y: 428, w: 196, h: 76, label: "Head Pitch Droop", sublabel: "Pitch <= -25.5 deg (1.5s)", pythonOrigin: "head_orientation/" },

    // Right column: Alert Coordinator, Audio Mixer, Telegram Messenger, Debug HUD
    { id: "alert_coord", kind: "box", x: 700, y: 100, w: 226, h: 142, label: "Alert Coordinator", sublabel: "10f mode + 2s timer &\ndrowsiness priority", badge: "mutex" },
    { id: "audio_lane", kind: "lane", x: 700, y: 266, w: 226, h: 98, label: "Audio Mixer", sublabel: "Always-on 48 kHz alerts", tag: "THREAD", tone: "amber" },
    { id: "tg_burst", kind: "box", x: 700, y: 382, w: 226, h: 66, label: "Telegram Messenger", sublabel: "Detached HTTP alert", badge: "async" },
    { id: "hud_rec", kind: "box", x: 700, y: 462, w: 226, h: 58, label: "Debug HUD & MP4", sublabel: "Overlay + 'r' record", badge: "debug" },

    // Non-crossing edges
    { id: "e_kbd_main", kind: "edge", points: [[200, 48], [230, 48]] },
    { id: "e_imu_gaze", kind: "edge", points: [[200, 143], [230, 143]], label: "motion" },
    { id: "e_cam_gaze", kind: "edge", points: [[200, 232], [230, 232]], label: "frames" },
    { id: "e_cam_fatigue", kind: "edge", points: [[200, 305], [230, 305]], label: "frames" },
    { id: "e_main_gaze", kind: "edge", points: [[450, 78], [450, 100]], label: "state" },
    { id: "e_main_alert", kind: "edge", points: [[813, 78], [813, 100]], label: "reset" },
    { id: "e_gaze_alert", kind: "edge", points: [[670, 170], [700, 170]] },
    { id: "e_fatigue_alert", kind: "edge", points: [[670, 330], [685, 330], [685, 214], [700, 214]] },
    { id: "e_alert_audio", kind: "edge", points: [[813, 242], [813, 266]], label: "sound" },
    { id: "e_alert_tg", kind: "edge", points: [[926, 170], [941, 170], [941, 415], [926, 415]], dashed: true, label: "detach", lx: 904, ly: 374 },
    { id: "e_main_hud", kind: "edge", points: [[926, 48], [952, 48], [952, 491], [926, 491]], dashed: true, label: "debug", lx: 904, ly: 456 }
  ],
  walkthrough: [
    {
      title: "1. Where Your Python Algorithms Live",
      show: ["py_loop", "norm_box", "backbone_box", "lstm_box", "yawn_box", "face_box", "ear_box", "head_box"],
      highlight: ["py_loop", "norm_box", "backbone_box", "lstm_box", "yawn_box", "face_box", "ear_box", "head_box"],
      edgeFlow: [],
      concept: "In the original Python prototypes, a single loop processed each frame sequentially through every detector. In C++, those familiar algorithms remain intact alongside a full-frame yawning neural net, split into gaze distraction above and physiological fatigue below."
    },
    {
      title: "2. Two Parallel Analysis Workers",
      show: ["py_loop", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box"],
      highlight: ["gaze_lane", "fatigue_lane", "norm_box", "backbone_box", "lstm_box", "yawn_box", "face_box", "ear_box", "head_box"],
      edgeFlow: [],
      concept: "The Gaze Worker runs contrast prep, spatial features, and a fast 10-frame sequence model (or 20-frame unified model) in parallel with the Fatigue Worker, which scores full-frame yawning independently alongside 68-landmark eye closure and head droop.",
      note: "For Raspberry Pi throughput, the gaze pipeline defaults to a pre-optimized 1-layer 10-frame sequence model, with a startup flag to switch to the 2-layer 20-frame unified model. Full-frame CNN yawning replaced the older geometric lip ratio (though the legacy class stays in the tree).",
      codeRef: "morphed_system6/distraction/modular/src/PipelineCoordinator.cpp (L68-92 build_model_config; L399-587)\ninclude/Types.hpp (L149-155, L193-197, L248-255) | src/YawnDetector.cpp (L112-171)"
    },
    {
      title: "3. Decoupled Camera Reader",
      show: ["py_loop", "cam_lane", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box", "e_cam_gaze", "e_cam_fatigue"],
      highlight: ["cam_lane"],
      edgeFlow: ["e_cam_gaze", "e_cam_fatigue"],
      concept: "A dedicated Camera Reader thread continuously pulls frames from the webcam or video file and passes isolated copies to both workers. Slow neural or landmark steps never stall frame capture.",
      codeRef: "morphed_system6/distraction/modular/src/CameraCapture.cpp (L45 start; L71-114 capture_loop)"
    },
    {
      title: "4. Supervisor, Keyboard & IMU Motion Worker",
      show: ["py_loop", "kbd_lane", "imu_lane", "cam_lane", "main_lane", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box", "e_cam_gaze", "e_cam_fatigue", "e_kbd_main", "e_imu_gaze", "e_main_gaze"],
      highlight: ["kbd_lane", "imu_lane", "main_lane"],
      edgeFlow: ["e_kbd_main", "e_imu_gaze", "e_main_gaze"],
      concept: "The Main Supervisor coordinates startup, mode switches, and shutdown while a Keyboard Listener watches for control keys. On Linux vehicles, an optional IMU Motion Worker also samples a physical motion sensor at 20 Hz to detect driving vs. parking automatically.",
      note: "On Windows or when the no-IMU flag is passed, the hardware sensor driver compiles to a clean stub that exits immediately, leaving manual keyboard switching active.",
      codeRef: "morphed_system6/distraction/modular/src/PipelineCoordinator.cpp (L232-397, L589-689 imu_worker_loop)\ninclude/MPU6050.hpp (L51-205) | src/ConsoleController.cpp (L47, L63-104)"
    },
    {
      title: "5. Shared Alert Coordinator & Audio Mixer",
      show: ["py_loop", "kbd_lane", "imu_lane", "cam_lane", "main_lane", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box", "alert_coord", "audio_lane", "e_cam_gaze", "e_cam_fatigue", "e_kbd_main", "e_imu_gaze", "e_main_gaze", "e_main_alert", "e_gaze_alert", "e_fatigue_alert", "e_alert_audio"],
      highlight: ["alert_coord", "audio_lane"],
      edgeFlow: ["e_gaze_alert", "e_fatigue_alert", "e_alert_audio", "e_main_alert"],
      concept: "Both workers report into one mutex-protected Alert Coordinator, which runs the two-stage distraction filter and drowsiness cue fusion before dispatching sound alerts to an always-on Audio Mixer thread.",
      note: "Counting the Audio Mixer engine initialized at startup, the system runs 7 persistent threads on Linux with IMU enabled (or 6 persistent threads on Windows / with no-IMU).",
      codeRef: "morphed_system6/distraction/modular/src/AlertManager.cpp (L83-204 update_gaze; L272-339 update_somnolence)\nsrc/SoundPlayer.cpp (L342-388 AudioStreamEngine::start)"
    },
    {
      title: "6. Detached Telegram & Optional Debug Recorder",
      show: ["py_loop", "kbd_lane", "imu_lane", "cam_lane", "main_lane", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box", "alert_coord", "audio_lane", "tg_burst", "hud_rec", "e_cam_gaze", "e_cam_fatigue", "e_kbd_main", "e_imu_gaze", "e_main_gaze", "e_main_alert", "e_gaze_alert", "e_fatigue_alert", "e_alert_audio", "e_alert_tg", "e_main_hud"],
      highlight: ["tg_burst", "hud_rec"],
      edgeFlow: ["e_alert_tg", "e_main_hud"],
      concept: "When alerts fire, short-lived Telegram Messenger threads send notifications in the background without blocking detection. In debug builds, the Main Supervisor also renders a live telemetry window and can record annotated MP4 videos on demand.",
      note: "In production Release builds, the debug window and MP4 recorder compile away into zero-cost stubs, so recording keys and flags become harmless no-ops.",
      codeRef: "morphed_system6/distraction/modular/src/TelegramNotifier.cpp (L36-71)\ninclude/DebugVisualizer.hpp (L17-179) | src/DebugVisualizer.cpp (L176-207, L563-598)"
    }
  ],
  traces: []
});
