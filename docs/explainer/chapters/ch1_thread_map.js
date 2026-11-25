window.EXPLAINER_CHAPTERS = window.EXPLAINER_CHAPTERS || [];
window.EXPLAINER_CHAPTERS.push({
  id: "ch1",
  title: "1. Thread Map",
  subtitle: "From one Python loop to parallel C++ workers",
  viewBox: "0 0 960 540",
  legend: ["thread", "mutex", "sync", "async"],
  elements: [
    // Left column: Keyboard Listener, Camera Reader, Python Reference Anchor
    { id: "kbd_lane", kind: "lane", x: 20, y: 18, w: 180, h: 86, label: "Keyboard Listener", sublabel: "Keys: m / c / q", tag: "THREAD" },
    { id: "cam_lane", kind: "lane", x: 20, y: 126, w: 180, h: 168, label: "Camera Reader", sublabel: "Webcam or video file", tag: "THREAD" },
    { id: "py_loop", kind: "box", x: 20, y: 320, w: 180, h: 190, label: "Python Prototype Loop", sublabel: "1 frame at a time:\nGaze -> Yawn -> Landmarks\n-> Eyes -> Head -> Alerts", pythonOrigin: "morphed_system2 / 3" },

    // Top bar: Main Supervisor
    { id: "main_lane", kind: "lane", x: 230, y: 18, w: 696, h: 64, label: "Main Supervisor", sublabel: "Startup, driving/parked mode switches, optional debug window & shutdown", tag: "THREAD" },

    // Center-top: Gaze Worker + 3 stages
    { id: "gaze_lane", kind: "lane", x: 230, y: 108, w: 440, h: 162, label: "Gaze Worker", sublabel: "Road-gaze distraction pipeline (~45 ms/frame)", tag: "THREAD" },
    { id: "norm_box", kind: "box", x: 244, y: 158, w: 124, h: 92, label: "Image Prep", sublabel: "Contrast + 224x224" },
    { id: "backbone_box", kind: "box", x: 388, y: 158, w: 124, h: 92, label: "ResNet-18", sublabel: "Spatial features", pythonOrigin: "distraction/" },
    { id: "lstm_box", kind: "box", x: 532, y: 158, w: 124, h: 92, label: "20-Frame LSTM", sublabel: "Road vs. away", pythonOrigin: "RTdetection_lstm" },

    // Center-bottom: Fatigue Worker + 4 stages
    { id: "fatigue_lane", kind: "lane", x: 230, y: 296, w: 440, h: 214, label: "Fatigue Worker", sublabel: "Physiological somnolence & calibration (~50 ms/frame)", tag: "THREAD" },
    { id: "yawn_box", kind: "box", x: 246, y: 344, w: 196, h: 70, label: "Full-Frame Yawn Net", sublabel: "112x112 neural net", pythonOrigin: "base_yawn_detector" },
    { id: "face_box", kind: "box", x: 458, y: 344, w: 196, h: 70, label: "68-Point Face Finder", sublabel: "Single-pass landmarks", pythonOrigin: "Dlib 68-pt" },
    { id: "ear_box", kind: "box", x: 246, y: 426, w: 196, h: 70, label: "Eye Closure & PERCLOS", sublabel: "Blinks + 30s closure %", pythonOrigin: "ear_mar/" },
    { id: "head_box", kind: "box", x: 458, y: 426, w: 196, h: 70, label: "Head Pitch Droop", sublabel: "Roll-invariant droop", pythonOrigin: "head_orientation/" },

    // Right column: Alert Coordinator, Audio Mixer, Telegram Messenger
    { id: "alert_coord", kind: "box", x: 700, y: 108, w: 226, h: 144, label: "Alert Coordinator", sublabel: "Shared rule evaluator &\ndrowsiness priority", badge: "mutex" },
    { id: "audio_lane", kind: "lane", x: 700, y: 278, w: 226, h: 112, label: "Audio Mixer", sublabel: "Always-on 48 kHz alerts", tag: "THREAD", tone: "amber" },
    { id: "tg_burst", kind: "box", x: 700, y: 416, w: 226, h: 94, label: "Telegram Messenger", sublabel: "Short-lived HTTP burst", badge: "async" },

    // Orthogonal non-crossing flow edges
    { id: "e_kbd_main", kind: "edge", points: [[200, 50], [230, 50]] },
    { id: "e_cam_gaze", kind: "edge", points: [[200, 188], [230, 188]], label: "frames" },
    { id: "e_cam_fatigue", kind: "edge", points: [[200, 265], [215, 265], [215, 360], [230, 360]] },
    { id: "e_main_gaze", kind: "edge", points: [[450, 82], [450, 108]], label: "state" },
    { id: "e_main_alert", kind: "edge", points: [[820, 82], [820, 108]], label: "reset" },
    { id: "e_gaze_alert", kind: "edge", points: [[670, 180], [700, 180]] },
    { id: "e_fatigue_alert", kind: "edge", points: [[670, 340], [685, 340], [685, 224], [700, 224]] },
    { id: "e_alert_audio", kind: "edge", points: [[820, 252], [820, 278]], label: "sound" },
    { id: "e_alert_tg", kind: "edge", points: [[926, 180], [944, 180], [944, 463], [926, 463]], dashed: true, label: "detach", lx: 904, ly: 405 }
  ],
  walkthrough: [
    {
      title: "1. Where Your Python Algorithms Live",
      show: ["py_loop", "norm_box", "backbone_box", "lstm_box", "yawn_box", "face_box", "ear_box", "head_box"],
      highlight: ["py_loop", "norm_box", "backbone_box", "lstm_box", "yawn_box", "face_box", "ear_box", "head_box"],
      edgeFlow: [],
      concept: "In the original Python prototypes, a single loop processed one camera frame through every detector in sequence. In C++, those exact four detection algorithms are preserved, split into road-gaze tracking on top and physiological fatigue below."
    },
    {
      title: "2. Two Parallel Analysis Workers",
      show: ["py_loop", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box"],
      highlight: ["gaze_lane", "fatigue_lane", "norm_box", "backbone_box", "lstm_box", "yawn_box", "face_box", "ear_box", "head_box"],
      edgeFlow: [],
      concept: "Instead of running sequentially, the algorithms split across two persistent worker threads: the Gaze Worker runs spatial-temporal road gaze inference, while the Fatigue Worker runs yawning, face landmarks, eye closure, and head droop in parallel.",
      note: "Older documentation mentions a geometric mouth-ratio yawn detector, which is still compiled in the project, but the active runtime replaced it with a full-frame yawning neural network that runs independently of face-landmark detection.",
      codeRef: "src/PipelineCoordinator.cpp (L363-429 gaze_worker_loop; L431-551 somnolence_worker_loop)\ninclude/YawnDetector.hpp | CMakeLists.txt (L56-57)"
    },
    {
      title: "3. Decoupled Camera Reader",
      show: ["py_loop", "cam_lane", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box", "e_cam_gaze", "e_cam_fatigue"],
      highlight: ["cam_lane"],
      edgeFlow: ["e_cam_gaze", "e_cam_fatigue"],
      concept: "A dedicated Camera Reader thread pulls frames continuously from the webcam or video file and pushes fresh copies to both workers. Slow neural or landmark steps never stall frame capture.",
      codeRef: "src/CameraCapture.cpp (L45 start; L71-114 capture_loop)"
    },
    {
      title: "4. Main Supervisor & Keyboard Listener",
      show: ["py_loop", "kbd_lane", "cam_lane", "main_lane", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box", "e_cam_gaze", "e_cam_fatigue", "e_kbd_main", "e_main_gaze"],
      highlight: ["kbd_lane", "main_lane"],
      edgeFlow: ["e_kbd_main", "e_main_gaze"],
      concept: "The Main Supervisor thread coordinates startup, driving vs. parked mode transitions, optional debug window rendering, and clean shutdown. Meanwhile, a tiny Keyboard Listener thread polls for movement, calibration, and quit keys without blocking video.",
      codeRef: "src/PipelineCoordinator.cpp (L210-284 run; L286-361 handle_console_command)\nsrc/ConsoleController.cpp (L47 start; L63-101 listener_loop)"
    },
    {
      title: "5. Shared Alert Coordinator & Audio Mixer",
      show: ["py_loop", "kbd_lane", "cam_lane", "main_lane", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box", "alert_coord", "audio_lane", "e_cam_gaze", "e_cam_fatigue", "e_kbd_main", "e_main_gaze", "e_main_alert", "e_gaze_alert", "e_fatigue_alert", "e_alert_audio"],
      highlight: ["alert_coord", "audio_lane"],
      edgeFlow: ["e_gaze_alert", "e_fatigue_alert", "e_alert_audio", "e_main_alert"],
      concept: "Both workers report their per-frame assessments into one mutex-protected Alert Coordinator, which applies distraction and drowsiness rules and dispatches sounds to an always-on Audio Mixer thread.",
      note: "Counting the Audio Mixer engine initialized at startup, the system runs 6 persistent background threads in steady state (Main Supervisor, Camera Reader, Gaze Worker, Fatigue Worker, Keyboard Listener, and Audio Mixer).",
      codeRef: "src/AlertManager.cpp (L83-204 update_gaze; L272-339 update_somnolence)\nsrc/SoundPlayer.cpp (L342-388 AudioStreamEngine::start)"
    },
    {
      title: "6. Fire-and-Forget Remote Notifications",
      show: ["py_loop", "kbd_lane", "cam_lane", "main_lane", "gaze_lane", "norm_box", "backbone_box", "lstm_box", "fatigue_lane", "yawn_box", "face_box", "ear_box", "head_box", "alert_coord", "audio_lane", "tg_burst", "e_cam_gaze", "e_cam_fatigue", "e_kbd_main", "e_main_gaze", "e_main_alert", "e_gaze_alert", "e_fatigue_alert", "e_alert_audio", "e_alert_tg"],
      highlight: ["tg_burst"],
      edgeFlow: ["e_alert_tg"],
      concept: "Whenever a distraction or drowsiness alarm triggers, the coordinator launches a short-lived Telegram Messenger thread to send the alert over HTTP and detaches immediately so network latency never stalls detection.",
      codeRef: "src/AlertManager.cpp (L177-179, L263-265, L331-333)\nsrc/TelegramNotifier.cpp (L36-71 send_telegram_alert)"
    }
  ],
  traces: []
});
