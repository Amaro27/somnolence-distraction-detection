window.EXPLAINER_CHAPTERS = window.EXPLAINER_CHAPTERS || [];
window.EXPLAINER_CHAPTERS.push({
  id: "ch2",
  title: "2. Mailboxes",
  subtitle: "Drop-oldest frame handoff for zero queue lag",
  viewBox: "0 0 960 540",
  legend: ["thread", "mailbox", "sync", "debug"],
  elements: [
    // Left column: Why queues fail + Camera Reader thread
    { id: "fifo_warn", kind: "box", x: 24, y: 24, w: 230, h: 120, label: "Why Not a Normal Queue?", sublabel: "30 FPS camera > 20 FPS worker\n= growing backlog & stale alarms", badge: "mutex" },
    { id: "cam_lane", kind: "lane", x: 24, y: 170, w: 230, h: 336, label: "Camera Reader", sublabel: "Reads frames & fans out", tag: "THREAD" },
    { id: "cam_grab", kind: "box", x: 40, y: 224, w: 198, h: 116, label: "Grab & Deep Clone", sublabel: "Copies pixels once per tray\n(no shared pixel buffers)" },
    { id: "cam_pace", kind: "box", x: 40, y: 364, w: 198, h: 122, label: "Source Pacing", sublabel: "Webcam: hardware rate\nVideo file: sleep to match FPS" },

    // Center column: Single-slot drop-oldest mailboxes
    { id: "mb_gaze", kind: "mailbox", x: 340, y: 60, w: 200, h: 102, label: "Gaze Mailbox", slotLabel: "1-slot: Fresh" },
    { id: "mb_somno", kind: "mailbox", x: 340, y: 218, w: 200, h: 102, label: "Fatigue Mailbox", slotLabel: "1-slot: Fresh" },
    { id: "mb_debug", kind: "mailbox", x: 340, y: 376, w: 200, h: 102, label: "Debug Mailbox (Opt.)", slotLabel: "1-slot: Fresh" },

    // Right column: Independent consumers
    { id: "gaze_lane", kind: "lane", x: 636, y: 34, w: 300, h: 154, label: "Gaze Worker", sublabel: "Wait up to 100 ms | ~45 ms work", tag: "THREAD" },
    { id: "gaze_wait", kind: "box", x: 654, y: 88, w: 264, h: 82, label: "Take Latest Frame", sublabel: "Wakes on arrival or checks\nshutdown every 100 ms" },
    { id: "fatigue_lane", kind: "lane", x: 636, y: 206, w: 300, h: 146, label: "Fatigue Worker", sublabel: "Independent speed | ~50 ms work", tag: "THREAD" },
    { id: "fatigue_wait", kind: "box", x: 654, y: 260, w: 264, h: 74, label: "Take Latest Frame", sublabel: "Never delayed by Gaze Worker" },
    { id: "hud_box", kind: "box", x: 636, y: 376, w: 300, h: 102, label: "Debug HUD (Main Loop)", sublabel: "Polls every 30 ms if built\n(omitted in Release)", badge: "debug" },

    // Non-crossing edges
    { id: "e_cam_mb_gaze", kind: "edge", points: [[254, 248], [296, 248], [296, 111], [340, 111]], label: "clone 1", lx: 296, ly: 101 },
    { id: "e_cam_mb_somno", kind: "edge", points: [[254, 269], [340, 269]], label: "clone 2" },
    { id: "e_cam_mb_debug", kind: "edge", points: [[254, 290], [296, 290], [296, 427], [340, 427]], dashed: true, label: "clone 3 (debug)", lx: 296, ly: 446 },
    { id: "e_mb_gaze_worker", kind: "edge", points: [[540, 111], [636, 111]], label: "latest" },
    { id: "e_mb_somno_worker", kind: "edge", points: [[540, 269], [636, 269]], label: "latest" },
    { id: "e_mb_debug_hud", kind: "edge", points: [[540, 427], [636, 427]], dashed: true, label: "optional" }
  ],
  walkthrough: [
    {
      title: "1. Why Normal Queues Fail for Safety",
      show: ["fifo_warn"],
      highlight: ["fifo_warn"],
      edgeFlow: [],
      concept: "If a camera produces 30 frames per second while a detector runs at 20 frames per second, a normal waiting queue grows every second. Soon the safety monitor would judge driver drowsiness on stale video from seconds ago."
    },
    {
      title: "2. Single-Slot Drop-Oldest Mailboxes",
      show: ["fifo_warn", "cam_lane", "mb_gaze", "mb_somno", "mb_debug", "e_cam_mb_gaze", "e_cam_mb_somno", "e_cam_mb_debug"],
      highlight: ["mb_gaze", "mb_somno", "mb_debug"],
      edgeFlow: ["e_cam_mb_gaze", "e_cam_mb_somno"],
      concept: "Each consumer gets its own single-slot mailbox tray instead of a growing queue. Whenever a newer frame arrives before a worker picks up the previous one, the old unread frame is overwritten in place."
    },
    {
      title: "3. Independent Frame Copies per Consumer",
      show: ["fifo_warn", "cam_lane", "cam_grab", "mb_gaze", "mb_somno", "mb_debug", "e_cam_mb_gaze", "e_cam_mb_somno", "e_cam_mb_debug"],
      highlight: ["cam_grab"],
      edgeFlow: ["e_cam_mb_gaze", "e_cam_mb_somno", "e_cam_mb_debug"],
      concept: "Because image objects normally share underlying pixel memory, the Camera Reader writes a fresh deep copy into each tray. Workers can resize, convert colors, or draw overlays without ever corrupting another thread.",
      note: "Cloning once per consumer means 2 full-frame copies per camera frame in Release builds, or 3 copies when the optional debug window is enabled—trading modest memory bandwidth for zero pixel races.",
      codeRef: "morphed_system6/distraction/modular/src/CameraCapture.cpp (L90-95 frame.clone)\nsrc/PipelineCoordinator.cpp (L115-124)"
    },
    {
      title: "4. Sleep Until Fresh (With Shutdown Escape)",
      show: ["cam_lane", "cam_grab", "mb_gaze", "mb_somno", "mb_debug", "gaze_lane", "gaze_wait", "fatigue_lane", "fatigue_wait", "hud_box", "e_cam_mb_gaze", "e_cam_mb_somno", "e_cam_mb_debug", "e_mb_gaze_worker", "e_mb_somno_worker", "e_mb_debug_hud"],
      highlight: ["gaze_lane", "gaze_wait", "fatigue_lane", "fatigue_wait", "hud_box"],
      edgeFlow: ["e_mb_gaze_worker", "e_mb_somno_worker", "e_mb_debug_hud"],
      concept: "Workers sleep with zero busy-spinning when no frame is ready, waking instantly when a tray is filled. If no frame arrives within 100 milliseconds, they wake briefly to check for program shutdown, or wake immediately when stopped.",
      note: "Because each worker owns its own mailbox, a temporary slowdown in 68-point face finding never delays road-gaze inference, and vice versa.",
      codeRef: "morphed_system6/distraction/modular/include/ThreadSafeMailbox.hpp (L30-69 put / get_latest / stop)\nsrc/PipelineCoordinator.cpp (L416, L507)"
    },
    {
      title: "5. Live Webcam vs. Video File Pacing",
      show: ["cam_lane", "cam_grab", "cam_pace", "mb_gaze", "mb_somno", "gaze_lane", "gaze_wait", "fatigue_lane", "fatigue_wait", "e_cam_mb_gaze", "e_cam_mb_somno", "e_mb_gaze_worker", "e_mb_somno_worker"],
      highlight: ["cam_pace"],
      edgeFlow: [],
      concept: "Against a live webcam, the Camera Reader runs at the hardware frame cadence. Against a recorded video file, it times each read and sleeps so playback matches the file's real-time frame rate instead of burning through the clip at maximum disk speed.",
      codeRef: "morphed_system6/distraction/modular/src/CameraCapture.cpp (L71-105 capture_loop; fallback 30 FPS)"
    }
  ],
  traces: [
    {
      id: "fast_cam",
      label: "Trace: Fast Camera, Busy Worker",
      steps: [
        {
          title: "Step 1: Frame #1 Arrives",
          show: ["cam_lane", "cam_grab", "mb_gaze", "gaze_lane", "gaze_wait", "e_cam_mb_gaze", "e_mb_gaze_worker"],
          highlight: ["cam_grab", "mb_gaze"],
          edgeFlow: ["e_cam_mb_gaze"],
          slotText: { mb_gaze: "Holds: Frame #1" },
          concept: "The Camera Reader deposits Frame #1 into the Gaze Mailbox and notifies the sleeping Gaze Worker."
        },
        {
          title: "Step 2: Gaze Worker Starts Work",
          show: ["cam_lane", "cam_grab", "mb_gaze", "gaze_lane", "gaze_wait", "e_cam_mb_gaze", "e_mb_gaze_worker"],
          highlight: ["gaze_lane", "gaze_wait"],
          edgeFlow: ["e_mb_gaze_worker"],
          slotText: { mb_gaze: "Empty (taken #1)" },
          concept: "The Gaze Worker moves Frame #1 out of the tray and starts its ~45 millisecond neural pass. The tray is now empty again."
        },
        {
          title: "Step 3: Frame #2 Arrives Mid-Inference",
          show: ["cam_lane", "cam_grab", "mb_gaze", "gaze_lane", "gaze_wait", "e_cam_mb_gaze", "e_mb_gaze_worker"],
          highlight: ["cam_grab", "mb_gaze"],
          edgeFlow: ["e_cam_mb_gaze"],
          slotText: { mb_gaze: "Holds: Frame #2" },
          concept: "Before Frame #1 finishes, the faster camera puts Frame #2 into the tray without waiting for the worker."
        },
        {
          title: "Step 4: Frame #3 Drops Frame #2",
          show: ["cam_lane", "cam_grab", "mb_gaze", "gaze_lane", "gaze_wait", "e_cam_mb_gaze", "e_mb_gaze_worker"],
          highlight: ["mb_gaze"],
          edgeFlow: ["e_cam_mb_gaze"],
          slotText: { mb_gaze: "Frame #3 (#2 dropped!)" },
          concept: "The next camera frame arrives while Frame #1 is still running. Frame #3 overwrites Frame #2 cleanly in the single slot, discarding #2 without any lag buildup."
        },
        {
          title: "Step 5: Worker Jumps Straight to #3",
          show: ["cam_lane", "cam_grab", "mb_gaze", "gaze_lane", "gaze_wait", "e_cam_mb_gaze", "e_mb_gaze_worker"],
          highlight: ["gaze_lane", "gaze_wait"],
          edgeFlow: ["e_mb_gaze_worker"],
          slotText: { mb_gaze: "Empty (taken #3)" },
          concept: "As soon as Frame #1 finishes, the Gaze Worker grabs Frame #3 immediately and stays synchronized with live reality."
        }
      ]
    }
  ]
});
