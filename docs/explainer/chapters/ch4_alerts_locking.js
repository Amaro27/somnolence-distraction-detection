window.EXPLAINER_CHAPTERS = window.EXPLAINER_CHAPTERS || [];
window.EXPLAINER_CHAPTERS.push({
  id: "ch4",
  title: "4. Alerts & Priority",
  subtitle: "2-stage gaze filter inside mutex, CNN yawn suppression, preemption & Telegram",
  viewBox: "0 0 960 540",
  legend: ["thread", "mutex", "atomic", "sync", "async"],
  elements: [
    // Left column: Per-frame worker outputs
    { id: "gaze_in", kind: "lane", x: 20, y: 18, w: 230, h: 138, label: "Gaze Worker", sublabel: "10f / 20f LSTM stream", tag: "THREAD" },
    { id: "gaze_res", kind: "box", x: 34, y: 66, w: 202, h: 76, label: "Per-Frame Gaze Result", sublabel: "Emits 1 vote per frame\n(0.54 conf, Binarize)" },

    { id: "fatigue_in", kind: "lane", x: 20, y: 178, w: 230, h: 342, label: "Fatigue Worker", sublabel: "CNN yawn + 68-pt eyes/head", tag: "THREAD" },
    { id: "yawn_sup", kind: "box", x: 34, y: 228, w: 202, h: 120, label: "0.2s Yawn Suppressor", sublabel: "Full-frame Yawn CNN >=0.2s\nmutes false blink &\ndroop cues" },
    { id: "cues_box", kind: "box", x: 34, y: 364, w: 202, h: 140, label: "Strong & Light Cues", sublabel: "Strong: >500ms / >=15%\nLight (2 in 2m): <=-25.5 deg\ndroop or >=1.0s Yawn CNN" },

    // Center column: Alert Coordinator lock boundary + outside side effects
    { id: "coord_lock", kind: "lane", x: 292, y: 18, w: 354, h: 374, label: "Alert Coordinator", sublabel: "Held only for microsecond math", tag: "MUTEX", tone: "amber" },
    { id: "gaze_eval", kind: "box", x: 310, y: 68, w: 318, h: 98, label: "2-Stage Distraction Filter", sublabel: "10f sliding mode vote +\n2.0s continuous away timer\n(skipped if Drowsy)", badge: "mutex" },
    { id: "drowsy_eval", kind: "box", x: 310, y: 180, w: 318, h: 112, label: "Drowsiness Fusion & Clear", sublabel: "1 Strong OR 2 Light (2m) -> Drowsy\nClears when eyes, head &\nyawning return to normal", badge: "mutex" },
    { id: "drv_state", kind: "pill", x: 326, y: 324, w: 286, h: 44, label: "Driver State: Safe / Distracted / Drowsy" },

    { id: "outside_lock", kind: "box", x: 292, y: 416, w: 354, h: 104, label: "Act Outside the Lock", sublabel: "Releases mutex BEFORE alerts\nso neither worker blocks on I/O" },

    // Right column: Audio outputs, preemption & detached Telegram
    { id: "dist_audio", kind: "box", x: 688, y: 24, w: 250, h: 116, label: "Distraction Warning", sublabel: "Blocking +6 dB on Gaze Worker\nThen wipes 10f mode & 2s timer" },
    { id: "drowsy_audio", kind: "lane", x: 688, y: 170, w: 250, h: 240, label: "Audio Mixer", sublabel: "Drowsiness always wins", tag: "PRIORITY", tone: "amber" },
    { id: "preempt_box", kind: "box", x: 704, y: 222, w: 218, h: 104, label: "Preempt & Escalate", sublabel: "Stops distraction clip;\n63.5 BPM: -21 to -3 dB" },
    { id: "clear_box", kind: "box", x: 704, y: 336, w: 218, h: 62, label: "Immediate Silence", sublabel: "Stops on recovery or park" },
    { id: "tg_burst", kind: "box", x: 688, y: 428, w: 250, h: 92, label: "Telegram Messenger", sublabel: "Detached HTTP alert\n(3s connect / 5s timeout)", badge: "async" },

    // Orthogonal non-crossing edges
    { id: "e_gaze_coord", kind: "edge", points: [[250, 96], [292, 96]], label: "vote" },
    { id: "e_fatigue_coord", kind: "edge", points: [[250, 240], [292, 240]], label: "cues" },
    { id: "e_lock_to_out", kind: "edge", points: [[469, 392], [469, 416]], label: "unlock" },
    { id: "e_out_dist", kind: "edge", points: [[646, 424], [660, 424], [660, 82], [688, 82]], label: "block", lx: 660, ly: 72 },
    { id: "e_out_drowsy", kind: "edge", points: [[646, 448], [674, 448], [674, 360], [688, 360]], label: "pulse", lx: 674, ly: 350 },
    { id: "e_out_tg", kind: "edge", points: [[646, 474], [688, 474]], dashed: true, label: "detach" },
    { id: "e_preempt_cut", kind: "edge", points: [[813, 170], [813, 140]], dashed: true, label: "cut off" }
  ],
  walkthrough: [
    {
      title: "1. One Shared Turnstile Lock",
      show: ["gaze_in", "gaze_res", "fatigue_in", "coord_lock", "drv_state", "e_gaze_coord", "e_fatigue_coord"],
      highlight: ["coord_lock", "drv_state"],
      edgeFlow: ["e_gaze_coord", "e_fatigue_coord"],
      concept: "Both workers report into one shared Alert Coordinator guarded by a single turnstile lock. This guarantees concurrent gaze and fatigue updates can never corrupt driver state or cue histories."
    },
    {
      title: "2. 2-Stage Distraction Filter (Inside Lock)",
      show: ["gaze_in", "gaze_res", "coord_lock", "gaze_eval", "drv_state", "e_gaze_coord"],
      highlight: ["gaze_res", "gaze_eval"],
      edgeFlow: ["e_gaze_coord"],
      concept: "The Gaze Worker emits one frame classification per pass into the Alert Coordinator. Inside the lock, a two-stage filter requires both a majority off-road vote across the sliding frame window and a continuous 2.0-second duration before triggering Distracted.",
      note: "By default, the sliding mode window matches the 10-frame pre-optimized sequence model (switching to 20 frames for the 20-frame unified model), using a 0.54 confidence cutoff under Binarize policy (with FilterUncertain and Argmax also available).",
      codeRef: "morphed_system6/distraction/modular/src/AlertManager.cpp (L83-145 update_gaze)\nsrc/AlertLogic.cpp (L5-49 evaluate_distraction_mode) | src/GazeDetector.cpp (L159-182) | include/Types.hpp (L172-179)"
    },
    {
      title: "3. Drowsiness Fusion & Yawn Suppression",
      show: ["fatigue_in", "yawn_sup", "cues_box", "coord_lock", "drowsy_eval", "drv_state", "e_fatigue_coord"],
      highlight: ["yawn_sup", "cues_box", "drowsy_eval"],
      edgeFlow: ["e_fatigue_coord"],
      concept: "Either one Strong cue (micro-sleep over 500 ms or High PERCLOS at 15%+) OR two Light cues within 2 minutes (head droop at or below -25.5 degrees for 1.5 s, or sustained neural yawn for 1.0 s) triggers Drowsy. Yawning at least 0.2 s mutes eye and head false alarms.",
      note: "Medium PERCLOS (7.5%-15.0%) Light cues are intentionally disabled in the eye monitor; only High PERCLOS (>=15.0%) Strong cues fire. Note that full-frame Yawn CNN runs even if Dlib misses the face.",
      codeRef: "morphed_system6/distraction/modular/src/AlertManager.cpp (L206-240, L272-315)\nsrc/AlertLogic.cpp (L51-59) | src/EarDetector.cpp (L223-235) | src/PipelineCoordinator.cpp (L526-559)"
    },
    {
      title: "4. Compute Inside Lock, Act Outside Lock",
      show: ["coord_lock", "gaze_eval", "drowsy_eval", "drv_state", "outside_lock", "dist_audio", "drowsy_audio", "preempt_box", "e_lock_to_out", "e_out_dist", "e_out_drowsy"],
      highlight: ["coord_lock", "outside_lock"],
      edgeFlow: ["e_lock_to_out", "e_out_dist", "e_out_drowsy"],
      concept: "The coordinator only holds the lock for fast window math and state flips, then copies out alert details and unlocks before touching logs, audio, or network calls. Neither worker ever stalls waiting on another thread's I/O.",
      codeRef: "morphed_system6/distraction/modular/src/AlertManager.cpp (L147-203, L317-338)"
    },
    {
      title: "5. Drowsiness Priority & Escalating Cadence",
      show: ["coord_lock", "gaze_eval", "drowsy_eval", "drv_state", "outside_lock", "dist_audio", "drowsy_audio", "preempt_box", "clear_box", "e_lock_to_out", "e_out_dist", "e_out_drowsy", "e_preempt_cut"],
      highlight: ["drowsy_audio", "preempt_box", "clear_box"],
      edgeFlow: ["e_out_drowsy", "e_preempt_cut"],
      concept: "Drowsiness strictly overrides distraction: while drowsy, distraction evaluation is skipped, active distraction audio is silenced immediately, and warning beeps pulse at roughly 63.5 beats per minute across five rising loudness levels until normal features return.",
      note: "Playing the +6 dB distraction clip intentionally blocks the Gaze Worker until playback finishes (or drowsiness cuts it off), then clears both the sliding mode window and the 2-second sustained timer.",
      codeRef: "morphed_system6/distraction/modular/src/AlertManager.cpp (L111-115, L182-192, L360-388, L413-421)\ninclude/Types.hpp (L160-167)"
    },
    {
      title: "6. Fire-and-Forget Remote Notifications",
      show: ["coord_lock", "drv_state", "outside_lock", "tg_burst", "e_lock_to_out", "e_out_tg"],
      highlight: ["outside_lock", "tg_burst"],
      edgeFlow: ["e_lock_to_out", "e_out_tg"],
      concept: "Whenever a distraction or high-alert drowsiness trigger occurs, the coordinator spawns a short-lived Telegram Messenger thread and detaches it immediately with tight connection timeouts so poor cell coverage never stalls driving safety.",
      note: "A legacy drowsiness-ticking helper for lost-face gaps still exists in the coordinator, but the Fatigue Worker now updates somnolence every frame with gradual face-lost decay instead.",
      codeRef: "morphed_system6/distraction/modular/src/AlertManager.cpp (L177-179, L330-334)\nsrc/TelegramNotifier.cpp (L36-71, 3s connect / 5s timeout)"
    }
  ],
  traces: [
    {
      id: "preempt_trace",
      label: "Trace: Drowsiness Preempts Distraction",
      steps: [
        {
          title: "Step 1: Gaze Worker Triggers Distraction",
          show: ["gaze_in", "gaze_res", "coord_lock", "gaze_eval", "drv_state", "outside_lock", "dist_audio", "tg_burst", "e_gaze_coord", "e_lock_to_out", "e_out_dist", "e_out_tg"],
          highlight: ["gaze_eval", "dist_audio", "tg_burst"],
          edgeFlow: ["e_gaze_coord", "e_out_dist", "e_out_tg"],
          concept: "Once the 10-frame majority mode stays off-road for 2.0 seconds, the Gaze Worker marks state Distracted, unlocks, launches a detached Telegram alert, and waits on blocking distraction audio."
        },
        {
          title: "Step 2: Fatigue Worker Detects Micro-Sleep",
          show: ["gaze_in", "fatigue_in", "cues_box", "coord_lock", "gaze_eval", "drowsy_eval", "drv_state", "dist_audio", "e_fatigue_coord"],
          highlight: ["cues_box", "drowsy_eval"],
          edgeFlow: ["e_fatigue_coord"],
          concept: "While the distraction warning plays, the Fatigue Worker spots eyes closed past 500 ms and enters the coordinator lock immediately because the Gaze Worker unlocked before playing sound."
        },
        {
          title: "Step 3: State Elevates to Drowsy",
          show: ["coord_lock", "drowsy_eval", "drv_state", "outside_lock", "dist_audio", "e_lock_to_out"],
          highlight: ["drowsy_eval", "drv_state", "outside_lock"],
          edgeFlow: ["e_lock_to_out"],
          concept: "One Strong micro-sleep cue immediately flips driver state to Drowsy, resets the escalation ladder to step zero (-21.0 dB), and releases the lock."
        },
        {
          title: "Step 4: Audio Cuts Off Distraction Instantly",
          show: ["outside_lock", "dist_audio", "drowsy_audio", "preempt_box", "tg_burst", "e_out_drowsy", "e_preempt_cut", "e_out_tg"],
          highlight: ["drowsy_audio", "preempt_box"],
          edgeFlow: ["e_out_drowsy", "e_preempt_cut", "e_out_tg"],
          concept: "Outside the lock, the coordinator stops active audio immediately—cutting off the distraction clip—emits the first -21.0 dB drowsiness beep, and detaches a drowsiness Telegram alert."
        },
        {
          title: "Step 5: Gaze Worker Unblocks & Yields",
          show: ["gaze_in", "gaze_res", "coord_lock", "gaze_eval", "drv_state", "dist_audio", "drowsy_audio", "e_gaze_coord"],
          highlight: ["gaze_in", "gaze_eval"],
          edgeFlow: [],
          concept: "Stopping audio immediately wakes the Gaze Worker, which clears its 10-frame mode window and 2-second timer, then skips distraction checks while Drowsy stays active."
        },
        {
          title: "Step 6: Recovery Silences Alarm",
          show: ["fatigue_in", "cues_box", "coord_lock", "drowsy_eval", "drv_state", "outside_lock", "drowsy_audio", "clear_box", "e_fatigue_coord", "e_lock_to_out", "e_out_drowsy"],
          highlight: ["drv_state", "clear_box"],
          edgeFlow: ["e_fatigue_coord", "e_out_drowsy"],
          concept: "Once eyes reopen, head posture recovers, and yawning stops, the next fatigue pass resets state to Safe, clears the 2-minute cue history, and stops drowsiness beeps."
        }
      ]
    }
  ]
});
