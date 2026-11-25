window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec4",
  title: "4. Yawning",
  subviews: [
    {
      id: "4A",
      label: "4A. Fused Algorithm",
      subtitle: "Full-frame 112x112 NHWC cnn_yawning.onnx, prob >= 0.80 for >= 1.0 s, -1.0*dt decay & 2-in-120s",
      viewBox: "0 0 1120 660",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "4A: Yawning — Fused Algorithm View",
        concept: "Resizes the full BGR camera frame to 112x112 RGB [0, 1] NHWC and runs cnn_yawning.onnx. When yawn_prob >= 0.80 persists for >= 1.0 s (with 2 s cooldown), it emits a Light Yawn cue into the 120-second window, triggering DROWSY once >= 2 Light cues occur in 2 minutes.",
        note: "Thresholds verified in YawnConfig (Types.hpp L248-255): input_dimension = 112x112, prob_threshold = 0.80, sustained_duration = 1000 ms, alarm_cooldown = 2 s. Decay rate is linear -1.0*dt when yawn_prob < 0.80.",
        codeRef: "src/YawnDetector.cpp (L89-171)\ninclude/MathUtils.hpp (L61-82)\ninclude/Types.hpp (L248-255)"
      },
      nodes: [
        {
          id: "in_bgr", kind: "process", x: 24, y: 34, w: 215, h: 68,
          label: "Full BGR Frame", sublabel: "No landmarks needed\ndt = clamp_dt(now)",
          details: {
            concept: "Operates directly on the full BGR frame without waiting for Dlib face bounding boxes.",
            codeRef: "src/YawnDetector.cpp (L112-120)"
          }
        },
        {
          id: "pre_nhwc", kind: "process", x: 275, y: 26, w: 265, h: 84,
          label: "Preprocess 112x112 NHWC", sublabel: "BGR -> RGB, INTER_AREA 112x112\nScale pixel / 255.0 -> [0, 1]",
          details: {
            concept: "Packs zero-allocation NHWC tensor (1 x 112 x 112 x 3) normalized to [0.0, 1.0].",
            codeRef: "src/YawnDetector.cpp (L89-110 pack_nhwc_blob)"
          }
        },
        {
          id: "run_cnn", kind: "process", x: 580, y: 32, w: 245, h: 72,
          label: "cnn_yawning.onnx Forward", sublabel: "OpenCV DNN CPU inference\nScalar output: yawn_prob",
          details: {
            concept: "Executes the lightweight yawning CNN and extracts scalar yawn_prob.",
            codeRef: "src/YawnDetector.cpp (L123-126)"
          }
        },
        {
          id: "chk_prob", kind: "decision", x: 580, y: 152, w: 245, h: 96,
          label: "yawn_prob >= 0.80?", sublabel: "CNN yawn confidence",
          details: {
            concept: "Checks whether the full-frame CNN yawning probability meets prob_threshold (0.80).",
            codeRef: "src/YawnDetector.cpp (L131)"
          }
        },
        {
          id: "yawn_up", kind: "process", x: 265, y: 162, w: 265, h: 76,
          label: "Accumulate Yawn Timer", sublabel: "in_yawn = true\naccum = min(1.0s, accum + dt)",
          details: {
            concept: "Caps accumulation cleanly at target_sec = 1.0 s while yawn_prob stays >= 0.80.",
            codeRef: "src/YawnDetector.cpp (L131-134)"
          }
        },
        {
          id: "chk_sustained", kind: "decision", x: 265, y: 288, w: 265, h: 102,
          label: "accum >= 1.0 s &&\n!emitted && cool >= 2s?", sublabel: "Sustained yawn gate",
          details: {
            concept: "Requires 1.0 s sustained yawning, no prior cue latch for this yawn, and >= 2.0 s elapsed since last_confirmed_yawn_.",
            codeRef: "src/YawnDetector.cpp (L135-138)"
          }
        },
        {
          id: "ret_wait", kind: "term_return", x: 24, y: 312, w: 195, h: 54,
          label: "RETURN", sublabel: "Building (<1s) or latched",
          details: {
            concept: "Returns without a new Light cue while accumulating toward 1.0 s or while yawn stays latched.",
            codeRef: "src/YawnDetector.cpp (L135-153)"
          }
        },
        {
          id: "emit_yawn_cue", kind: "process", x: 265, y: 442, w: 265, h: 72,
          label: "Record Light Yawn Cue", sublabel: "yawn_cue_emitted = true\nPush to 120 s cue window",
          details: {
            concept: "Emits CueEvent(Yawn, Light) and updates last_confirmed_yawn_ = now.",
            codeRef: "src/YawnDetector.cpp (L138-152) | src/AlertManager.cpp (L220-238)"
          }
        },
        {
          id: "chk_two_light", kind: "decision", x: 580, y: 430, w: 255, h: 96,
          label: ">= 2 Light cues\nin 120 s window?", sublabel: "Multimodal light rule",
          details: {
            concept: "Two Light cues (Yawn or HeadDroop) within 120 seconds trigger Drowsy.",
            codeRef: "src/AlertLogic.cpp (L51-59) | src/AlertManager.cpp (L223-238)"
          }
        },
        {
          id: "term_drowsy", kind: "term_alarm", x: 890, y: 448, w: 200, h: 60,
          label: "DROWSY", sublabel: "2 Light cues in 120 s",
          details: {
            concept: "Two Light fatigue cues in 2 minutes elevate DriverState to Drowsy.",
            codeRef: "src/AlertManager.cpp (L229-238)"
          }
        },
        {
          id: "ret_one_light", kind: "term_return", x: 890, y: 556, w: 200, h: 54,
          label: "RETURN", sublabel: "1 Light cue banked",
          details: {
            concept: "First Light cue stored in 120 s window; returns until a second Light cue arrives.",
            codeRef: "src/AlertLogic.cpp (L51-59)"
          }
        },
        {
          id: "yawn_decay", kind: "process", x: 870, y: 162, w: 225, h: 76,
          label: "Linear Decay (-1.0*dt)", sublabel: "accum = max(0, accum - dt)",
          details: {
            concept: "When yawn_prob drops below 0.80, yawn_accum drains linearly at 1.0*dt.",
            codeRef: "src/YawnDetector.cpp (L154-157)"
          }
        },
        {
          id: "chk_zero", kind: "decision", x: 870, y: 274, w: 225, h: 92,
          label: "accum <= 0.0 s?", sublabel: "Yawn ended check",
          details: {
            concept: "Clears in_yawn and yawn_cue_emitted once the accumulator drains to zero.",
            codeRef: "src/YawnDetector.cpp (L158-162)"
          }
        },
        {
          id: "clear_yawn", kind: "process", x: 580, y: 286, w: 245, h: 68,
          label: "Clear Yawn Flags", sublabel: "accum=0; in_yawn=false\nyawn_cue_emitted=false",
          details: {
            concept: "Re-arms yawn cue detection for the next yawning episode.",
            codeRef: "src/YawnDetector.cpp (L158-162)"
          }
        },
        {
          id: "ret_closed", kind: "term_return", x: 580, y: 556, w: 245, h: 54,
          label: "RETURN", sublabel: "Not yawning / decaying",
          details: {
            concept: "Exits cycle with no cue emitted.",
            codeRef: "src/YawnDetector.cpp (L165-170)"
          }
        }
      ],
      edges: [
        { id: "e_in_pre", points: [[239, 68], [275, 68]], tone: "default" },
        { id: "e_pre_cnn", points: [[540, 68], [580, 68]], tone: "default" },
        { id: "e_cnn_chk", points: [[702, 104], [702, 152]], tone: "default" },
        { id: "e_prob_yes", points: [[580, 200], [530, 200]], label: "YES", tone: "yes" },
        { id: "e_prob_no", points: [[825, 200], [870, 200]], label: "NO", tone: "no" },
        { id: "e_up_sus", points: [[397, 238], [397, 288]], tone: "default" },
        { id: "e_sus_no", points: [[265, 339], [219, 339]], label: "NO", tone: "no" },
        { id: "e_sus_yes", points: [[397, 390], [397, 442]], label: "YES", tone: "yes" },
        { id: "e_cue_2l", points: [[530, 478], [580, 478]], tone: "default" },
        { id: "e_2l_yes", points: [[835, 478], [890, 478]], label: "YES", tone: "yes" },
        { id: "e_2l_no", points: [[707, 526], [707, 540], [862, 540], [862, 583], [890, 583]], label: "NO", tone: "no", lx: 785, ly: 535 },
        { id: "e_dec_chk", points: [[982, 238], [982, 274]], tone: "default" },
        { id: "e_z_yes", points: [[870, 320], [825, 320]], label: "YES", tone: "yes" },
        { id: "e_z_no", points: [[982, 366], [982, 404], [552, 404], [552, 583], [580, 583]], label: "NO", tone: "no", lx: 775, ly: 396 },
        { id: "e_clr_ret", points: [[580, 320], [552, 320], [552, 583], [580, 583]], tone: "default" }
      ],
      traces: [
        {
          id: "yawn_trigger",
          label: "Trace: Sustained Yawn (>= 1.0 s) -> Light Cue -> DROWSY",
          steps: [
            {
              title: "Step 1: Full-Frame CNN >= 0.80",
              highlightNodes: ["in_bgr", "pre_nhwc", "run_cnn", "chk_prob", "yawn_up"],
              highlightEdges: ["e_in_pre", "e_pre_cnn", "e_cnn_chk", "e_prob_yes"],
              concept: "112x112 RGB NHWC blob runs through cnn_yawning.onnx; yawn_prob >= 0.80 increments yawn_accum by clamped dt.",
              codeRef: "src/YawnDetector.cpp (L89-134)"
            },
            {
              title: "Step 2: 1.0 s + 2 s Cooldown -> Light Cue -> 2-in-120s",
              highlightNodes: ["chk_sustained", "emit_yawn_cue", "chk_two_light", "term_drowsy"],
              highlightEdges: ["e_up_sus", "e_sus_yes", "e_cue_2l", "e_2l_yes"],
              concept: "At >= 1.0 s, a Light Yawn cue records into the 120 s cue window; >= 2 Light cues within 2 minutes trigger DROWSY.",
              codeRef: "src/YawnDetector.cpp (L135-153) | src/AlertLogic.cpp (L51-59)"
            }
          ]
        },
        {
          id: "yawn_short",
          label: "Trace: Brief Speech / Mouth Open (< 1.0 s) -> Decay",
          steps: [
            {
              title: "Below 0.80 Decays Linearly (-1.0*dt) to Clear",
              highlightNodes: ["chk_prob", "yawn_decay", "chk_zero", "clear_yawn", "ret_closed"],
              highlightEdges: ["e_prob_no", "e_dec_chk", "e_z_yes", "e_clr_ret"],
              concept: "Brief mouth openings drop below 0.80 before reaching 1.0 s, draining linearly at -1.0*dt and resetting in_yawn.",
              codeRef: "src/YawnDetector.cpp (L154-163)"
            }
          ]
        }
      ]
    },
    {
      id: "4B",
      label: "4B. System & FSM Cut-Off",
      subtitle: "Calibration/Dlib independence, early >= 0.2 s yawn suppressor & MovingCalibrated cut-off",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "4B: Yawning — System & FSM Cut-Off View",
        concept: "YawnDetector runs on the full camera frame every somnolence_worker_loop iteration (outside Calibrating), needing zero calibration baselines and zero Dlib landmarks. At >= 0.2 s it fires suppress_by_yawn to mute Head Droop, Blink, and PERCLOS before forwarding YawnAssessment at MovingCalibrated.",
        note: "Legacy landmark-based MarDetector (MAR > 0.55 for 4 s in src/MarDetector.cpp) remains compiled in the repository but is unused in MS6.",
        codeRef: "src/PipelineCoordinator.cpp (L480-505, L526-560)\nsrc/YawnDetector.cpp (L80-171)"
      },
      nodes: [
        {
          id: "read_calibrating", kind: "decision", x: 30, y: 34, w: 235, h: 96,
          label: "READ 1A: SystemState\n== Calibrating?", sublabel: "Click -> 1A FSM",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_calib" },
          details: {
            concept: "While SystemState == Calibrating, somnolence_worker_loop jumps into Calibration (1B) instead of running YawnDetector.",
            codeRef: "src/PipelineCoordinator.cpp (L480-505)"
          }
        },
        {
          id: "goto_1b_take", kind: "portal_out", x: 30, y: 180, w: 235, h: 62,
          label: "GOTO 1B: CALIBRATION", sublabel: "Yawn paused during calib",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "entry_calib" },
          details: {
            concept: "run_calibration() hijacks the Fatigue Worker until calibration completes or aborts.",
            codeRef: "src/PipelineCoordinator.cpp (L481-488)"
          }
        },
        {
          id: "from_calib_yawn_reset", kind: "portal_in", x: 30, y: 290, w: 235, h: 64,
          label: "FROM 1B: YAWN_RESET", sublabel: "yawn_detector_.reset()",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_4b_yawn_rst" },
          details: {
            concept: "Clears yawn_accum_sec_, in_yawn_, and cooldown when calibration finishes (even though Yawn CNN needs no personal baseline).",
            codeRef: "src/PipelineCoordinator.cpp (L493) | src/YawnDetector.cpp (L80-87)"
          }
        },
        {
          id: "cnn_process", kind: "process", x: 325, y: 42, w: 270, h: 84,
          label: "YawnDetector::process()", sublabel: "Runs every frame (no Dlib,\nno calibration needed);\nupdates yawn_accum & cue",
          details: {
            concept: "Executes 112x112 CNN inference on the raw BGR frame even when Dlib is missing the face or throttled to 500 ms.",
            codeRef: "src/PipelineCoordinator.cpp (L526-527) | src/YawnDetector.cpp (L112-171)"
          }
        },
        {
          id: "chk_sup_02", kind: "decision", x: 325, y: 176, w: 270, h: 98,
          label: "yawn_duration_sec\n>= 0.2 s?", sublabel: "Early suppressor threshold",
          details: {
            concept: "Triggers cross-estimator yawn suppression after just 0.2 s (well before the 1.0 s Light cue threshold).",
            codeRef: "src/PipelineCoordinator.cpp (L528-531) | include/Types.hpp (L252)"
          }
        },
        {
          id: "sig_yawn_sup", kind: "portal_out", x: 655, y: 146, w: 245, h: 66,
          label: "SIGNAL: SUPPRESS_DROOP", sublabel: "suppress_by_yawn = true\nClick -> 2B Head Droop",
          target: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_yawn_sup" },
          details: {
            concept: "Sets suppress_by_yawn = true, feeding HeadOrientationDetector (2B) so yawning head tilt does not register as HeadDroop.",
            codeRef: "src/PipelineCoordinator.cpp (L528-531, L551)"
          }
        },
        {
          id: "sig_yawn_eyes", kind: "portal_out", x: 655, y: 234, w: 245, h: 66,
          label: "SIGNAL: SUPPRESS_EYES", sublabel: "Mutes 3C Blink, 3D PERCLOS\n& 6B AlertManager",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "from_yawn_sup" },
          details: {
            concept: "Same suppress_by_yawn = true wire clears EarDetector blink streaks (3C), zeroes PERCLOS closed samples (3D), and filters eye cues in AlertManager (6B).",
            codeRef: "src/PipelineCoordinator.cpp (L528-531, L550) | src/AlertManager.cpp (L287-292)"
          }
        },
        {
          id: "to_1c_dlib", kind: "portal_out", x: 325, y: 344, w: 270, h: 64,
          label: "CONTINUE -> 1C: DLIB", sublabel: "500 ms face gate + landmarks",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "entry_frame" },
          details: {
            concept: "Sequential next step in somnolence_worker_loop: passes frame + suppress_by_yawn flag to Dlib (1C) -> Blink/Head.",
            codeRef: "src/PipelineCoordinator.cpp (L533-555)"
          }
        },
        {
          id: "read_mov_cal", kind: "decision", x: 325, y: 462, w: 270, h: 98,
          label: "READ 1A: SystemState\n== MovingCalibrated?", sublabel: "Click -> 1A FSM",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            concept: "YawnAssessment (including Light cue if >= 1.0 s) enters AlertManager only when MovingCalibrated.",
            codeRef: "src/PipelineCoordinator.cpp (L515, L557-559)"
          }
        },
        {
          id: "goto_6b", kind: "portal_out", x: 655, y: 478, w: 245, h: 66,
          label: "GOTO 6B: CUE_FUSION", sublabel: "update_somnolence(...yawn)",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
          details: {
            concept: "Passes YawnAssessment into AlertManager::mutex_ to register Light Yawn cues and check recovery (!yawn.is_yawning).",
            codeRef: "src/PipelineCoordinator.cpp (L558) | src/AlertManager.cpp (L287-303)"
          }
        },
        {
          id: "ret_hud_only", kind: "term_return", x: 655, y: 364, w: 245, h: 58,
          label: "RETURN (HUD Only)", sublabel: "Live Yawn % without alerts",
          details: {
            concept: "DebugVisualizer displays yawn probability & duration even when parked or uncalibrated.",
            codeRef: "src/PipelineCoordinator.cpp (L557-575)"
          }
        }
      ],
      edges: [
        { id: "e_cal_yes", points: [[147, 130], [147, 180]], label: "YES", tone: "yes" },
        { id: "e_cal_no", points: [[265, 82], [325, 82]], label: "NO", tone: "no" },
        { id: "e_cnn_sup", points: [[460, 126], [460, 176]], tone: "default" },
        { id: "e_sup_yes1", points: [[595, 214], [625, 214], [625, 179], [655, 179]], label: "YES", tone: "portal", lx: 625, ly: 168 },
        { id: "e_sup_yes2", points: [[595, 232], [625, 232], [625, 267], [655, 267]], tone: "portal" },
        { id: "e_sup_1c", points: [[460, 274], [460, 344]], label: "next step", tone: "default" },
        { id: "e_1c_mov", points: [[460, 408], [460, 462]], tone: "default" },
        { id: "e_mov_no", points: [[595, 492], [625, 492], [625, 393], [655, 393]], label: "NO", tone: "no", lx: 625, ly: 382 },
        { id: "e_mov_yes", points: [[595, 511], [655, 511]], label: "YES", tone: "yes" }
      ],
      traces: [
        {
          id: "yawn_early_suppress",
          label: "Trace: 0.2 s Early Suppressor -> 1.0 s Light Cue -> 6B",
          steps: [
            {
              title: "Step 1: Full-Frame CNN Crosses 0.2 s Suppressor",
              highlightNodes: ["read_calibrating", "cnn_process", "chk_sup_02", "sig_yawn_sup", "sig_yawn_eyes"],
              highlightEdges: ["e_cal_no", "e_cnn_sup", "e_sup_yes1", "e_sup_yes2"],
              concept: "After 0.2 s of yawning, suppress_by_yawn flips true and immediately mutes Head Droop (2B), Blink (3C), PERCLOS (3D), and AlertManager eye cues (6B).",
              portalTarget: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_yawn_sup" },
              codeRef: "src/PipelineCoordinator.cpp (L527-531, L550-551)"
            },
            {
              title: "Step 2: MovingCalibrated Forwards YawnAssessment -> 6B",
              highlightNodes: ["to_1c_dlib", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_sup_1c", "e_1c_mov", "e_mov_yes"],
              concept: "Once sustained >= 1.0 s, the Light Yawn cue passes through the MovingCalibrated check into AlertManager::update_somnolence (6B).",
              portalTarget: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
              codeRef: "src/PipelineCoordinator.cpp (L557-559)"
            }
          ]
        },
        {
          id: "yawn_face_lost",
          label: "Trace: Dlib Face Lost (500 ms) While Yawn CNN Runs",
          steps: [
            {
              title: "Yawn CNN Uninterrupted by Missing Landmarks",
              highlightNodes: ["cnn_process", "chk_sup_02", "to_1c_dlib", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_cnn_sup", "e_sup_1c", "e_1c_mov", "e_mov_yes"],
              concept: "Even if a wide yawn causes Dlib (1C) to miss the face and enter 500 ms throttling, YawnDetector already processed the full BGR frame and still enters 6B.",
              codeRef: "src/PipelineCoordinator.cpp (L526-559)"
            }
          ]
        },
        {
          id: "yawn_parked",
          label: "Trace: Parked (Calibrated) -> HUD Telemetry Only",
          steps: [
            {
              title: "Stationary State Gates AlertManager",
              highlightNodes: ["cnn_process", "read_mov_cal", "ret_hud_only"],
              highlightEdges: ["e_cnn_sup", "e_sup_1c", "e_1c_mov", "e_mov_no"],
              concept: "While parked, YawnDetector keeps running for live HUD telemetry, but SystemState != MovingCalibrated prevents cues from entering AlertManager.",
              codeRef: "src/PipelineCoordinator.cpp (L557-575)"
            }
          ]
        }
      ]
    }
  ]
});
