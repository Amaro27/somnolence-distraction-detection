window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec2",
  title: "2. Head Drooping",
  subviews: [
    {
      id: "2A",
      label: "2A. Fused Algorithm",
      subtitle: "Roll-invariant eye-nose-chin proportion (rho), pitch <= -25.5 deg for 1.5 s, -2.0*dt recovery",
      viewBox: "0 0 1120 660",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "2A: Head Drooping — Fused Algorithm View",
        concept: "Pure algorithmic flow for head-pitch droop detection: projects nose tip (p33) and chin (p8) onto the downward eye-corner perpendicular axis u_y to obtain roll-invariant proportion rho, maps (rho - rho_0) to pitch_deg, and requires pitch <= -25.5 deg for 1.5 s (with fast -2.0*dt recovery) to emit a Light cue.",
        note: "Thresholds verified in HeadOrientationConfig (Types.hpp L193-197): kp_pitch = 0.18, pitch_droop_thresh = -25.5 deg (adjusted for BU HT dataset 5.48 deg MAE), droop_sustained_duration = 1500 ms. Comments in HeadOrientationDetector.cpp L59 still mention -20 deg.",
        codeRef: "src/FaceGeometry.cpp (L22-49)\nsrc/HeadOrientationDetector.cpp (L42-94)\ninclude/MathUtils.hpp (L61-82)\ninclude/Types.hpp (L193-197)"
      },
      nodes: [
        {
          id: "in_lm", kind: "process", x: 24, y: 34, w: 220, h: 66,
          label: "68 Landmarks + rho_0", sublabel: "p36, p45 (eyes), p33, p8",
          details: {
            concept: "Takes outer eye corners (p36, p45), nose tip (p33), chin (p8), and calibrated neutral proportion rho_0.",
            codeRef: "include/FaceGeometry.hpp (L10-28)"
          }
        },
        {
          id: "calc_rho", kind: "process", x: 280, y: 24, w: 270, h: 86,
          label: "Roll-Invariant rho", sublabel: "e_mid=(p36+p45)/2; u_y perp\nd_top=(p33-e_mid).u_y; d_bot=(p8-p33).u_y\nrho = d_top / max(d_top + d_bot, 1e-6)",
          details: {
            concept: "Projecting onto the downward normal u_y perpendicular to the eye line makes rho invariant to in-plane head roll.",
            codeRef: "src/FaceGeometry.cpp (L22-49 compute_rho)"
          }
        },
        {
          id: "calc_pitch", kind: "process", x: 590, y: 26, w: 250, h: 82,
          label: "Estimate Pitch & dt", sublabel: "pitch = atan2(-(rho - rho_0), 0.18)\n* 180 / pi;  dt = clamp_dt(now)",
          details: {
            concept: "Converts vertical shift delta_rho = rho - rho_0 into pitch degrees (negative = head tilting downward). Clamps frame gaps > 0.15 s down to 0.033 s.",
            codeRef: "src/HeadOrientationDetector.cpp (L47-57) | include/MathUtils.hpp (L61-82)"
          }
        },
        {
          id: "chk_pitch", kind: "decision", x: 590, y: 158, w: 250, h: 96,
          label: "pitch <= -25.5 deg?", sublabel: "Sustained droop threshold",
          details: {
            concept: "Checks whether pitch has dropped to or below -25.5 degrees relative to the driver's calibrated upright posture.",
            note: "Types.hpp L195 sets pitch_droop_thresh = -25.5 deg (whereas legacy comments in HeadOrientationDetector.cpp L59 still mention -20 deg).",
            codeRef: "src/HeadOrientationDetector.cpp (L60) | include/Types.hpp (L195)"
          }
        },
        {
          id: "accum_up", kind: "process", x: 270, y: 168, w: 265, h: 76,
          label: "Accumulate Droop Timer", sublabel: "in_droop = true\naccum = min(2.0s, accum + dt)",
          details: {
            concept: "Integrates clamped dt up to target_sec + 0.5 s = 2.0 s cap while head remains tilted down.",
            codeRef: "src/HeadOrientationDetector.cpp (L61-63)"
          }
        },
        {
          id: "chk_sustained", kind: "decision", x: 270, y: 294, w: 265, h: 98,
          label: "accum >= 1.5 s &&\n!droop_cue_emitted?", sublabel: "Single cue per droop",
          details: {
            concept: "Requires at least 1.5 s of downward pitch before emitting one HeadDroop Light cue (droop_cue_emitted prevents repeated firing on the same nod).",
            codeRef: "src/HeadOrientationDetector.cpp (L65-73)"
          }
        },
        {
          id: "emit_light", kind: "process", x: 270, y: 440, w: 265, h: 72,
          label: "Record Light Cue", sublabel: "droop_cue_emitted = true\nPush to 120 s cue window",
          details: {
            concept: "Emits a Light HeadDroop cue into the 2-minute sliding cue window.",
            codeRef: "src/HeadOrientationDetector.cpp (L66-72) | src/AlertManager.cpp (L220-238)"
          }
        },
        {
          id: "chk_two_light", kind: "decision", x: 590, y: 428, w: 265, h: 96,
          label: ">= 2 Light cues\nin 120 s window?", sublabel: "Multimodal light rule",
          details: {
            concept: "A single Light cue alone does not sound an alarm; >= 2 Light cues (HeadDroop or Yawn) within 120 seconds trigger Drowsy.",
            codeRef: "src/AlertLogic.cpp (L51-59) | src/AlertManager.cpp (L223-238)"
          }
        },
        {
          id: "ret_hold", kind: "term_return", x: 24, y: 316, w: 195, h: 54,
          label: "RETURN", sublabel: "Hold / wait 1.5 s",
          details: {
            concept: "Returns without emitting a new cue while waiting for 1.5 s or after the cue has already been emitted for this nod.",
            codeRef: "src/HeadOrientationDetector.cpp (L65-73)"
          }
        },
        {
          id: "term_drowsy", kind: "term_alarm", x: 905, y: 446, w: 190, h: 60,
          label: "DROWSY", sublabel: "2 Light cues in 120 s",
          details: {
            concept: "Threshold reached: driver enters Drowsy alert state.",
            codeRef: "src/AlertManager.cpp (L229-238)"
          }
        },
        {
          id: "ret_one_light", kind: "term_return", x: 905, y: 556, w: 190, h: 54,
          label: "RETURN", sublabel: "1 Light cue banked",
          details: {
            concept: "First Light cue is stored in the 120 s window; no alarm triggers yet unless a second Light cue occurs within 2 minutes.",
            codeRef: "src/AlertLogic.cpp (L51-59)"
          }
        },
        {
          id: "decay_fast", kind: "process", x: 875, y: 168, w: 220, h: 76,
          label: "Fast Recovery (-2.0*dt)", sublabel: "accum = max(0, accum - 2.0*dt)",
          details: {
            concept: "When head lifts above -25.5 deg, the droop accumulator drains at 2x real-time speed.",
            codeRef: "src/HeadOrientationDetector.cpp (L75)"
          }
        },
        {
          id: "chk_zero", kind: "decision", x: 875, y: 280, w: 220, h: 90,
          label: "accum <= 0.0 s?", sublabel: "Full recovery check",
          details: {
            concept: "Once the accumulator drains completely to zero, in_droop and droop_cue_emitted reset for future nods.",
            codeRef: "src/HeadOrientationDetector.cpp (L76-79)"
          }
        },
        {
          id: "reset_flags", kind: "process", x: 605, y: 292, w: 220, h: 66,
          label: "Clear Droop Flags", sublabel: "in_droop = false\ndroop_cue_emitted = false",
          details: {
            concept: "Marks head posture normal again and re-arms the cue latch.",
            codeRef: "src/HeadOrientationDetector.cpp (L77-78)"
          }
        },
        {
          id: "ret_upright", kind: "term_return", x: 605, y: 556, w: 220, h: 54,
          label: "RETURN", sublabel: "Head upright / recovering",
          details: {
            concept: "Exits normal cycle with no cue emitted.",
            codeRef: "src/HeadOrientationDetector.cpp (L88-93)"
          }
        }
      ],
      edges: [
        { id: "e_lm_rho", points: [[244, 67], [280, 67]], tone: "default" },
        { id: "e_rho_pit", points: [[550, 67], [590, 67]], tone: "default" },
        { id: "e_pit_chk", points: [[715, 108], [715, 158]], tone: "default" },
        { id: "e_droop_yes", points: [[590, 206], [535, 206]], label: "YES", tone: "yes" },
        { id: "e_droop_no", points: [[840, 206], [875, 206]], label: "NO", tone: "no" },
        { id: "e_acc_sus", points: [[402, 244], [402, 294]], tone: "default" },
        { id: "e_sus_no", points: [[270, 343], [219, 343]], label: "NO", tone: "no" },
        { id: "e_sus_yes", points: [[402, 392], [402, 440]], label: "YES", tone: "yes" },
        { id: "e_cue_2l", points: [[535, 476], [590, 476]], tone: "default" },
        { id: "e_2l_yes", points: [[855, 476], [905, 476]], label: "YES", tone: "yes" },
        { id: "e_2l_no", points: [[722, 524], [722, 540], [875, 540], [875, 583], [905, 583]], label: "NO", tone: "no", lx: 800, ly: 534 },
        { id: "e_dec_chk", points: [[985, 244], [985, 280]], tone: "default" },
        { id: "e_z_yes", points: [[875, 325], [825, 325]], label: "YES", tone: "yes" },
        { id: "e_z_no", points: [[985, 370], [985, 408], [568, 408], [568, 583], [605, 583]], label: "NO", tone: "no", lx: 785, ly: 400 },
        { id: "e_rst_ret", points: [[605, 340], [568, 340], [568, 583], [605, 583]], tone: "default" }
      ],
      traces: [
        {
          id: "droop_trigger",
          label: "Trace: Sustained Nod <= -25.5 deg (>= 1.5 s)",
          steps: [
            {
              title: "Step 1: Compute Roll-Invariant rho & Pitch",
              highlightNodes: ["in_lm", "calc_rho", "calc_pitch", "chk_pitch"],
              highlightEdges: ["e_lm_rho", "e_rho_pit", "e_pit_chk"],
              concept: "Landmark geometry measures nose-to-chin ratio rho along the eye-orthogonal axis u_y, then maps delta_rho via atan2 to pitch_deg <= -25.5 deg.",
              codeRef: "src/FaceGeometry.cpp (L22-49) | src/HeadOrientationDetector.cpp (L55-60)"
            },
            {
              title: "Step 2: Hold >= 1.5 s -> Light Cue -> 2-in-120 s Check",
              highlightNodes: ["chk_pitch", "accum_up", "chk_sustained", "emit_light", "chk_two_light", "term_drowsy"],
              highlightEdges: ["e_droop_yes", "e_acc_sus", "e_sus_yes", "e_cue_2l", "e_2l_yes"],
              concept: "Once droop_accum reaches 1.5 s, droop_cue_emitted latches true and emits a Light cue. A second Light cue within 120 s triggers DROWSY.",
              codeRef: "src/HeadOrientationDetector.cpp (L61-73) | src/AlertLogic.cpp (L51-59)"
            }
          ]
        },
        {
          id: "droop_recover",
          label: "Trace: Brief Bob -> Fast -2.0*dt Recovery",
          steps: [
            {
              title: "Head Lifts Above -25.5 deg -> 2x Decay & Clear",
              highlightNodes: ["chk_pitch", "decay_fast", "chk_zero", "reset_flags", "ret_upright"],
              highlightEdges: ["e_droop_no", "e_dec_chk", "e_z_yes", "e_rst_ret"],
              concept: "Raising the head drains droop_accum at 2.0*dt (twice real-time speed). Once it hits 0, in_droop and droop_cue_emitted clear cleanly.",
              codeRef: "src/HeadOrientationDetector.cpp (L74-80)"
            }
          ]
        }
      ]
    },
    {
      id: "2B",
      label: "2B. System & FSM Cut-Off",
      subtitle: "Dlib face-loss -1.0*dt decay, calibration gate, 0.2 s yawn suppression & MovingCalibrated cut-off",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "2B: Head Drooping — System & FSM Cut-Off View",
        concept: "Exposes every external boundary around HeadOrientationDetector: Dlib 68-landmark vs Face Lost inputs (1C), neutral rho calibration injection (1B), >= 0.2 s yawn cross-suppression (4B), and the MovingCalibrated FSM gate before calling AlertManager (6B).",
        note: "When parked (StationaryCalibrated), HeadOrientationDetector::process() still runs for live pitch HUD telemetry, but PipelineCoordinator skips alert_manager_.update_somnolence().",
        codeRef: "src/PipelineCoordinator.cpp (L492, L528-560)\nsrc/HeadOrientationDetector.cpp (L13-114)"
      },
      nodes: [
        {
          id: "from_face_gate", kind: "portal_in", x: 20, y: 120, w: 225, h: 66,
          label: "FROM 1C: DLIB_GATE", sublabel: "68 Landmarks | Face Lost",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "out_landmarks_head" },
          details: {
            concept: "Receives either 68 landmarks (calling process()) or Face Lost (calling on_face_lost(), including during 500 ms Dlib throttling).",
            codeRef: "src/PipelineCoordinator.cpp (L547-555)"
          }
        },
        {
          id: "from_calib_rho", kind: "portal_in", x: 20, y: 26, w: 225, h: 64,
          label: "FROM 1B: APPLY_HEAD", sublabel: "baseline_rho_ (fallback 0.5)",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_2b_head_base" },
          details: {
            concept: "apply_baseline(neutral_rho) stores baseline_rho_, sets calibrated_ = true, and resets droop accumulators.",
            codeRef: "src/HeadOrientationDetector.cpp (L13-25)"
          }
        },
        {
          id: "from_yawn_sup", kind: "portal_in", x: 20, y: 410, w: 225, h: 64,
          label: "FROM 4B: YAWN_SUP", sublabel: "suppress_by_yawn >= 0.2s",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "sig_yawn_sup" },
          details: {
            concept: "When YawnDetector accumulates >= 0.2 s, suppress_droop = true forces HeadOrientationDetector into the -2.0*dt decay path.",
            codeRef: "src/PipelineCoordinator.cpp (L528-531, L551) | src/HeadOrientationDetector.cpp (L60)"
          }
        },
        {
          id: "chk_face_path", kind: "decision", x: 295, y: 105, w: 215, h: 96,
          label: "Landmarks present?", sublabel: "1C output branch",
          details: {
            concept: "Branches between active landmark processing and gradual missing-face decay.",
            codeRef: "src/PipelineCoordinator.cpp (L547-555)"
          }
        },
        {
          id: "face_lost_decay", kind: "process", x: 560, y: 26, w: 255, h: 74,
          label: "on_face_lost(now)", sublabel: "accum = max(0, accum - 1.0*dt)\nKeep last pitch; cue = nullopt",
          details: {
            concept: "Decays droop_accum_sec_ at -1.0*dt (slower than upright -2.0*dt) so temporary occlusion neither increments droop nor snaps straight to 0.",
            codeRef: "src/HeadOrientationDetector.cpp (L96-114)"
          }
        },
        {
          id: "chk_cal", kind: "decision", x: 295, y: 245, w: 215, h: 94,
          label: "calibrated_ == true?", sublabel: "Has neutral rho_0",
          details: {
            concept: "Before calibration completes, pitch is clamped to 0.0 deg and droop tracking flags stay cleared.",
            codeRef: "src/HeadOrientationDetector.cpp (L54-86)"
          }
        },
        {
          id: "uncal_zero", kind: "process", x: 560, y: 256, w: 235, h: 72,
          label: "Uncalibrated Guard", sublabel: "pitch = 0.0 deg; reset\nin_droop & droop_accum = 0",
          details: {
            concept: "Prevents uncalibrated rho shifts from triggering droop flags.",
            codeRef: "src/HeadOrientationDetector.cpp (L81-86)"
          }
        },
        {
          id: "calc_pitch_cut", kind: "process", x: 295, y: 374, w: 215, h: 68,
          label: "Compute rho & Pitch", sublabel: "atan2(-(rho-rho_0), 0.18)\n+ compute_clamped_dt",
          details: {
            concept: "Computes instantaneous pitch_deg and updates last_rho_ and last_pitch_deg_.",
            codeRef: "src/HeadOrientationDetector.cpp (L47-57)"
          }
        },
        {
          id: "chk_guard_droop", kind: "decision", x: 550, y: 362, w: 260, h: 94,
          label: "!suppress_by_yawn &&\npitch <= -25.5 deg?", sublabel: "Yawn-guarded check",
          details: {
            concept: "Combines yawn suppression and angle check: if yawning >= 0.2 s OR head is above -25.5 deg, enters fast -2.0*dt decay.",
            codeRef: "src/HeadOrientationDetector.cpp (L60)"
          }
        },
        {
          id: "droop_acc_cut", kind: "process", x: 550, y: 496, w: 260, h: 72,
          label: "Droop +dt -> Cue at 1.5s", sublabel: "accum = min(2.0s, accum + dt)\n>=1.5s -> Light HeadDroop cue",
          details: {
            concept: "Accumulates droop timer up to 1.5 s and attaches optional CueEvent(HeadDroop, Light) to HeadOrientationAssessment.",
            codeRef: "src/HeadOrientationDetector.cpp (L61-73)"
          }
        },
        {
          id: "droop_dec_cut", kind: "process", x: 295, y: 496, w: 215, h: 72,
          label: "Decay (-2.0*dt) & Clear", sublabel: "accum = max(0, accum - 2.0*dt)\nClear flags at <= 0.0",
          details: {
            concept: "Drains droop_accum_sec_ at twice real time when head lifts or yawn suppression activates.",
            codeRef: "src/HeadOrientationDetector.cpp (L74-80)"
          }
        },
        {
          id: "read_mov_cal", kind: "decision", x: 855, y: 245, w: 235, h: 98,
          label: "READ 1A: SystemState\n== MovingCalibrated?", sublabel: "Click -> 1A FSM",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            concept: "PipelineCoordinator checks is_moving_calibrated before sending HeadOrientationAssessment into AlertManager::update_somnolence().",
            note: "Parked Latch Caveat: if parked and calibrated, drooping >= 1.5 s sets droop_cue_emitted_ = true even though AlertManager isn't called; lift head to re-arm.",
            codeRef: "src/PipelineCoordinator.cpp (L515, L557-559)"
          }
        },
        {
          id: "goto_6b", kind: "portal_out", x: 865, y: 412, w: 225, h: 66,
          label: "GOTO 6B: CUE_FUSION", sublabel: "alert_manager_.update_\nsomnolence(ear, head, yawn)",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
          details: {
            concept: "Passes HeadOrientationAssessment into AlertManager::mutex_ for Light cue fusion and recovery evaluation.",
            codeRef: "src/PipelineCoordinator.cpp (L558) | src/AlertManager.cpp (L272-315)"
          }
        },
        {
          id: "ret_hud_only", kind: "term_return", x: 865, y: 122, w: 225, h: 58,
          label: "RETURN (HUD Only)", sublabel: "Telemetry without alerts",
          details: {
            concept: "Updates DebugVisualizer HUD pitch readout while suppressing cue registration when parked or uncalibrated.",
            codeRef: "src/PipelineCoordinator.cpp (L557-575)"
          }
        }
      ],
      edges: [
        { id: "e_in_chk", points: [[245, 153], [295, 153]], tone: "default" },
        { id: "e_fl_yes", points: [[402, 105], [402, 63], [560, 63]], label: "NO (Lost)", tone: "no", lx: 475, ly: 55 },
        { id: "e_fl_gate", points: [[815, 63], [835, 63], [835, 294], [855, 294]], tone: "default" },
        { id: "e_ff_cal", points: [[402, 201], [402, 245]], label: "YES", tone: "yes" },
        { id: "e_cal_no", points: [[510, 292], [560, 292]], label: "NO", tone: "no" },
        { id: "e_unc_gate", points: [[795, 294], [855, 294]], tone: "default" },
        { id: "e_cal_yes", points: [[402, 339], [402, 374]], label: "YES", tone: "yes" },
        { id: "e_pit_grd", points: [[510, 408], [550, 408]], tone: "default" },
        { id: "e_yawn_in", points: [[245, 442], [530, 442], [530, 424], [550, 424]], label: "suppress", tone: "portal", lx: 380, ly: 454 },
        { id: "e_grd_yes", points: [[680, 456], [680, 496]], label: "YES", tone: "yes" },
        { id: "e_grd_no", points: [[640, 456], [640, 474], [402, 474], [402, 496]], label: "NO / Yawn", tone: "no", lx: 515, ly: 474 },
        { id: "e_acc_gate", points: [[810, 532], [835, 532], [835, 294], [855, 294]], tone: "default" },
        { id: "e_dec_gate", points: [[402, 568], [402, 606], [835, 606], [835, 294], [855, 294]], tone: "default" },
        { id: "e_mov_no", points: [[972, 245], [972, 180]], label: "NO", tone: "no", lx: 990, ly: 215 },
        { id: "e_mov_yes", points: [[972, 343], [972, 412]], label: "YES", tone: "yes", lx: 992, ly: 378 }
      ],
      traces: [
        {
          id: "cut_droop_ok",
          label: "Trace: MovingCalibrated Droop -> GOTO 6B",
          steps: [
            {
              title: "Step 1: Landmarks & Calibration Active",
              highlightNodes: ["from_face_gate", "from_calib_rho", "chk_face_path", "chk_cal", "calc_pitch_cut"],
              highlightEdges: ["e_in_chk", "e_ff_cal", "e_cal_yes", "e_pit_grd"],
              concept: "68 landmarks arrive from 1C with calibrated baseline_rho_ applied from 1B, computing current pitch_deg.",
              codeRef: "src/HeadOrientationDetector.cpp (L42-57)"
            },
            {
              title: "Step 2: Not Yawning & <= -25.5 deg -> MovingCalibrated -> GOTO 6B",
              highlightNodes: ["chk_guard_droop", "droop_acc_cut", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_grd_yes", "e_acc_gate", "e_mov_yes"],
              concept: "When 1.5 s accumulates without yawn suppression, MovingCalibrated allows HeadOrientationAssessment into AlertManager::update_somnolence (6B).",
              portalTarget: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
              codeRef: "src/HeadOrientationDetector.cpp (L60-73) | src/PipelineCoordinator.cpp (L557-558)"
            }
          ]
        },
        {
          id: "cut_yawn_sup",
          label: "Trace: Yawn >= 0.2 s Suppresses Pitch Drop",
          steps: [
            {
              title: "Yawn Signal Forces -2.0*dt Decay",
              highlightNodes: ["from_yawn_sup", "chk_guard_droop", "droop_dec_cut", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_yawn_in", "e_grd_no", "e_dec_gate", "e_mov_yes"],
              concept: "Even if pitch dips <= -25.5 deg during a deep yawn, suppress_by_yawn forces the -2.0*dt decay branch so no false HeadDroop cue emits.",
              codeRef: "src/PipelineCoordinator.cpp (L528-531, L551) | src/HeadOrientationDetector.cpp (L60, L74-80)"
            }
          ]
        },
        {
          id: "cut_face_lost",
          label: "Trace: Face Lost Mid-Droop (-1.0*dt Decay)",
          steps: [
            {
              title: "on_face_lost Decays Accumulator Gradually",
              highlightNodes: ["from_face_gate", "chk_face_path", "face_lost_decay", "read_mov_cal"],
              highlightEdges: ["e_in_chk", "e_fl_yes", "e_fl_gate"],
              concept: "If Dlib loses the face (e.g., extreme pitch or occlusion), on_face_lost() decays droop_accum at -1.0*dt instead of inflating or snapping to 0.",
              codeRef: "src/HeadOrientationDetector.cpp (L96-114)"
            }
          ]
        },
        {
          id: "cut_parked",
          label: "Trace: StationaryCalibrated (HUD Telemetry Only)",
          steps: [
            {
              title: "Parked Vehicle Gates AlertManager",
              highlightNodes: ["calc_pitch_cut", "read_mov_cal", "ret_hud_only"],
              highlightEdges: ["e_pit_grd", "e_mov_no"],
              concept: "While parked, pitch is still computed for DebugVisualizer HUD, but SystemState != MovingCalibrated skips AlertManager.",
              codeRef: "src/PipelineCoordinator.cpp (L551, L557-575)"
            }
          ]
        }
      ]
    }
  ]
});
