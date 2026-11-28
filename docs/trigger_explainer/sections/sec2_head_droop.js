window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec2",
  title: "2. Head Drooping",
  subviews: [
    {
      id: "2A",
      label: "2A. Fused Algorithm",
      subtitle: "Roll-invariant vertical face proportion, pitch <= -25.5 deg for 1.5 s, and 2x fast recovery",
      viewBox: "0 0 1080 580",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "Head Drooping — Fused Algorithm View",
        concept: "Projects the nose tip and chin onto a roll-invariant vertical face axis anchored by the eye corners, maps shift from the calibrated upright posture to pitch angle, and requires pitch at or below -25.5 degrees for 1.5 seconds (with 2x fast recovery when upright) to bank a Light fatigue cue.",
        code: "double delta_rho = rho - baseline_rho_;\ndouble pitch_deg = std::atan2(-delta_rho, 0.18) * (180.0 / CV_PI);\nif (pitch_deg <= -25.5) { droop_accum_sec_ = std::min(2.0, droop_accum_sec_ + dt); }\nelse { droop_accum_sec_ = std::max(0.0, droop_accum_sec_ - 2.0 * dt); }",
        note: "Thresholds in HeadOrientationConfig: pitch gain = 0.18, droop threshold = -25.5 deg (tuned for BU HT dataset 5.48 deg MAE), sustained duration = 1.5 s. Legacy comments in HeadOrientationDetector.cpp L59 still mention -20 deg.",
        codeRef: "src/FaceGeometry.cpp (L22-49)\nsrc/HeadOrientationDetector.cpp (L42-94)\ninclude/MathUtils.hpp (L61-82)\ninclude/Types.hpp (L193-197)"
      },
      nodes: [
        {
          id: "in_lm", kind: "process", x: 24, y: 28, w: 205, h: 56,
          label: "Eye, Nose & Chin Pts", sublabel: "Plus calibrated neutral pose",
          details: {
            title: "Landmark Geometry Inputs",
            concept: "Takes the outer eye corners, nose tip, and chin coordinates along with the driver's calibrated upright vertical ratio.",
            code: "// Dlib indices: outer eyes p36 & p45, nose tip p33, chin p8\n// Baseline: baseline_rho_ from Phase 1 calibration",
            codeRef: "include/FaceGeometry.hpp (L10-28)"
          }
        },
        {
          id: "calc_rho", kind: "process", x: 265, y: 28, w: 245, h: 56,
          label: "Roll-Invariant Ratio", sublabel: "Project along eye-perp axis",
          details: {
            title: "Roll-Invariant Vertical Ratio",
            concept: "Computes the downward unit axis perpendicular to the eye line and divides upper nose distance by total nose-to-chin face height, making the metric immune to head tilt (roll).",
            code: "u_y = (-u_x.y, u_x.x); // Downward perpendicular to eye vector\nd_top = (p33 - e_mid).dot(u_y); d_bot = (p8 - p33).dot(u_y);\nrho = d_top / std::max(d_top + d_bot, 1e-6);",
            codeRef: "src/FaceGeometry.cpp (L22-49)"
          }
        },
        {
          id: "calc_pitch", kind: "process", x: 550, y: 28, w: 240, h: 56,
          label: "Compute Pitch & Time Step", sublabel: "Map shift to degrees; clamp gap",
          details: {
            title: "Pitch Estimation & Clamped Step",
            concept: "Converts vertical proportion shift into pitch angle in degrees (negative means chin dropping downward) and clamps frame time gaps above 0.15 s down to 0.033 s.",
            code: "const double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);\ndouble pitch_deg = std::atan2(-(rho - baseline_rho_), 0.18) * (180.0 / CV_PI);",
            codeRef: "src/HeadOrientationDetector.cpp (L47-57) | include/MathUtils.hpp (L61-82)"
          }
        },
        {
          id: "chk_pitch", kind: "decision", x: 550, y: 126, w: 240, h: 84,
          label: "Pitch <= -25.5 deg?", sublabel: "Droop angle check",
          details: {
            title: "Head Droop Angle Check",
            concept: "Tests whether estimated head pitch has fallen at or below -25.5 degrees relative to calibrated neutral posture.",
            code: "if (pitch_deg <= config_.pitch_droop_thresh) { // -25.5 deg",
            note: "Configured as -25.5 deg in Types.hpp L195 (older code comments mention -20 deg).",
            codeRef: "src/HeadOrientationDetector.cpp (L60) | include/Types.hpp (L195)"
          }
        },
        {
          id: "accum_up", kind: "process", x: 255, y: 140, w: 245, h: 56,
          label: "Accumulate Droop Timer", sublabel: "Add time step up to 2.0 s cap",
          details: {
            title: "Integrate Droop Duration",
            concept: "Marks head drooping active and adds the clamped frame time step up to a 2.0-second ceiling.",
            code: "in_droop_ = true;\ndroop_accum_sec_ = std::min(1.5 + 0.5, droop_accum_sec_ + dt);",
            codeRef: "src/HeadOrientationDetector.cpp (L61-63)"
          }
        },
        {
          id: "chk_sustained", kind: "decision", x: 255, y: 242, w: 245, h: 86,
          label: "Held >= 1.5 s &\nNot Yet Latched?", sublabel: "One cue per nod",
          details: {
            title: "Sustained 1.5-Second Check",
            concept: "Waits until downward pitch persists for at least 1.5 seconds and ensures only one Light cue fires per continuous nod.",
            code: "if (droop_accum_sec_ >= 1.5 && !droop_cue_emitted_) { ... }",
            codeRef: "src/HeadOrientationDetector.cpp (L65-73)"
          }
        },
        {
          id: "ret_hold", kind: "term_return", x: 24, y: 260, w: 185, h: 50,
          label: "RETURN", sublabel: "Waiting or already latched",
          details: {
            title: "Return Without New Cue",
            concept: "Exits cleanly while accumulating toward 1.5 seconds or after the current nod has already recorded its cue.",
            code: "return HeadOrientationAssessment{pitch_deg, rho, in_droop_, std::nullopt};",
            codeRef: "src/HeadOrientationDetector.cpp (L65-73)"
          }
        },
        {
          id: "emit_light", kind: "process", x: 255, y: 374, w: 245, h: 56,
          label: "Record Light Cue", sublabel: "Latch nod & log to 120 s window",
          details: {
            title: "Record Light Fatigue Cue",
            concept: "Latches the droop cue flag and adds a Light Head Droop event into the 2-minute sliding cue window.",
            code: "droop_cue_emitted_ = true;\ncue = CueEvent{CueType::HeadDroop, CueStrength::Light, now, ...};",
            codeRef: "src/HeadOrientationDetector.cpp (L66-72) | src/AlertManager.cpp (L220-238)"
          }
        },
        {
          id: "chk_two_light", kind: "decision", x: 550, y: 358, w: 245, h: 86,
          label: "2+ Light Cues\nin 120 Seconds?", sublabel: "Multimodal light rule",
          details: {
            title: "2-Minute Light Cue Check",
            concept: "A single Light cue alone is banked quietly; two or more Light cues (head droop or yawning) within 120 seconds trigger Drowsy.",
            code: "if (cue_window_.size() >= 2) { current_state_ = DriverState::Drowsy; }",
            codeRef: "src/AlertLogic.cpp (L51-59) | src/AlertManager.cpp (L223-238)"
          }
        },
        {
          id: "term_drowsy", kind: "term_alarm", x: 855, y: 375, w: 195, h: 52,
          label: "DROWSY", sublabel: "2 Light cues in 120 s",
          details: {
            title: "Drowsiness Verdict",
            concept: "Two Light cues within 2 minutes escalate driver state to Drowsy.",
            code: "current_state_ = DriverState::Drowsy; // Trigger escalating alarm",
            codeRef: "src/AlertManager.cpp (L229-238)"
          }
        },
        {
          id: "ret_one_light", kind: "term_return", x: 855, y: 486, w: 195, h: 50,
          label: "RETURN", sublabel: "1 Light cue banked",
          details: {
            title: "Banked First Light Cue",
            concept: "Stores the cue in the 2-minute history; no alarm fires unless a second Light cue occurs within 120 seconds.",
            code: "cue_window_.push_back(cue); // Size < 2 -> keep Safe",
            codeRef: "src/AlertLogic.cpp (L51-59)"
          }
        },
        {
          id: "decay_fast", kind: "process", x: 840, y: 140, w: 215, h: 56,
          label: "Fast 2x Recovery Decay", sublabel: "Drain droop timer at 2x speed",
          details: {
            title: "Asymmetric Fast Recovery",
            concept: "When the head lifts above -25.5 deg, the droop timer drains at twice real-time speed.",
            code: "droop_accum_sec_ = std::max(0.0, droop_accum_sec_ - 2.0 * dt);",
            codeRef: "src/HeadOrientationDetector.cpp (L75)"
          }
        },
        {
          id: "chk_zero", kind: "decision", x: 840, y: 234, w: 215, h: 82,
          label: "Droop Timer\nDrained to 0?", sublabel: "Full recovery check",
          details: {
            title: "Full Posture Recovery Check",
            concept: "Once the timer drains back to zero, clears drooping state and re-arms the cue latch for future nods.",
            code: "if (droop_accum_sec_ <= 0.0) {\n  in_droop_ = false; droop_cue_emitted_ = false;\n}",
            codeRef: "src/HeadOrientationDetector.cpp (L76-79)"
          }
        },
        {
          id: "reset_flags", kind: "process", x: 565, y: 247, w: 215, h: 56,
          label: "Clear Droop Flags", sublabel: "Posture normal; re-arm latch",
          details: {
            title: "Re-Arm Droop Latch",
            concept: "Clears active drooping state and allows future downward nods to emit a new cue.",
            code: "in_droop_ = false;\ndroop_cue_emitted_ = false;",
            codeRef: "src/HeadOrientationDetector.cpp (L77-78)"
          }
        },
        {
          id: "ret_upright", kind: "term_return", x: 565, y: 486, w: 215, h: 50,
          label: "RETURN", sublabel: "Upright / recovering",
          details: {
            title: "Return Upright",
            concept: "Exits cycle cleanly with head posture normal or recovering.",
            code: "return HeadOrientationAssessment{pitch_deg, rho, in_droop_, std::nullopt};",
            codeRef: "src/HeadOrientationDetector.cpp (L88-93)"
          }
        }
      ],
      edges: [
        { id: "e_lm_rho", points: [[229, 56], [265, 56]], tone: "default" },
        { id: "e_rho_pit", points: [[510, 56], [550, 56]], tone: "default" },
        { id: "e_pit_chk", points: [[670, 84], [670, 126]], tone: "default" },
        { id: "e_droop_yes", points: [[550, 168], [500, 168]], label: "YES", tone: "yes" },
        { id: "e_droop_no", points: [[790, 168], [840, 168]], label: "NO", tone: "no" },
        { id: "e_acc_sus", points: [[377, 196], [377, 242]], tone: "default" },
        { id: "e_sus_no", points: [[255, 285], [209, 285]], label: "NO", tone: "no" },
        { id: "e_sus_yes", points: [[377, 328], [377, 374]], label: "YES", tone: "yes" },
        { id: "e_cue_2l", points: [[500, 402], [550, 402]], tone: "default" },
        { id: "e_2l_yes", points: [[795, 402], [855, 402]], label: "YES", tone: "yes" },
        { id: "e_2l_no", points: [[672, 444], [672, 464], [824, 464], [824, 511], [855, 511]], label: "NO", tone: "no", lx: 755, ly: 458 },
        { id: "e_dec_chk", points: [[947, 196], [947, 234]], tone: "default" },
        { id: "e_z_yes", points: [[840, 275], [780, 275]], label: "YES", tone: "yes" },
        { id: "e_z_no", points: [[947, 316], [947, 344], [526, 344], [526, 511], [565, 511]], label: "NO", tone: "no", lx: 740, ly: 336 },
        { id: "e_rst_ret", points: [[565, 286], [526, 286], [526, 511], [565, 511]], tone: "default" }
      ],
      traces: [
        {
          id: "droop_trigger",
          label: "Trace: Sustained Nod <= -25.5 deg (>= 1.5 s)",
          steps: [
            {
              title: "Step 1: Compute Roll-Invariant Ratio & Pitch",
              highlightNodes: ["in_lm", "calc_rho", "calc_pitch", "chk_pitch"],
              highlightEdges: ["e_lm_rho", "e_rho_pit", "e_pit_chk"],
              concept: "Face geometry projects nose and chin points onto the downward eye-perpendicular axis and maps deviation from calibration into degrees.",
              code: "rho = d_top / std::max(d_top + d_bot, 1e-6);\npitch_deg = std::atan2(-(rho - baseline_rho_), 0.18) * (180.0 / CV_PI);"
            },
            {
              title: "Step 2: Hold 1.5 s -> Light Cue -> 2-in-120 s Check",
              highlightNodes: ["chk_pitch", "accum_up", "chk_sustained", "emit_light", "chk_two_light", "term_drowsy"],
              highlightEdges: ["e_droop_yes", "e_acc_sus", "e_sus_yes", "e_cue_2l", "e_2l_yes"],
              concept: "After 1.5 seconds of downward tilt, a Light Head Droop cue records into the 2-minute window; two Light cues within 2 minutes trigger Drowsy.",
              code: "if (droop_accum_sec_ >= 1.5 && !droop_cue_emitted_) {\n  droop_cue_emitted_ = true; // Emit Light cue -> Drowsy if >= 2 in 120 s\n}"
            }
          ]
        },
        {
          id: "droop_recover",
          label: "Trace: Brief Bob -> Fast 2x Recovery",
          steps: [
            {
              title: "Head Lifts -> 2x Decay & Clear Flags",
              highlightNodes: ["chk_pitch", "decay_fast", "chk_zero", "reset_flags", "ret_upright"],
              highlightEdges: ["e_droop_no", "e_dec_chk", "e_z_yes", "e_rst_ret"],
              concept: "Lifting the head above -25.5 degrees drains the droop timer twice as fast as real time and clears drooping flags at zero.",
              code: "droop_accum_sec_ = std::max(0.0, droop_accum_sec_ - 2.0 * dt);\nif (droop_accum_sec_ <= 0.0) { in_droop_ = false; droop_cue_emitted_ = false; }"
            }
          ]
        }
      ]
    },
    {
      id: "2B",
      label: "2B. System & FSM Cut-Off",
      subtitle: "Face-loss 1x decay, calibration gate, 0.2 s yawn suppression & Moving-Calibrated cut-off",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "Head Drooping — System & FSM Cut-Off View",
        concept: "Exposes the runtime system boundaries around head pose monitoring: landmark vs. face-lost branches (1C), neutral calibration injection (1B), early yawn suppression (4B), and the moving-calibrated gate before entering cue fusion (6B).",
        code: "if (landmarks_opt) head_res = head_detector_.process(*landmarks_opt, now, suppress_by_yawn);\nelse head_res = head_detector_.on_face_lost(now);\nif (is_moving_calibrated) alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
        note: "When parked and calibrated, pitch estimation still updates for live telemetry while alerts stay gated off.",
        codeRef: "src/PipelineCoordinator.cpp (L492, L528-560)\nsrc/HeadOrientationDetector.cpp (L13-114)"
      },
      nodes: [
        {
          id: "from_calib_rho", kind: "portal_in", x: 20, y: 24, w: 215, h: 56,
          label: "FROM 1B: Apply Head Base", sublabel: "Save neutral vertical ratio",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_2b_head_base" },
          details: {
            title: "Calibration Baseline Injection",
            concept: "Applies the median neutral vertical face ratio measured in calibration Phase 1 (with 0.50 fallback) and clears droop timers.",
            code: "void HeadOrientationDetector::apply_baseline(double neutral_rho) {\n  baseline_rho_ = (neutral_rho > 0.0) ? neutral_rho : 0.50;\n  calibrated_ = true; droop_accum_sec_ = 0.0;\n}",
            codeRef: "src/HeadOrientationDetector.cpp (L13-25)"
          }
        },
        {
          id: "from_face_gate", kind: "portal_in", x: 20, y: 112, w: 215, h: 56,
          label: "FROM 1C: Face Gate", sublabel: "68 landmarks or face lost",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "out_landmarks_head" },
          details: {
            title: "Shared Landmark Input",
            concept: "Receives 68 landmarks when a face is verified, or enters the face-lost handler when absent or throttled.",
            code: "if (landmarks_opt.has_value()) head_res = head_detector_.process(...);\nelse head_res = head_detector_.on_face_lost(t_work_start);",
            codeRef: "src/PipelineCoordinator.cpp (L547-555)"
          }
        },
        {
          id: "from_yawn_sup", kind: "portal_in", x: 20, y: 368, w: 215, h: 56,
          label: "FROM 4B: Yawn Suppressor", sublabel: "Active when yawning >= 0.2 s",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "sig_yawn_sup" },
          details: {
            title: "Yawn Suppression Signal",
            concept: "When the full-frame yawn model has detected yawning for at least 0.2 seconds, head droop accumulation is blocked to prevent false alarms from head tilt during yawns.",
            code: "const bool suppress_by_yawn = (yawn_res.yawn_duration_sec >= 0.2);",
            codeRef: "src/PipelineCoordinator.cpp (L528-531, L551)"
          }
        },
        {
          id: "chk_face_path", kind: "decision", x: 280, y: 98, w: 210, h: 84,
          label: "Landmarks\nPresent?", sublabel: "Face found vs lost",
          details: {
            title: "Landmark Availability Branch",
            concept: "Routes frames with verified landmarks to pitch evaluation and missing frames to gradual 1x timer decay.",
            code: "if (landmarks_opt.has_value()) { /* process */ } else { /* on_face_lost */ }",
            codeRef: "src/PipelineCoordinator.cpp (L547-555)"
          }
        },
        {
          id: "face_lost_decay", kind: "process", x: 540, y: 24, w: 245, h: 56,
          label: "Gradual Face-Lost Decay", sublabel: "Drain at 1x speed; keep pitch",
          details: {
            title: "Face-Lost Handler",
            concept: "Drains the accumulated droop timer at 1x real-time speed during missing frames (slower than the 2x upright recovery) and preserves the last known pitch angle without emitting cues.",
            code: "droop_accum_sec_ = std::max(0.0, droop_accum_sec_ - 1.0 * dt);\nif (droop_accum_sec_ <= 0.0) { in_droop_ = false; droop_cue_emitted_ = false; }\nreturn HeadOrientationAssessment{last_pitch_deg_, last_rho_, in_droop_, std::nullopt};",
            codeRef: "src/HeadOrientationDetector.cpp (L96-114)"
          }
        },
        {
          id: "chk_cal", kind: "decision", x: 280, y: 218, w: 210, h: 82,
          label: "Head Baseline\nCalibrated?", sublabel: "Has neutral pose",
          details: {
            title: "Calibration Readiness Check",
            concept: "Before calibration completes, pitch stays zeroed and droop tracking stays inactive.",
            code: "if (!calibrated_) { last_pitch_deg_ = 0.0; in_droop_ = false; droop_accum_sec_ = 0.0; }",
            codeRef: "src/HeadOrientationDetector.cpp (L54-86)"
          }
        },
        {
          id: "uncal_zero", kind: "process", x: 540, y: 231, w: 225, h: 56,
          label: "Uncalibrated Guard", sublabel: "Zero pitch & keep timer reset",
          details: {
            title: "Uncalibrated Guard",
            concept: "Keeps pitch at 0 degrees and droop flags cleared until calibration provides a personal baseline.",
            code: "pitch_deg = 0.0; in_droop_ = false; droop_cue_emitted_ = false; droop_accum_sec_ = 0.0;",
            codeRef: "src/HeadOrientationDetector.cpp (L81-86)"
          }
        },
        {
          id: "calc_pitch_cut", kind: "process", x: 280, y: 332, w: 210, h: 56,
          label: "Compute Ratio & Pitch", sublabel: "Calculate angle & time delta",
          details: {
            title: "Pitch & Time Step Calculation",
            concept: "Computes instantaneous head pitch in degrees relative to baseline and updates clamped frame delta time.",
            code: "double rho = compute_rho(landmarks);\npitch_deg = std::atan2(-(rho - baseline_rho_), 0.18) * (180.0 / CV_PI);",
            codeRef: "src/HeadOrientationDetector.cpp (L47-57)"
          }
        },
        {
          id: "chk_guard_droop", kind: "decision", x: 530, y: 318, w: 245, h: 84,
          label: "Not Yawning &\nPitch <= -25.5 deg?", sublabel: "Yawn-guarded test",
          details: {
            title: "Yawn-Guarded Droop Test",
            concept: "Only increments droop when yawning suppression is inactive AND head pitch is at or below -25.5 degrees.",
            code: "if (!suppress_droop && pitch_deg <= config_.pitch_droop_thresh) { ... }",
            codeRef: "src/HeadOrientationDetector.cpp (L60)"
          }
        },
        {
          id: "droop_acc_cut", kind: "process", x: 530, y: 444, w: 245, h: 56,
          label: "Accumulate & Emit at 1.5 s", sublabel: "Attach Light cue if latched",
          details: {
            title: "Droop Accumulation & Cue Output",
            concept: "Increments the droop timer up to 1.5 seconds and attaches a Light head-droop cue once per nod.",
            code: "droop_accum_sec_ = std::min(1.5 + 0.5, droop_accum_sec_ + dt);\nif (droop_accum_sec_ >= 1.5 && !droop_cue_emitted_) cue = CueEvent{HeadDroop, Light, ...};",
            codeRef: "src/HeadOrientationDetector.cpp (L61-73)"
          }
        },
        {
          id: "droop_dec_cut", kind: "process", x: 280, y: 444, w: 210, h: 56,
          label: "2x Recovery Decay", sublabel: "Drain timer & clear flags at 0",
          details: {
            title: "Fast 2x Decay Branch",
            concept: "When head posture is upright OR yawn suppression is active, drains the droop timer at 2x speed and clears flags at zero.",
            code: "droop_accum_sec_ = std::max(0.0, droop_accum_sec_ - 2.0 * dt);\nif (droop_accum_sec_ <= 0.0) { in_droop_ = false; droop_cue_emitted_ = false; }",
            codeRef: "src/HeadOrientationDetector.cpp (L74-80)"
          }
        },
        {
          id: "read_mov_cal", kind: "decision", x: 825, y: 216, w: 230, h: 86,
          label: "READ 1A: Moving\n& Calibrated?", sublabel: "Lifecycle gate",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            title: "Vehicle Movement & Calibration Gate",
            concept: "The pipeline coordinator forwards head posture assessments into cue fusion only when the vehicle is both moving and calibrated.",
            code: "const bool is_moving_calibrated = (system_state_.load() == SystemState::MovingCalibrated);\nif (is_moving_calibrated) alert_manager_.update_somnolence(...);",
            note: "Parked Latch Caveat: drooping for 1.5 s while parked sets the cue-emitted latch locally even though the alert coordinator isn't called; lift head to re-arm.",
            codeRef: "src/PipelineCoordinator.cpp (L515, L557-559)"
          }
        },
        {
          id: "goto_6b", kind: "portal_out", x: 830, y: 368, w: 220, h: 56,
          label: "GOTO 6B: Cue Fusion", sublabel: "Forward head posture result",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
          details: {
            title: "Forward to Alert Coordinator",
            concept: "Sends the frame's head posture assessment into the mutex-protected cue fusion logic.",
            code: "alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
            codeRef: "src/PipelineCoordinator.cpp (L558) | src/AlertManager.cpp (L272-315)"
          }
        },
        {
          id: "ret_hud_only", kind: "term_return", x: 830, y: 106, w: 220, h: 50,
          label: "RETURN (HUD Only)", sublabel: "Live pitch telemetry only",
          details: {
            title: "Telemetry Only Exit",
            concept: "Updates live on-screen pitch diagnostics while suppressing alerts whenever parked or uncalibrated.",
            code: "debug_visualizer_.update_somnolence(..., head_res, ...);",
            codeRef: "src/PipelineCoordinator.cpp (L557-575)"
          }
        }
      ],
      edges: [
        { id: "e_in_chk", points: [[235, 140], [280, 140]], tone: "default" },
        { id: "e_fl_yes", points: [[385, 98], [385, 52], [540, 52]], label: "NO", tone: "no", lx: 460, ly: 44 },
        { id: "e_fl_gate", points: [[785, 52], [805, 52], [805, 259], [825, 259]], tone: "default" },
        { id: "e_ff_cal", points: [[385, 182], [385, 218]], label: "YES", tone: "yes" },
        { id: "e_cal_no", points: [[490, 259], [540, 259]], label: "NO", tone: "no" },
        { id: "e_unc_gate", points: [[765, 259], [825, 259]], tone: "default" },
        { id: "e_cal_yes", points: [[385, 300], [385, 332]], label: "YES", tone: "yes" },
        { id: "e_pit_grd", points: [[490, 360], [530, 360]], tone: "default" },
        { id: "e_yawn_in", points: [[235, 396], [510, 396], [510, 376], [530, 376]], label: "suppress", tone: "portal", lx: 375, ly: 408 },
        { id: "e_grd_yes", points: [[652, 402], [652, 444]], label: "YES", tone: "yes" },
        { id: "e_grd_no", points: [[620, 402], [620, 422], [385, 422], [385, 444]], label: "NO / Yawn", tone: "no", lx: 500, ly: 420 },
        { id: "e_acc_gate", points: [[775, 472], [805, 472], [805, 259], [825, 259]], tone: "default" },
        { id: "e_dec_gate", points: [[385, 500], [385, 534], [805, 534], [805, 259], [825, 259]], tone: "default" },
        { id: "e_mov_no", points: [[940, 216], [940, 156]], label: "NO", tone: "no", lx: 958, ly: 186 },
        { id: "e_mov_yes", points: [[940, 302], [940, 368]], label: "YES", tone: "yes", lx: 960, ly: 335 }
      ],
      traces: [
        {
          id: "cut_droop_ok",
          label: "Trace: Moving & Calibrated Droop -> GOTO 6B",
          steps: [
            {
              title: "Step 1: Landmarks & Calibration Active",
              highlightNodes: ["from_face_gate", "from_calib_rho", "chk_face_path", "chk_cal", "calc_pitch_cut"],
              highlightEdges: ["e_in_chk", "e_ff_cal", "e_cal_yes", "e_pit_grd"],
              concept: "68 landmarks arrive with calibrated neutral pose active, allowing pitch calculation.",
              code: "double pitch_deg = std::atan2(-(rho - baseline_rho_), 0.18) * (180.0 / CV_PI);"
            },
            {
              title: "Step 2: Sustained Droop -> Cue Fusion (6B)",
              highlightNodes: ["chk_guard_droop", "droop_acc_cut", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_grd_yes", "e_acc_gate", "e_mov_yes"],
              concept: "Without yawn suppression, 1.5 s at or below -25.5 deg emits a Light cue and passes through the moving-calibrated gate into cue fusion (6B).",
              code: "if (is_moving_calibrated) alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
              portalTarget: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" }
            }
          ]
        },
        {
          id: "cut_yawn_sup",
          label: "Trace: Yawn >= 0.2 s Suppresses Pitch Drop",
          steps: [
            {
              title: "Yawn Signal Forces 2x Timer Decay",
              highlightNodes: ["from_yawn_sup", "chk_guard_droop", "droop_dec_cut", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_yawn_in", "e_grd_no", "e_dec_gate", "e_mov_yes"],
              concept: "Even if head pitch dips below -25.5 degrees during a deep yawn, yawn suppression forces the 2x decay branch so no false cue emits.",
              code: "if (!suppress_droop && pitch_deg <= -25.5) { ... } else { droop_accum_sec_ -= 2.0 * dt; }"
            }
          ]
        },
        {
          id: "cut_face_lost",
          label: "Trace: Face Lost Mid-Droop (1x Decay)",
          steps: [
            {
              title: "Missing Landmarks Trigger Gradual 1x Decay",
              highlightNodes: ["from_face_gate", "chk_face_path", "face_lost_decay", "read_mov_cal"],
              highlightEdges: ["e_in_chk", "e_fl_yes", "e_fl_gate"],
              concept: "When the face detector loses the driver, the timer drains gradually at 1x speed rather than jumping to zero or inflating.",
              code: "droop_accum_sec_ = std::max(0.0, droop_accum_sec_ - 1.0 * dt);"
            }
          ]
        },
        {
          id: "cut_parked",
          label: "Trace: Parked & Calibrated (Telemetry Only)",
          steps: [
            {
              title: "Parked State Blocks Cue Fusion",
              highlightNodes: ["calc_pitch_cut", "read_mov_cal", "ret_hud_only"],
              highlightEdges: ["e_pit_grd", "e_mov_no"],
              concept: "While stationary, live pitch telemetry still updates the debug display, but alerts are bypassed.",
              code: "if (is_moving_calibrated) alert_manager_.update_somnolence(...); // Skipped"
            }
          ]
        }
      ]
    }
  ]
});
