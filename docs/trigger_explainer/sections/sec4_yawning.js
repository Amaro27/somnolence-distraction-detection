window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec4",
  title: "4. Yawning",
  subviews: [
    {
      id: "4A",
      label: "4A. Fused Algorithm",
      subtitle: "Full-frame 112x112 yawning neural network, 80% confidence for 1.0 s, 1x linear decay & 2-in-120 s rule",
      viewBox: "0 0 1080 580",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "Yawning — Fused Algorithm View",
        concept: "Resizes the full camera image to 112x112 normalized color pixels and evaluates the yawning neural network. When yawning probability stays at 80% or above for 1.0 second (after a 2-second cooldown), it records a Light fatigue cue into the 2-minute window, flagging Drowsy once two Light cues occur within 120 seconds.",
        code: "if (yawn_prob >= 0.80f) {\n  yawn_accum_sec_ = std::min(1.0, yawn_accum_sec_ + dt);\n  if (yawn_accum_sec_ >= 1.0 && !yawn_cue_emitted_ && cooldown >= 2s)\n    cue = CueEvent{CueType::Yawn, CueStrength::Light, now, ...};\n} else {\n  yawn_accum_sec_ = std::max(0.0, yawn_accum_sec_ - 1.0 * dt);\n}",
        note: "Full-frame neural yawning replaced the legacy geometric mouth aspect ratio detector because wide jaw opening distorts facial landmark fitting.",
        codeRef: "src/YawnDetector.cpp (L89-171)\ninclude/MathUtils.hpp (L61-82)\ninclude/Types.hpp (L248-255)"
      },
      nodes: [
        {
          id: "in_bgr", kind: "process", x: 24, y: 28, w: 205, h: 56,
          label: "Full Camera Frame", sublabel: "No landmarks needed; clamp dt",
          details: {
            title: "Full-Frame Input",
            concept: "Takes the raw camera frame directly without needing facial landmark extraction, and calculates clamped frame delta time.",
            code: "const double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);",
            codeRef: "src/YawnDetector.cpp (L112-120)"
          }
        },
        {
          id: "pre_nhwc", kind: "process", x: 265, y: 28, w: 245, h: 56,
          label: "Resize 112x112 & Scale", sublabel: "Convert to RGB [0, 1] tensor",
          details: {
            title: "Image Preprocessing",
            concept: "Converts color channels to RGB, resizes to 112x112 using area interpolation, and divides pixel values by 255 into pre-allocated memory.",
            code: "cv::cvtColor(bgr_frame, rgb_frame_, cv::COLOR_BGR2RGB);\ncv::resize(rgb_frame_, resized_rgb_, cv::Size(112, 112), 0, 0, cv::INTER_AREA);\ndst[i] = static_cast<float>(src[i]) * (1.0f / 255.0f);",
            codeRef: "src/YawnDetector.cpp (L89-110)"
          }
        },
        {
          id: "run_cnn", kind: "process", x: 550, y: 28, w: 240, h: 56,
          label: "Yawning Neural Net", sublabel: "Predict frame yawn confidence",
          details: {
            title: "CNN Forward Pass",
            concept: "Runs the pre-warmed yawning convolutional network on CPU and reads the single scalar yawning probability.",
            code: "net_.setInput(input_blob_);\ncv::Mat outputs = net_.forward();\nconst float yawn_prob = outputs.ptr<float>(0)[0];",
            codeRef: "src/YawnDetector.cpp (L123-126)"
          }
        },
        {
          id: "chk_prob", kind: "decision", x: 550, y: 126, w: 240, h: 84,
          label: "Yawn Confidence\n>= 80%?", sublabel: "Frame yawn threshold",
          details: {
            title: "80% Probability Gate",
            concept: "Tests whether the network's yawning probability meets the 0.80 threshold.",
            code: "if (yawn_prob >= config_.prob_threshold) { // 0.80f",
            codeRef: "src/YawnDetector.cpp (L131)"
          }
        },
        {
          id: "yawn_up", kind: "process", x: 255, y: 140, w: 245, h: 56,
          label: "Accumulate Yawn Timer", sublabel: "Add time step up to 1.0 s cap",
          details: {
            title: "Increment Yawn Timer",
            concept: "Marks yawning active and integrates frame time step up to the 1.0-second target.",
            code: "yawn_accum_sec_ = std::min(1.0, yawn_accum_sec_ + dt);\nin_yawn_ = true;",
            codeRef: "src/YawnDetector.cpp (L131-134)"
          }
        },
        {
          id: "chk_sustained", kind: "decision", x: 255, y: 240, w: 245, h: 88,
          label: "Held >= 1.0 s &\n2 s Cooldown Clear?", sublabel: "First cue per yawn",
          details: {
            title: "Sustained 1-Second Check",
            concept: "Requires 1.0 second of sustained yawning, no prior latch for this yawn, and at least 2 seconds since the previous confirmed yawn.",
            code: "if (yawn_accum_sec_ >= 1.0 && !yawn_cue_emitted_ && cooldown >= 2s)",
            codeRef: "src/YawnDetector.cpp (L135-138)"
          }
        },
        {
          id: "ret_wait", kind: "term_return", x: 24, y: 259, w: 185, h: 50,
          label: "RETURN", sublabel: "Under 1.0 s or latched",
          details: {
            title: "Return Without New Cue",
            concept: "Returns while accumulating toward 1.0 second or while this yawn episode already recorded its cue.",
            code: "return YawnAssessment{yawn_prob, yawn_accum_sec_, in_yawn_, std::nullopt};",
            codeRef: "src/YawnDetector.cpp (L135-153)"
          }
        },
        {
          id: "emit_yawn_cue", kind: "process", x: 255, y: 374, w: 245, h: 56,
          label: "Record Light Cue", sublabel: "Latch yawn & log to 120 s window",
          details: {
            title: "Record Light Yawn Cue",
            concept: "Latches the cue flag, updates last-confirmed timestamp, and logs a Light yawn cue into the 2-minute window.",
            code: "yawn_cue_emitted_ = true; last_confirmed_yawn_ = now;\ncue = CueEvent{CueType::Yawn, CueStrength::Light, now, ...};",
            codeRef: "src/YawnDetector.cpp (L138-152) | src/AlertManager.cpp (L220-238)"
          }
        },
        {
          id: "chk_two_light", kind: "decision", x: 550, y: 358, w: 245, h: 86,
          label: "2+ Light Cues\nin 120 Seconds?", sublabel: "Multimodal light rule",
          details: {
            title: "2-Minute Light Cue Fusion",
            concept: "Two or more Light cues (yawn or head droop) accumulated within 120 seconds elevate state to Drowsy.",
            code: "if (cue_window_.size() >= 2) current_state_ = DriverState::Drowsy;",
            codeRef: "src/AlertLogic.cpp (L51-59) | src/AlertManager.cpp (L223-238)"
          }
        },
        {
          id: "term_drowsy", kind: "term_alarm", x: 855, y: 375, w: 195, h: 52,
          label: "DROWSY", sublabel: "2 Light cues in 120 s",
          details: {
            title: "Drowsiness Verdict",
            concept: "Two Light fatigue cues in 2 minutes trigger the Drowsy alert state.",
            code: "current_state_ = DriverState::Drowsy;",
            codeRef: "src/AlertManager.cpp (L229-238)"
          }
        },
        {
          id: "ret_one_light", kind: "term_return", x: 855, y: 486, w: 195, h: 50,
          label: "RETURN", sublabel: "1 Light cue banked",
          details: {
            title: "Banked First Light Cue",
            concept: "Keeps the first Light cue in the 2-minute window and awaits a second cue.",
            code: "cue_window_.push_back(cue); // Size < 2",
            codeRef: "src/AlertLogic.cpp (L51-59)"
          }
        },
        {
          id: "yawn_decay", kind: "process", x: 840, y: 140, w: 215, h: 56,
          label: "1x Linear Decay", sublabel: "Drain yawn timer at 1x speed",
          details: {
            title: "Linear Accumulator Decay",
            concept: "When probability falls below 80%, subtracts clamped time step linearly from the yawn timer.",
            code: "yawn_accum_sec_ = std::max(0.0, yawn_accum_sec_ - dt);",
            codeRef: "src/YawnDetector.cpp (L154-157)"
          }
        },
        {
          id: "chk_zero", kind: "decision", x: 840, y: 234, w: 215, h: 82,
          label: "Yawn Timer\nDrained to 0?", sublabel: "Yawn ended check",
          details: {
            title: "Yawn End Check",
            concept: "Tests whether the yawn timer has drained completely to zero.",
            code: "if (yawn_accum_sec_ <= 0.0) { ... }",
            codeRef: "src/YawnDetector.cpp (L158-162)"
          }
        },
        {
          id: "clear_yawn", kind: "process", x: 565, y: 247, w: 215, h: 56,
          label: "Clear Yawn Flags", sublabel: "Mouth closed; re-arm latch",
          details: {
            title: "Reset Yawn State",
            concept: "Clears active yawning state and re-arms the cue latch for the next yawn.",
            code: "yawn_accum_sec_ = 0.0; in_yawn_ = false; yawn_cue_emitted_ = false;",
            codeRef: "src/YawnDetector.cpp (L158-162)"
          }
        },
        {
          id: "ret_closed", kind: "term_return", x: 565, y: 486, w: 215, h: 50,
          label: "RETURN", sublabel: "Not yawning / decaying",
          details: {
            title: "No-Yawn Return",
            concept: "Exits cycle with no Light cue emitted.",
            code: "return YawnAssessment{yawn_prob, yawn_accum_sec_, in_yawn_, std::nullopt};",
            codeRef: "src/YawnDetector.cpp (L165-170)"
          }
        }
      ],
      edges: [
        { id: "e_in_pre", points: [[229, 56], [265, 56]], tone: "default" },
        { id: "e_pre_cnn", points: [[510, 56], [550, 56]], tone: "default" },
        { id: "e_cnn_chk", points: [[670, 84], [670, 126]], tone: "default" },
        { id: "e_prob_yes", points: [[550, 168], [500, 168]], label: "YES", tone: "yes" },
        { id: "e_prob_no", points: [[790, 168], [840, 168]], label: "NO", tone: "no" },
        { id: "e_up_sus", points: [[377, 196], [377, 240]], tone: "default" },
        { id: "e_sus_no", points: [[255, 284], [209, 284]], label: "NO", tone: "no" },
        { id: "e_sus_yes", points: [[377, 328], [377, 374]], label: "YES", tone: "yes" },
        { id: "e_cue_2l", points: [[500, 402], [550, 402]], tone: "default" },
        { id: "e_2l_yes", points: [[795, 402], [855, 402]], label: "YES", tone: "yes" },
        { id: "e_2l_no", points: [[672, 444], [672, 464], [824, 464], [824, 511], [855, 511]], label: "NO", tone: "no", lx: 755, ly: 458 },
        { id: "e_dec_chk", points: [[947, 196], [947, 234]], tone: "default" },
        { id: "e_z_yes", points: [[840, 275], [780, 275]], label: "YES", tone: "yes" },
        { id: "e_z_no", points: [[947, 316], [947, 344], [526, 344], [526, 511], [565, 511]], label: "NO", tone: "no", lx: 740, ly: 336 },
        { id: "e_clr_ret", points: [[565, 286], [526, 286], [526, 511], [565, 511]], tone: "default" }
      ],
      traces: [
        {
          id: "yawn_trigger",
          label: "Trace: Sustained Yawn (>= 1.0 s) -> Light Cue -> DROWSY",
          steps: [
            {
              title: "Step 1: Full-Frame Neural Confidence >= 80%",
              highlightNodes: ["in_bgr", "pre_nhwc", "run_cnn", "chk_prob", "yawn_up"],
              highlightEdges: ["e_in_pre", "e_pre_cnn", "e_cnn_chk", "e_prob_yes"],
              concept: "The 112x112 resized camera frame scores 80% or higher on the yawning model, advancing the yawn duration timer.",
              code: "if (yawn_prob >= 0.80f) { yawn_accum_sec_ = std::min(1.0, yawn_accum_sec_ + dt); }"
            },
            {
              title: "Step 2: 1.0 s + Cooldown -> Light Cue -> 2-in-120 s",
              highlightNodes: ["chk_sustained", "emit_yawn_cue", "chk_two_light", "term_drowsy"],
              highlightEdges: ["e_up_sus", "e_sus_yes", "e_cue_2l", "e_2l_yes"],
              concept: "After 1.0 second sustained and a clear cooldown, a Light yawn cue records into the 2-minute window; two Light cues trigger Drowsy.",
              code: "if (yawn_accum_sec_ >= 1.0 && !yawn_cue_emitted_ && cooldown >= 2s) { /* Light cue */ }"
            }
          ]
        },
        {
          id: "yawn_short",
          label: "Trace: Brief Speech / Mouth Open (< 1.0 s) -> Decay",
          steps: [
            {
              title: "Below 80% Decays Linearly to Clear",
              highlightNodes: ["chk_prob", "yawn_decay", "chk_zero", "clear_yawn", "ret_closed"],
              highlightEdges: ["e_prob_no", "e_dec_chk", "e_z_yes", "e_clr_ret"],
              concept: "Brief speech or mouth opening drops below 80% before reaching 1.0 s and drains linearly back to zero.",
              code: "yawn_accum_sec_ = std::max(0.0, yawn_accum_sec_ - dt);"
            }
          ]
        }
      ]
    },
    {
      id: "4B",
      label: "4B. System & FSM Cut-Off",
      subtitle: "Calibration/landmark independence, early 0.2 s yawn suppressor & Moving-Calibrated cut-off",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "Yawning — System & FSM Cut-Off View",
        concept: "Yawning runs on the full camera frame on every Fatigue Worker pass outside calibration, requiring zero personal calibration and zero facial landmarks. Once yawning reaches 0.2 seconds, it raises a suppression signal to mute eye and head false alarms before continuing to landmark detection and cue fusion.",
        code: "const YawnAssessment yawn_res = yawn_detector_.process(frame, t_work_start);\nconst bool suppress_by_yawn = (yawn_res.yawn_duration_sec >= 0.2);\nif (is_moving_calibrated) alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
        note: "Why 0.2 s matters: yawning naturally squints eyelids and tilts the head, so early yawn suppression blocks false blink, PERCLOS, and droop triggers before their timers complete.",
        codeRef: "src/PipelineCoordinator.cpp (L480-505, L526-560)\nsrc/YawnDetector.cpp (L80-171)"
      },
      nodes: [
        {
          id: "read_calibrating", kind: "decision", x: 24, y: 26, w: 225, h: 86,
          label: "READ 1A: Currently\nCalibrating?", sublabel: "Lifecycle check",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_calib" },
          details: {
            title: "Calibration Takeover Gate",
            concept: "When the system enters Calibrating, the Fatigue Worker pauses normal frame monitoring (including yawning) to run the 2-phase audio sequence.",
            code: "if (system_state_.load() == SystemState::Calibrating) { run_calibration(...); continue; }",
            codeRef: "src/PipelineCoordinator.cpp (L480-505)"
          }
        },
        {
          id: "goto_1b_take", kind: "portal_out", x: 24, y: 158, w: 225, h: 56,
          label: "GOTO 1B: Calibration", sublabel: "Pause yawn during setup",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "entry_calib" },
          details: {
            title: "Calibration Takeover",
            concept: "Transfers control of the Fatigue Worker thread into the acoustic calibration routine.",
            code: "auto result = run_calibration(calib_ctx, calib_audio_config_);",
            codeRef: "src/PipelineCoordinator.cpp (L481-488)"
          }
        },
        {
          id: "from_calib_yawn_reset", kind: "portal_in", x: 24, y: 256, w: 225, h: 56,
          label: "FROM 1B: Reset Yawn", sublabel: "Clear timer after calibration",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_4b_yawn_rst" },
          details: {
            title: "Post-Calibration Reset",
            concept: "Clears accumulated yawn duration and cooldown state once calibration finishes.",
            code: "yawn_detector_.reset();",
            codeRef: "src/PipelineCoordinator.cpp (L493) | src/YawnDetector.cpp (L80-87)"
          }
        },
        {
          id: "cnn_process", kind: "process", x: 300, y: 36, w: 255, h: 64,
          label: "Run Yawn Network", sublabel: "Independent of landmarks & calib",
          details: {
            title: "Full-Frame Yawn Inference",
            concept: "Runs on every non-calibrating frame regardless of vehicle movement or missing-face throttling.",
            code: "const YawnAssessment yawn_res = yawn_detector_.process(frame, t_work_start);",
            codeRef: "src/PipelineCoordinator.cpp (L526-527) | src/YawnDetector.cpp (L112-171)"
          }
        },
        {
          id: "chk_sup_02", kind: "decision", x: 300, y: 146, w: 255, h: 86,
          label: "Yawn Duration\n>= 0.2 Seconds?", sublabel: "Early suppressor check",
          details: {
            title: "0.2-Second Suppression Threshold",
            concept: "Raises cross-detector yawn suppression early at 0.2 seconds, well ahead of the 1.0-second fatigue cue.",
            code: "const double yawn_suppress_sec = duration<double>(config().time_for_ear_suppression_by_yawn).count();\nconst bool suppress_by_yawn = (yawn_res.yawn_duration_sec >= yawn_suppress_sec);",
            codeRef: "src/PipelineCoordinator.cpp (L528-531) | include/Types.hpp (L252)"
          }
        },
        {
          id: "sig_yawn_sup", kind: "portal_out", x: 615, y: 122, w: 235, h: 56,
          label: "SIGNAL: Mute Head Droop", sublabel: "Send suppress flag to 2B",
          target: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_yawn_sup" },
          details: {
            title: "Suppress Head Droop",
            concept: "Passes the active yawn suppression flag into Head Orientation processing so yawning posture changes decay safely.",
            code: "head_res = head_detector_.process(landmarks, t_work_start, suppress_by_yawn);",
            codeRef: "src/PipelineCoordinator.cpp (L551)"
          }
        },
        {
          id: "sig_yawn_eyes", kind: "portal_out", x: 615, y: 200, w: 235, h: 56,
          label: "SIGNAL: Mute Eyes & Fusion", sublabel: "Send suppress flag to 3C/3D/6B",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "from_yawn_sup" },
          details: {
            title: "Suppress Blink, PERCLOS & Alert Fusion",
            concept: "Clears eye-squint blink streaks (3C), skips PERCLOS closed samples (3D), and filters eye cues in cue fusion (6B).",
            code: "ear_res = ear_detector_.process(landmarks, now, suppress_by_yawn, is_moving_cal);\n// And in AlertManager: if (yawn.yawn_duration_sec >= 0.2) skip eye cues",
            codeRef: "src/PipelineCoordinator.cpp (L550) | src/AlertManager.cpp (L287-292)"
          }
        },
        {
          id: "to_1c_dlib", kind: "portal_out", x: 300, y: 294, w: 255, h: 56,
          label: "CONTINUE -> 1C: Face Gate", sublabel: "500 ms face search + landmarks",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "entry_frame" },
          details: {
            title: "Proceed to Face Gate",
            concept: "Continues the Fatigue Worker iteration into the 500 ms throttled facial landmark check.",
            code: "if (should_probe) landmarks_opt = face_detector_.detect(frame, t_work_start);",
            codeRef: "src/PipelineCoordinator.cpp (L533-555)"
          }
        },
        {
          id: "read_mov_cal", kind: "decision", x: 300, y: 400, w: 255, h: 86,
          label: "READ 1A: Moving\n& Calibrated?", sublabel: "Lifecycle gate",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            title: "Vehicle Movement & Calibration Gate",
            concept: "Forwards yawn cues and status into Alert Manager only while moving and calibrated.",
            code: "if (is_moving_calibrated) alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
            codeRef: "src/PipelineCoordinator.cpp (L515, L557-559)"
          }
        },
        {
          id: "goto_6b", kind: "portal_out", x: 615, y: 415, w: 235, h: 56,
          label: "GOTO 6B: Cue Fusion", sublabel: "Send yawn assessment",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
          details: {
            title: "Forward Yawn Result to Cue Fusion",
            concept: "Submits yawn status and any 1.0-second Light cue into the Alert Coordinator.",
            code: "alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
            codeRef: "src/PipelineCoordinator.cpp (L558) | src/AlertManager.cpp (L287-303)"
          }
        },
        {
          id: "ret_hud_only", kind: "term_return", x: 615, y: 312, w: 235, h: 50,
          label: "RETURN (HUD Only)", sublabel: "Live yawn probability only",
          details: {
            title: "Telemetry Only Return",
            concept: "Shows live yawn probability and timer on the debug display while alerts stay off.",
            code: "debug_visualizer_.update_somnolence(..., yawn_res, ...);",
            codeRef: "src/PipelineCoordinator.cpp (L557-575)"
          }
        }
      ],
      edges: [
        { id: "e_cal_yes", points: [[136, 112], [136, 158]], label: "YES", tone: "yes" },
        { id: "e_cal_no", points: [[249, 68], [300, 68]], label: "NO", tone: "no" },
        { id: "e_cnn_sup", points: [[427, 100], [427, 146]], tone: "default" },
        { id: "e_sup_yes1", points: [[555, 178], [585, 178], [585, 150], [615, 150]], label: "YES", tone: "portal", lx: 585, ly: 138 },
        { id: "e_sup_yes2", points: [[555, 198], [585, 198], [585, 228], [615, 228]], tone: "portal" },
        { id: "e_sup_1c", points: [[427, 232], [427, 294]], label: "next", tone: "default" },
        { id: "e_1c_mov", points: [[427, 350], [427, 400]], tone: "default" },
        { id: "e_mov_no", points: [[555, 428], [585, 428], [585, 337], [615, 337]], label: "NO", tone: "no", lx: 585, ly: 326 },
        { id: "e_mov_yes", points: [[555, 443], [615, 443]], label: "YES", tone: "yes" }
      ],
      traces: [
        {
          id: "yawn_early_suppress",
          label: "Trace: 0.2 s Early Suppressor -> 1.0 s Light Cue -> 6B",
          steps: [
            {
              title: "Step 1: Yawn Crosses 0.2 s Early Suppressor",
              highlightNodes: ["read_calibrating", "cnn_process", "chk_sup_02", "sig_yawn_sup", "sig_yawn_eyes"],
              highlightEdges: ["e_cal_no", "e_cnn_sup", "e_sup_yes1", "e_sup_yes2"],
              concept: "After 0.2 seconds of yawning, the suppression signal mutes Head Droop (2B), Blink (3C), PERCLOS (3D), and Alert Manager eye cues (6B).",
              code: "const bool suppress_by_yawn = (yawn_res.yawn_duration_sec >= 0.2);",
              portalTarget: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_yawn_sup" }
            },
            {
              title: "Step 2: Moving & Calibrated Forwards Cue to 6B",
              highlightNodes: ["to_1c_dlib", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_sup_1c", "e_1c_mov", "e_mov_yes"],
              concept: "Once sustained at 1.0 s or longer, the Light yawn cue passes through the moving-calibrated gate into cue fusion (6B).",
              code: "if (is_moving_calibrated) alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
              portalTarget: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" }
            }
          ]
        },
        {
          id: "yawn_face_lost",
          label: "Trace: Face Lost (500 ms) While Yawn Net Still Runs",
          steps: [
            {
              title: "Yawn Network Keeps Tracking Without Landmarks",
              highlightNodes: ["cnn_process", "chk_sup_02", "to_1c_dlib", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_cnn_sup", "e_sup_1c", "e_1c_mov", "e_mov_yes"],
              concept: "Even if wide yawning causes landmark detection (1C) to drop into 500 ms throttling, the full-frame yawn network still evaluates every frame.",
              code: "// YawnDetector::process(frame, now) runs before FaceLandmarkDetector"
            }
          ]
        },
        {
          id: "yawn_parked",
          label: "Trace: Parked (Calibrated) -> Telemetry Only",
          steps: [
            {
              title: "Stationary State Gates Alert Manager",
              highlightNodes: ["cnn_process", "read_mov_cal", "ret_hud_only"],
              highlightEdges: ["e_cnn_sup", "e_sup_1c", "e_1c_mov", "e_mov_no"],
              concept: "While parked, yawn metrics still update on the live display while cue fusion stays paused.",
              code: "if (is_moving_calibrated) { /* skipped */ }"
            }
          ]
        }
      ]
    }
  ]
});
