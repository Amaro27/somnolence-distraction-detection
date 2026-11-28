window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec1",
  title: "1. Vehicle & Calib FSM",
  subviews: [
    {
      id: "1A",
      label: "1A. 5-State Movement & IMU FSM",
      subtitle: "System lifecycle states, keyboard commands, 20 Hz IMU motion hysteresis, and parking resets",
      viewBox: "0 0 1080 600",
      legend: ["state", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "5-State Vehicle & Calibration Lifecycle FSM",
        concept: "All fatigue and distraction triggers are governed by a 5-state safety finite-state machine. The system boots parked and uncalibrated by default, locks out movement changes while calibration runs, and clears detector histories whenever the vehicle stops.",
        code: "enum class SystemState {\n  StationaryUncalibrated, MovingUncalibrated,\n  Calibrating, StationaryCalibrated, MovingCalibrated\n};",
        note: "On Linux vehicles, a 20 Hz IMU motion thread automates parked-to-moving transitions using smoothed acceleration hysteresis. On Windows/macOS or when hardware sensing is disabled, manual keyboard toggles control movement.",
        codeRef: "include/Types.hpp (L66-72)\nsrc/PipelineCoordinator.cpp (L130-131, L319-384, L589-689)\ninclude/MPU6050.hpp (L38-205)"
      },
      nodes: [
        {
          id: "startup_opt", kind: "decision", x: 20, y: 20, w: 200, h: 80,
          label: "Start Moving\nFlag Active?", sublabel: "Boot option",
          details: {
            title: "Startup Movement Option",
            concept: "At launch, the pipeline checks whether the user requested starting in the moving state; otherwise it boots safely parked and uncalibrated.",
            code: "system_state_(options_.start_moving\n  ? SystemState::MovingUncalibrated\n  : SystemState::StationaryUncalibrated)",
            codeRef: "src/PipelineCoordinator.cpp (L130-131)"
          }
        },
        {
          id: "st_stat_uncal", kind: "state", x: 255, y: 32, w: 200, h: 56,
          label: "Parked (Uncalibrated)", sublabel: "Default boot; alerts off",
          details: {
            title: "Parked (Uncalibrated)",
            concept: "Safe startup state: the car is parked and no personal eye or posture baselines exist yet, so gaze inference and alarms stay disabled.",
            code: "SystemState::StationaryUncalibrated // Default startup",
            codeRef: "include/Types.hpp (L67)"
          }
        },
        {
          id: "st_mov_uncal", kind: "state", x: 255, y: 140, w: 200, h: 56,
          label: "Moving (Uncalibrated)", sublabel: "Alerts paused; warns user",
          details: {
            title: "Moving (Uncalibrated)",
            concept: "Vehicle motion occurred before calibration. Alerts remain paused and the console warns the driver to stop and calibrate first.",
            code: "SystemState::MovingUncalibrated // Detections disabled",
            codeRef: "include/Types.hpp (L68) | src/PipelineCoordinator.cpp (L347-353)"
          }
        },
        {
          id: "st_calib", kind: "state", tone: "calib", x: 530, y: 32, w: 200, h: 56,
          label: "Calibrating", sublabel: "Locks out 'm' & IMU",
          details: {
            title: "Calibrating State",
            concept: "Entered only via the calibrate command while parked. Both manual movement toggles and automatic IMU transitions are locked out during calibration.",
            code: "if (s == StationaryUncalibrated || s == StationaryCalibrated)\n  system_state_.store(SystemState::Calibrating);",
            codeRef: "include/Types.hpp (L69) | src/PipelineCoordinator.cpp (L323-327, L363-383)"
          }
        },
        {
          id: "goto_1b_calib", kind: "portal_out", x: 530, y: 122, w: 200, h: 52,
          label: "GOTO 1B: Calibration", sublabel: "Fatigue thread takeover",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "entry_calib" },
          details: {
            title: "Jump to Calibration Sequence",
            concept: "When the fatigue worker sees the calibrating state, it pauses monitoring and executes the 2-phase acoustic calibration sequence in-place.",
            code: "if (system_state_.load() == SystemState::Calibrating) {\n  auto result = run_calibration(calib_ctx, calib_audio_config_);\n}",
            codeRef: "src/PipelineCoordinator.cpp (L480-505)"
          }
        },
        {
          id: "st_stat_cal", kind: "state", doubleBorder: true, x: 810, y: 32, w: 225, h: 56,
          label: "Parked (Calibrated)", sublabel: "Baselines set; paused",
          details: {
            title: "Parked (Calibrated)",
            concept: "Personal baselines are saved and the vehicle is stationary. Live eye, posture, and yawn telemetry stays visible on screen, while PERCLOS and alarms stay paused.",
            code: "SystemState::StationaryCalibrated // Paused telemetry mode",
            codeRef: "include/Types.hpp (L70) | src/PipelineCoordinator.cpp (L496, L515-575)"
          }
        },
        {
          id: "st_mov_cal", kind: "state", tone: "active", doubleBorder: true, x: 810, y: 140, w: 225, h: 56,
          label: "Moving (Calibrated)", sublabel: "Full monitoring active",
          details: {
            title: "Moving (Calibrated)",
            concept: "Full real-time safety monitoring: gaze neural inference runs, PERCLOS accumulates, and distraction or drowsiness alarms can fire.",
            code: "SystemState::MovingCalibrated // Full multimodal active",
            codeRef: "include/Types.hpp (L71) | src/PipelineCoordinator.cpp (L429, L515, L557-559)"
          }
        },
        {
          id: "key_gate", kind: "process", tone: "neutral", x: 20, y: 244, w: 250, h: 58,
          label: "Keyboard / Window Commands", sublabel: "'m' toggles move; 'c' calibrates",
          details: {
            title: "Console & Window Commands",
            concept: "The supervisory loop polls keyboard and GUI inputs. Pressing 'm' switches movement unless calibrating; pressing 'c' starts calibration only if parked.",
            code: "switch (cmd) {\n  case ConsoleCommand::ToggleMovement: ...\n  case ConsoleCommand::Calibrate: ...\n}",
            codeRef: "src/PipelineCoordinator.cpp (L282-384)"
          }
        },
        {
          id: "grp_imu", kind: "group", x: 20, y: 326, w: 485, h: 254,
          label: "Hardware Motion Sensor (20 Hz Linux Bus)", sublabel: "Cleanly skipped on Windows/macOS or when disabled",
          details: {
            title: "20 Hz IMU Motion Worker",
            concept: "Polls the accelerometer magnitude every 50 ms using a 5-sample sliding average and hysteresis thresholds so short road vibrations never flip vehicle state.",
            code: "MPU6050 imu(options_.imu_bus, options_.imu_address);\nif (!imu.init()) return;",
            codeRef: "src/PipelineCoordinator.cpp (L589-689) | include/MPU6050.hpp (L51-205)"
          }
        },
        {
          id: "imu_smooth", kind: "process", x: 38, y: 384, w: 205, h: 56,
          label: "Smooth Acceleration", sublabel: "5-sample avg minus gravity",
          details: {
            title: "5-Sample Acceleration Smoothing",
            concept: "Averages the last 5 total acceleration readings and subtracts standard gravity (9.81 m/s^2) to measure true vehicle motion.",
            code: "float smoothed_mag = sum(mag_buffer[0..4]) / 5.0f;\nfloat dynamic_motion = std::abs(smoothed_mag - MPU6050::GRAVITY_CONST);",
            codeRef: "src/PipelineCoordinator.cpp (L606-634)"
          }
        },
        {
          id: "imu_high_gate", kind: "decision", x: 278, y: 372, w: 210, h: 82,
          label: "> 0.80 m/s^2\nfor 4 Samples?", sublabel: "200 ms motion",
          details: {
            title: "Motion Onset Hysteresis",
            concept: "Requires 4 consecutive high-motion readings above 0.80 m/s^2 before switching the vehicle state from Parked to Moving.",
            code: "if (dynamic_motion > thresh_high) {\n  count_motion++; count_still = 0;\n  if (count_motion >= 4 && !is_moving) { is_moving = true; ... }\n}",
            codeRef: "src/PipelineCoordinator.cpp (L610-655)"
          }
        },
        {
          id: "imu_low_gate", kind: "decision", x: 278, y: 478, w: 210, h: 82,
          label: "< 0.35 m/s^2\nfor 15 Samples?", sublabel: "750 ms stillness",
          details: {
            title: "Parking Stillness Hysteresis",
            concept: "Requires 15 consecutive low-motion readings below 0.35 m/s^2 before switching the vehicle state back to Parked and triggering cleanup.",
            code: "else if (dynamic_motion < thresh_low) {\n  count_still++; count_motion = 0;\n  if (count_still >= 15 && is_moving) { is_moving = false; ... }\n}",
            codeRef: "src/PipelineCoordinator.cpp (L611-684)"
          }
        },
        {
          id: "imu_calib_lock", kind: "decision", x: 38, y: 478, w: 205, h: 82,
          label: "Currently\nCalibrating?", sublabel: "Lockout check",
          details: {
            title: "Calibration Lockout Guard",
            concept: "The motion worker only alters Parked or Moving states; if calibration is active, motion transitions are ignored.",
            code: "if (current == StationaryCalibrated) system_state_.store(MovingCalibrated);\nelse if (current == StationaryUncalibrated) system_state_.store(MovingUncalibrated);",
            codeRef: "src/PipelineCoordinator.cpp (L643-683)"
          }
        },
        {
          id: "grp_cleanup", kind: "group", x: 530, y: 244, w: 525, h: 336,
          label: "Parking Cleanup (Moving -> Parked)", sublabel: "Triggered by 'm' keypress or IMU stillness",
          details: {
            title: "Parking Cleanup Routine",
            concept: "Whenever the vehicle stops, active alerts silence immediately and detector windows clear.",
            code: "alert_manager_.reset();\near_detector_.clear_perclos(); // if MovingCalibrated\ngaze_reset_requested_.store(true, std::memory_order_release);",
            codeRef: "src/PipelineCoordinator.cpp (L328-346, L664-682)"
          }
        },
        {
          id: "park_cleanup", kind: "process", x: 548, y: 300, w: 215, h: 58,
          label: "Stop-State Cleanup", sublabel: "Reset alerts, HUD & histories",
          details: {
            title: "Stop-State Cleanup",
            concept: "Main or IMU threads clear active alarms, reset the debug HUD, and trigger eye/gaze resets whenever transitioning from Moving to Parked.",
            code: "alert_manager_.reset();\near_detector_.clear_perclos();\ngaze_reset_requested_.store(true, std::memory_order_release);\ndebug_visualizer_.reset();",
            note: "Concurrency caveat: Main (L340) and IMU (L668) call clear_perclos directly without a mutex against the fatigue thread, whereas gaze resets safely via an atomic flag.",
            codeRef: "src/PipelineCoordinator.cpp (L328-346, L518-522, L664-682)"
          }
        },
        {
          id: "goto_6c_reset", kind: "portal_out", x: 805, y: 300, w: 230, h: 54,
          label: "GOTO 6C: Alert Reset", sublabel: "Silence audio & clear cues",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_reset" },
          details: {
            title: "Alert Manager Reset",
            concept: "Clears both gaze and fatigue cue windows, resets driver state to Safe, and stops any playing audio immediately.",
            code: "alert_manager_.reset(); // Clears gaze_window_, cue_window_, calls audio_.stop()",
            codeRef: "src/AlertManager.cpp (L390-406)"
          }
        },
        {
          id: "goto_3d_clear", kind: "portal_out", x: 805, y: 376, w: 230, h: 54,
          label: "GOTO 3D: Clear PERCLOS", sublabel: "Wipe 30 s eye history",
          target: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "from_park_clear" },
          details: {
            title: "Clear PERCLOS History",
            concept: "Clears the 30-second rolling eye-closure window and restarts warmup when parking from Moving (Calibrated).",
            code: "if (current == SystemState::MovingCalibrated)\n  ear_detector_.clear_perclos();",
            note: "Called directly from Main (L340) and IMU (L668); the fatigue worker also executes a fallback clear when leaving MovingCalibrated (L520).",
            codeRef: "src/PipelineCoordinator.cpp (L340, L520, L668)"
          }
        },
        {
          id: "set_gaze_atomic", kind: "portal_out", x: 805, y: 452, w: 230, h: 54,
          label: "SET ATOMIC: Gaze Reset", sublabel: "Signal gaze worker safely",
          target: { sectionId: "sec5", subviewId: "5B", highlightNodeId: "from_atomic_reset" },
          details: {
            title: "Atomic Gaze Reset Signal",
            concept: "Instead of wiping the gaze sequence buffer from another thread, Main and IMU set a lightweight atomic flag so the Gaze Worker clears its own memory.",
            code: "gaze_reset_requested_.store(true, std::memory_order_release);",
            codeRef: "src/PipelineCoordinator.cpp (L331, L341, L669, L677)"
          }
        },
        {
          id: "hud_paused", kind: "term_return", x: 548, y: 454, w: 215, h: 50,
          label: "Detections Paused", sublabel: "Await move or calibrate",
          details: {
            title: "Stationary Idle State",
            concept: "Monitoring stays paused until the vehicle moves again or calibration begins.",
            code: "// Stationary: Gaze skips inference; Fatigue updates HUD only",
            codeRef: "src/PipelineCoordinator.cpp (L429, L557)"
          }
        }
      ],
      edges: [
        { id: "e_boot_no", points: [[220, 60], [255, 60]], label: "NO", tone: "no" },
        { id: "e_boot_yes", points: [[120, 100], [120, 168], [255, 168]], label: "YES", tone: "yes", lx: 190, ly: 158 },
        { id: "e_uncal_move", points: [[332, 88], [332, 140]], label: "move", tone: "default", lx: 300, ly: 114 },
        { id: "e_uncal_park", points: [[380, 140], [380, 88]], label: "park", tone: "default", lx: 406, ly: 114 },
        { id: "e_c_start", points: [[455, 60], [530, 60]], label: "'c'", tone: "yes" },
        { id: "e_calib_take", points: [[630, 88], [630, 122]], label: "takeover", tone: "portal" },
        { id: "e_c_done", points: [[730, 60], [810, 60]], label: "done", tone: "yes" },
        { id: "e_cal_move", points: [[895, 88], [895, 140]], label: "move", tone: "default", lx: 864, ly: 114 },
        { id: "e_cal_park", points: [[950, 140], [950, 88]], label: "park", tone: "default", lx: 976, ly: 114 },
        { id: "e_imu_hi", points: [[243, 412], [278, 412]], tone: "default" },
        { id: "e_imu_lo", points: [[383, 454], [383, 478]], label: "NO", tone: "no", lx: 400, ly: 466 },
        { id: "e_imu_chk", points: [[278, 519], [243, 519]], label: "trigger", tone: "default" },
        { id: "e_mov_cleanup", points: [[810, 178], [786, 178], [786, 280], [655, 280], [655, 300]], label: "on park", tone: "portal", lx: 720, ly: 272 },
        { id: "e_cln_alert", points: [[763, 327], [805, 327]], tone: "portal" },
        { id: "e_cln_perc", points: [[763, 342], [784, 342], [784, 403], [805, 403]], tone: "portal" },
        { id: "e_cln_gaze", points: [[763, 352], [774, 352], [774, 479], [805, 479]], tone: "portal" },
        { id: "e_cln_ret", points: [[655, 358], [655, 454]], tone: "yes" }
      ],
      traces: [
        {
          id: "boot_calib_drive",
          label: "Trace: Safe Boot -> Calibrate ('c') -> Drive ('m')",
          steps: [
            {
              title: "Step 1: Safe Stationary Boot",
              highlightNodes: ["startup_opt", "st_stat_uncal"],
              highlightEdges: ["e_boot_no"],
              concept: "By default, the system boots parked and uncalibrated with all detections and alarms disabled.",
              code: "system_state_(options_.start_moving ? MovingUncalibrated : StationaryUncalibrated);"
            },
            {
              title: "Step 2: Driver Starts Calibration While Parked",
              highlightNodes: ["key_gate", "st_stat_uncal", "st_calib", "goto_1b_calib"],
              highlightEdges: ["e_c_start", "e_calib_take"],
              concept: "Calibration is accepted only while stationary. Entering the calibrating state hands control of the Fatigue Worker to the 2-phase acoustic calibration sequence.",
              code: "system_state_.store(SystemState::Calibrating); // Fatigue thread runs run_calibration()",
              portalTarget: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "entry_calib" }
            },
            {
              title: "Step 3: Baselines Applied -> Parked (Calibrated)",
              highlightNodes: ["st_calib", "st_stat_cal"],
              highlightEdges: ["e_c_done"],
              concept: "Once 15 or more open-eye and closed-eye samples are collected, personal baselines are applied and the system enters Parked (Calibrated).",
              code: "ear_detector_.apply_baseline(...); head_detector_.apply_baseline(...);\nsystem_state_.store(SystemState::StationaryCalibrated);"
            },
            {
              title: "Step 4: Drive ('m' or IMU) -> Moving (Calibrated)",
              highlightNodes: ["st_stat_cal", "st_mov_cal"],
              highlightEdges: ["e_cal_move"],
              concept: "Pressing the movement key or accelerating switches the state to Moving (Calibrated), activating gaze tracking, PERCLOS, and alerts.",
              code: "system_state_.store(SystemState::MovingCalibrated);"
            }
          ]
        },
        {
          id: "imu_hysteresis",
          label: "Trace: IMU Auto-Move & Auto-Park Cleanup",
          steps: [
            {
              title: "Step 1: 5-Sample Dynamic Acceleration",
              highlightNodes: ["grp_imu", "imu_smooth", "imu_high_gate"],
              highlightEdges: ["e_imu_hi"],
              concept: "At 20 Hz, the motion sensor worker averages 5 accelerometer magnitudes and subtracts gravity to reject bumps and noise.",
              code: "dynamic_motion = std::abs(smoothed_mag - MPU6050::GRAVITY_CONST);"
            },
            {
              title: "Step 2: 4 High-Motion Samples -> Moving",
              highlightNodes: ["imu_high_gate", "st_stat_cal", "st_mov_cal"],
              highlightEdges: ["e_cal_move"],
              concept: "Sustaining more than 0.80 m/s^2 for 4 consecutive samples (200 ms) promotes Parked (Calibrated) to Moving (Calibrated).",
              code: "if (dynamic_motion > 0.80f && ++count_motion >= 4) system_state_.store(MovingCalibrated);"
            },
            {
              title: "Step 3: 15 Still Samples -> Park & Cleanup",
              highlightNodes: ["imu_low_gate", "imu_calib_lock", "st_mov_cal", "st_stat_cal", "park_cleanup", "goto_6c_reset", "goto_3d_clear", "set_gaze_atomic", "hud_paused"],
              highlightEdges: ["e_imu_lo", "e_imu_chk", "e_cal_park", "e_mov_cleanup", "e_cln_alert", "e_cln_perc", "e_cln_gaze", "e_cln_ret"],
              concept: "Holding below 0.35 m/s^2 for 15 samples (750 ms) parks the vehicle, resets alerts (6C), clears PERCLOS (3D), and sets the atomic gaze-reset flag (5B).",
              code: "alert_manager_.reset(); ear_detector_.clear_perclos();\ngaze_reset_requested_.store(true, std::memory_order_release);",
              note: "Note the concurrency caveat: PERCLOS is cleared directly from the IMU thread here, whereas the gaze sequence uses an atomic flag."
            }
          ]
        },
        {
          id: "calib_lockout",
          label: "Trace: Lockout During Calibration",
          steps: [
            {
              title: "Calibration Locks Out 'm' and IMU",
              highlightNodes: ["st_calib", "key_gate", "imu_calib_lock"],
              highlightEdges: [],
              concept: "While calibration runs, manual movement commands are ignored and the motion sensor skips state transitions.",
              code: "if (system_state_.load() == SystemState::Calibrating) return; // Lockout"
            }
          ]
        }
      ]
    },
    {
      id: "1B",
      label: "1B. 2-Phase Calibration Routine",
      subtitle: "In-place Fatigue Worker takeover: +6 dB voice, 415 Hz prep tone, 8x500 ms beeps & baselines",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "2-Phase Acoustic Calibration Routine",
        concept: "Runs directly on the Fatigue Worker thread whenever calibration starts. Phase 1 captures open-eye aperture and neutral head posture; Phase 2 captures closed-eye baseline, repeating 4-second beep windows until at least 15 face frames are measured per phase.",
        code: "auto res = run_calibration(calib_ctx, calib_audio_config_);\n// Phase 1: open_samples >= 15, rho_samples\n// Phase 2: closed_samples >= 15\n// Return: {mean(open), mean(closed), median(rho)}",
        note: "The 1-second 415 Hz preparation tone plays before measurement beeps in BOTH Phase 1 and Phase 2. Face detection is skipped if <= 30 ms remain before the next 500 ms beep so audio cadence stays accurate.",
        codeRef: "src/PipelineCoordinator.cpp (L470-505)\nsrc/Calibration.cpp (L14-220)\ninclude/Types.hpp (L204-217)"
      },
      nodes: [
        {
          id: "entry_calib", kind: "portal_in", x: 20, y: 26, w: 205, h: 56,
          label: "FROM 1A: Calibrating", sublabel: "Fatigue Worker takeover",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "goto_1b_calib" },
          details: {
            title: "Calibration Entry",
            concept: "The Fatigue Worker checks the system state before normal frame processing and runs calibration when active.",
            code: "if (system_state_.load() == SystemState::Calibrating) {\n  auto result = run_calibration(calib_ctx, calib_audio_config_);\n}",
            codeRef: "src/PipelineCoordinator.cpp (L480-488)"
          }
        },
        {
          id: "p1_intro", kind: "process", x: 260, y: 26, w: 230, h: 56,
          label: "Phase 1: Voice + Tone", sublabel: "Voice -> 1 s -> 415 Hz -> 1 s",
          details: {
            title: "Phase 1 Audio Guidance",
            concept: "Plays spoken open-eye instructions at +6 dB, pauses 1 second, plays the 1-second 415 Hz preparation tone, and pauses 1 second.",
            code: "ctx.audio.play_blocking(cfg.phase1_instructions_path, 6.0f);\npause_interruptible(1s);\nctx.audio.play_blocking(cfg.phase1_tone_path);\npause_interruptible(1s);",
            codeRef: "src/Calibration.cpp (L95-104)"
          }
        },
        {
          id: "p1_window", kind: "process", x: 530, y: 26, w: 245, h: 56,
          label: "Phase 1 Beeps (8 x 500 ms)", sublabel: "Skip face detect if <= 30 ms left",
          details: {
            title: "Phase 1 Measurement Window",
            concept: "Emits 8 cadence beeps at +3 dB spaced 500 ms apart (4 seconds total) while gathering open-eye ratio and head posture samples.",
            code: "for (size_t i = 0; i < 8; ++i) {\n  ctx.audio.play(cfg.cadence_beep_path, 3.0f);\n  if (remaining > 30ms && mailbox.get_latest(frame, wait_dur)) { ... }\n}",
            codeRef: "src/Calibration.cpp (L25-76, L127-134)"
          }
        },
        {
          id: "goto_1c_detect", kind: "portal_out", x: 825, y: 26, w: 230, h: 56,
          label: "GOTO 1C: Direct Detect", sublabel: "Extract open-eye & head data",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "dlib_hog" },
          details: {
            title: "Direct Landmark Detection",
            concept: "Calls the 68-point facial landmark detector directly without the normal 500 ms missing-face throttle.",
            code: "auto lm_opt = ctx.face.detect(frame, steady_clock::now());\nif (lm_opt) { open_samples.push_back(compute_ear(*lm_opt)); rho_samples.push_back(compute_rho(*lm_opt)); }",
            codeRef: "src/Calibration.cpp (L62-66, L127-132)"
          }
        },
        {
          id: "p1_count", kind: "decision", x: 530, y: 120, w: 245, h: 82,
          label: "Open Samples >= 15?", sublabel: "Repeat window if fewer",
          details: {
            title: "Phase 1 Sample Minimum",
            concept: "Repeats the 4-second beep window automatically if fewer than 15 valid face samples were recorded.",
            code: "while (open_samples.size() < cfg.min_samples_per_window) { /* 15 */ }",
            codeRef: "src/Calibration.cpp (L116-136)"
          }
        },
        {
          id: "abort_check", kind: "decision", x: 260, y: 120, w: 230, h: 82,
          label: "Shutdown Requested?", sublabel: "Polled in pauses",
          details: {
            title: "Interruptible Abort Check",
            concept: "Polls the shutdown flag every 20 ms during silences and beep loops so quitting exits calibration immediately.",
            code: "if (ctx.shutdown.load()) return std::nullopt;",
            codeRef: "src/Calibration.cpp (L14-23, L36-48)"
          }
        },
        {
          id: "abort_restore", kind: "term_return", x: 20, y: 135, w: 205, h: 52,
          label: "ABORT: Restore State", sublabel: "Revert to prior parked state",
          details: {
            title: "Calibration Aborted",
            concept: "Restores Parked (Calibrated) if calibrated earlier in the session, or Parked (Uncalibrated) otherwise.",
            code: "system_state_.store(was_calibrated\n  ? SystemState::StationaryCalibrated\n  : SystemState::StationaryUncalibrated);",
            codeRef: "src/PipelineCoordinator.cpp (L500-503)"
          }
        },
        {
          id: "trans_gap", kind: "process", x: 825, y: 133, w: 230, h: 56,
          label: "1.0 s Transition Pause", sublabel: "Cue driver to close eyes",
          details: {
            title: "Eye-Closure Transition",
            concept: "Logs and pauses 1 second to let the driver gently close their eyes before Phase 2 instructions begin.",
            code: "pause_interruptible(cfg.transition_silence, ctx.shutdown); // 1000 ms",
            codeRef: "src/Calibration.cpp (L141-148)"
          }
        },
        {
          id: "p2_intro", kind: "process", x: 825, y: 236, w: 230, h: 56,
          label: "Phase 2: Voice + Tone", sublabel: "Voice -> 1 s -> 415 Hz -> 1 s",
          details: {
            title: "Phase 2 Audio Guidance",
            concept: "Plays closed-eye spoken instructions (+6 dB), 1 s gap, 415 Hz preparation tone, and 1 s gap.",
            code: "ctx.audio.play_blocking(cfg.phase2_instructions_path, 6.0f);\npause_interruptible(1s);\nctx.audio.play_blocking(cfg.phase2_tone_path);\npause_interruptible(1s);",
            codeRef: "src/Calibration.cpp (L161-170)"
          }
        },
        {
          id: "p2_window", kind: "process", x: 530, y: 236, w: 245, h: 56,
          label: "Phase 2 Beeps (8 x 500 ms)", sublabel: "Measure closed-eye ratio",
          details: {
            title: "Phase 2 Measurement Window",
            concept: "Plays 8 cadence beeps at +3 dB while sampling closed-eye aspect ratios through the landmark detector.",
            code: "measure_window(ctx, cfg, \"Phase 2...\", closed_samples.size(), [&](const auto& lm) {\n  double ear = FaceGeometry::compute_ear(lm);\n  if (ear > 0.0) closed_samples.push_back(ear);\n});",
            codeRef: "src/Calibration.cpp (L178-199)"
          }
        },
        {
          id: "p2_count", kind: "decision", x: 260, y: 223, w: 230, h: 82,
          label: "Closed Samples >= 15?", sublabel: "Repeat window if fewer",
          details: {
            title: "Phase 2 Sample Minimum",
            concept: "Repeats the 4-second closed-eye beep window until at least 15 valid samples are collected.",
            code: "while (closed_samples.size() < cfg.min_samples_per_window) { /* 15 */ }",
            codeRef: "src/Calibration.cpp (L181-199)"
          }
        },
        {
          id: "calib_complete", kind: "process", x: 20, y: 236, w: 205, h: 56,
          label: "Compute Baselines", sublabel: "Mean open/closed + median pose",
          details: {
            title: "Compute Personal Baselines",
            concept: "Plays the completion chime and computes mean open-eye ratio, mean closed-eye ratio, and median neutral head proportion.",
            code: "ctx.audio.play(cfg.success_sound_path);\nreturn CalibrationResult{\n  compute_mean(open_samples), compute_mean(closed_samples), compute_median(rho_samples)\n};",
            codeRef: "src/Calibration.cpp (L213-219)"
          }
        },
        {
          id: "goto_3c_ear_base", kind: "portal_out", x: 20, y: 374, w: 235, h: 54,
          label: "GOTO 3C: Apply Eye Base", sublabel: "Set 70% & 20% thresholds",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "from_calib_ear" },
          details: {
            title: "Apply Eye Baselines",
            concept: "Configures the blink and PERCLOS detector with personal open and closed eye thresholds.",
            code: "ear_detector_.apply_baseline(result->ear_open, result->ear_closed, now_calib);",
            codeRef: "src/PipelineCoordinator.cpp (L491) | src/EarDetector.cpp (L42-83)"
          }
        },
        {
          id: "goto_2b_head_base", kind: "portal_out", x: 285, y: 374, w: 235, h: 54,
          label: "GOTO 2B: Apply Head Base", sublabel: "Set neutral pitch posture",
          target: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_calib_rho" },
          details: {
            title: "Apply Head Posture Baseline",
            concept: "Stores the driver's neutral vertical face proportion and clears head droop timers.",
            code: "head_detector_.apply_baseline(result->neutral_rho);",
            codeRef: "src/PipelineCoordinator.cpp (L492) | src/HeadOrientationDetector.cpp (L13-25)"
          }
        },
        {
          id: "goto_4b_yawn_rst", kind: "portal_out", x: 550, y: 374, w: 235, h: 54,
          label: "GOTO 4B: Reset Yawn", sublabel: "Clear yawn accumulator",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "from_calib_yawn_reset" },
          details: {
            title: "Reset Yawn State",
            concept: "Resets any stale yawn accumulator and cooldown timer before monitoring resumes.",
            code: "yawn_detector_.reset();",
            codeRef: "src/PipelineCoordinator.cpp (L493) | src/YawnDetector.cpp (L80-87)"
          }
        },
        {
          id: "goto_1c_face_rst", kind: "portal_out", x: 815, y: 374, w: 235, h: 54,
          label: "GOTO 1C: Reset Face Gate", sublabel: "Clear missing-face state",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "from_calib_face_reset" },
          details: {
            title: "Reset Face Gate",
            concept: "Clears face-missing flags and 3-second lost-face warnings before monitoring starts.",
            code: "face_detector_.reset();",
            codeRef: "src/PipelineCoordinator.cpp (L494) | src/FaceLandmarkDetector.cpp (L117-121)"
          }
        },
        {
          id: "calib_exit", kind: "term_return", x: 380, y: 492, w: 320, h: 52,
          label: "Set Parked (Calibrated)", sublabel: "Return to Fatigue Worker loop",
          details: {
            title: "Calibration Complete",
            concept: "Marks the system as calibrated and parked; full detection activates once vehicle movement begins.",
            code: "was_calibrated = true;\nsystem_state_.store(SystemState::StationaryCalibrated);",
            codeRef: "src/PipelineCoordinator.cpp (L495-499)"
          }
        }
      ],
      edges: [
        { id: "e_en_p1", points: [[225, 54], [260, 54]], tone: "default" },
        { id: "e_p1_win", points: [[490, 54], [530, 54]], tone: "default" },
        { id: "e_win_1c", points: [[775, 54], [825, 54]], tone: "portal" },
        { id: "e_win_cnt", points: [[652, 82], [652, 120]], tone: "default" },
        { id: "e_p1_retry", points: [[775, 161], [800, 161], [800, 96], [674, 96], [674, 82]], label: "NO (<15)", tone: "no", lx: 742, ly: 100 },
        { id: "e_p1_ok", points: [[775, 161], [825, 161]], label: "YES", tone: "yes" },
        { id: "e_p1_ab", points: [[530, 161], [490, 161]], label: "abort?", tone: "default" },
        { id: "e_ab_yes", points: [[260, 161], [225, 161]], label: "YES", tone: "no" },
        { id: "e_tr_p2", points: [[940, 189], [940, 236]], tone: "default" },
        { id: "e_p2_win", points: [[825, 264], [775, 264]], tone: "default" },
        { id: "e_p2_cnt", points: [[530, 264], [490, 264]], tone: "default" },
        { id: "e_p2_retry", points: [[375, 223], [375, 210], [652, 210], [652, 236]], label: "NO (<15)", tone: "no", lx: 515, ly: 204 },
        { id: "e_p2_ok", points: [[260, 264], [225, 264]], label: "YES", tone: "yes" },
        { id: "e_comp_ear", points: [[122, 292], [122, 374]], tone: "portal" },
        { id: "e_comp_head", points: [[122, 334], [402, 334], [402, 374]], tone: "portal" },
        { id: "e_comp_yawn", points: [[122, 334], [667, 334], [667, 374]], tone: "portal" },
        { id: "e_comp_face", points: [[122, 334], [932, 334], [932, 374]], tone: "portal" },
        { id: "e_exit", points: [[540, 428], [540, 492]], tone: "yes" }
      ],
      traces: [
        {
          id: "calib_happy",
          label: "Trace: Happy Path (>=15 Samples Both Phases)",
          steps: [
            {
              title: "Step 1: Phase 1 Voice, Tone & Open-Eye Beeps",
              highlightNodes: ["entry_calib", "p1_intro", "p1_window", "goto_1c_detect"],
              highlightEdges: ["e_en_p1", "e_p1_win", "e_win_1c"],
              concept: "The driver hears Phase 1 open-eye instructions, a 1-second preparation tone, and 8 cadence beeps while face detection samples open eyes and neutral head posture.",
              code: "ctx.audio.play_blocking(phase1_instructions_path, 6.0f);\nctx.audio.play_blocking(phase1_tone_path);\nmeasure_window(...);"
            },
            {
              title: "Step 2: Phase 1 Meets 15 Samples -> Phase 2",
              highlightNodes: ["p1_count", "trans_gap", "p2_intro", "p2_window", "p2_count"],
              highlightEdges: ["e_win_cnt", "e_p1_ok", "e_tr_p2", "e_p2_win", "e_p2_cnt"],
              concept: "After 15+ open samples, a 1-second silence cues eye closure, Phase 2 plays guidance and the 415 Hz tone, and 8 beeps collect 15+ closed-eye samples.",
              code: "if (open_samples.size() >= 15) { /* transition & run Phase 2 */ }"
            },
            {
              title: "Step 3: Apply Baselines & Enter Parked (Calibrated)",
              highlightNodes: ["calib_complete", "goto_3c_ear_base", "goto_2b_head_base", "goto_4b_yawn_rst", "goto_1c_face_rst", "calib_exit"],
              highlightEdges: ["e_p2_ok", "e_comp_ear", "e_comp_head", "e_comp_yawn", "e_comp_face", "e_exit"],
              concept: "After playing the success chime, baselines are sent to the eye and head detectors, yawn/face timers reset, and the system enters Parked (Calibrated).",
              code: "ear_detector_.apply_baseline(res->ear_open, res->ear_closed, now);\nhead_detector_.apply_baseline(res->neutral_rho);\nsystem_state_.store(SystemState::StationaryCalibrated);"
            }
          ]
        },
        {
          id: "calib_retry",
          label: "Trace: Insufficient Samples -> Repeat 4 s Window",
          steps: [
            {
              title: "Fewer Than 15 Samples Repeats Window",
              highlightNodes: ["p2_window", "p2_count"],
              highlightEdges: ["e_p2_cnt", "e_p2_retry"],
              concept: "If occlusion prevents 15 valid detections in 4 seconds, the 8-beep measurement window repeats automatically until the quota is reached.",
              code: "while (closed_samples.size() < cfg.min_samples_per_window) { measure_window(...); }"
            }
          ]
        },
        {
          id: "calib_abort",
          label: "Trace: Shutdown Abort Restores Prior State",
          steps: [
            {
              title: "Shutdown Interrupt Restores Prior Stationary State",
              highlightNodes: ["p1_window", "abort_check", "abort_restore"],
              highlightEdges: ["e_win_cnt", "e_p1_ab", "e_ab_yes"],
              concept: "If a quit signal arrives mid-calibration, calibration cancels cleanly and restores the previous stationary state.",
              code: "if (ctx.shutdown.load()) return std::nullopt;\nsystem_state_.store(was_calibrated ? StationaryCalibrated : StationaryUncalibrated);"
            }
          ]
        }
      ]
    },
    {
      id: "1C",
      label: "1C. Shared Dlib & 500 ms Gate",
      subtitle: "500 ms missing-face CPU throttle, frontal face detector, 68 landmarks & Face Found vs Lost fan-out",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "process", "decision", "portal_out", "yes", "no", "portal_edge"],
      overview: {
        title: "Shared 68-Landmark Face Detector & 500 ms Gate",
        concept: "Both Head Droop and Blink/PERCLOS share one facial landmark pass on the Fatigue Worker. When no face is visible, searches are throttled to once every 500 ms so the Gaze Worker retains full CPU bandwidth.",
        code: "const bool should_probe = !face_missing || (elapsed >= 500ms);\nif (should_probe) landmarks_opt = face_detector_.detect(frame, now);",
        note: "The 68-point shape predictor only runs after the frontal face detector confirms a bounding box, preventing hallucinated landmarks on empty seats.",
        codeRef: "src/PipelineCoordinator.cpp (L494, L533-555)\nsrc/FaceLandmarkDetector.cpp (L56-121)\ninclude/Types.hpp (L184-188)"
      },
      nodes: [
        {
          id: "entry_frame", kind: "portal_in", x: 24, y: 34, w: 225, h: 56,
          label: "FROM Fatigue Loop", sublabel: "After full-frame yawn check",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "cnn_process" },
          details: {
            title: "Normal Loop Entry",
            concept: "Every camera frame in the Fatigue Worker first passes through the full-frame yawn network, then reaches this face gate.",
            code: "const YawnAssessment yawn_res = yawn_detector_.process(frame, t_work_start);\n// Next: 500 ms missing-face probe gate",
            codeRef: "src/PipelineCoordinator.cpp (L526-544)"
          }
        },
        {
          id: "from_calib_face_reset", kind: "portal_in", x: 24, y: 120, w: 225, h: 56,
          label: "FROM 1B: Reset Face Gate", sublabel: "Clear missing-face timer",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_1c_face_rst" },
          details: {
            title: "Calibration Reset Hook",
            concept: "Resets face-missing flags and 3-second warning state once calibration finishes.",
            code: "void FaceLandmarkDetector::reset() {\n  face_missing_ = false; warned_3s_ = false; face_lost_start_ = now();\n}",
            codeRef: "src/FaceLandmarkDetector.cpp (L117-121)"
          }
        },
        {
          id: "calib_bypass", kind: "portal_in", x: 24, y: 206, w: 225, h: 56,
          label: "FROM 1B: Direct Detect", sublabel: "Calibration skips 500 ms gate",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_1c_detect" },
          details: {
            title: "Direct Calibration Probe",
            concept: "During calibration, the measurement loop calls face detection directly without waiting 500 ms between misses.",
            code: "auto lm_opt = ctx.face.detect(frame, steady_clock::now());",
            codeRef: "src/Calibration.cpp (L63)"
          }
        },
        {
          id: "throttle_gate", kind: "decision", x: 300, y: 20, w: 255, h: 84,
          label: "Face Missing &\nUnder 500 ms?", sublabel: "CPU throttle gate",
          details: {
            title: "500 ms Probe Throttle",
            concept: "If the previous frame had no face and fewer than 500 ms have passed since the last search, skips expensive face detection entirely.",
            code: "const bool should_probe = !face_missing ||\n  (duration_cast<milliseconds>(t_work_start - last_face_probe_time) >= 500ms);",
            codeRef: "src/PipelineCoordinator.cpp (L535-544) | include/Types.hpp (L187)"
          }
        },
        {
          id: "dlib_hog", kind: "process", x: 300, y: 150, w: 255, h: 58,
          label: "Grayscale Face Search", sublabel: "Run frontal face detector",
          details: {
            title: "Frontal Face Detector",
            concept: "Updates the probe timestamp, converts the image to grayscale, and searches for frontal faces.",
            code: "cv::cvtColor(frame, gray_buf_, cv::COLOR_BGR2GRAY);\nstd::vector<dlib::rectangle> faces = detector_(cimg, 0);",
            codeRef: "src/FaceLandmarkDetector.cpp (L69-78)"
          }
        },
        {
          id: "face_found_gate", kind: "decision", x: 300, y: 248, w: 255, h: 84,
          label: "Face Detected?", sublabel: "Guards 68-pt prediction",
          details: {
            title: "Face Presence Verification",
            concept: "Proceeds to 68-point landmark prediction if a face box exists; otherwise enters the face-lost branch.",
            code: "if (faces.empty()) { /* mark missing */ return std::nullopt; }\ndlib::full_object_detection shape = predictor_(cimg, faces[0]);",
            codeRef: "src/FaceLandmarkDetector.cpp (L80-104)"
          }
        },
        {
          id: "dlib_shape", kind: "process", x: 605, y: 261, w: 215, h: 58,
          label: "Predict 68 Landmarks", sublabel: "Clear missing flag & extract pts",
          details: {
            title: "68-Point Landmark Extraction",
            concept: "Marks the face present and extracts 68 facial coordinates for eye and head posture calculations.",
            code: "face_missing_ = false; warned_3s_ = false;\nfor (size_t i = 0; i < 68; ++i) points[i] = Point2f(shape.part(i));",
            codeRef: "src/FaceLandmarkDetector.cpp (L100-114)"
          }
        },
        {
          id: "out_landmarks", kind: "portal_out", x: 865, y: 218, w: 195, h: 56,
          label: "GOTO 3C: Eye Check", sublabel: "Send 68 landmarks",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "from_face_gate" },
          details: {
            title: "Send Landmarks to Eye Monitor",
            concept: "Passes the 68 landmarks into Blink Duration and PERCLOS estimation.",
            code: "ear_res = ear_detector_.process(landmarks, t_work_start, suppress_by_yawn, is_moving_calibrated);",
            codeRef: "src/PipelineCoordinator.cpp (L550)"
          }
        },
        {
          id: "out_landmarks_head", kind: "portal_out", x: 865, y: 298, w: 195, h: 56,
          label: "GOTO 2B: Head Pose", sublabel: "Send 68 landmarks",
          target: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_face_gate" },
          details: {
            title: "Send Landmarks to Head Monitor",
            concept: "Passes the same 68 landmarks into Head Orientation pitch and droop estimation.",
            code: "head_res = head_detector_.process(landmarks, t_work_start, suppress_by_yawn);",
            codeRef: "src/PipelineCoordinator.cpp (L551)"
          }
        },
        {
          id: "mark_missing", kind: "process", x: 300, y: 404, w: 255, h: 58,
          label: "Mark Face Missing", sublabel: "Warn once after 3.0 s absent",
          details: {
            title: "Face-Lost Tracking",
            concept: "Sets the face-missing state so the next frames throttle to 500 ms, and logs one warning if absent for 3 seconds.",
            code: "face_missing_ = true;\nif (elapsed >= 3.0 && !warned_3s_) { warned_3s_ = true; /* warn */ }",
            codeRef: "src/FaceLandmarkDetector.cpp (L80-96) | include/Types.hpp (L186)"
          }
        },
        {
          id: "out_face_lost", kind: "portal_out", x: 605, y: 406, w: 215, h: 56,
          label: "GOTO 3C: Eyes Lost", sublabel: "Clear unconfirmed blink",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "from_face_gate" },
          details: {
            title: "Eye Face-Lost Handler",
            concept: "Wipes active blink streaks immediately so missing-face gaps never trigger false micro-sleeps.",
            code: "ear_res = ear_detector_.on_face_lost(t_work_start);",
            codeRef: "src/PipelineCoordinator.cpp (L553) | src/EarDetector.cpp (L266-331)"
          }
        },
        {
          id: "out_face_lost_head", kind: "portal_out", x: 865, y: 406, w: 195, h: 56,
          label: "GOTO 2B: Head Lost", sublabel: "1x gradual timer decay",
          target: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_face_gate" },
          details: {
            title: "Head Face-Lost Handler",
            concept: "Decays any accumulated droop timer smoothly at 1x speed instead of snapping to zero.",
            code: "head_res = head_detector_.on_face_lost(t_work_start);",
            codeRef: "src/PipelineCoordinator.cpp (L554) | src/HeadOrientationDetector.cpp (L96-114)"
          }
        }
      ],
      edges: [
        { id: "e_in_thr", points: [[249, 62], [300, 62]], tone: "default" },
        { id: "e_cal_hog", points: [[249, 234], [275, 234], [275, 179], [300, 179]], tone: "portal" },
        { id: "e_thr_no", points: [[427, 104], [427, 150]], label: "NO", tone: "no" },
        { id: "e_thr_yes", points: [[555, 62], [580, 62], [580, 380], [427, 380], [427, 404]], label: "YES (Skip)", tone: "yes", lx: 625, ly: 62 },
        { id: "e_hog_chk", points: [[427, 208], [427, 248]], tone: "default" },
        { id: "e_fnd_yes", points: [[555, 290], [605, 290]], label: "YES", tone: "yes" },
        { id: "e_fnd_no", points: [[427, 332], [427, 404]], label: "NO", tone: "no" },
        { id: "e_shp_ear", points: [[820, 276], [842, 276], [842, 246], [865, 246]], tone: "portal" },
        { id: "e_shp_head", points: [[820, 304], [842, 304], [842, 326], [865, 326]], tone: "portal" },
        { id: "e_miss_ear", points: [[555, 434], [605, 434]], tone: "portal" },
        { id: "e_miss_head", points: [[555, 450], [580, 450], [580, 488], [842, 488], [842, 434], [865, 434]], tone: "portal" }
      ],
      traces: [
        {
          id: "face_found_trace",
          label: "Trace: Face Found -> 68 Landmarks -> Blink & Head",
          steps: [
            {
              title: "Step 1: Search Gate Allows Face Detection",
              highlightNodes: ["entry_frame", "throttle_gate", "dlib_hog", "face_found_gate"],
              highlightEdges: ["e_in_thr", "e_thr_no", "e_hog_chk"],
              concept: "Either a face was recently present or 500 ms have passed, so the detector searches the grayscale image.",
              code: "if (!face_missing || elapsed >= 500ms) landmarks_opt = face_detector_.detect(frame, now);"
            },
            {
              title: "Step 2: Predict 68 Landmarks & Fan Out",
              highlightNodes: ["face_found_gate", "dlib_shape", "out_landmarks", "out_landmarks_head"],
              highlightEdges: ["e_fnd_yes", "e_shp_ear", "e_shp_head"],
              concept: "With a verified face bounding box, 68 landmarks are predicted and passed to the Eye Monitor (3C) and Head Pose Monitor (2B).",
              code: "ear_res = ear_detector_.process(landmarks, now, suppress_by_yawn, is_moving_cal);\nhead_res = head_detector_.process(landmarks, now, suppress_by_yawn);"
            }
          ]
        },
        {
          id: "face_lost_throttle",
          label: "Trace: Face Missing -> 500 ms CPU-Saving Throttle",
          steps: [
            {
              title: "Step 1: Empty Search Marks Face Missing",
              highlightNodes: ["dlib_hog", "face_found_gate", "mark_missing", "out_face_lost", "out_face_lost_head"],
              highlightEdges: ["e_hog_chk", "e_fnd_no", "e_miss_ear", "e_miss_head"],
              concept: "When no face is found, the missing flag turns on and both detectors switch to their face-lost handlers.",
              code: "if (faces.empty()) { face_missing_ = true; return std::nullopt; }"
            },
            {
              title: "Step 2: Next Frames Under 500 ms Skip Search",
              highlightNodes: ["entry_frame", "throttle_gate", "mark_missing", "out_face_lost", "out_face_lost_head"],
              highlightEdges: ["e_in_thr", "e_thr_yes", "e_miss_ear", "e_miss_head"],
              concept: "Subsequent frames within 500 ms skip face search completely to save CPU for gaze tracking, while full-frame yawn detection keeps running.",
              code: "const bool should_probe = !face_missing || (elapsed >= 500ms);"
            }
          ]
        }
      ]
    }
  ]
});
