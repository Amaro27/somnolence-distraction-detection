window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec1",
  title: "1. Vehicle & Calib FSM",
  subviews: [
    {
      id: "1A",
      label: "1A. 5-State Movement & IMU FSM",
      subtitle: "SystemState lifecycle, keyboard 'm'/'c', 20 Hz MPU-6050 hysteresis & parking resets",
      viewBox: "0 0 1120 660",
      legend: ["state", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "1A: 5-State Vehicle & Calibration Lifecycle FSM",
        concept: "All trigger estimation runs inside a 5-state safety FSM (SystemState). The vehicle boots stationary and uncalibrated (unless --start-moving is passed), locks out movement switches during calibration, and resets temporal buffers whenever the car stops.",
        note: "On Linux, a dedicated 20 Hz MPU-6050 IMU thread automates Stationary <-> Moving transitions using smoothed acceleration hysteresis. On Windows/macOS or with --no-imu, MPU6050::init() returns false cleanly and manual 'm' keypresses control movement.",
        codeRef: "morphed_system6/distraction/modular/include/Types.hpp (L66-72)\nsrc/PipelineCoordinator.cpp (L130-131, L319-384, L589-689)\ninclude/MPU6050.hpp (L38-205)"
      },
      nodes: [
        {
          id: "startup_opt", kind: "decision", x: 20, y: 24, w: 205, h: 86,
          label: "--start-moving?", sublabel: "Boot state check",
          details: {
            concept: "At PipelineCoordinator construction, the system reads options_.start_moving to pick its initial SystemState.",
            codeRef: "src/PipelineCoordinator.cpp (L130-131)"
          }
        },
        {
          id: "st_stat_uncal", kind: "state", x: 265, y: 34, w: 205, h: 66,
          label: "StationaryUncalibrated", sublabel: "Default boot; alerts off",
          details: {
            concept: "Safe default startup state: vehicle is parked and no personal baselines exist. Gaze inference is skipped and all alerts are suppressed.",
            codeRef: "include/Types.hpp (L67)"
          }
        },
        {
          id: "st_mov_uncal", kind: "state", x: 265, y: 156, w: 205, h: 66,
          label: "MovingUncalibrated", sublabel: "Alerts paused; warns user",
          details: {
            concept: "Vehicle is moving before calibration. Both distraction and drowsiness alerts remain paused because personal EAR and head-pitch thresholds are unknown.",
            codeRef: "include/Types.hpp (L68) | src/PipelineCoordinator.cpp (L347-353)"
          }
        },
        {
          id: "st_calib", kind: "state", tone: "calib", x: 545, y: 34, w: 205, h: 66,
          label: "Calibrating", sublabel: "Lockout 'm' & IMU",
          details: {
            concept: "Entered only via 'c' while stationary. Both manual 'm' movement toggles and automatic IMU state changes are locked out while calibration runs.",
            codeRef: "include/Types.hpp (L69) | src/PipelineCoordinator.cpp (L323-327, L363-383, L644-682)"
          }
        },
        {
          id: "goto_1b_calib", kind: "portal_out", x: 545, y: 132, w: 205, h: 56,
          label: "GOTO 1B: CALIBRATION", sublabel: "Hijacks Fatigue Worker",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "entry_calib" },
          details: {
            concept: "When somnolence_worker_loop observes SystemState == Calibrating, it pauses monitoring and runs the 2-phase acoustic calibration sequence in-place.",
            codeRef: "src/PipelineCoordinator.cpp (L480-505)"
          }
        },
        {
          id: "st_stat_cal", kind: "state", doubleBorder: true, x: 830, y: 34, w: 230, h: 66,
          label: "StationaryCalibrated", sublabel: "Baselines set; alerts off",
          details: {
            concept: "Baselines are applied and the vehicle is parked. Fatigue Worker keeps computing EAR/pitch/yawn for HUD telemetry while PERCLOS and alarms stay paused.",
            codeRef: "include/Types.hpp (L70) | src/PipelineCoordinator.cpp (L496, L515-575)"
          }
        },
        {
          id: "st_mov_cal", kind: "state", tone: "active", doubleBorder: true, x: 830, y: 156, w: 230, h: 66,
          label: "MovingCalibrated", sublabel: "Full detection active",
          details: {
            concept: "Full multimodal monitoring state: Gaze Worker runs ResNet-18+LSTM, Fatigue Worker tracks micro-sleep, PERCLOS, droop & yawn, and AlertManager is active.",
            codeRef: "include/Types.hpp (L71) | src/PipelineCoordinator.cpp (L429, L515, L557-559)"
          }
        },
        {
          id: "key_gate", kind: "process", tone: "neutral", x: 20, y: 274, w: 255, h: 74,
          label: "Console / GUI ('m' / 'c')", sublabel: "Main loop polls commands\n'c' rejected unless stationary",
          details: {
            concept: "Main supervisor polls stdin and OpenCV window keys. 'm' toggles movement unless Calibrating; 'c' enters Calibrating only from Stationary* states.",
            codeRef: "src/PipelineCoordinator.cpp (L282-384 handle_console_command)"
          }
        },
        {
          id: "grp_imu", kind: "group", x: 20, y: 374, w: 505, h: 264,
          label: "IMU Motion Worker (20 Hz Linux I2C 0x68)", sublabel: "Exits cleanly on non-Linux hosts or --no-imu",
          details: {
            concept: "Background thread polling MPU-6050 accelerometer magnitude at 50 ms intervals with a 5-sample moving average and asymmetric hysteresis counters.",
            codeRef: "src/PipelineCoordinator.cpp (L589-689) | include/MPU6050.hpp (L51-205)"
          }
        },
        {
          id: "imu_smooth", kind: "process", x: 38, y: 432, w: 215, h: 68,
          label: "5-Sample Smoothed Mag", sublabel: "dyn = |mean5(mag) - 9.81|",
          details: {
            concept: "Averages the last 5 total acceleration magnitudes (m/s^2) and subtracts GRAVITY_CONST (9.80665 m/s^2) to isolate vehicle dynamic acceleration.",
            codeRef: "src/PipelineCoordinator.cpp (L606-634)"
          }
        },
        {
          id: "imu_high_gate", kind: "decision", x: 290, y: 420, w: 215, h: 92,
          label: "dyn > 0.80 m/s^2\nfor 4 samples?", sublabel: "200 ms sustained",
          details: {
            concept: "Requires 4 consecutive high-motion readings (count_motion >= 4) before flipping vehicle_moving_ to true and advancing Stationary* -> Moving*.",
            codeRef: "src/PipelineCoordinator.cpp (L610-612, L635-655)"
          }
        },
        {
          id: "imu_low_gate", kind: "decision", x: 290, y: 534, w: 215, h: 92,
          label: "dyn < 0.35 m/s^2\nfor 15 samples?", sublabel: "750 ms stillness",
          details: {
            concept: "Requires 15 consecutive low-motion readings (count_still >= 15) before flipping vehicle_moving_ to false and parking Moving* -> Stationary*.",
            codeRef: "src/PipelineCoordinator.cpp (L611-613, L656-684)"
          }
        },
        {
          id: "imu_calib_lock", kind: "decision", x: 38, y: 534, w: 215, h: 92,
          label: "SystemState ==\nCalibrating?", sublabel: "Lockout check",
          details: {
            concept: "IMU transition blocks only act on Stationary* (when accelerating) or Moving* (when stopping), leaving Calibrating untouched until calibration finishes.",
            codeRef: "src/PipelineCoordinator.cpp (L643-683)"
          }
        },
        {
          id: "grp_cleanup", kind: "group", x: 550, y: 274, w: 545, h: 364,
          label: "Parking Cleanup (Moving* -> Stationary*)", sublabel: "Executed on stop via 'm' key or IMU stillness",
          details: {
            concept: "Stopping the car immediately silences active alarms, wipes alert windows, and schedules detector resets.",
            codeRef: "src/PipelineCoordinator.cpp (L328-346, L664-682)"
          }
        },
        {
          id: "park_cleanup", kind: "process", x: 568, y: 332, w: 230, h: 74,
          label: "Park State Cleanup", sublabel: "Reset alerts + debug HUD;\nclear PERCLOS if calibrated",
          details: {
            concept: "Main or IMU thread performs cleanup immediately upon transitioning from Moving* to Stationary*.",
            note: "Concurrency caveat: clear_perclos() is invoked directly from Main (L340) and IMU (L668) threads without a mutex against somnolence_worker_loop, whereas GazeDetector uses atomic gaze_reset_requested_.",
            codeRef: "src/PipelineCoordinator.cpp (L328-346, L518-522, L664-682)"
          }
        },
        {
          id: "goto_6c_reset", kind: "portal_out", x: 842, y: 332, w: 235, h: 60,
          label: "GOTO 6C: ALERT_RESET", sublabel: "AlertManager::reset()",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_reset" },
          details: {
            concept: "Clears gaze_window_, cue_window_, distracted_since_, resets DriverState to Safe, and immediately stops audio playback.",
            codeRef: "src/AlertManager.cpp (L390-406)"
          }
        },
        {
          id: "goto_3d_clear", kind: "portal_out", x: 842, y: 418, w: 235, h: 64,
          label: "GOTO 3D: CLEAR_PERCLOS", sublabel: "MovingCal -> StatCal only",
          target: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "from_park_clear" },
          details: {
            concept: "Wipes the 30-second rolling PERCLOS buffer and warmup clock whenever stopping from MovingCalibrated.",
            note: "Direct cross-thread call from Main (L340) and IMU (L668); somnolence_worker_loop also performs a fallback clear_perclos() when leaving MovingCalibrated (L520).",
            codeRef: "src/PipelineCoordinator.cpp (L340, L520, L668) | src/EarDetector.cpp (L17-22)"
          }
        },
        {
          id: "set_gaze_atomic", kind: "portal_out", x: 842, y: 508, w: 235, h: 64,
          label: "SET ATOMIC: gaze_reset", sublabel: "gaze_reset_requested_=true",
          target: { sectionId: "sec5", subviewId: "5B", highlightNodeId: "from_atomic_reset" },
          details: {
            concept: "Instead of touching GazeDetector's ring buffer across threads, Main/IMU raise an atomic flag consumed safely by GazeWorker on its next loop turn.",
            codeRef: "src/PipelineCoordinator.cpp (L331, L341, L412, L424, L669, L677)"
          }
        },
        {
          id: "hud_paused", kind: "term_return", x: 568, y: 512, w: 230, h: 54,
          label: "Detections Paused", sublabel: "Await 'm', IMU, or 'c'",
          details: {
            concept: "Stationary state active; Fatigue Worker keeps HUD telemetry alive if calibrated, while Gaze Worker skips neural inference.",
            codeRef: "src/PipelineCoordinator.cpp (L429, L557)"
          }
        }
      ],
      edges: [
        { id: "e_boot_no", points: [[225, 67], [265, 67]], label: "NO", tone: "no" },
        { id: "e_boot_yes", points: [[122, 110], [122, 189], [265, 189]], label: "YES", tone: "yes", lx: 195, ly: 179 },
        { id: "e_uncal_move", points: [[345, 100], [345, 156]], label: "'m' / IMU", tone: "default", lx: 308, ly: 128 },
        { id: "e_uncal_park", points: [[395, 156], [395, 100]], label: "stop", tone: "default", lx: 418, ly: 128 },
        { id: "e_c_start", points: [[470, 67], [545, 67]], label: "'c'", tone: "yes" },
        { id: "e_calib_take", points: [[647, 100], [647, 132]], label: "takeover", tone: "portal" },
        { id: "e_c_done", points: [[750, 67], [830, 67]], label: "done", tone: "yes" },
        { id: "e_cal_move", points: [[920, 100], [920, 156]], label: "'m' / IMU", tone: "default", lx: 882, ly: 128 },
        { id: "e_cal_park", points: [[975, 156], [975, 100]], label: "park", tone: "default", lx: 1002, ly: 128 },
        { id: "e_imu_hi", points: [[253, 466], [290, 466]], tone: "default" },
        { id: "e_imu_lo", points: [[397, 512], [397, 534]], label: "NO", tone: "no", lx: 416, ly: 523 },
        { id: "e_imu_chk", points: [[290, 580], [253, 580]], label: "trigger", tone: "default" },
        { id: "e_mov_cleanup", points: [[830, 200], [810, 200], [810, 312], [683, 312], [683, 332]], label: "on park", tone: "portal", lx: 745, ly: 302 },
        { id: "e_cln_alert", points: [[798, 362], [842, 362]], tone: "portal" },
        { id: "e_cln_perc", points: [[798, 382], [820, 382], [820, 450], [842, 450]], tone: "portal" },
        { id: "e_cln_gaze", points: [[798, 396], [812, 396], [812, 540], [842, 540]], tone: "portal" },
        { id: "e_cln_ret", points: [[683, 406], [683, 512]], tone: "yes" }
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
              concept: "By default (--start-moving omitted), the FSM boots into StationaryUncalibrated with all detections and alarms disabled.",
              codeRef: "src/PipelineCoordinator.cpp (L130-131)"
            },
            {
              title: "Step 2: Driver Presses 'c' While Parked",
              highlightNodes: ["key_gate", "st_stat_uncal", "st_calib", "goto_1b_calib"],
              highlightEdges: ["e_c_start", "e_calib_take"],
              concept: "Calibration is accepted only from a Stationary* state. SystemState stores Calibrating, which hijacks the Fatigue Worker into 2-phase acoustic calibration (1B).",
              portalTarget: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "entry_calib" },
              codeRef: "src/PipelineCoordinator.cpp (L363-384, L480-488)"
            },
            {
              title: "Step 3: Baselines Applied -> StationaryCalibrated",
              highlightNodes: ["st_calib", "st_stat_cal"],
              highlightEdges: ["e_c_done"],
              concept: "Once 15+ open-eye and 15+ closed-eye samples are collected, EAR and neutral head-pitch baselines are applied and SystemState becomes StationaryCalibrated.",
              codeRef: "src/PipelineCoordinator.cpp (L489-499)"
            },
            {
              title: "Step 4: Drive ('m' or IMU) -> MovingCalibrated",
              highlightNodes: ["st_stat_cal", "st_mov_cal"],
              highlightEdges: ["e_cal_move"],
              concept: "Pressing 'm' or detecting sustained vehicle acceleration switches SystemState to MovingCalibrated, activating gaze inference, PERCLOS, and alert fusion.",
              codeRef: "src/PipelineCoordinator.cpp (L354-359, L643-648)"
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
              concept: "At 20 Hz, the MPU-6050 worker computes |mean5(accel_mag) - 9.81 m/s^2| to reject short road bumps and sensor jitter.",
              codeRef: "src/PipelineCoordinator.cpp (L621-634)"
            },
            {
              title: "Step 2: 4 High-Motion Samples -> Moving",
              highlightNodes: ["imu_high_gate", "st_stat_cal", "st_mov_cal"],
              highlightEdges: ["e_cal_move"],
              concept: "Holding dynamic acceleration > 0.80 m/s^2 for 4 consecutive samples (200 ms) flips StationaryCalibrated -> MovingCalibrated.",
              codeRef: "src/PipelineCoordinator.cpp (L635-655)"
            },
            {
              title: "Step 3: 15 Still Samples -> Park & Cleanup",
              highlightNodes: ["imu_low_gate", "imu_calib_lock", "st_mov_cal", "st_stat_cal", "park_cleanup", "goto_6c_reset", "goto_3d_clear", "set_gaze_atomic", "hud_paused"],
              highlightEdges: ["e_imu_lo", "e_imu_chk", "e_cal_park", "e_mov_cleanup", "e_cln_alert", "e_cln_perc", "e_cln_gaze", "e_cln_ret"],
              concept: "Holding < 0.35 m/s^2 for 15 samples (750 ms) parks the vehicle, resets AlertManager (6C), clears PERCLOS (3D), and raises the atomic gaze-reset flag (5B).",
              note: "Note the concurrency caveat: clear_perclos() is called directly from the IMU thread here (L668), whereas GazeWorker's reset uses atomic gaze_reset_requested_.",
              codeRef: "src/PipelineCoordinator.cpp (L656-683)"
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
              concept: "While SystemState == Calibrating, manual 'm' toggles print an ignore notice and IMU motion checks refuse to transition out of Calibrating.",
              codeRef: "src/PipelineCoordinator.cpp (L323-327, L643-683)"
            }
          ]
        }
      ]
    },
    {
      id: "1B",
      label: "1B. 2-Phase Calibration Routine",
      subtitle: "In-place Fatigue Worker takeover: +6 dB voice, 415 Hz prep tone, 8x500 ms beeps & baselines",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "1B: 2-Phase Acoustic Calibration Routine",
        concept: "Triggered inside somnolence_worker_loop whenever SystemState == Calibrating. Phase 1 records open-eye EAR and neutral head proportion rho; Phase 2 records closed-eye EAR floor, repeating 4-second beep windows until >= 15 face samples are captured per phase.",
        note: "Calibration.cpp plays the 1-second 415 Hz preparation tone before measurement beeps in BOTH Phase 1 and Phase 2 (whereas legacy README implies Phase 2 skips it). Dlib probes are skipped if <= 30 ms remain before the next 500 ms beep.",
        codeRef: "src/PipelineCoordinator.cpp (L470-505)\nsrc/Calibration.cpp (L14-220)\ninclude/Types.hpp (L204-217)"
      },
      nodes: [
        {
          id: "entry_calib", kind: "portal_in", x: 20, y: 26, w: 215, h: 64,
          label: "FROM 1A: Calibrating", sublabel: "Fatigue Worker takeover",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "goto_1b_calib" },
          details: {
            concept: "somnolence_worker_loop detects SystemState == Calibrating before popping normal frames and invokes run_calibration().",
            codeRef: "src/PipelineCoordinator.cpp (L480-488)"
          }
        },
        {
          id: "p1_intro", kind: "process", x: 270, y: 24, w: 235, h: 68,
          label: "Phase 1: Voice + Tone", sublabel: "+6dB audio -> 1s gap ->\n415Hz 1s tone -> 1s gap",
          details: {
            concept: "Guides driver to look straight ahead with open eyes using blocking +6 dB speech, a 1 s pause, a 1 s 415 Hz tone, and another 1 s pause.",
            codeRef: "src/Calibration.cpp (L95-104)"
          }
        },
        {
          id: "p1_window", kind: "process", x: 545, y: 24, w: 255, h: 68,
          label: "Phase 1 Window (8x500ms)", sublabel: "+3dB beeps (4.0s); skip Dlib\nif <=30ms to next beep",
          details: {
            concept: "Plays 8 cadence beeps 500 ms apart at +3 dB while fetching latest frames (<= 50 ms wait) and collecting EAR and rho when landmarks succeed.",
            codeRef: "src/Calibration.cpp (L25-76 measure_window, L127-134)"
          }
        },
        {
          id: "goto_1c_detect", kind: "portal_out", x: 855, y: 26, w: 240, h: 64,
          label: "GOTO 1C: DLIB_DETECT", sublabel: "Direct detect (no throttle)",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "dlib_hog" },
          details: {
            concept: "Calls ctx.face.detect(frame, now) directly without the normal loop's 500 ms missing-face throttle (guarded only by the 30 ms beep-deadline check).",
            codeRef: "src/Calibration.cpp (L55-66)"
          }
        },
        {
          id: "p1_count", kind: "decision", x: 545, y: 132, w: 255, h: 92,
          label: "open_samples >= 15?", sublabel: "Min 15 valid face frames",
          details: {
            concept: "Repeats the 8-beep measurement window automatically if fewer than 15 valid face samples were captured in Phase 1.",
            codeRef: "src/Calibration.cpp (L116-136)"
          }
        },
        {
          id: "abort_check", kind: "decision", x: 270, y: 132, w: 235, h: 92,
          label: "shutdown_requested_?", sublabel: "Pauses & beeps check",
          details: {
            concept: "Checked continuously in pause_interruptible (20 ms steps) and measure_window so Ctrl+C or 'q' exits calibration without hanging.",
            codeRef: "src/Calibration.cpp (L14-23, L36-48, L96-205)"
          }
        },
        {
          id: "abort_restore", kind: "term_return", x: 20, y: 148, w: 215, h: 60,
          label: "ABORT: Restore State", sublabel: "was_calibrated ? Cal : Uncal",
          details: {
            concept: "Returns std::nullopt; PipelineCoordinator restores StationaryCalibrated if previously calibrated, otherwise StationaryUncalibrated.",
            codeRef: "src/PipelineCoordinator.cpp (L500-503)"
          }
        },
        {
          id: "trans_gap", kind: "process", x: 855, y: 146, w: 240, h: 64,
          label: "1.0s Transition Silence", sublabel: "CLOSE EYES NOW AND HOLD",
          details: {
            concept: "Signals the driver to close their eyes during a 1.0 s pause before Phase 2 instructions play.",
            codeRef: "src/Calibration.cpp (L141-148)"
          }
        },
        {
          id: "p2_intro", kind: "process", x: 855, y: 262, w: 240, h: 68,
          label: "Phase 2: Voice + Tone", sublabel: "+6dB audio -> 1s -> 415Hz\n1s prep tone -> 1s gap",
          details: {
            concept: "Plays Phase 2 closed-eye spoken instructions (+6 dB), 1 s gap, 415 Hz preparation tone (played in both phases!), and 1 s gap.",
            codeRef: "src/Calibration.cpp (L161-170)"
          }
        },
        {
          id: "p2_window", kind: "process", x: 545, y: 262, w: 255, h: 68,
          label: "Phase 2 Window (8x500ms)", sublabel: "+3dB beeps; collect closed\nEAR via 1C until >=15",
          details: {
            concept: "Runs 8 cadence beeps at +3 dB while sampling closed-eye EAR into closed_samples.",
            codeRef: "src/Calibration.cpp (L178-199)"
          }
        },
        {
          id: "p2_count", kind: "decision", x: 270, y: 250, w: 235, h: 92,
          label: "closed_samples >= 15?", sublabel: "Min 15 closed frames",
          details: {
            concept: "Repeats the Phase 2 4-second beep window until at least 15 valid closed-eye EAR samples are recorded.",
            codeRef: "src/Calibration.cpp (L181-199)"
          }
        },
        {
          id: "calib_complete", kind: "process", x: 20, y: 262, w: 215, h: 68,
          label: "Compute Baselines + Chime", sublabel: "mean(open/closed),\nmedian(rho) + tada-LOUD",
          details: {
            concept: "Plays tada-LOUD.wav and computes ear_open = mean(open_samples), ear_closed = mean(closed_samples), and neutral_rho = median(rho_samples).",
            codeRef: "src/Calibration.cpp (L213-219)"
          }
        },
        {
          id: "goto_3c_ear_base", kind: "portal_out", x: 20, y: 405, w: 235, h: 62,
          label: "GOTO 3C: APPLY_EAR", sublabel: "ear_open, ear_closed -> T",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "from_calib_ear" },
          details: {
            concept: "Applies calibrated open/closed EAR baselines to compute 70% descent and 20% closure floor thresholds.",
            codeRef: "src/PipelineCoordinator.cpp (L491) | src/EarDetector.cpp (L42-83)"
          }
        },
        {
          id: "goto_2b_head_base", kind: "portal_out", x: 290, y: 405, w: 235, h: 62,
          label: "GOTO 2B: APPLY_HEAD", sublabel: "baseline_rho = median(rho)",
          target: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_calib_rho" },
          details: {
            concept: "Sets calibrated neutral head proportion baseline_rho_ (fallback 0.50) and clears droop accumulators.",
            codeRef: "src/PipelineCoordinator.cpp (L492) | src/HeadOrientationDetector.cpp (L13-25)"
          }
        },
        {
          id: "goto_4b_yawn_rst", kind: "portal_out", x: 560, y: 405, w: 235, h: 62,
          label: "GOTO 4B: YAWN_RESET", sublabel: "yawn_detector_.reset()",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "from_calib_yawn_reset" },
          details: {
            concept: "Resets YawnDetector accumulator and cooldown timers cleanly after calibration.",
            codeRef: "src/PipelineCoordinator.cpp (L493) | src/YawnDetector.cpp (L80-87)"
          }
        },
        {
          id: "goto_1c_face_rst", kind: "portal_out", x: 830, y: 405, w: 235, h: 62,
          label: "GOTO 1C: FACE_RESET", sublabel: "face_detector_.reset()",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "from_calib_face_reset" },
          details: {
            concept: "Clears face_missing_ and 3 s lost-face timer before normal monitoring resumes.",
            codeRef: "src/PipelineCoordinator.cpp (L494) | src/FaceLandmarkDetector.cpp (L117-121)"
          }
        },
        {
          id: "calib_exit", kind: "term_return", x: 390, y: 536, w: 340, h: 58,
          label: "SET StationaryCalibrated", sublabel: "Return to somnolence_worker_loop",
          details: {
            concept: "Marks was_calibrated = true and sets SystemState = StationaryCalibrated; monitoring activates once the driver moves ('m' or IMU).",
            codeRef: "src/PipelineCoordinator.cpp (L495-499)"
          }
        }
      ],
      edges: [
        { id: "e_en_p1", points: [[235, 58], [270, 58]], tone: "default" },
        { id: "e_p1_win", points: [[505, 58], [545, 58]], tone: "default" },
        { id: "e_win_1c", points: [[800, 58], [855, 58]], tone: "portal" },
        { id: "e_win_cnt", points: [[672, 92], [672, 132]], tone: "default" },
        { id: "e_p1_retry", points: [[800, 178], [824, 178], [824, 108], [695, 108], [695, 92]], label: "NO (<15)", tone: "no", lx: 760, ly: 112 },
        { id: "e_p1_ok", points: [[800, 178], [855, 178]], label: "YES", tone: "yes" },
        { id: "e_p1_ab", points: [[545, 178], [505, 178]], label: "shutdown?", tone: "default" },
        { id: "e_ab_yes", points: [[270, 178], [235, 178]], label: "YES", tone: "no" },
        { id: "e_tr_p2", points: [[975, 210], [975, 262]], tone: "default" },
        { id: "e_p2_win", points: [[855, 296], [800, 296]], tone: "default" },
        { id: "e_p2_cnt", points: [[545, 296], [505, 296]], tone: "default" },
        { id: "e_p2_retry", points: [[387, 250], [387, 232], [672, 232], [672, 262]], label: "NO (<15)", tone: "no", lx: 530, ly: 226 },
        { id: "e_p2_ok", points: [[270, 296], [235, 296]], label: "YES", tone: "yes" },
        { id: "e_comp_ear", points: [[127, 330], [127, 405]], tone: "portal" },
        { id: "e_comp_head", points: [[127, 366], [407, 366], [407, 405]], tone: "portal" },
        { id: "e_comp_yawn", points: [[127, 366], [677, 366], [677, 405]], tone: "portal" },
        { id: "e_comp_face", points: [[127, 366], [947, 366], [947, 405]], tone: "portal" },
        { id: "e_exit", points: [[560, 467], [560, 536]], tone: "yes" }
      ],
      traces: [
        {
          id: "calib_happy",
          label: "Trace: Happy Path (>=15 Samples Both Phases)",
          steps: [
            {
              title: "Step 1: Phase 1 Voice, Tone & Open-Eye Window",
              highlightNodes: ["entry_calib", "p1_intro", "p1_window", "goto_1c_detect"],
              highlightEdges: ["e_en_p1", "e_p1_win", "e_win_1c"],
              concept: "Driver hears Phase 1 voice instructions (+6 dB), 1 s pause, 415 Hz prep tone, and 8 cadence beeps (+3 dB) while 1C extracts open-eye EAR and neutral rho.",
              codeRef: "src/Calibration.cpp (L95-136)"
            },
            {
              title: "Step 2: Phase 1 >= 15 Samples -> Transition",
              highlightNodes: ["p1_count", "trans_gap", "p2_intro", "p2_window", "p2_count"],
              highlightEdges: ["e_win_cnt", "e_p1_ok", "e_tr_p2", "e_p2_win", "e_p2_cnt"],
              concept: "With >= 15 open samples, 1 s silence cues eye closure, Phase 2 plays voice + 415 Hz prep tone, and 8 beeps collect >= 15 closed-eye EAR samples.",
              codeRef: "src/Calibration.cpp (L141-199)"
            },
            {
              title: "Step 3: Apply Baselines & Set StationaryCalibrated",
              highlightNodes: ["calib_complete", "goto_3c_ear_base", "goto_2b_head_base", "goto_4b_yawn_rst", "goto_1c_face_rst", "calib_exit"],
              highlightEdges: ["e_p2_ok", "e_comp_ear", "e_comp_head", "e_comp_yawn", "e_comp_face", "e_exit"],
              concept: "Success chime plays, mean/median baselines are injected into EarDetector and HeadOrientationDetector, Yawn/Face states reset, and SystemState becomes StationaryCalibrated.",
              codeRef: "src/PipelineCoordinator.cpp (L489-499) | src/Calibration.cpp (L213-219)"
            }
          ]
        },
        {
          id: "calib_retry",
          label: "Trace: Insufficient Samples -> Repeat 4 s Window",
          steps: [
            {
              title: "Phase 2 < 15 Samples Loops Automatically",
              highlightNodes: ["p2_window", "p2_count"],
              highlightEdges: ["e_p2_cnt", "e_p2_retry"],
              concept: "If glare or occlusion prevents 15 valid Dlib face detections during the 8-beep window, the loop immediately replays the 4-second beep window until >= 15 samples accumulate.",
              codeRef: "src/Calibration.cpp (L181-199)"
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
              concept: "If shutdown_requested_ becomes true mid-calibration, run_calibration() returns std::nullopt and PipelineCoordinator restores StationaryCalibrated (if previously calibrated) or StationaryUncalibrated.",
              codeRef: "src/PipelineCoordinator.cpp (L500-503)"
            }
          ]
        }
      ]
    },
    {
      id: "1C",
      label: "1C. Shared Dlib & 500 ms Gate",
      subtitle: "500 ms missing-face CPU throttle, Dlib frontal HOG + 68-pt landmarks, Face Found vs Lost fan-out",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "process", "decision", "portal_out", "yes", "no", "portal_edge"],
      overview: {
        title: "1C: Shared Dlib 68-Landmark & 500 ms Missing-Face Gate",
        concept: "Both HeadOrientationDetector and EarDetector consume the same Dlib 68-point facial landmark pass inside somnolence_worker_loop. When no face is visible, PipelineCoordinator throttles Dlib HOG searches to once every 500 ms to protect CPU throughput for GazeWorker.",
        note: "Dlib's 68-point shape_predictor is only invoked if frontal_face_detector returns at least one bounding box (faces[0]), guaranteeing zero ghost/hallucinated landmarks on empty frames.",
        codeRef: "src/PipelineCoordinator.cpp (L494, L533-555)\nsrc/FaceLandmarkDetector.cpp (L56-121)\ninclude/Types.hpp (L184-188)"
      },
      nodes: [
        {
          id: "entry_frame", kind: "portal_in", x: 24, y: 40, w: 235, h: 64,
          label: "FROM Somnolence Loop", sublabel: "After full-frame Yawn CNN",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "cnn_process" },
          details: {
            concept: "Every normal frame inside somnolence_worker_loop first runs YawnDetector::process(), then reaches the missing-face throttle check before invoking Dlib.",
            codeRef: "src/PipelineCoordinator.cpp (L526-544)"
          }
        },
        {
          id: "from_calib_face_reset", kind: "portal_in", x: 24, y: 142, w: 235, h: 62,
          label: "FROM 1B: FACE_RESET", sublabel: "Reset face_missing_ = false",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_1c_face_rst" },
          details: {
            concept: "Called on calibration completion so normal monitoring starts with a fresh face-missing timer.",
            codeRef: "src/FaceLandmarkDetector.cpp (L117-121)"
          }
        },
        {
          id: "calib_bypass", kind: "portal_in", x: 24, y: 242, w: 235, h: 64,
          label: "FROM 1B: Direct Detect", sublabel: "Calibration bypasses throttle",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_1c_detect" },
          details: {
            concept: "Calibration's measure_window calls face_detector_.detect() directly (unless <= 30 ms remain before the next beep).",
            codeRef: "src/Calibration.cpp (L56-64)"
          }
        },
        {
          id: "throttle_gate", kind: "decision", x: 310, y: 24, w: 280, h: 96,
          label: "face_missing_ &&\nelapsed < 500 ms?", sublabel: "CPU-saving search gate",
          details: {
            concept: "When face_missing_ is true, PipelineCoordinator skips Dlib entirely until lost_face_probe_interval (500 ms) has elapsed, preserving CPU cycles for GazeWorker.",
            codeRef: "src/PipelineCoordinator.cpp (L535-544) | include/Types.hpp (L187)"
          }
        },
        {
          id: "dlib_hog", kind: "process", x: 310, y: 170, w: 280, h: 72,
          label: "Grayscale + Dlib HOG", sublabel: "frontal_face_detector(gray, 0)",
          details: {
            concept: "Updates last_face_probe_time, converts BGR to 8-bit grayscale, and runs Dlib's frontal HOG detector without upsampling.",
            codeRef: "src/FaceLandmarkDetector.cpp (L69-78)"
          }
        },
        {
          id: "face_found_gate", kind: "decision", x: 310, y: 286, w: 280, h: 96,
          label: "faces.empty() == true?", sublabel: "Verify real face present",
          details: {
            concept: "Branches to Face Lost if no bounding box is detected, or runs the 68-landmark shape predictor on faces[0] if found.",
            codeRef: "src/FaceLandmarkDetector.cpp (L80-104)"
          }
        },
        {
          id: "dlib_shape", kind: "process", x: 640, y: 298, w: 225, h: 72,
          label: "Dlib 68-Pt Predictor", sublabel: "face_missing_ = false\nExtract 68 (x, y) coords",
          details: {
            concept: "Runs shape_predictor on primary face faces[0] and populates FacialLandmarks68 (array of 68 cv::Point2f).",
            codeRef: "src/FaceLandmarkDetector.cpp (L100-114)"
          }
        },
        {
          id: "out_landmarks", kind: "portal_out", x: 905, y: 246, w: 195, h: 64,
          label: "GOTO 3C: BLINK (68 pt)", sublabel: "EarDetector::process()",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "from_face_gate" },
          details: {
            concept: "Hands 68 facial landmarks to EarDetector::process() for blink duration and PERCLOS updates.",
            codeRef: "src/PipelineCoordinator.cpp (L550)"
          }
        },
        {
          id: "out_landmarks_head", kind: "portal_out", x: 905, y: 334, w: 195, h: 64,
          label: "GOTO 2B: HEAD (68 pt)", sublabel: "HeadDetector::process()",
          target: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_face_gate" },
          details: {
            concept: "Hands 68 facial landmarks to HeadOrientationDetector::process() for rho and pitch estimation.",
            codeRef: "src/PipelineCoordinator.cpp (L551)"
          }
        },
        {
          id: "mark_missing", kind: "process", x: 310, y: 454, w: 280, h: 74,
          label: "Face Lost State", sublabel: "face_missing_ = true\nWarn once if lost >= 3.0s",
          details: {
            concept: "Tracks face_lost_start_ timestamp; logs a console warning once after lost_face_timeout (3 seconds).",
            codeRef: "src/FaceLandmarkDetector.cpp (L80-96) | include/Types.hpp (L186)"
          }
        },
        {
          id: "out_face_lost", kind: "portal_out", x: 640, y: 458, w: 215, h: 64,
          label: "GOTO 3C: EAR_LOST", sublabel: "Clears unconfirmed blink",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "from_face_gate" },
          details: {
            concept: "Calls ear_detector_.on_face_lost(t_work_start) to reset active blink/micro-sleep streaks so a gap never triggers a false micro-sleep.",
            codeRef: "src/PipelineCoordinator.cpp (L553) | src/EarDetector.cpp (L266-331)"
          }
        },
        {
          id: "out_face_lost_head", kind: "portal_out", x: 885, y: 458, w: 215, h: 64,
          label: "GOTO 2B: HEAD_LOST", sublabel: "-1.0*dt droop decay",
          target: { sectionId: "sec2", subviewId: "2B", highlightNodeId: "from_face_gate" },
          details: {
            concept: "Calls head_detector_.on_face_lost(t_work_start) to gradually decay any accumulated droop timer without snapping to zero.",
            codeRef: "src/PipelineCoordinator.cpp (L554) | src/HeadOrientationDetector.cpp (L96-114)"
          }
        }
      ],
      edges: [
        { id: "e_in_thr", points: [[259, 72], [310, 72]], tone: "default" },
        { id: "e_cal_hog", points: [[259, 274], [285, 274], [285, 206], [310, 206]], tone: "portal" },
        { id: "e_thr_no", points: [[450, 120], [450, 170]], label: "NO", tone: "no" },
        { id: "e_thr_yes", points: [[590, 72], [615, 72], [615, 424], [450, 424], [450, 454]], label: "YES (Skip Dlib)", tone: "yes", lx: 690, ly: 72 },
        { id: "e_hog_chk", points: [[450, 242], [450, 286]], tone: "default" },
        { id: "e_fnd_no", points: [[590, 334], [640, 334]], label: "NO", tone: "yes" },
        { id: "e_fnd_yes", points: [[450, 382], [450, 454]], label: "YES", tone: "no" },
        { id: "e_shp_ear", points: [[865, 320], [885, 320], [885, 278], [905, 278]], tone: "portal" },
        { id: "e_shp_head", points: [[865, 348], [885, 348], [885, 366], [905, 366]], tone: "portal" },
        { id: "e_miss_ear", points: [[590, 490], [640, 490]], tone: "portal" },
        { id: "e_miss_head", points: [[590, 505], [615, 505], [615, 545], [868, 545], [868, 490], [885, 490]], tone: "portal" }
      ],
      traces: [
        {
          id: "face_found_trace",
          label: "Trace: Face Found -> 68 Landmarks -> Blink & Head",
          steps: [
            {
              title: "Step 1: Probe Gate Allows Dlib HOG",
              highlightNodes: ["entry_frame", "throttle_gate", "dlib_hog", "face_found_gate"],
              highlightEdges: ["e_in_thr", "e_thr_no", "e_hog_chk"],
              concept: "Either face_missing_ is false or >= 500 ms passed since the last probe, so BGR is converted to grayscale and frontal_face_detector searches the frame.",
              codeRef: "src/PipelineCoordinator.cpp (L535-544) | src/FaceLandmarkDetector.cpp (L69-78)"
            },
            {
              title: "Step 2: Predict 68 Landmarks & Fan Out",
              highlightNodes: ["face_found_gate", "dlib_shape", "out_landmarks", "out_landmarks_head"],
              highlightEdges: ["e_fnd_no", "e_shp_ear", "e_shp_head"],
              concept: "With a confirmed bounding box, Dlib predicts 68 landmarks and sends them to EarDetector (3C) and HeadOrientationDetector (2B).",
              codeRef: "src/FaceLandmarkDetector.cpp (L100-114) | src/PipelineCoordinator.cpp (L548-551)"
            }
          ]
        },
        {
          id: "face_lost_throttle",
          label: "Trace: Face Missing -> 500 ms CPU-Saving Throttle",
          steps: [
            {
              title: "Step 1: Initial Empty HOG Marks Face Missing",
              highlightNodes: ["dlib_hog", "face_found_gate", "mark_missing", "out_face_lost", "out_face_lost_head"],
              highlightEdges: ["e_hog_chk", "e_fnd_yes", "e_miss_ear", "e_miss_head"],
              concept: "When faces.empty() is true, face_missing_ turns true and both detectors switch to their on_face_lost() handlers.",
              codeRef: "src/FaceLandmarkDetector.cpp (L80-96) | src/PipelineCoordinator.cpp (L552-555)"
            },
            {
              title: "Step 2: Subsequent Frames < 500 ms Bypass Dlib",
              highlightNodes: ["entry_frame", "throttle_gate", "mark_missing", "out_face_lost", "out_face_lost_head"],
              highlightEdges: ["e_in_thr", "e_thr_yes", "e_miss_ear", "e_miss_head"],
              concept: "Until 500 ms elapse, Dlib is skipped entirely so GazeWorker gets full CPU priority while Yawn CNN keeps running on the full frame.",
              codeRef: "src/PipelineCoordinator.cpp (L527-544)"
            }
          ]
        }
      ]
    }
  ]
});
