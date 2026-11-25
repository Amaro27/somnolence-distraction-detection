window.EXPLAINER_CHAPTERS = window.EXPLAINER_CHAPTERS || [];
window.EXPLAINER_CHAPTERS.push({
  id: "ch3",
  title: "3. Lifecycle & Calibration",
  subtitle: "5-state safety FSM, IMU motion gating, in-place calibration & resets",
  viewBox: "0 0 960 540",
  legend: ["thread", "atomic", "sync"],
  elements: [
    // Top FSM band: 5 system states
    { id: "st_stat_uncal", kind: "pill", x: 40, y: 24, w: 196, h: 40, label: "Parked (Uncalibrated)" },
    { id: "st_calib", kind: "pill", x: 382, y: 24, w: 160, h: 40, label: "Calibrating" },
    { id: "st_stat_cal", kind: "pill", x: 690, y: 24, w: 196, h: 40, label: "Parked (Calibrated)" },
    { id: "st_mov_uncal", kind: "pill", x: 40, y: 126, w: 196, h: 40, label: "Moving (Uncalibrated)" },
    { id: "st_mov_cal", kind: "pill", x: 690, y: 126, w: 196, h: 40, label: "Moving (Calibrated)" },

    // Middle control band: IMU Auto-Motion + Atomic Reset Flag + Main Supervisor
    { id: "imu_auto", kind: "lane", x: 24, y: 194, w: 174, h: 82, label: "IMU Auto-Motion", sublabel: "20 Hz hysteresis (Linux)", tag: "THREAD" },
    { id: "reset_flag", kind: "pill", x: 224, y: 216, w: 170, h: 40, label: "Atomic Reset Flag" },
    { id: "main_sup", kind: "lane", x: 430, y: 194, w: 506, h: 82, label: "Main Supervisor", sublabel: "Handles 'm' / 'c' keys, updates atomic state & resets alerts", tag: "THREAD" },

    // Bottom worker band
    { id: "gaze_lane", kind: "lane", x: 24, y: 306, w: 390, h: 214, label: "Gaze Worker", sublabel: "Active only in Moving (Calibrated)", tag: "THREAD" },
    { id: "gaze_gate", kind: "box", x: 42, y: 358, w: 170, h: 142, label: "State Gate", sublabel: "Discards frames when\nparked or uncalibrated\n(skips neural nets)" },
    { id: "gaze_reset", kind: "box", x: 226, y: 358, w: 170, h: 142, label: "Self-Reset Memory", sublabel: "Checks atomic flag\nand clears its own\nLSTM ring buffer" },

    { id: "fatigue_lane", kind: "lane", x: 444, y: 306, w: 492, h: 214, label: "Fatigue Worker", sublabel: "Switches in-place between monitoring & calibration", tag: "THREAD" },
    { id: "fatigue_mon", kind: "box", x: 462, y: 358, w: 218, h: 142, label: "Normal Monitoring", sublabel: "Runs Yawn CNN, landmarks,\nEAR & pitch; alerts &\nPERCLOS only if Moving+Cal" },
    { id: "calib_run", kind: "box", x: 696, y: 358, w: 222, h: 142, label: "Calibration Takeover", sublabel: "Same thread! Phase 1 open\n+ Phase 2 closed, 8x500 ms\nbeeps until >=15 samples" },

    // Non-crossing edges
    { id: "e_m_uncal_dn", kind: "edge", points: [[115, 64], [115, 126]], label: "'m' / IMU", lx: 80, ly: 98 },
    { id: "e_m_uncal_up", kind: "edge", points: [[160, 126], [160, 64]], label: "'m' / IMU", lx: 196, ly: 98 },
    { id: "e_c_start", kind: "edge", points: [[236, 44], [382, 44]], label: "'c' (parked)" },
    { id: "e_c_done", kind: "edge", points: [[542, 44], [690, 44]], label: "done -> apply" },
    { id: "e_m_cal_dn", kind: "edge", points: [[765, 64], [765, 126]], label: "move", lx: 736, ly: 98 },
    { id: "e_m_cal_up", kind: "edge", points: [[815, 126], [815, 64]], label: "park", lx: 848, ly: 98 },
    { id: "e_imu_fsm", kind: "edge", points: [[110, 194], [110, 166]], label: "auto" },
    { id: "e_imu_reset", kind: "edge", points: [[198, 236], [224, 236]] },
    { id: "e_imu_gaze", kind: "edge", points: [[110, 276], [110, 306]], label: "state" },
    { id: "e_main_fsm", kind: "edge", points: [[500, 194], [500, 102], [462, 102], [462, 64]], label: "state" },
    { id: "e_main_reset", kind: "edge", points: [[430, 236], [394, 236]], label: "set", lx: 412, ly: 226 },
    { id: "e_reset_gaze", kind: "edge", points: [[310, 256], [310, 306]], label: "consume" },
    { id: "e_main_gaze", kind: "edge", points: [[430, 266], [404, 266], [404, 306]] },
    { id: "e_main_fatigue", kind: "edge", points: [[690, 276], [690, 306]], label: "mode" },
    { id: "e_calib_apply", kind: "edge", points: [[696, 430], [680, 430]] }
  ],
  walkthrough: [
    {
      title: "1. Safe Stationary Startup",
      show: ["st_stat_uncal", "st_calib", "st_stat_cal", "st_mov_uncal", "st_mov_cal"],
      highlight: ["st_stat_uncal"],
      edgeFlow: [],
      concept: "By default, the system boots parked and uncalibrated with all alerts disabled. This guarantees no false alarms trigger before personal open-eye, closed-eye, and neutral head-pitch baselines are measured."
    },
    {
      title: "2. Moving via Keyboard or Hardware IMU",
      show: ["st_stat_uncal", "st_calib", "st_stat_cal", "st_mov_uncal", "st_mov_cal", "imu_auto", "e_m_uncal_dn", "e_m_uncal_up", "e_imu_fsm"],
      highlight: ["st_mov_uncal", "imu_auto"],
      edgeFlow: ["e_imu_fsm", "e_m_uncal_dn"],
      concept: "Drivers can toggle movement manually, or let the IMU Motion Worker switch states automatically on Linux vehicles using smoothed acceleration hysteresis. If moving happens before calibration, detections remain paused and warn the user.",
      note: "The hardware sensor thread averages 5 readings at 20 Hz, requiring 4 high-motion samples (>0.80 m/s^2) to enter Moving and 15 still samples (<0.35 m/s^2) to park; on non-Linux hosts it exits cleanly at startup.",
      codeRef: "morphed_system6/distraction/modular/src/PipelineCoordinator.cpp (L589-689 imu_worker_loop)\ninclude/MPU6050.hpp (L51-205) | include/Types.hpp (L66-72)"
    },
    {
      title: "3. Stationary Calibration Gate & Lockout",
      show: ["st_stat_uncal", "st_calib", "st_stat_cal", "st_mov_uncal", "st_mov_cal", "imu_auto", "main_sup", "e_m_uncal_dn", "e_m_uncal_up", "e_c_start", "e_main_fsm"],
      highlight: ["st_calib", "main_sup"],
      edgeFlow: ["e_c_start", "e_main_fsm"],
      concept: "Calibration can only begin while parked. Requests while moving or already calibrating are ignored, and both manual movement toggles and automatic IMU state changes stay locked out while calibration runs.",
      codeRef: "morphed_system6/distraction/modular/src/PipelineCoordinator.cpp (L319-384 handle_console_command; L644-682)"
    },
    {
      title: "4. In-Place Calibration Takeover (No Extra Thread)",
      show: ["st_stat_uncal", "st_calib", "st_stat_cal", "main_sup", "fatigue_lane", "fatigue_mon", "calib_run", "e_c_start", "e_c_done", "e_main_fatigue", "e_calib_apply"],
      highlight: ["fatigue_lane", "calib_run"],
      edgeFlow: ["e_main_fatigue", "e_calib_apply", "e_c_done"],
      concept: "Instead of spawning a separate thread, the Fatigue Worker pauses normal monitoring and runs the 2-phase audio sequence top-to-bottom: Phase 1 (open eyes, neutral posture) and Phase 2 (eyes closed), repeating 4-second beep windows until at least 15 face samples are captured per phase.",
      note: "Spoken instructions play at +6 dB, cadence beeps play at +3 dB, and both Phase 1 and Phase 2 play the 1-second preparation tone before measurement beeps begin.",
      codeRef: "morphed_system6/distraction/modular/src/PipelineCoordinator.cpp (L480-505)\nsrc/Calibration.cpp (L25-76 measure_window; L80-220 run_calibration)"
    },
    {
      title: "5. Gaze vs. Fatigue Idle Behavior",
      show: ["st_stat_cal", "st_mov_cal", "main_sup", "gaze_lane", "gaze_gate", "fatigue_lane", "fatigue_mon", "e_m_cal_dn", "e_m_cal_up", "e_main_gaze", "e_main_fatigue"],
      highlight: ["gaze_gate", "fatigue_mon"],
      edgeFlow: ["e_main_gaze", "e_m_cal_dn"],
      concept: "When parked or uncalibrated, the Gaze Worker discards frames and skips all neural work, whereas the Fatigue Worker keeps running its yawning net, landmarks, eye ratio, and head pitch for live telemetry while suppressing alerts and closure history.",
      codeRef: "morphed_system6/distraction/modular/src/PipelineCoordinator.cpp (L428-431 gaze skip; L515-559 somnolence loop)"
    },
    {
      title: "6. Mutex-Free Reset When Stopping",
      show: ["st_stat_cal", "st_mov_cal", "imu_auto", "main_sup", "reset_flag", "gaze_lane", "gaze_gate", "gaze_reset", "fatigue_lane", "fatigue_mon", "e_m_cal_up", "e_main_reset", "e_imu_reset", "e_reset_gaze"],
      highlight: ["reset_flag", "gaze_reset"],
      edgeFlow: ["e_m_cal_up", "e_main_reset", "e_imu_reset", "e_reset_gaze"],
      concept: "When the vehicle stops via keyboard or IMU, neither thread wipes the Gaze Worker's sequence memory directly. Instead, they flip a lightweight atomic flag so the Gaze Worker clears its own buffer safely on its next pass.",
      note: "Concurrency caveat: while gaze sequence reset uses this safe atomic handoff, stopping from Moving (Calibrated) currently clears the eye-closure PERCLOS history directly from both Main Supervisor and IMU threads without a lock—worth guarding with a matching atomic flag.",
      codeRef: "morphed_system6/distraction/modular/src/PipelineCoordinator.cpp (L337-346, L411-426, L665-673; L340 & L668 clear_perclos race)"
    }
  ],
  traces: [
    {
      id: "driver_session",
      label: "Trace: Startup -> Calibrate -> Drive -> Park",
      steps: [
        {
          title: "Step 1: Boot Parked & Uncalibrated",
          show: ["st_stat_uncal", "st_calib", "st_stat_cal", "st_mov_uncal", "st_mov_cal", "imu_auto", "gaze_lane", "gaze_gate", "fatigue_lane", "fatigue_mon"],
          highlight: ["st_stat_uncal"],
          edgeFlow: [],
          concept: "Application boots in Parked (Uncalibrated). Both workers pull frames from their mailboxes, and on Linux the IMU Motion Worker starts watching for vehicle movement, while all alarms stay disabled."
        },
        {
          title: "Step 2: User Triggers Calibration ('c')",
          show: ["st_stat_uncal", "st_calib", "st_stat_cal", "main_sup", "gaze_lane", "gaze_gate", "fatigue_lane", "calib_run", "e_c_start", "e_main_fsm", "e_main_fatigue"],
          highlight: ["st_calib", "calib_run"],
          edgeFlow: ["e_c_start", "e_main_fatigue"],
          concept: "Pressing calibrate while parked moves state to Calibrating. The Fatigue Worker runs its 2-phase audio sequence in place while the Gaze Worker skips inference."
        },
        {
          title: "Step 3: Baselines Applied -> Parked (Calibrated)",
          show: ["st_stat_uncal", "st_calib", "st_stat_cal", "fatigue_lane", "fatigue_mon", "calib_run", "e_c_done", "e_calib_apply"],
          highlight: ["st_stat_cal", "fatigue_mon"],
          edgeFlow: ["e_calib_apply", "e_c_done"],
          concept: "Once >=15 samples are gathered for open and closed eyes, eye thresholds and neutral head pitch are applied, and state transitions to Parked (Calibrated)."
        },
        {
          title: "Step 4: Car Accelerates (IMU or 'm')",
          show: ["st_stat_cal", "st_mov_cal", "imu_auto", "main_sup", "gaze_lane", "gaze_gate", "fatigue_lane", "fatigue_mon", "e_m_cal_dn", "e_imu_gaze", "e_main_gaze", "e_main_fatigue"],
          highlight: ["st_mov_cal", "gaze_lane", "fatigue_lane"],
          edgeFlow: ["e_m_cal_dn", "e_imu_gaze", "e_main_gaze", "e_main_fatigue"],
          concept: "Whether triggered by key press or IMU motion hysteresis, state becomes Moving (Calibrated). Gaze inference activates, PERCLOS tracking begins, and both workers feed the Alert Coordinator."
        },
        {
          title: "Step 5: Car Parks (IMU or 'm')",
          show: ["st_stat_cal", "st_mov_cal", "imu_auto", "main_sup", "reset_flag", "gaze_lane", "gaze_reset", "e_m_cal_up", "e_main_reset", "e_imu_reset"],
          highlight: ["st_stat_cal", "reset_flag"],
          edgeFlow: ["e_m_cal_up", "e_main_reset", "e_imu_reset"],
          concept: "Parking transitions state back to Parked (Calibrated), resets active alerts, and sets the atomic gaze-reset flag so no cross-thread collision occurs."
        },
        {
          title: "Step 6: Gaze Worker Cleans Up Safely",
          show: ["st_stat_cal", "reset_flag", "gaze_lane", "gaze_gate", "gaze_reset", "e_reset_gaze"],
          highlight: ["gaze_reset", "gaze_gate"],
          edgeFlow: ["e_reset_gaze"],
          concept: "On its next loop turn, the Gaze Worker consumes the atomic reset flag, clears its own sequence memory safely on its thread, and returns to idle gating."
        }
      ]
    }
  ]
});
