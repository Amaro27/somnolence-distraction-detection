window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec6",
  title: "6. Alarms & Dispatcher",
  subviews: [
    {
      id: "6A",
      label: "6A. Gaze Filter (Mutex)",
      subtitle: "Inside Alert Manager mutex: 10f mode, drowsiness preemption, 2.0 s timer & post-alarm reset",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "mutex", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "Gaze 2-Stage Filter (Inside Mutex)",
        concept: "Landing target for the Gaze Worker (5B). Under the Alert Coordinator's mutex, appends each single-frame vote to the 10-frame sliding window, aborts immediately if Drowsy is active, runs Stage 1 majority mode and Stage 2 continuous 2.0-second timing, unlocks before triggering audio (6C), and clears history upon completion.",
        code: "{ std::lock_guard<std::mutex> lock(mutex_);\n  gaze_window_.push_back(detection);\n  if (is_drowsiness_alert_reproducing_locked() || current_state_ == Drowsy) return;\n  /* Stage 1 mode + Stage 2 >= 2000 ms */ }\n// Outside lock: dispatch audio (6C), then re-lock to clear gaze_window_",
        note: "Once the distraction warning clip finishes or is preempted, the worker re-acquires the mutex, clears all 10 votes and the timer, and reverts Distracted to Safe.",
        codeRef: "src/AlertManager.cpp (L83-204, L413-421)\nsrc/AlertLogic.cpp (L5-49)"
      },
      nodes: [
        {
          id: "grp_mutex_gaze", kind: "group", tone: "mutex", x: 235, y: 12, w: 590, h: 556,
          label: "Alert Coordinator Turnstile Lock", sublabel: "Held only for microsecond math; unlocked before audio or network I/O",
          details: {
            title: "Alert Manager Mutex Scope",
            concept: "Protects shared driver state and sliding windows between Gaze and Fatigue workers without ever blocking one thread on another's I/O.",
            code: "std::lock_guard<std::mutex> lock(mutex_);",
            codeRef: "src/AlertManager.cpp (L91-145, L160-202)"
          }
        },
        {
          id: "from_gaze_vote", kind: "portal_in", x: 16, y: 40, w: 195, h: 56,
          label: "FROM 5B: 1 Frame Vote", sublabel: "Acquire mutex lock",
          target: { sectionId: "sec5", subviewId: "5B", highlightNodeId: "goto_6a" },
          details: {
            title: "Incoming Gaze Classification",
            concept: "Receives one frame classification from the Gaze Worker and acquires the coordinator mutex.",
            code: "void AlertManager::update_gaze(const DistractionDetectionResult& detection) {\n  std::lock_guard<std::mutex> lock(mutex_);\n}",
            codeRef: "src/PipelineCoordinator.cpp (L443) | src/AlertManager.cpp (L83-92)"
          }
        },
        {
          id: "push_win", kind: "process", x: 255, y: 40, w: 225, h: 56,
          label: "Append to Vote Window", sublabel: "Keep newest 10 (or 20) votes",
          details: {
            title: "Update Sliding Gaze Window",
            concept: "Pushes the newest vote and pops older entries beyond the configured mode window size.",
            code: "gaze_window_.push_back(detection);\nwhile (gaze_window_.size() > mode_frames) gaze_window_.pop_front();",
            codeRef: "src/AlertManager.cpp (L94-100)"
          }
        },
        {
          id: "chk_warm", kind: "decision", x: 525, y: 27, w: 240, h: 82,
          label: "Fewer Than 10\nVotes?", sublabel: "Window warmup check",
          details: {
            title: "Mode Window Warmup Check",
            concept: "Returns immediately until the mode window fills with 10 votes (or 20 for unified).",
            code: "if (gaze_window_.size() < mode_frames) return;",
            codeRef: "src/AlertManager.cpp (L103-105)"
          }
        },
        {
          id: "ret_warm", kind: "term_return", x: 850, y: 43, w: 210, h: 50,
          label: "Unlock & RETURN", sublabel: "Warming up (< 10 frames)",
          details: {
            title: "Warmup Exit",
            concept: "Releases the mutex and awaits enough frames to fill the window.",
            code: "return; // Lock guard releases mutex_ on exit",
            codeRef: "src/AlertManager.cpp (L103-105)"
          }
        },
        {
          id: "gate_drowsy_preempt", kind: "decision", x: 525, y: 142, w: 240, h: 84,
          label: "Drowsy Active or\nAlarm Playing?", sublabel: "Drowsiness priority lock",
          details: {
            title: "Drowsiness Preemption Check",
            concept: "Drowsiness always takes priority: if Drowsy is active or drowsiness audio is playing, resets the distraction timer and exits.",
            code: "if (is_drowsiness_alert_reproducing_locked() || current_state_ == DriverState::Drowsy) {\n  distracted_since_.reset(); return;\n}",
            codeRef: "src/AlertManager.cpp (L111-115, L413-421)"
          }
        },
        {
          id: "ret_preempt", kind: "term_return", x: 850, y: 159, w: 210, h: 50,
          label: "Clear Timer & RETURN", sublabel: "Preempted by Drowsy",
          details: {
            title: "Preempted Exit",
            concept: "Clears the distraction timer and leaves drowsiness alarms untouched.",
            code: "distracted_since_.reset(); return;",
            codeRef: "src/AlertManager.cpp (L112-115)"
          }
        },
        {
          id: "eval_mode", kind: "decision", x: 255, y: 142, w: 225, h: 84,
          label: "Stage 1: Off-Road\nMode Majority?", sublabel: "Compare window votes",
          details: {
            title: "Stage 1 Majority Vote",
            concept: "Counts attentive vs. off-road votes in the sliding window and checks for strict off-road majority.",
            code: "assessment = AlertLogic::evaluate_distraction_mode(gaze_window_, policy, min_conf);\nbool ok = (assessment.total_votes > 0 && assessment.is_distracted);",
            codeRef: "src/AlertManager.cpp (L117-129) | src/AlertLogic.cpp (L5-49)"
          }
        },
        {
          id: "chk_was_dist", kind: "decision", x: 255, y: 268, w: 225, h: 84,
          label: "Reset Timer; Was\nState Distracted?", sublabel: "Revert to Safe",
          details: {
            title: "Attentive Recovery Check",
            concept: "Clears the distraction timer; if the driver had been marked Distracted, flips back to Safe and requests audio stop.",
            code: "distracted_since_.reset();\nif (current_state_ == DriverState::Distracted) { current_state_ = Safe; transitioned_to_safe = true; }",
            codeRef: "src/AlertManager.cpp (L123-128, L138-144)"
          }
        },
        {
          id: "ret_safe", kind: "term_return", x: 16, y: 285, w: 195, h: 50,
          label: "Unlock & RETURN", sublabel: "Attentive (Safe)",
          details: {
            title: "Attentive Exit",
            concept: "Unlocks mutex with driver in Safe state.",
            code: "// Already Safe; unlock and return",
            codeRef: "src/AlertManager.cpp (L138-144)"
          }
        },
        {
          id: "goto_6c_silence", kind: "portal_out", x: 850, y: 282, w: 210, h: 56,
          label: "GOTO 6C: Silence", sublabel: "Unlock & stop audio",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_reset" },
          details: {
            title: "Silence Distraction Audio",
            concept: "Unlocks and stops active audio (after verifying drowsiness did not just start).",
            code: "if (!is_drowsiness_alert_reproducing_locked()) audio_.stop();",
            codeRef: "src/AlertManager.cpp (L193-203)"
          }
        },
        {
          id: "chk_stage2", kind: "decision", x: 525, y: 268, w: 240, h: 84,
          label: "Stage 2: Held\n>= 2.0 Seconds?", sublabel: "Continuous off-road",
          details: {
            title: "Stage 2 Continuous Timer",
            concept: "Starts the timer if unset and checks whether off-road mode persisted continuously for 2.0 seconds.",
            code: "if (!distracted_since_) distracted_since_ = now;\nif (now - *distracted_since_ >= 2000ms) { current_state_ = Distracted; trigger = true; }",
            codeRef: "src/AlertManager.cpp (L129-137)"
          }
        },
        {
          id: "ret_count", kind: "term_return", x: 525, y: 392, w: 240, h: 50,
          label: "Unlock & RETURN", sublabel: "Timer counting (< 2.0 s)",
          details: {
            title: "Timer Counting Exit",
            concept: "Releases mutex while the 2-second timer continues accumulating.",
            code: "return; // sustained < 2000 ms",
            codeRef: "src/AlertManager.cpp (L133-137)"
          }
        },
        {
          id: "set_distracted", kind: "process", x: 525, y: 478, w: 240, h: 56,
          label: "Set Distracted & Unlock", sublabel: "Release lock before I/O",
          details: {
            title: "Flip State & Unlock",
            concept: "Sets state to Distracted and releases the mutex before running blocking audio or network calls.",
            code: "current_state_ = DriverState::Distracted;\ntrigger_distraction_alert = true;\n// Exit lock_guard scope",
            codeRef: "src/AlertManager.cpp (L134-145)"
          }
        },
        {
          id: "goto_6c_dist", kind: "portal_out", x: 850, y: 478, w: 210, h: 56,
          label: "GOTO 6C: Dispatch", sublabel: "Telegram + blocking sound",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_distraction" },
          details: {
            title: "Dispatch Distraction Warning",
            concept: "Calls outside-lock dispatch (6C) to send a Telegram message and play the blocking distraction sound.",
            code: "std::thread(...).detach(); // Telegram\naudio_.play_blocking(distraction_sound_path, +6.0f);",
            codeRef: "src/AlertManager.cpp (L159-185)"
          }
        },
        {
          id: "post_alarm_wipe", kind: "process", x: 255, y: 478, w: 225, h: 58,
          label: "Post-Alarm Wipe", sublabel: "Re-lock, clear votes & reset",
          details: {
            title: "Post-Alarm Window Wipe",
            concept: "After warning audio ends or is cut off, re-locks the mutex, clears all 10 votes and the timer, and returns Distracted to Safe.",
            code: "std::lock_guard<std::mutex> lock(mutex_);\ngaze_window_.clear(); distracted_since_.reset();\nif (current_state_ == DriverState::Distracted) current_state_ = DriverState::Safe;",
            codeRef: "src/AlertManager.cpp (L187-192)"
          }
        }
      ],
      edges: [
        { id: "e_in_push", points: [[211, 68], [255, 68]], tone: "portal" },
        { id: "e_push_wu", points: [[480, 68], [525, 68]], tone: "default" },
        { id: "e_wu_yes", points: [[765, 68], [850, 68]], label: "YES", tone: "yes" },
        { id: "e_wu_no", points: [[645, 109], [645, 142]], label: "NO", tone: "no" },
        { id: "e_dr_yes", points: [[765, 184], [850, 184]], label: "YES", tone: "yes" },
        { id: "e_dr_no", points: [[525, 184], [480, 184]], label: "NO", tone: "no" },
        { id: "e_s1_no", points: [[367, 226], [367, 268]], label: "NO", tone: "no" },
        { id: "e_wd_no", points: [[255, 310], [211, 310]], label: "NO", tone: "no" },
        { id: "e_wd_yes", points: [[367, 352], [367, 372], [835, 372], [835, 310], [850, 310]], label: "YES", tone: "yes", lx: 600, ly: 366 },
        { id: "e_s1_yes", points: [[480, 206], [502, 206], [502, 310], [525, 310]], label: "YES", tone: "yes" },
        { id: "e_s2_no", points: [[645, 352], [645, 392]], label: "NO", tone: "no" },
        { id: "e_s2_yes", points: [[645, 352], [502, 352], [502, 506], [525, 506]], label: "YES", tone: "yes", lx: 502, ly: 446 },
        { id: "e_set_6c", points: [[765, 506], [850, 506]], tone: "portal" },
        { id: "e_6c_wipe", points: [[955, 534], [955, 554], [367, 554], [367, 536]], label: "after audio", tone: "portal", lx: 655, ly: 548 }
      ],
      traces: [
        {
          id: "filter_alarm_trace",
          label: "Trace: 10f + 2.0 s Met -> GOTO 6C -> Post-Alarm Wipe",
          steps: [
            {
              title: "Step 1: Window Ready & No Drowsiness -> Stage 1 & 2",
              highlightNodes: ["grp_mutex_gaze", "from_gaze_vote", "push_win", "chk_warm", "gate_drowsy_preempt", "eval_mode", "chk_stage2"],
              highlightEdges: ["e_in_push", "e_push_wu", "e_wu_no", "e_dr_no", "e_s1_yes"],
              concept: "All 10 votes are present, no drowsiness alert is active, off-road votes hold the majority, and the sustained timer reaches 2.0 seconds.",
              code: "if (assessment.is_distracted && sustained >= 2000ms) current_state_ = Distracted;"
            },
            {
              title: "Step 2: Unlock -> Blocking Audio -> Re-Lock Wipe",
              highlightNodes: ["set_distracted", "goto_6c_dist", "post_alarm_wipe"],
              highlightEdges: ["e_s2_yes", "e_set_6c", "e_6c_wipe"],
              concept: "Driver state flips to Distracted, the mutex unlocks for outside audio/Telegram dispatch (6C), and post-playback re-locking clears the vote window.",
              code: "gaze_window_.clear(); distracted_since_.reset();",
              portalTarget: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_distraction" }
            }
          ]
        },
        {
          id: "filter_preempt_trace",
          label: "Trace: Preempted by Active Drowsiness",
          steps: [
            {
              title: "Drowsy Priority Gate Aborts Distraction",
              highlightNodes: ["from_gaze_vote", "push_win", "chk_warm", "gate_drowsy_preempt", "ret_preempt"],
              highlightEdges: ["e_in_push", "e_push_wu", "e_wu_no", "e_dr_yes"],
              concept: "Whenever Drowsy is active or drowsiness audio is playing, gaze updates clear the distraction timer and return immediately.",
              code: "if (is_drowsiness_alert_reproducing_locked() || current_state_ == Drowsy) { distracted_since_.reset(); return; }"
            }
          ]
        },
        {
          id: "filter_clear_trace",
          label: "Trace: Driver Returns Gaze -> Clear & Silence",
          steps: [
            {
              title: "Attentive Majority Clears Timer & Stops Audio",
              highlightNodes: ["eval_mode", "chk_was_dist", "goto_6c_silence"],
              highlightEdges: ["e_s1_no", "e_wd_yes"],
              concept: "Returning gaze to the road clears the sustained timer and stops distraction sound if the alert had been active.",
              code: "if (current_state_ == Distracted) { current_state_ = Safe; audio_.stop(); }"
            }
          ]
        }
      ]
    },
    {
      id: "6B",
      label: "6B. Cue Fusion (Mutex)",
      subtitle: "Inside Alert Manager mutex: Strong vs 2-in-120 s Light cues, recovery clear & 63.5 BPM gain ladder",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "mutex", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "Somnolence Cue Fusion & Recovery (Inside Mutex)",
        concept: "Landing target for Head Droop (2B), Blink/PERCLOS (3C/3D), and Yawning (4B) while driving calibrated. Under mutex protection, ignores eye cues during yawning >= 0.2 s, triggers Drowsy on 1 Strong cue or 2+ Light cues within 120 s, silences immediately when all features recover, and steps a 5-level 63.5 BPM loudness ladder.",
        code: "if (cue.strength == Strong || cue_window_.size() >= 2) {\n  current_state_ = Drowsy; drowsiness_step_idx_ = 0;\n}\nif (features_normal && no_new_cues) { current_state_ = Safe; cue_window_.clear(); }\nelse { gain_db = steps[min(step_idx++, 4)]; } // -21 to -3 dB at 63.5 BPM",
        note: "Recovery requires eyes open (or suppressed by yawn), head upright, AND mouth closed with no newly triggered cues, clearing the 2-minute Light cue window on exit.",
        codeRef: "src/AlertManager.cpp (L206-339, L360-373)\nsrc/AlertLogic.cpp (L51-59)"
      },
      nodes: [
        {
          id: "grp_mutex_somno", kind: "group", tone: "mutex", x: 235, y: 12, w: 590, h: 556,
          label: "Alert Coordinator Turnstile Lock", sublabel: "Evaluates cues & recovery atomically; unlocks before audio/Telegram",
          details: {
            title: "Somnolence Fusion Lock",
            concept: "Shares the same mutex as gaze distraction so Drowsy escalation overrides distraction cleanly.",
            code: "std::lock_guard<std::mutex> lock(mutex_);",
            codeRef: "src/AlertManager.cpp (L285-315)"
          }
        },
        {
          id: "from_detectors", kind: "portal_in", x: 16, y: 38, w: 195, h: 58,
          label: "FROM 2B, 3C/3D, 4B", sublabel: "Eye, head & yawn results",
          target: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "goto_6b" },
          details: {
            title: "Incoming Fatigue Assessments",
            concept: "Called once per Fatigue Worker frame while Moving (Calibrated), passing eye, head, and yawn structs.",
            code: "alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
            codeRef: "src/PipelineCoordinator.cpp (L557-559)"
          }
        },
        {
          id: "from_yawn_sup", kind: "portal_in", x: 16, y: 128, w: 195, h: 58,
          label: "FROM 4B: Yawn Suppress", sublabel: "Filter eye cues if >= 0.2 s",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "sig_yawn_eyes" },
          details: {
            title: "Redundant Yawn Suppression Guard",
            concept: "Filters out Blink Duration and PERCLOS cues inside Alert Manager whenever yawn duration is 0.2 seconds or longer.",
            code: "const bool is_yawning_suppression = (yawn.yawn_duration_sec >= 0.2);\nif (is_yawning_suppression && (cue.type == BlinkDuration || cue.type == Perclos)) continue;",
            codeRef: "src/AlertManager.cpp (L287-292)"
          }
        },
        {
          id: "chk_strong", kind: "decision", x: 255, y: 26, w: 235, h: 84,
          label: "Any Strong Cue?\n(Micro-Sleep | PERCLOS)", sublabel: "Immediate trigger",
          details: {
            title: "Strong Cue Check",
            concept: "Either micro-sleep (>= 500 ms) or High PERCLOS (>= 15%) triggers Drowsy immediately on its own.",
            code: "if (cue.strength == CueStrength::Strong) { /* Immediate Drowsy */ }",
            codeRef: "src/AlertManager.cpp (L213-219)"
          }
        },
        {
          id: "light_window", kind: "process", x: 255, y: 146, w: 235, h: 60,
          label: "Light Cue 120 s Window", sublabel: "Prune > 120 s; check 2+ cues",
          details: {
            title: "Light Cue 2-Minute Sliding Window",
            concept: "Appends Light Head Droop or Yawn cues, prunes entries older than 120 seconds, and triggers Drowsy if at least 2 cues exist.",
            code: "cue_window_.push_back(cue);\nwhile (!cue_window_.empty() && cue_window_.front().timestamp < cutoff) cue_window_.pop_front();\n// evaluate_somnolence_cues(): cue_window_.size() >= 2",
            codeRef: "src/AlertManager.cpp (L220-239) | src/AlertLogic.cpp (L51-59)"
          }
        },
        {
          id: "enter_drowsy", kind: "process", x: 540, y: 38, w: 245, h: 58,
          label: "Set State = Drowsy", sublabel: "Reset ladder & flag Telegram",
          details: {
            title: "Activate Drowsy State",
            concept: "Sets Driver State to Drowsy, resets the 5-step volume ladder to -21 dB, and rewinds the sound timer so the first beep fires immediately.",
            code: "if (current_state_ != DriverState::Drowsy) {\n  current_state_ = DriverState::Drowsy; drowsiness_step_idx_ = 0;\n  last_alert_sound_time_ = now - 10s;\n}",
            codeRef: "src/AlertManager.cpp (L214-237)"
          }
        },
        {
          id: "chk_drowsy_state", kind: "decision", x: 540, y: 146, w: 245, h: 82,
          label: "Driver State\nIs Drowsy?", sublabel: "Active alarm state",
          details: {
            title: "Drowsy State Check",
            concept: "Checks whether the driver is currently Drowsy (newly entered or continuing from prior frames).",
            code: "if (current_state_ == DriverState::Drowsy) { ... }",
            codeRef: "src/AlertManager.cpp (L304)"
          }
        },
        {
          id: "ret_safe", kind: "term_return", x: 850, y: 162, w: 205, h: 50,
          label: "Unlock & RETURN", sublabel: "Safe (no active alarm)",
          details: {
            title: "Safe Return",
            concept: "Unlocks mutex with driver Safe (keeping any single banked Light cue inside the 120 s window).",
            code: "// Safe; unlock and return",
            codeRef: "src/AlertManager.cpp (L304-315)"
          }
        },
        {
          id: "chk_recovery", kind: "decision", x: 255, y: 274, w: 245, h: 88,
          label: "All Features Normal\n& No New Cue?", sublabel: "Eyes, head & mouth normal",
          details: {
            title: "Physiological Recovery Check",
            concept: "Triggers recovery when blink (unless yawn-suppressed), head droop, and active yawning are all clear and no cue fired this frame.",
            code: "const bool blink_active = !is_yawning_suppression && ear.in_blink_cycle;\nconst bool features_normal = !blink_active && !head.is_drooping && !yawn.is_yawning;\nif (features_normal && strong_cues_triggered.empty() && !trigger_high_alert)",
            codeRef: "src/AlertManager.cpp (L302-310)"
          }
        },
        {
          id: "recover_safe", kind: "process", x: 540, y: 288, w: 245, h: 58,
          label: "Clear Drowsiness State", sublabel: "Set Safe, step 0 & clear cues",
          details: {
            title: "Clear Drowsiness State",
            concept: "Resets state to Safe, clears the 2-minute Light cue history, and flags immediate audio stop outside the lock.",
            code: "current_state_ = DriverState::Safe; drowsiness_step_idx_ = 0;\ncue_window_.clear(); cleared_drowsiness = true;",
            codeRef: "src/AlertManager.cpp (L305-310)"
          }
        },
        {
          id: "goto_6c_silence", kind: "portal_out", x: 850, y: 290, w: 205, h: 56,
          label: "GOTO 6C: Silence", sublabel: "Stop drowsiness pulses",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_reset" },
          details: {
            title: "Silence Drowsiness Audio",
            concept: "Clears active playback flags and stops the warning tone immediately upon recovery.",
            code: "drowsiness_audio_active_ = false; audio_.stop();",
            codeRef: "src/AlertManager.cpp (L320-329)"
          }
        },
        {
          id: "chk_bpm", kind: "decision", x: 255, y: 416, w: 245, h: 88,
          label: "63.5 BPM Cooldown\n(~945 ms) Elapsed?", sublabel: "Cadence timer",
          details: {
            title: "63.5 BPM Repetition Gate",
            concept: "Spaces escalating warning pulses at 63.5 beats per minute (roughly 944.9 ms apart).",
            code: "consume_rate_cooldown(63.5 / 60.0, 60.0 / 63.5, last_alert_sound_time_);",
            codeRef: "src/AlertManager.cpp (L50-63, L312, L360-366)"
          }
        },
        {
          id: "ret_wait_beat", kind: "term_return", x: 16, y: 435, w: 195, h: 50,
          label: "Unlock & RETURN", sublabel: "Wait next pulse beat",
          details: {
            title: "Wait Next Pulse Beat",
            concept: "Releases mutex while waiting for the next 63.5 BPM beat (still sending Telegram if newly triggered).",
            code: "// emit_audio == false; wait next cadence tick",
            codeRef: "src/AlertManager.cpp (L360-366)"
          }
        },
        {
          id: "step_ladder", kind: "process", x: 540, y: 430, w: 245, h: 58,
          label: "Step Gain Ladder & Unlock", sublabel: "Escalate -21 to -3 dB",
          details: {
            title: "5-Step Loudness Escalation",
            concept: "Picks the next loudness step (-21.0, -16.5, -12.0, -7.5, or -3.0 dB), increments the index, and unlocks the mutex.",
            code: "static constexpr array<float, 5> steps{-21.0f, -16.5f, -12.0f, -7.5f, -3.0f};\nout_gain_db = steps[std::min(drowsiness_step_idx_, size_t(4))];\n++drowsiness_step_idx_;",
            codeRef: "src/AlertManager.cpp (L368-373) | include/Types.hpp (L166)"
          }
        },
        {
          id: "goto_6c_drowsy", kind: "portal_out", x: 850, y: 432, w: 205, h: 56,
          label: "GOTO 6C: Pulse + TG", sublabel: "Preempt & play beep",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_drowsiness" },
          details: {
            title: "Dispatch Drowsiness Pulse",
            concept: "Unlocks and invokes 6C to stop any active distraction clip, emit the non-blocking pulse, and send Telegram if newly triggered.",
            code: "dispatch_auditory_alert(drowsiness_sound_path, gain_db, reason);",
            codeRef: "src/AlertManager.cpp (L330-338, L375-388)"
          }
        }
      ],
      edges: [
        { id: "e_in_str", points: [[211, 67], [255, 67]], tone: "portal" },
        { id: "e_yawn_flt", points: [[211, 157], [233, 157], [233, 86], [255, 86]], tone: "portal" },
        { id: "e_str_yes", points: [[490, 68], [540, 68]], label: "YES", tone: "yes" },
        { id: "e_str_no", points: [[372, 110], [372, 146]], label: "NO", tone: "no" },
        { id: "e_lgt_ent", points: [[490, 172], [515, 172], [515, 92], [540, 92]], label: ">= 2", tone: "yes", lx: 515, ly: 122 },
        { id: "e_lgt_chk", points: [[490, 187], [540, 187]], label: "< 2", tone: "no" },
        { id: "e_ent_chk", points: [[662, 96], [662, 146]], tone: "default" },
        { id: "e_dr_no", points: [[785, 187], [850, 187]], label: "NO", tone: "no" },
        { id: "e_dr_yes", points: [[662, 228], [662, 250], [377, 250], [377, 274]], label: "YES", tone: "yes", lx: 518, ly: 243 },
        { id: "e_rec_yes", points: [[500, 318], [540, 318]], label: "YES", tone: "yes" },
        { id: "e_rec_sil", points: [[785, 318], [850, 318]], tone: "portal" },
        { id: "e_rec_no", points: [[377, 362], [377, 416]], label: "NO", tone: "no" },
        { id: "e_bpm_no", points: [[255, 460], [211, 460]], label: "NO", tone: "no" },
        { id: "e_bpm_yes", points: [[500, 460], [540, 460]], label: "YES", tone: "yes" },
        { id: "e_lad_6c", points: [[785, 460], [850, 460]], tone: "portal" }
      ],
      traces: [
        {
          id: "fusion_strong",
          label: "Trace: Single Strong Cue -> Drowsy -> -21 dB (6C)",
          steps: [
            {
              title: "Step 1: Strong Cue Immediately Enters Drowsy",
              highlightNodes: ["grp_mutex_somno", "from_detectors", "chk_strong", "enter_drowsy", "chk_drowsy_state"],
              highlightEdges: ["e_in_str", "e_str_yes", "e_ent_chk"],
              concept: "A Blink Duration or High PERCLOS Strong cue elevates state to Drowsy immediately and resets the loudness ladder to step zero.",
              code: "if (cue.strength == CueStrength::Strong) { current_state_ = Drowsy; drowsiness_step_idx_ = 0; }"
            },
            {
              title: "Step 2: 63.5 BPM Cooldown -> -21 dB -> GOTO 6C",
              highlightNodes: ["chk_recovery", "chk_bpm", "step_ladder", "goto_6c_drowsy"],
              highlightEdges: ["e_dr_yes", "e_rec_no", "e_bpm_yes", "e_lad_6c"],
              concept: "With cooldown rewound, the first -21 dB tone step fires immediately outside the mutex in the Alarm Dispatcher (6C).",
              code: "gain_db = steps[0]; // -21.0 dB -> dispatch outside lock",
              portalTarget: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_drowsiness" }
            }
          ]
        },
        {
          id: "fusion_light",
          label: "Trace: Two Light Cues in 120 s (Yawn + Droop)",
          steps: [
            {
              title: "Two Light Cues Within 2 Minutes Trigger Drowsy",
              highlightNodes: ["from_detectors", "chk_strong", "light_window", "enter_drowsy", "chk_drowsy_state", "chk_bpm", "step_ladder", "goto_6c_drowsy"],
              highlightEdges: ["e_in_str", "e_str_no", "e_lgt_ent", "e_ent_chk", "e_dr_yes", "e_rec_no", "e_bpm_yes", "e_lad_6c"],
              concept: "A yawn plus a head droop within 120 seconds satisfy the two-cue threshold, flipping state to Drowsy and dispatching 6C.",
              code: "if (cue_window_.size() >= 2) { current_state_ = Drowsy; }"
            }
          ]
        },
        {
          id: "fusion_recover",
          label: "Trace: Normal Recovery Silences Alarm Immediately",
          steps: [
            {
              title: "All Features Normal -> Safe & Immediate Stop",
              highlightNodes: ["chk_drowsy_state", "chk_recovery", "recover_safe", "goto_6c_silence"],
              highlightEdges: ["e_dr_yes", "e_rec_yes", "e_rec_sil"],
              concept: "As soon as eyes open, posture recovers, and yawning ends, state returns to Safe, the 2-minute cue history clears, and 6C silences audio.",
              code: "if (features_normal && no_new_cues) { current_state_ = Safe; cue_window_.clear(); audio_.stop(); }",
              portalTarget: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_reset" }
            }
          ]
        }
      ]
    },
    {
      id: "6C",
      label: "6C. Outside-Lock Dispatcher",
      subtitle: "Outside-lock audio preemption, 48 kHz ring-buffer mixer, async Telegram HTTP & park reset",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "Outside-Lock Alarm Dispatcher (Audio Preemption & Telegram)",
        concept: "Runs all slow audio and network I/O outside the Alert Coordinator's mutex: blocking distraction playback on the Gaze Worker, non-blocking 63.5 BPM drowsiness tones that preempt distraction immediately, detached HTTP Telegram alerts (3 s connect / 5 s total timeout), and clean silence/reset handlers.",
        code: "// Distraction: detached Telegram + audio_.play_blocking(distraction_wav, +6 dB)\n// Drowsiness: audio_.stop() (playback_id_++ preempts) + audio_.play(beep_wav, gain_db)\n// Reset / Recovery: audio_.stop()",
        note: "A persistent 48 kHz stereo 6x10 ms audio mixer increments a generation counter on stop, waking any blocked distraction thread within one 10 ms block.",
        codeRef: "src/AlertManager.cpp (L159-203, L320-338, L375-406)\nsrc/AudioManager.cpp (L12-30)\nsrc/SoundPlayer.cpp (L253-295, L503-563)\nsrc/TelegramNotifier.cpp (L36-71)"
      },
      nodes: [
        {
          id: "path_distraction", kind: "portal_in", x: 20, y: 28, w: 210, h: 56,
          label: "FROM 6A: Distraction", sublabel: "Gaze Worker (unlocked)",
          target: { sectionId: "sec6", subviewId: "6A", highlightNodeId: "goto_6c_dist" },
          details: {
            title: "Distraction Dispatch Entry",
            concept: "Entered after Stage 1 and Stage 2 confirm distraction and unlock the coordinator mutex.",
            code: "if (trigger_distraction_alert) { /* re-check & play blocking */ }",
            codeRef: "src/AlertManager.cpp (L159)"
          }
        },
        {
          id: "chk_preempt_dist", kind: "decision", x: 265, y: 14, w: 225, h: 84,
          label: "Drowsiness\nReproducing?", sublabel: "Brief safety re-check",
          details: {
            title: "Pre-Playback Safety Check",
            concept: "Briefly re-locks to ensure drowsiness did not activate in the microsecond after releasing the mutex.",
            code: "std::lock_guard<std::mutex> lock(mutex_);\nif (is_drowsiness_alert_reproducing_locked()) return;\ndrowsiness_audio_active_ = false;",
            codeRef: "src/AlertManager.cpp (L160-166)"
          }
        },
        {
          id: "dist_abort", kind: "term_return", x: 265, y: 132, w: 225, h: 48,
          label: "ABORT Distraction", sublabel: "Drowsy took priority",
          details: {
            title: "Abort Distraction Playback",
            concept: "Skips distraction sound and Telegram immediately because drowsiness already holds priority.",
            code: "return; // Drowsiness took priority",
            codeRef: "src/AlertManager.cpp (L162-164)"
          }
        },
        {
          id: "tg_dist", kind: "process", tone: "async", x: 530, y: 28, w: 235, h: 56,
          label: "ASYNC: Telegram Alert", sublabel: "Detached HTTP (3 s / 5 s max)",
          details: {
            title: "Asynchronous Telegram Notification",
            concept: "Spawns a detached background thread to send the distraction warning message with strict 3-second connect and 5-second total timeouts.",
            code: "std::thread([]() { send_telegram_alert(\"Distraction Alert...\"); }).detach();",
            codeRef: "src/AlertManager.cpp (L177-179) | src/TelegramNotifier.cpp (L36-71)"
          }
        },
        {
          id: "play_dist_block", kind: "process", x: 810, y: 28, w: 245, h: 56,
          label: "Play Distraction Warning", sublabel: "Blocks Gaze Worker at +6 dB",
          details: {
            title: "Blocking Distraction Playback",
            concept: "Plays the distraction audio clip at +6.0 dB and blocks the Gaze Worker on a condition variable until playback ends or drowsiness cuts it off.",
            code: "audio_.play_blocking(config_.audio_config.distraction_sound_path, +6.0f);",
            codeRef: "src/AlertManager.cpp (L182-185) | src/AudioManager.cpp (L20-26) | src/SoundPlayer.cpp (L277-284)"
          }
        },
        {
          id: "ret_to_6a", kind: "portal_out", x: 810, y: 128, w: 245, h: 56,
          label: "RETURN TO 6A: Wipe", sublabel: "Clear 10f mode & 2.0 s timer",
          target: { sectionId: "sec6", subviewId: "6A", highlightNodeId: "post_alarm_wipe" },
          details: {
            title: "Post-Audio Wipe in 6A",
            concept: "Upon waking from blocking playback, returns to 6A to clear the vote window and reset Distracted to Safe.",
            code: "std::lock_guard<std::mutex> lock(mutex_);\ngaze_window_.clear(); distracted_since_.reset();",
            codeRef: "src/AlertManager.cpp (L187-192)"
          }
        },
        {
          id: "path_drowsiness", kind: "portal_in", x: 20, y: 228, w: 210, h: 56,
          label: "FROM 6B: Drowsiness", sublabel: "Fatigue Worker (unlocked)",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "goto_6c_drowsy" },
          details: {
            title: "Drowsiness Dispatch Entry",
            concept: "Entered when Drowsy is newly triggered or when the next 63.5 BPM warning pulse is due.",
            code: "// After AlertManager::update_somnolence unlocks mutex_",
            codeRef: "src/AlertManager.cpp (L330-338)"
          }
        },
        {
          id: "chk_new_drowsy", kind: "decision", x: 265, y: 214, w: 225, h: 84,
          label: "Newly Triggered\nDrowsy?", sublabel: "Send Telegram once",
          details: {
            title: "Initial Trigger Check",
            concept: "Dispatches a remote Telegram alert on the transition into Drowsy rather than repeating messages on every beep pulse.",
            code: "if (!strong_cues_triggered.empty() || trigger_high_alert) {\n  std::thread([msg]() { send_telegram_alert(msg); }).detach();\n}",
            codeRef: "src/AlertManager.cpp (L330-334)"
          }
        },
        {
          id: "tg_drowsy", kind: "process", tone: "async", x: 530, y: 228, w: 235, h: 56,
          label: "ASYNC: Telegram Alert", sublabel: "Detached HTTP somnolence msg",
          details: {
            title: "Detached Drowsiness Telegram",
            concept: "Posts the drowsiness reason asynchronously with tight timeouts so cellular latency never blocks safety alerts.",
            code: "std::thread([txt = \"Somnolence Alert: \" + reason]() {\n  send_telegram_alert(txt);\n}).detach();",
            codeRef: "src/AlertManager.cpp (L331-333) | src/TelegramNotifier.cpp (L36-71)"
          }
        },
        {
          id: "preempt_stop", kind: "process", x: 530, y: 322, w: 235, h: 58,
          label: "Cut Off Active Audio", sublabel: "Preempts distraction immediately",
          details: {
            title: "Priority Audio Preemption",
            concept: "Marks drowsiness audio active and stops the mixer immediately, cutting off any playing distraction clip and waking the Gaze Worker.",
            code: "{ std::lock_guard<std::mutex> lock(mutex_); drowsiness_audio_active_ = true; }\naudio_.stop(); // Increments playback_id_ & notifies state_cv_",
            codeRef: "src/AlertManager.cpp (L375-386) | src/SoundPlayer.cpp (L286-295)"
          }
        },
        {
          id: "play_drowsy_pulse", kind: "process", x: 810, y: 322, w: 245, h: 58,
          label: "Play Drowsiness Pulse", sublabel: "Non-blocking mix (-21 to -3 dB)",
          details: {
            title: "Non-Blocking Escalating Beep",
            concept: "Queues the 1850/975 Hz warning pulse on the 48 kHz ring buffer without blocking the Fatigue Worker thread.",
            code: "audio_.play(sound_path, gain_db); // -21.0, -16.5, -12.0, -7.5, -3.0 dB",
            codeRef: "src/AlertManager.cpp (L387) | src/AudioManager.cpp (L12-18) | src/SoundPlayer.cpp (L253-275)"
          }
        },
        {
          id: "path_reset", kind: "portal_in", x: 20, y: 444, w: 210, h: 60,
          label: "FROM 1A / 6A / 6B: Stop", sublabel: "Park reset, attentive or normal",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "goto_6c_reset" },
          details: {
            title: "Silence & Reset Entry",
            concept: "Entered when the vehicle parks (1A), gaze returns to road (6A), or physiological fatigue recovers (6B).",
            code: "// Called via reset(), update_gaze() Safe transition, or update_somnolence() recovery",
            codeRef: "src/AlertManager.cpp (L193-203, L320-329, L390-406)"
          }
        },
        {
          id: "clear_state_stop", kind: "process", x: 280, y: 446, w: 255, h: 56,
          label: "Clear State & Windows", sublabel: "Wipe history on park; clear flag",
          details: {
            title: "State & Window Cleanup",
            concept: "Full park reset wipes both gaze and cue windows and sets state to Safe; recovery/attentive paths clear active playback flags.",
            code: "gaze_window_.clear(); cue_window_.clear(); current_state_ = DriverState::Safe;\ndrowsiness_step_idx_ = 0; drowsiness_audio_active_ = false;",
            codeRef: "src/AlertManager.cpp (L199-202, L320-328, L390-402)"
          }
        },
        {
          id: "audio_stop_box", kind: "process", x: 580, y: 446, w: 205, h: 56,
          label: "Stop Audio Stream", sublabel: "Zero mixer & wake threads",
          details: {
            title: "Stop Mixer Playback",
            concept: "Clears active PCM samples so the next 10 ms audio block outputs silence and notifies any waiting threads.",
            code: "audio_.stop(); // SoundPlayer::stop() -> stop_playback()",
            codeRef: "src/AudioManager.cpp (L28-30) | src/SoundPlayer.cpp (L286-295)"
          }
        },
        {
          id: "ret_done", kind: "term_return", x: 850, y: 449, w: 205, h: 50,
          label: "RETURN (Silent)", sublabel: "Audio mixer idle",
          details: {
            title: "Mixer Idle",
            concept: "The 48 kHz audio mixer streams silence ready for the next alert.",
            code: "std::memset(dst, 0, BLOCK_SAMPLES * sizeof(int16_t));",
            codeRef: "src/SoundPlayer.cpp (L334-335)"
          }
        }
      ],
      edges: [
        { id: "e_d_chk", points: [[230, 56], [265, 56]], tone: "portal" },
        { id: "e_d_ab", points: [[377, 98], [377, 132]], label: "YES", tone: "yes" },
        { id: "e_d_tg", points: [[490, 56], [530, 56]], label: "NO", tone: "no" },
        { id: "e_d_blk", points: [[765, 56], [810, 56]], tone: "default" },
        { id: "e_blk_6a", points: [[932, 84], [932, 128]], tone: "portal" },
        { id: "e_s_chk", points: [[230, 256], [265, 256]], tone: "portal" },
        { id: "e_s_tg", points: [[490, 256], [530, 256]], label: "YES", tone: "yes" },
        { id: "e_s_no_aud", points: [[377, 298], [377, 351], [530, 351]], label: "emit audio", tone: "default", lx: 452, ly: 342 },
        { id: "e_tg_pre", points: [[647, 284], [647, 322]], label: "emit audio", tone: "default" },
        { id: "e_pre_ply", points: [[765, 351], [810, 351]], tone: "default" },
        { id: "e_pre_cut_dist", points: [[765, 336], [786, 336], [786, 156], [810, 156]], label: "wakes 6A!", tone: "portal", dashed: true, lx: 786, ly: 204 },
        { id: "e_r_clr", points: [[230, 474], [280, 474]], tone: "portal" },
        { id: "e_clr_stp", points: [[535, 474], [580, 474]], tone: "default" },
        { id: "e_stp_ret", points: [[785, 474], [850, 474]], tone: "yes" }
      ],
      traces: [
        {
          id: "disp_dist",
          label: "Trace: Distraction Dispatch (Telegram + Blocking Audio)",
          steps: [
            {
              title: "Unlock, Send Telegram, Play Blocking Clip",
              highlightNodes: ["path_distraction", "chk_preempt_dist", "tg_dist", "play_dist_block", "ret_to_6a"],
              highlightEdges: ["e_d_chk", "e_d_tg", "e_d_blk", "e_blk_6a"],
              concept: "After re-checking that drowsiness is inactive, detaches a Telegram thread and blocks the Gaze Worker on the +6 dB warning before wiping 6A's window.",
              code: "audio_.play_blocking(distraction_sound_path, +6.0f);\n// Then re-lock in 6A to clear gaze_window_"
            }
          ]
        },
        {
          id: "disp_preempt",
          label: "Trace: Drowsiness Preemption Cuts Off Distraction",
          steps: [
            {
              title: "Step 1: Drowsiness Reaches Dispatcher Unlocked",
              highlightNodes: ["path_drowsiness", "chk_new_drowsy", "tg_drowsy"],
              highlightEdges: ["e_s_chk", "e_s_tg"],
              concept: "Because the Gaze Worker unlocked before playing audio, the Fatigue Worker enters promptly and dispatches a detached Telegram alert.",
              code: "std::thread([msg]() { send_telegram_alert(msg); }).detach();"
            },
            {
              title: "Step 2: Stop Preempts Distraction & Emits Pulse",
              highlightNodes: ["preempt_stop", "play_drowsy_pulse", "play_dist_block", "ret_to_6a"],
              highlightEdges: ["e_tg_pre", "e_pre_ply", "e_pre_cut_dist", "e_blk_6a"],
              concept: "Stopping playback increments the mixer generation ID, cutting off distraction immediately, waking the Gaze Worker to wipe 6A, and playing the drowsiness beep.",
              code: "audio_.stop(); // Interrupts play_blocking immediately\naudio_.play(drowsiness_sound_path, gain_db);"
            }
          ]
        },
        {
          id: "disp_park_stop",
          label: "Trace: Park Reset (1A) Silences Everything",
          steps: [
            {
              title: "Reset Clears State & Stops Mixer",
              highlightNodes: ["path_reset", "clear_state_stop", "audio_stop_box", "ret_done"],
              highlightEdges: ["e_r_clr", "e_clr_stp", "e_stp_ret"],
              concept: "Parking, returning road gaze, or recovering from fatigue stops audio immediately.",
              code: "alert_manager_.reset(); // Clears windows & calls audio_.stop()"
            }
          ]
        }
      ]
    }
  ]
});
