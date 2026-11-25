window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec6",
  title: "6. Alarms & Dispatcher",
  subviews: [
    {
      id: "6A",
      label: "6A. Gaze Filter (Mutex)",
      subtitle: "AlertManager::update_gaze inside mutex_: 10f mode, Drowsy preemption, 2.0 s timer & post-alarm wipe",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "mutex", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "6A: Gaze 2-Stage Filter (AlertManager::update_gaze, Inside Mutex)",
        concept: "Landing target for GazeDetector (5B). Inside AlertManager::mutex_, accumulates the 10-frame (or 20-frame) sliding mode window, aborts immediately if Drowsy is active/reproducing, evaluates majority mode (Stage 1) and continuous 2.0 s duration (Stage 2), unlocks before dispatching audio (6C), and wipes the window after playback.",
        note: "When distraction audio finishes (or is interrupted by drowsiness), update_gaze() re-acquires mutex_, clears gaze_window_ and distracted_since_, and reverts DriverState::Distracted -> Safe (AlertManager.cpp L187-192).",
        codeRef: "src/AlertManager.cpp (L83-204, L413-421)\nsrc/AlertLogic.cpp (L5-49)"
      },
      nodes: [
        {
          id: "grp_mutex_gaze", kind: "group", tone: "mutex", x: 245, y: 14, w: 605, h: 630,
          label: "AlertManager::mutex_ (Turnstile Lock)", sublabel: "Held only for fast deque + state math; released before audio/HTTP",
          details: {
            concept: "Ensures GazeWorker and FatigueWorker never race on DriverState, timers, or windows while avoiding lock-hold during blocking I/O.",
            codeRef: "src/AlertManager.cpp (L91-145, L160-202)"
          }
        },
        {
          id: "from_gaze_vote", kind: "portal_in", x: 16, y: 50, w: 205, h: 66,
          label: "FROM 5B: 1 VOTE/FRAME", sublabel: "DistractionDetectionResult\nLock AlertManager::mutex_",
          target: { sectionId: "sec5", subviewId: "5B", highlightNodeId: "goto_6a" },
          details: {
            concept: "Called on GazeWorker after every single-frame LSTM evaluation while MovingCalibrated.",
            codeRef: "src/PipelineCoordinator.cpp (L443) | src/AlertManager.cpp (L83-92)"
          }
        },
        {
          id: "push_win", kind: "process", x: 268, y: 48, w: 235, h: 70,
          label: "Push to gaze_window_", sublabel: "Trim to mode_window_frames\n(10 default, 20 if unified)",
          details: {
            concept: "Pushes newest vote into sliding deque and pops oldest entries beyond mode_window_frames.",
            codeRef: "src/AlertManager.cpp (L94-100)"
          }
        },
        {
          id: "chk_warm", kind: "decision", x: 545, y: 38, w: 245, h: 90,
          label: "window.size() < 10?", sublabel: "Initial warmup check",
          details: {
            concept: "Waits until mode_window_frames predictions accumulate before checking distraction.",
            codeRef: "src/AlertManager.cpp (L103-105)"
          }
        },
        {
          id: "ret_warm", kind: "term_return", x: 875, y: 54, w: 220, h: 56,
          label: "Unlock & RETURN", sublabel: "Warming up (<10f)",
          details: {
            concept: "Releases mutex and returns while mode window fills.",
            codeRef: "src/AlertManager.cpp (L103-105)"
          }
        },
        {
          id: "gate_drowsy_preempt", kind: "decision", x: 545, y: 168, w: 245, h: 98,
          label: "DriverState == Drowsy\n|| drowsy_audio_playing?", sublabel: "Drowsiness priority lock",
          details: {
            concept: "Drowsiness strictly overrides distraction: if drowsy or drowsiness audio is reproducing, resets distracted_since_ and exits.",
            codeRef: "src/AlertManager.cpp (L111-115, L413-421)"
          }
        },
        {
          id: "ret_preempt", kind: "term_return", x: 875, y: 190, w: 220, h: 56,
          label: "Reset Timer & RETURN", sublabel: "Preempted by Drowsy",
          details: {
            concept: "Clears distracted_since_ and unlocks without triggering or clearing drowsiness audio.",
            codeRef: "src/AlertManager.cpp (L112-115)"
          }
        },
        {
          id: "eval_mode", kind: "decision", x: 268, y: 170, w: 235, h: 96,
          label: "Stage 1: total > 0 &&\nnot_looking > looking?", sublabel: "AlertLogic mode vote",
          details: {
            concept: "Evaluates sliding window mode per policy (BinarizeThreshold, Argmax, or FilterUncertain where total_votes can be 0).",
            codeRef: "src/AlertManager.cpp (L117-129) | src/AlertLogic.cpp (L5-49)"
          }
        },
        {
          id: "chk_was_dist", kind: "decision", x: 268, y: 316, w: 235, h: 96,
          label: "Reset distracted_since\nWas State == Distracted?", sublabel: "Transition to Safe",
          details: {
            concept: "Clears distracted_since_; if DriverState had been Distracted, reverts to Safe and flags transitioned_to_safe.",
            codeRef: "src/AlertManager.cpp (L123-128, L138-144)"
          }
        },
        {
          id: "ret_safe", kind: "term_return", x: 16, y: 336, w: 205, h: 56,
          label: "Unlock & RETURN", sublabel: "Attentive (Safe)",
          details: {
            concept: "No alarm active; exits cleanly.",
            codeRef: "src/AlertManager.cpp (L138-144)"
          }
        },
        {
          id: "goto_6c_silence", kind: "portal_out", x: 875, y: 332, w: 220, h: 64,
          label: "GOTO 6C: SILENCE", sublabel: "Unlock -> stop() if safe",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_reset" },
          details: {
            concept: "If DriverState transitioned Distracted -> Safe, unlocks and stops distraction audio (re-checking !drowsy reproducing).",
            codeRef: "src/AlertManager.cpp (L193-203)"
          }
        },
        {
          id: "chk_stage2", kind: "decision", x: 545, y: 316, w: 245, h: 98,
          label: "Stage 2: now - since\n>= 2000 ms?", sublabel: "Start timer if empty",
          details: {
            concept: "Sets distracted_since_ = now if unset, then checks whether sustained duration meets distraction_window (2000 ms).",
            codeRef: "src/AlertManager.cpp (L129-137)"
          }
        },
        {
          id: "ret_count", kind: "term_return", x: 545, y: 454, w: 245, h: 54,
          label: "Unlock & RETURN", sublabel: "Timer counting (< 2.0 s)",
          details: {
            concept: "Keeps distracted_since_ advancing while waiting for 2000 ms.",
            codeRef: "src/AlertManager.cpp (L133-137)"
          }
        },
        {
          id: "set_distracted", kind: "process", x: 545, y: 548, w: 245, h: 68,
          label: "Set State = Distracted", sublabel: "Unlock mutex_ BEFORE I/O",
          details: {
            concept: "Flips DriverState = Distracted, sets trigger_distraction_alert = true, and exits mutex_ scope.",
            codeRef: "src/AlertManager.cpp (L134-145)"
          }
        },
        {
          id: "goto_6c_dist", kind: "portal_out", x: 875, y: 548, w: 220, h: 68,
          label: "GOTO 6C: DISPATCH", sublabel: "Telegram + play_blocking()",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_distraction" },
          details: {
            concept: "Outside lock: launches detached Telegram HTTP thread and blocks GazeWorker in play_blocking(+6 dB) until clip ends or drowsiness interrupts.",
            codeRef: "src/AlertManager.cpp (L159-185)"
          }
        },
        {
          id: "post_alarm_wipe", kind: "process", x: 268, y: 548, w: 235, h: 72,
          label: "Post-Alarm Wipe (Re-lock)", sublabel: "gaze_window_.clear()\ndistracted_since_.reset()\nDistracted -> Safe",
          details: {
            concept: "After play_blocking() returns (finished or cut off by drowsiness), re-locks mutex_, wipes all 10 votes + timer, and reverts Distracted -> Safe.",
            codeRef: "src/AlertManager.cpp (L187-192)"
          }
        }
      ],
      edges: [
        { id: "e_in_push", points: [[221, 83], [268, 83]], tone: "portal" },
        { id: "e_push_wu", points: [[503, 83], [545, 83]], tone: "default" },
        { id: "e_wu_yes", points: [[790, 83], [875, 83]], label: "YES", tone: "yes" },
        { id: "e_wu_no", points: [[667, 128], [667, 168]], label: "NO", tone: "no" },
        { id: "e_dr_yes", points: [[790, 217], [875, 217]], label: "YES", tone: "yes" },
        { id: "e_dr_no", points: [[545, 217], [503, 217]], label: "NO", tone: "no" },
        { id: "e_s1_no", points: [[385, 266], [385, 316]], label: "NO", tone: "no" },
        { id: "e_wd_no", points: [[268, 364], [221, 364]], label: "NO", tone: "no" },
        { id: "e_wd_yes", points: [[385, 412], [385, 434], [858, 434], [858, 364], [875, 364]], label: "YES", tone: "yes", lx: 630, ly: 428 },
        { id: "e_s1_yes", points: [[503, 240], [524, 240], [524, 365], [545, 365]], label: "YES", tone: "yes" },
        { id: "e_s2_no", points: [[667, 414], [667, 454]], label: "NO", tone: "no" },
        { id: "e_s2_yes", points: [[667, 414], [524, 414], [524, 582], [545, 582]], label: "YES", tone: "yes", lx: 524, ly: 520 },
        { id: "e_set_6c", points: [[790, 582], [875, 582]], tone: "portal" },
        { id: "e_6c_wipe", points: [[985, 616], [985, 634], [385, 634], [385, 620]], label: "after audio", tone: "portal", lx: 685, ly: 628 }
      ],
      traces: [
        {
          id: "filter_alarm_trace",
          label: "Trace: 10f + 2.0 s Met -> GOTO 6C -> Post-Alarm Wipe",
          steps: [
            {
              title: "Step 1: Warmed Up & Not Preempted -> Stage 1 & 2",
              highlightNodes: ["grp_mutex_gaze", "from_gaze_vote", "push_win", "chk_warm", "gate_drowsy_preempt", "eval_mode", "chk_stage2"],
              highlightEdges: ["e_in_push", "e_push_wu", "e_wu_no", "e_dr_no", "e_s1_yes"],
              concept: "10-frame mode window is full, no drowsiness alert is active, majority is NotLookingRoad, and distracted_since_ reaches >= 2000 ms.",
              codeRef: "src/AlertManager.cpp (L92-137)"
            },
            {
              title: "Step 2: Unlock -> Dispatch Blocking Audio -> Re-Lock & Wipe",
              highlightNodes: ["set_distracted", "goto_6c_dist", "post_alarm_wipe"],
              highlightEdges: ["e_s2_yes", "e_set_6c", "e_6c_wipe"],
              concept: "DriverState flips to Distracted, mutex unlocks before calling 6C (Telegram + play_blocking), then re-locks to clear gaze_window_ and timer.",
              portalTarget: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_distraction" },
              codeRef: "src/AlertManager.cpp (L134-192)"
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
              concept: "Whenever DriverState == Drowsy or drowsiness audio is playing, update_gaze clears distracted_since_ and returns immediately.",
              codeRef: "src/AlertManager.cpp (L111-115)"
            }
          ]
        },
        {
          id: "filter_clear_trace",
          label: "Trace: Driver Returns Gaze -> Clear & Silence",
          steps: [
            {
              title: "On-Road Majority Clears Timer & Silences Distraction",
              highlightNodes: ["eval_mode", "chk_was_dist", "goto_6c_silence"],
              highlightEdges: ["e_s1_no", "e_wd_yes"],
              concept: "Returning gaze to the road resets distracted_since_; if DriverState was Distracted, it reverts to Safe and stops audio (6C).",
              codeRef: "src/AlertManager.cpp (L138-144, L193-203)"
            }
          ]
        }
      ]
    },
    {
      id: "6B",
      label: "6B. Cue Fusion (Mutex)",
      subtitle: "AlertManager::update_somnolence inside mutex_: Strong vs 2-in-120s Light cues, recovery & 63.5 BPM ladder",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "mutex", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "6B: Somnolence Cue Fusion & Recovery (Inside Mutex)",
        concept: "Landing target for Head Droop (2B), Blink/PERCLOS (3C/3D), and Yawning (4B) while MovingCalibrated. Acquires mutex_, drops eye cues if yawning >= 0.2 s, triggers Drowsy on 1 Strong cue or >= 2 Light cues within 120 s, checks physiological recovery to silence audio, and steps a 5-level 63.5 BPM gain ladder.",
        note: "Recovery requires ALL physiological flags normal (!blink_active && !head.is_drooping && !yawn.is_yawning) and clears cue_window_ completely on recovery (AlertManager.cpp L302-310).",
        codeRef: "src/AlertManager.cpp (L206-339, L360-373)\nsrc/AlertLogic.cpp (L51-59)"
      },
      nodes: [
        {
          id: "grp_mutex_somno", kind: "group", tone: "mutex", x: 245, y: 14, w: 605, h: 630,
          label: "AlertManager::mutex_ (Somnolence Fusion)", sublabel: "Acquires turnstile lock for cue window math, then unlocks before audio/Telegram",
          details: {
            concept: "Shares mutex_ with update_gaze() so Drowsy preemption is atomic with respect to distraction state.",
            codeRef: "src/AlertManager.cpp (L285-315)"
          }
        },
        {
          id: "from_detectors", kind: "portal_in", x: 16, y: 46, w: 210, h: 72,
          label: "FROM 2B, 3C/3D, 4B", sublabel: "Ear, Head & Yawn structs\n(MovingCalibrated only)",
          target: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "goto_6b" },
          details: {
            concept: "PipelineCoordinator invokes alert_manager_.update_somnolence(ear_res, head_res, yawn_res) once per somnolence frame when MovingCalibrated.",
            codeRef: "src/PipelineCoordinator.cpp (L557-559)"
          }
        },
        {
          id: "from_yawn_sup", kind: "portal_in", x: 16, y: 148, w: 210, h: 64,
          label: "FROM 4B: YAWN_SUP", sublabel: "yawn_duration >= 0.2 s\nFilter Blink/PERCLOS cues",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "sig_yawn_eyes" },
          details: {
            concept: "Defense-in-depth guard inside AlertManager (L287-292): skips BlinkDuration and Perclos cues if yawn_duration_sec >= 0.2.",
            codeRef: "src/AlertManager.cpp (L287-292)"
          }
        },
        {
          id: "chk_strong", kind: "decision", x: 268, y: 36, w: 245, h: 94,
          label: "Any Strong Cue?\n(Blink>=500ms | PERCLOS)", sublabel: "Immediate trigger",
          details: {
            concept: "One Strong cue immediately enters Drowsy, resets drowsiness_step_idx_ = 0, and rewinds sound cooldown so beep 0 plays immediately.",
            codeRef: "src/AlertManager.cpp (L213-219)"
          }
        },
        {
          id: "light_window", kind: "process", x: 268, y: 168, w: 245, h: 80,
          label: "Light Cue 120 s Window", sublabel: "HeadDroop or Yawn -> push\nPrune < ts - 120s; >= 2 cues?",
          details: {
            concept: "Pushes Light cues into cue_window_, prunes entries older than 120 s, and triggers Drowsy if size() >= 2.",
            codeRef: "src/AlertManager.cpp (L220-239) | src/AlertLogic.cpp (L51-59)"
          }
        },
        {
          id: "enter_drowsy", kind: "process", x: 560, y: 50, w: 255, h: 76,
          label: "Set DriverState = Drowsy", sublabel: "step_idx = 0; rewind cooldown\nFlag Telegram notification",
          details: {
            concept: "Resets the 5-step loudness ladder to -21.0 dB and flags strong_cues / trigger_high_alert for Telegram dispatch outside lock.",
            codeRef: "src/AlertManager.cpp (L214-237)"
          }
        },
        {
          id: "chk_drowsy_state", kind: "decision", x: 560, y: 174, w: 255, h: 96,
          label: "DriverState == Drowsy?", sublabel: "Active alarm state",
          details: {
            concept: "Checks whether the driver is currently in Drowsy state (newly triggered or ongoing).",
            codeRef: "src/AlertManager.cpp (L304)"
          }
        },
        {
          id: "ret_safe", kind: "term_return", x: 875, y: 194, w: 220, h: 56,
          label: "Unlock & RETURN", sublabel: "Safe (no active alarm)",
          details: {
            concept: "Exits mutex without sounding audio (any single Light cue stays banked in the 120 s window).",
            codeRef: "src/AlertManager.cpp (L304-315)"
          }
        },
        {
          id: "chk_recovery", kind: "decision", x: 268, y: 310, w: 260, h: 102,
          label: "features_normal &&\nno cue this frame?", sublabel: "!blink & !droop & !yawn",
          details: {
            concept: "Physiological recovery holds when blink_active (suppressed by yawn >=0.2s), head.is_drooping, and yawn.is_yawning are all false.",
            codeRef: "src/AlertManager.cpp (L302-310)"
          }
        },
        {
          id: "recover_safe", kind: "process", x: 560, y: 324, w: 255, h: 74,
          label: "Clear Drowsiness State", sublabel: "State = Safe; step_idx = 0\ncue_window_.clear(); unlock",
          details: {
            concept: "Resets DriverState to Safe, wipes the 2-minute Light cue history, and flags cleared_drowsiness.",
            codeRef: "src/AlertManager.cpp (L305-310)"
          }
        },
        {
          id: "goto_6c_silence", kind: "portal_out", x: 875, y: 330, w: 220, h: 64,
          label: "GOTO 6C: SILENCE", sublabel: "Immediate audio_.stop()",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_reset" },
          details: {
            concept: "Clears drowsiness_audio_active_ and stops warning pulses immediately upon physiological recovery.",
            codeRef: "src/AlertManager.cpp (L320-329)"
          }
        },
        {
          id: "chk_bpm", kind: "decision", x: 268, y: 468, w: 260, h: 98,
          label: "63.5 BPM (~944.9 ms)\ncooldown elapsed?", sublabel: "Rate-controlled pulse",
          details: {
            concept: "Limits warning pulses to 63.5 BPM (60 / 63.5 = ~0.945 s interval).",
            codeRef: "src/AlertManager.cpp (L50-63, L312, L360-366)"
          }
        },
        {
          id: "ret_wait_beat", kind: "term_return", x: 16, y: 490, w: 210, h: 56,
          label: "Unlock & RETURN", sublabel: "Wait next 63.5 BPM beat",
          details: {
            concept: "Driver remains Drowsy, waiting for the next ~945 ms cadence beat (or Telegram if initial trigger).",
            codeRef: "src/AlertManager.cpp (L360-366)"
          }
        },
        {
          id: "step_ladder", kind: "process", x: 560, y: 482, w: 255, h: 72,
          label: "Step Gain Ladder & Unlock", sublabel: "{-21, -16.5, -12, -7.5, -3} dB\nmin(step_idx++, 4)",
          details: {
            concept: "Escalates beep volume across 5 steps from -21.0 dB up to -3.0 dB before unlocking mutex_.",
            codeRef: "src/AlertManager.cpp (L368-373) | include/Types.hpp (L166)"
          }
        },
        {
          id: "goto_6c_drowsy", kind: "portal_out", x: 875, y: 484, w: 220, h: 68,
          label: "GOTO 6C: PULSE + TG", sublabel: "Preempt dist + play(gain)",
          target: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_drowsiness" },
          details: {
            concept: "Outside lock: stops any active distraction audio immediately, plays non-blocking pulse at gain_db, and sends Telegram if newly triggered.",
            codeRef: "src/AlertManager.cpp (L330-338, L375-388)"
          }
        }
      ],
      edges: [
        { id: "e_in_str", points: [[226, 82], [268, 82]], tone: "portal" },
        { id: "e_yawn_flt", points: [[226, 180], [248, 180], [248, 100], [268, 100]], tone: "portal" },
        { id: "e_str_yes", points: [[513, 83], [560, 83]], label: "YES", tone: "yes" },
        { id: "e_str_no", points: [[390, 130], [390, 168]], label: "NO", tone: "no" },
        { id: "e_lgt_ent", points: [[513, 208], [536, 208], [536, 108], [560, 108]], label: ">=2", tone: "yes", lx: 536, ly: 140 },
        { id: "e_lgt_chk", points: [[513, 222], [560, 222]], label: "<2", tone: "no" },
        { id: "e_ent_chk", points: [[687, 126], [687, 174]], tone: "default" },
        { id: "e_dr_no", points: [[815, 222], [875, 222]], label: "NO", tone: "no" },
        { id: "e_dr_yes", points: [[687, 270], [687, 290], [398, 290], [398, 310]], label: "YES", tone: "yes", lx: 540, ly: 284 },
        { id: "e_rec_yes", points: [[528, 361], [560, 361]], label: "YES", tone: "yes" },
        { id: "e_rec_sil", points: [[815, 361], [875, 361]], tone: "portal" },
        { id: "e_rec_no", points: [[398, 412], [398, 468]], label: "NO", tone: "no" },
        { id: "e_bpm_no", points: [[268, 517], [226, 517]], label: "NO", tone: "no" },
        { id: "e_bpm_yes", points: [[528, 517], [560, 517]], label: "YES", tone: "yes" },
        { id: "e_lad_6c", points: [[815, 517], [875, 517]], tone: "portal" }
      ],
      traces: [
        {
          id: "fusion_strong",
          label: "Trace: Single Strong Cue -> Drowsy -> -21 dB (6C)",
          steps: [
            {
              title: "Step 1: Strong Cue Immediately Flips State = Drowsy",
              highlightNodes: ["grp_mutex_somno", "from_detectors", "chk_strong", "enter_drowsy", "chk_drowsy_state"],
              highlightEdges: ["e_in_str", "e_str_yes", "e_ent_chk"],
              concept: "A BlinkDuration (>= 500 ms) or High PERCLOS (>= 15%) cue enters Drowsy immediately and resets the step index to 0.",
              codeRef: "src/AlertManager.cpp (L213-219, L289-294)"
            },
            {
              title: "Step 2: 63.5 BPM Cooldown -> Step 0 (-21 dB) -> GOTO 6C",
              highlightNodes: ["chk_recovery", "chk_bpm", "step_ladder", "goto_6c_drowsy"],
              highlightEdges: ["e_dr_yes", "e_rec_no", "e_bpm_yes", "e_lad_6c"],
              concept: "Because cooldown was rewound, step 0 (-21.0 dB) fires right away and hands off outside mutex_ to 6C for preemption + audio + Telegram.",
              portalTarget: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_drowsiness" },
              codeRef: "src/AlertManager.cpp (L312, L330-338, L360-373)"
            }
          ]
        },
        {
          id: "fusion_light",
          label: "Trace: Two Light Cues in 120 s (Yawn + Droop)",
          steps: [
            {
              title: "2 Light Cues Within 120 s Trigger Drowsy",
              highlightNodes: ["from_detectors", "chk_strong", "light_window", "enter_drowsy", "chk_drowsy_state", "chk_bpm", "step_ladder", "goto_6c_drowsy"],
              highlightEdges: ["e_in_str", "e_str_no", "e_lgt_ent", "e_ent_chk", "e_dr_yes", "e_rec_no", "e_bpm_yes", "e_lad_6c"],
              concept: "A Yawn (>= 1.0 s) plus a HeadDroop (>= 1.5 s) inside 120 s satisfy cue_window_.size() >= 2, raising Drowsy and dispatching 6C.",
              codeRef: "src/AlertManager.cpp (L220-238) | src/AlertLogic.cpp (L51-59)"
            }
          ]
        },
        {
          id: "fusion_recover",
          label: "Trace: Normal Recovery Silences Alarm Immediately",
          steps: [
            {
              title: "Eyes Open + Upright Head + Mouth Closed -> Safe + Stop",
              highlightNodes: ["chk_drowsy_state", "chk_recovery", "recover_safe", "goto_6c_silence"],
              highlightEdges: ["e_dr_yes", "e_rec_yes", "e_rec_sil"],
              concept: "When blink, droop, and yawn all return to normal with no new cue, DriverState reverts to Safe, cue_window_ clears, and 6C silences audio immediately.",
              portalTarget: { sectionId: "sec6", subviewId: "6C", highlightNodeId: "path_reset" },
              codeRef: "src/AlertManager.cpp (L302-329)"
            }
          ]
        }
      ]
    },
    {
      id: "6C",
      label: "6C. Outside-Lock Dispatcher",
      subtitle: "Priority audio_.stop() preemption, 48 kHz ring-buffer playback, detached Telegram HTTP & park reset",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "6C: Outside-Lock Alarm Dispatcher (Audio Preemption & Telegram)",
        concept: "All slow or blocking I/O executes strictly outside AlertManager's evaluation mutex: blocking distraction playback on GazeWorker, non-blocking 63.5 BPM drowsiness pulses that preempt active distraction audio via generation ticket++, detached libcurl Telegram alerts (3 s / 5 s timeout), and clean silence/reset paths.",
        note: "AudioStreamEngine runs a persistent 48 kHz stereo 6x10 ms ring-buffer mixer (waveOut on Windows, aplay raw pipe on Linux). Calling audio_.stop() increments playback_id_ and wakes GazeWorker's play_blocking() immediately via condition_variable.",
        codeRef: "src/AlertManager.cpp (L159-203, L320-338, L375-406)\nsrc/AudioManager.cpp (L12-30)\nsrc/SoundPlayer.cpp (L253-295, L503-563)\nsrc/TelegramNotifier.cpp (L36-71)"
      },
      nodes: [
        {
          id: "path_distraction", kind: "portal_in", x: 20, y: 34, w: 225, h: 68,
          label: "FROM 6A: DISTRACTION", sublabel: "GazeWorker (unlocked)",
          target: { sectionId: "sec6", subviewId: "6A", highlightNodeId: "goto_6c_dist" },
          details: {
            concept: "Triggered when 10f mode stays NotLookingRoad for >= 2.0 s.",
            codeRef: "src/AlertManager.cpp (L159)"
          }
        },
        {
          id: "chk_preempt_dist", kind: "decision", x: 280, y: 20, w: 235, h: 96,
          label: "Drowsiness reproducing?\n(brief re-lock)", sublabel: "Last-second safety check",
          details: {
            concept: "Re-checks is_drowsiness_alert_reproducing_locked() before launching distraction audio; if false, sets drowsiness_audio_active_ = false and unlocks.",
            codeRef: "src/AlertManager.cpp (L160-166)"
          }
        },
        {
          id: "dist_abort", kind: "term_return", x: 280, y: 146, w: 235, h: 52,
          label: "ABORT Distraction", sublabel: "Drowsy took priority",
          details: {
            concept: "Aborts distraction sound & Telegram if FatigueWorker flipped Drowsy in the microsecond after unlock.",
            codeRef: "src/AlertManager.cpp (L162-164)"
          }
        },
        {
          id: "tg_dist", kind: "process", tone: "async", x: 555, y: 32, w: 245, h: 72,
          label: "ASYNC: Telegram Alert", sublabel: "Detached std::thread\nlibcurl (3s conn / 5s tot)",
          details: {
            concept: "Fires HTTP POST in a detached thread so poor cellular connectivity never stalls GazeWorker.",
            codeRef: "src/AlertManager.cpp (L177-179) | src/TelegramNotifier.cpp (L36-71)"
          }
        },
        {
          id: "play_dist_block", kind: "process", x: 840, y: 28, w: 255, h: 80,
          label: "audio_.play_blocking()", sublabel: "distraction_warning.wav +6 dB\nWaits on state_cv_ until done\nor audio_.stop() wakes it",
          details: {
            concept: "Intentionally blocks GazeWorker while distraction audio plays, WITHOUT holding AlertManager::mutex_ so FatigueWorker can interrupt at any instant.",
            codeRef: "src/AlertManager.cpp (L182-185) | src/AudioManager.cpp (L20-26) | src/SoundPlayer.cpp (L277-284)"
          }
        },
        {
          id: "ret_to_6a", kind: "portal_out", x: 840, y: 138, w: 255, h: 64,
          label: "RETURN TO 6A: WIPE", sublabel: "Clear 10f mode + 2s timer",
          target: { sectionId: "sec6", subviewId: "6A", highlightNodeId: "post_alarm_wipe" },
          details: {
            concept: "When play_blocking finishes or wakes from preemption, GazeWorker returns to 6A to clear gaze_window_ and distracted_since_.",
            codeRef: "src/AlertManager.cpp (L187-192)"
          }
        },
        {
          id: "path_drowsiness", kind: "portal_in", x: 20, y: 256, w: 225, h: 68,
          label: "FROM 6B: DROWSINESS", sublabel: "FatigueWorker (unlocked)",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "goto_6c_drowsy" },
          details: {
            concept: "Entered when DriverState == Drowsy and either a new cue triggered or the 63.5 BPM pulse timer elapsed.",
            codeRef: "src/AlertManager.cpp (L330-338)"
          }
        },
        {
          id: "chk_new_drowsy", kind: "decision", x: 280, y: 242, w: 235, h: 96,
          label: "Newly triggered?\n(Strong or 2 Light)", sublabel: "Telegram edge trigger",
          details: {
            concept: "Sends a Telegram notification on initial Strong / 2-Light triggers rather than spamming on every 63.5 BPM beep.",
            codeRef: "src/AlertManager.cpp (L330-334)"
          }
        },
        {
          id: "tg_drowsy", kind: "process", tone: "async", x: 555, y: 254, w: 245, h: 72,
          label: "ASYNC: Telegram Alert", sublabel: "Detached std::thread\nSomnolence reason payload",
          details: {
            concept: "Posts Critical/Sustained Drowsiness alert asynchronously with 3 s connect / 5 s total timeouts.",
            codeRef: "src/AlertManager.cpp (L331-333) | src/TelegramNotifier.cpp (L36-71)"
          }
        },
        {
          id: "preempt_stop", kind: "process", x: 555, y: 360, w: 245, h: 78,
          label: "Preempt Active Audio", sublabel: "Set drowsy_audio_active=true\naudio_.stop() -> playback_id_++\n(Cuts off distraction clip!)",
          details: {
            concept: "Increments playback_id_ on the 48 kHz mixer, instantly silencing any playing distraction clip and waking GazeWorker's play_blocking().",
            codeRef: "src/AlertManager.cpp (L375-386) | src/SoundPlayer.cpp (L286-295)"
          }
        },
        {
          id: "play_drowsy_pulse", kind: "process", x: 840, y: 360, w: 255, h: 78,
          label: "audio_.play() Pulse", sublabel: "single-beep-1850-975.wav\nNon-blocking 48 kHz mix at\ngain_db (-21.0 to -3.0 dB)",
          details: {
            concept: "Triggers non-blocking 1850/975 Hz beep on the 6x10 ms ring buffer so FatigueWorker continues processing frames at full rate.",
            codeRef: "src/AlertManager.cpp (L387) | src/AudioManager.cpp (L12-18) | src/SoundPlayer.cpp (L253-275)"
          }
        },
        {
          id: "path_reset", kind: "portal_in", x: 20, y: 492, w: 225, h: 74,
          label: "FROM 1A / 6A / 6B: STOP", sublabel: "Park reset, 6A attentive,\nor 6B normal recovery",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "goto_6c_reset" },
          details: {
            concept: "Three callers reach silence/reset: Vehicle Park (1A AlertManager::reset), Gaze Attentive (6A), or Somnolence Recovery (6B).",
            codeRef: "src/AlertManager.cpp (L193-203, L320-329, L390-406)"
          }
        },
        {
          id: "clear_state_stop", kind: "process", x: 295, y: 492, w: 270, h: 74,
          label: "Clear State & Windows", sublabel: "If reset(): clear gaze & cue win,\nSafe, step=0; else clear\ndrowsiness_audio_active_",
          details: {
            concept: "Full reset wipes all windows and DriverState; recovery/attentive clears the matching active playback flag.",
            codeRef: "src/AlertManager.cpp (L199-202, L320-328, L390-402)"
          }
        },
        {
          id: "audio_stop_box", kind: "process", x: 615, y: 494, w: 215, h: 70,
          label: "audio_.stop()", sublabel: "Zeroes active mixer buffer\n& wakes blocked threads",
          details: {
            concept: "Clears active_samples_ and wakes any condition_variable wait on the next 10 ms audio block.",
            codeRef: "src/AudioManager.cpp (L28-30) | src/SoundPlayer.cpp (L286-295)"
          }
        },
        {
          id: "ret_done", kind: "term_return", x: 875, y: 500, w: 220, h: 58,
          label: "RETURN (Silent)", sublabel: "Mixer idle",
          details: {
            concept: "48 kHz audio mixer continues outputting silent PCM blocks ready for the next alert.",
            codeRef: "src/SoundPlayer.cpp (L334-335)"
          }
        }
      ],
      edges: [
        { id: "e_d_chk", points: [[245, 68], [280, 68]], tone: "portal" },
        { id: "e_d_ab", points: [[397, 116], [397, 146]], label: "YES", tone: "yes" },
        { id: "e_d_tg", points: [[515, 68], [555, 68]], label: "NO", tone: "no" },
        { id: "e_d_blk", points: [[800, 68], [840, 68]], tone: "default" },
        { id: "e_blk_6a", points: [[967, 108], [967, 138]], tone: "portal" },
        { id: "e_s_chk", points: [[245, 290], [280, 290]], tone: "portal" },
        { id: "e_s_tg", points: [[515, 290], [555, 290]], label: "YES", tone: "yes" },
        { id: "e_s_no_aud", points: [[397, 338], [397, 399], [555, 399]], label: "emit_audio", tone: "default", lx: 472, ly: 390 },
        { id: "e_tg_pre", points: [[677, 326], [677, 360]], label: "emit_audio", tone: "default" },
        { id: "e_pre_ply", points: [[800, 399], [840, 399]], tone: "default" },
        { id: "e_pre_cut_dist", points: [[800, 376], [820, 376], [820, 170], [840, 170]], label: "wakes 6A!", tone: "portal", dashed: true, lx: 820, ly: 222 },
        { id: "e_r_clr", points: [[245, 529], [295, 529]], tone: "portal" },
        { id: "e_clr_stp", points: [[565, 529], [615, 529]], tone: "default" },
        { id: "e_stp_ret", points: [[830, 529], [875, 529]], tone: "yes" }
      ],
      traces: [
        {
          id: "disp_dist",
          label: "Trace: Distraction Dispatch (Telegram + Blocking Audio)",
          steps: [
            {
              title: "GazeWorker Unlocks, Sends Telegram, Blocks on +6 dB Clip",
              highlightNodes: ["path_distraction", "chk_preempt_dist", "tg_dist", "play_dist_block", "ret_to_6a"],
              highlightEdges: ["e_d_chk", "e_d_tg", "e_d_blk", "e_blk_6a"],
              concept: "AlertManager confirms drowsiness is not active, detaches a Telegram thread, and blocks GazeWorker on distraction_warning.wav (+6 dB) before returning to 6A to wipe gaze_window_.",
              codeRef: "src/AlertManager.cpp (L159-192)"
            }
          ]
        },
        {
          id: "disp_preempt",
          label: "Trace: Drowsiness Preemption Cuts Off Distraction",
          steps: [
            {
              title: "Step 1: Drowsiness Enters Outside-Lock Dispatcher",
              highlightNodes: ["path_drowsiness", "chk_new_drowsy", "tg_drowsy"],
              highlightEdges: ["e_s_chk", "e_s_tg"],
              concept: "FatigueWorker enters AlertManager immediately (since GazeWorker did not hold mutex_ during play_blocking) and fires a detached Telegram alert.",
              codeRef: "src/AlertManager.cpp (L330-338)"
            },
            {
              title: "Step 2: audio_.stop() Cuts Off Distraction & Emits Pulse",
              highlightNodes: ["preempt_stop", "play_drowsy_pulse", "play_dist_block", "ret_to_6a"],
              highlightEdges: ["e_tg_pre", "e_pre_ply", "e_pre_cut_dist", "e_blk_6a"],
              concept: "audio_.stop() bumps playback_id_, immediately silencing distraction_warning.wav and waking GazeWorker to wipe its window (6A), while the first -21 dB beep plays.",
              codeRef: "src/AlertManager.cpp (L375-388) | src/SoundPlayer.cpp (L277-295)"
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
              concept: "Stopping the car (1A), returning gaze (6A), or recovering normal eyes/posture/mouth (6B) stops active playback immediately.",
              codeRef: "src/AlertManager.cpp (L193-203, L320-329, L390-406)"
            }
          ]
        }
      ]
    }
  ]
});
