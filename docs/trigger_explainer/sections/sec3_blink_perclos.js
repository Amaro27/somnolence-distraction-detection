window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec3",
  title: "3. Blinking & PERCLOS",
  subviews: [
    {
      id: "3A",
      label: "3A. Blink Duration (Fused)",
      subtitle: "6-pt EAR, cascaded 5-median -> 40-MA adaptive baseline (70%/20%) & >= 500 ms micro-sleep",
      viewBox: "0 0 1120 660",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "3A: Blink Duration & Micro-Sleep — Fused Algorithm View",
        concept: "Computes average 6-point Eye Aspect Ratio (EAR), continuously adapts open-eye baseline through a cascaded 5-sample median -> 40-sample moving-average filter when eyes are open, and tracks eye closure below T_close (20% aperture) to emit a Strong micro-sleep verdict at >= 500 ms.",
        note: "Thresholds verified in EarConfig (Types.hpp L222-233): factor_descenso = 0.70, sensibilidad_cierre = 0.20, min_dynamic_range = 0.05, median_window = 5, mov_avg_window = 40, min_samples_for_update = 15, microsleep_thresh = 500 ms. Code checks closure_accum_sec_ >= 0.500 (EarDetector.cpp L153).",
        codeRef: "src/FaceGeometry.cpp (L7-20)\nsrc/EarDetector.cpp (L85-182)\ninclude/Types.hpp (L222-233)"
      },
      nodes: [
        {
          id: "calc_ear", kind: "process", x: 20, y: 26, w: 235, h: 72,
          label: "6-Point Average EAR", sublabel: "EAR = (EAR_R + EAR_L) * 0.5\ndt = compute_clamped_dt(now)",
          details: {
            concept: "Computes vertical-to-horizontal eyelid ratio for right (36..41) and left (42..47) eyes, averages them, and clamps frame time delta dt.",
            codeRef: "src/FaceGeometry.cpp (L7-20) | src/EarDetector.cpp (L92, L97)"
          }
        },
        {
          id: "chk_open_adapt", kind: "decision", x: 295, y: 18, w: 245, h: 88,
          label: "!in_blink &&\nEAR > T_desc?", sublabel: "Open-eye adaptation gate",
          details: {
            concept: "Adaptive baseline only updates while eyes are genuinely open above the 70% descent threshold, freezing thresholds during blinks.",
            codeRef: "src/EarDetector.cpp (L100)"
          }
        },
        {
          id: "adapt_filter", kind: "process", x: 585, y: 18, w: 270, h: 88,
          label: "Cascaded Open-Eye Filter", sublabel: "5-med -> 40-MA (update >=15)\nT_desc = closed + 0.70*range\nT_close = max(floor, +0.20*range)",
          details: {
            concept: "Pushes EAR to a 5-frame median deque, feeds medians to a 40-frame moving-average deque, and recomputes T_desc (70%) and T_close (20%, floored by calibration).",
            codeRef: "src/EarDetector.cpp (L101-126)"
          }
        },
        {
          id: "chk_descent", kind: "decision", x: 295, y: 152, w: 245, h: 90,
          label: "EAR < T_desc (70%)?", sublabel: "Blink cycle entry test",
          details: {
            concept: "Eyes dropping below 70% aperture enter a blink cycle (in_blink_cycle = true); returning above T_desc finishes the blink.",
            codeRef: "src/EarDetector.cpp (L137-182)"
          }
        },
        {
          id: "enter_blink", kind: "process", x: 20, y: 162, w: 235, h: 70,
          label: "Blink Cycle Active", sublabel: "If new: in_blink = true,\nreached_close=false, accum=0",
          details: {
            concept: "Initializes closure accumulator on the first frame dipping below T_desc.",
            codeRef: "src/EarDetector.cpp (L138-142)"
          }
        },
        {
          id: "chk_closure", kind: "decision", x: 20, y: 278, w: 235, h: 92,
          label: "EAR <= T_close (20%)?", sublabel: "Effective eye closure",
          details: {
            concept: "Distinguishes full eyelid closure (<= 20% aperture) from mid-blink hover between 20% and 70%.",
            codeRef: "src/EarDetector.cpp (L144)"
          }
        },
        {
          id: "acc_close", kind: "process", x: 20, y: 414, w: 235, h: 70,
          label: "Accumulate Closure", sublabel: "accum = !reached ? dt : accum+dt\nreached_close = true",
          details: {
            concept: "First closed frame sets accum = dt; subsequent consecutive closed frames add dt.",
            codeRef: "src/EarDetector.cpp (L145-150)"
          }
        },
        {
          id: "chk_microsleep", kind: "decision", x: 295, y: 402, w: 245, h: 94,
          label: "accum >= 500 ms &&\n!microsleep_active?", sublabel: "Strong cue trigger",
          details: {
            concept: "Micro-sleep is a Strong cue: triggers an immediate Drowsy alert once closure persists for >= 500 ms.",
            codeRef: "src/EarDetector.cpp (L152-167)"
          }
        },
        {
          id: "term_drowsy", kind: "term_alarm", x: 600, y: 420, w: 225, h: 60,
          label: "DROWSY", sublabel: "Strong: Micro-sleep >=500ms",
          details: {
            concept: "Sets microsleep_active = true and immediately triggers Drowsy (1 Strong cue suffices).",
            codeRef: "src/EarDetector.cpp (L155-165) | src/AlertManager.cpp (L213-219)"
          }
        },
        {
          id: "ret_closed_wait", kind: "term_return", x: 295, y: 544, w: 245, h: 54,
          label: "RETURN", sublabel: "Closing (<500ms or latched)",
          details: {
            concept: "Normal blink still under 500 ms, or micro-sleep already latched for this closure.",
            codeRef: "src/EarDetector.cpp (L153-167)"
          }
        },
        {
          id: "hover_decay", kind: "process", x: 295, y: 288, w: 245, h: 72,
          label: "Mid-Lid Hover Decay", sublabel: "T_close < EAR < T_desc\naccum = max(0, accum - 1.5*dt)",
          details: {
            concept: "If eyelids hover between 20% and 70% aperture, accumulator decays at -1.5*dt rather than wiping immediately.",
            codeRef: "src/EarDetector.cpp (L168-171)"
          }
        },
        {
          id: "ret_hover", kind: "term_return", x: 600, y: 296, w: 205, h: 56,
          label: "RETURN", sublabel: "Partial closure decay",
          details: {
            concept: "Returns while blink cycle stays active above T_close.",
            codeRef: "src/EarDetector.cpp (L168-171)"
          }
        },
        {
          id: "blink_end", kind: "process", x: 600, y: 160, w: 255, h: 74,
          label: "Finish Blink Cycle", sublabel: "If reached_close: save blink_ms\nin_blink=false, accum=0",
          details: {
            concept: "When EAR rises back >= T_desc, logs last_blink_duration_ms = accum * 1000 and clears blink state.",
            codeRef: "src/EarDetector.cpp (L172-182)"
          }
        },
        {
          id: "ret_open", kind: "term_return", x: 900, y: 170, w: 190, h: 54,
          label: "RETURN", sublabel: "Eyes open",
          details: {
            concept: "Normal open-eye frame completes.",
            codeRef: "src/EarDetector.cpp (L172-182)"
          }
        }
      ],
      edges: [
        { id: "e_ear_ad", points: [[255, 62], [295, 62]], tone: "default" },
        { id: "e_ad_yes", points: [[540, 62], [585, 62]], label: "YES", tone: "yes" },
        { id: "e_ad_no", points: [[417, 106], [417, 152]], label: "NO", tone: "no", lx: 434, ly: 129 },
        { id: "e_ad_desc", points: [[720, 106], [720, 128], [417, 128], [417, 152]], tone: "default" },
        { id: "e_desc_yes", points: [[295, 197], [255, 197]], label: "YES", tone: "yes" },
        { id: "e_desc_no", points: [[540, 197], [600, 197]], label: "NO", tone: "no" },
        { id: "e_end_ret", points: [[855, 197], [900, 197]], tone: "yes" },
        { id: "e_ent_cls", points: [[137, 232], [137, 278]], tone: "default" },
        { id: "e_cls_yes", points: [[137, 370], [137, 414]], label: "YES", tone: "yes", lx: 158, ly: 392 },
        { id: "e_cls_no", points: [[255, 324], [295, 324]], label: "NO", tone: "no" },
        { id: "e_hov_ret", points: [[540, 324], [600, 324]], tone: "default" },
        { id: "e_acc_mic", points: [[255, 449], [295, 449]], tone: "default" },
        { id: "e_mic_yes", points: [[540, 449], [600, 449]], label: "YES", tone: "yes" },
        { id: "e_mic_no", points: [[417, 496], [417, 544]], label: "NO", tone: "no", lx: 436, ly: 520 }
      ],
      traces: [
        {
          id: "open_adapt_trace",
          label: "Trace: Open Eye Drift -> Adaptive Filter",
          steps: [
            {
              title: "Step 1: Open Eyes Update 5-Med -> 40-MA Filter",
              highlightNodes: ["calc_ear", "chk_open_adapt", "adapt_filter"],
              highlightEdges: ["e_ear_ad", "e_ad_yes"],
              concept: "While !in_blink_cycle and EAR > T_desc, the 5-frame median and 40-frame moving average gently adapt T_desc (70%) and T_close (20%, floored by calibration).",
              codeRef: "src/EarDetector.cpp (L99-126)"
            },
            {
              title: "Step 2: Exit Open",
              highlightNodes: ["chk_descent", "blink_end", "ret_open"],
              highlightEdges: ["e_ad_desc", "e_desc_no", "e_end_ret"],
              concept: "Because EAR >= T_desc, any finished blink duration is saved and the cycle returns.",
              codeRef: "src/EarDetector.cpp (L172-182)"
            }
          ]
        },
        {
          id: "normal_blink_trace",
          label: "Trace: Normal Blink (< 500 ms)",
          steps: [
            {
              title: "Step 1: Enter Closure (< 500 ms)",
              highlightNodes: ["calc_ear", "chk_open_adapt", "chk_descent", "enter_blink", "chk_closure", "acc_close", "chk_microsleep", "ret_closed_wait"],
              highlightEdges: ["e_ear_ad", "e_ad_no", "e_desc_yes", "e_ent_cls", "e_cls_yes", "e_acc_mic", "e_mic_no"],
              concept: "EAR drops below T_desc and T_close, accumulating ~150-300 ms without reaching 500 ms.",
              codeRef: "src/EarDetector.cpp (L137-167)"
            },
            {
              title: "Step 2: Re-Open Records Blink Duration",
              highlightNodes: ["chk_descent", "blink_end", "ret_open"],
              highlightEdges: ["e_desc_no", "e_end_ret"],
              concept: "Upon opening above T_desc, last_blink_duration_ms updates and blink state resets.",
              codeRef: "src/EarDetector.cpp (L173-181)"
            }
          ]
        },
        {
          id: "microsleep_trace",
          label: "Trace: Micro-Sleep (>= 500 ms) -> DROWSY",
          steps: [
            {
              title: "Closure >= 500 ms Fires Strong Cue -> DROWSY",
              highlightNodes: ["enter_blink", "chk_closure", "acc_close", "chk_microsleep", "term_drowsy"],
              highlightEdges: ["e_ent_cls", "e_cls_yes", "e_acc_mic", "e_mic_yes"],
              concept: "Holding EAR <= T_close for >= 500 ms sets microsleep_alarm_active_ and immediately triggers Drowsy.",
              codeRef: "src/EarDetector.cpp (L152-167)"
            }
          ]
        },
        {
          id: "hover_trace",
          label: "Trace: Partial Lid Hover (-1.5*dt Decay)",
          steps: [
            {
              title: "Mid-Aperture Hover Decays Accumulator",
              highlightNodes: ["enter_blink", "chk_closure", "hover_decay", "ret_hover"],
              highlightEdges: ["e_ent_cls", "e_cls_no", "e_hov_ret"],
              concept: "Between T_close and T_desc, closure_accum decays smoothly at -1.5*dt instead of dropping abruptly to 0.",
              codeRef: "src/EarDetector.cpp (L168-171)"
            }
          ]
        }
      ]
    },
    {
      id: "3B",
      label: "3B. PERCLOS (Fused)",
      subtitle: "O(1) 30-second rolling closure percentage, 30 s warmup, >= 15.0% High PERCLOS trigger & buffer wipe",
      viewBox: "0 0 1120 660",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "3B: 30-Second Rolling PERCLOS — Fused Algorithm View",
        concept: "Maintains a 30-second sliding time window of closed-eye samples (EAR <= T_close) with an O(1) running counter. After 30 seconds of warmup, if PERCLOS >= 15.0% while in a blink cycle (with 5 s cooldown), it emits a Strong cue and wipes the PERCLOS window.",
        note: "High PERCLOS (>= 15.0%, Strong cue) is the only PERCLOS alarm active in MS6; Medium PERCLOS (7.5%-15.0%) is intentionally disabled in EarDetector.cpp.",
        codeRef: "src/EarDetector.cpp (L17-22, L184-247)\ninclude/Types.hpp (L230-232)"
      },
      nodes: [
        {
          id: "in_ear", kind: "process", x: 30, y: 38, w: 235, h: 68,
          label: "Inputs from EAR Step", sublabel: "EAR, T_close (20%),\nin_blink_cycle, now",
          details: {
            concept: "Uses the instantaneous EAR and adaptive T_close threshold computed in step 3A.",
            codeRef: "src/EarDetector.cpp (L186-188)"
          }
        },
        {
          id: "push_sample", kind: "process", x: 305, y: 38, w: 255, h: 68,
          label: "Classify & Push Sample", sublabel: "is_closed = (EAR <= T_close)\nPush (now, closed); ++count",
          details: {
            concept: "Records binary eye-closure sample with timestamp and increments O(1) closed_samples_count_.",
            codeRef: "src/EarDetector.cpp (L187-191)"
          }
        },
        {
          id: "prune_30s", kind: "process", x: 605, y: 38, w: 255, h: 68,
          label: "Prune > 30 s Old Samples", sublabel: "While front.t < now - 30s:\ndecrement closed if 1, pop front",
          details: {
            concept: "Evicts samples older than perclos_window (30 s) in amortized O(1) while keeping closed_samples_count_ exact.",
            codeRef: "src/EarDetector.cpp (L193-201)"
          }
        },
        {
          id: "calc_pct", kind: "process", x: 605, y: 152, w: 255, h: 72,
          label: "Compute PERCLOS %", sublabel: "PERCLOS = (closed / size) * 100%\nelapsed = now - perclos_start_time",
          details: {
            concept: "Starts warmup timer on the first sample and computes percentage of closed samples over the current window.",
            codeRef: "src/EarDetector.cpp (L203-217)"
          }
        },
        {
          id: "chk_warmup", kind: "decision", x: 605, y: 270, w: 255, h: 96,
          label: "elapsed >= 30 s?\n(Warmed Up)", sublabel: "Full 30s baseline",
          details: {
            concept: "Prevents transient blinks during the first 30 seconds from artificially spiking PERCLOS above 15%.",
            codeRef: "src/EarDetector.cpp (L208-211, L223)"
          }
        },
        {
          id: "ret_warmup", kind: "term_return", x: 905, y: 290, w: 190, h: 56,
          label: "RETURN", sublabel: "Warming up (< 30 s)",
          details: {
            concept: "Continues filling the 30-second window; reports warmup_remaining_sec to HUD.",
            codeRef: "src/EarDetector.cpp (L210-211, L237-247)"
          }
        },
        {
          id: "chk_high_perclos", kind: "decision", x: 260, y: 266, w: 295, h: 104,
          label: "in_blink && cooldown>=5s\n&& PERCLOS >= 15.0%?", sublabel: "High PERCLOS trigger",
          details: {
            concept: "Fires High PERCLOS (>= 15.0%) only during an active blink cycle and at least 5 seconds after the last PERCLOS cue.",
            codeRef: "src/EarDetector.cpp (L223-225)"
          }
        },
        {
          id: "ret_normal", kind: "term_return", x: 30, y: 290, w: 185, h: 56,
          label: "RETURN", sublabel: "PERCLOS < 15.0%",
          details: {
            concept: "Returns current perclos_pct_ with no cue emitted.",
            codeRef: "src/EarDetector.cpp (L237-247)"
          }
        },
        {
          id: "wipe_perclos", kind: "process", x: 260, y: 422, w: 295, h: 72,
          label: "Emit Strong Cue & Wipe", sublabel: "last_cue = now; clear_perclos()\n(Restarts 30 s warmup)",
          details: {
            concept: "Attaches CueEvent(Perclos, Strong) and calls clear_perclos(), resetting buffer, closed_count, and has_perclos_start_.",
            codeRef: "src/EarDetector.cpp (L226-234)"
          }
        },
        {
          id: "term_drowsy", kind: "term_alarm", x: 615, y: 428, w: 235, h: 60,
          label: "DROWSY", sublabel: "Strong: High PERCLOS >=15%",
          details: {
            concept: "High PERCLOS is a Strong cue, triggering an immediate Drowsy state.",
            codeRef: "src/EarDetector.cpp (L227-232) | src/AlertManager.cpp (L213-219)"
          }
        }
      ],
      edges: [
        { id: "e_in_push", points: [[265, 72], [305, 72]], tone: "default" },
        { id: "e_push_pr", points: [[560, 72], [605, 72]], tone: "default" },
        { id: "e_pr_pct", points: [[732, 106], [732, 152]], tone: "default" },
        { id: "e_pct_wu", points: [[732, 224], [732, 270]], tone: "default" },
        { id: "e_wu_no", points: [[860, 318], [905, 318]], label: "NO", tone: "no" },
        { id: "e_wu_yes", points: [[605, 318], [555, 318]], label: "YES", tone: "yes" },
        { id: "e_hi_no", points: [[260, 318], [215, 318]], label: "NO", tone: "no" },
        { id: "e_hi_yes", points: [[407, 370], [407, 422]], label: "YES", tone: "yes" },
        { id: "e_wipe_al", points: [[555, 458], [615, 458]], tone: "yes" }
      ],
      traces: [
        {
          id: "perclos_warmup",
          label: "Trace: Initial 30 s Warmup Accumulation",
          steps: [
            {
              title: "Samples Accumulate Until 30 s Elapsed",
              highlightNodes: ["in_ear", "push_sample", "prune_30s", "calc_pct", "chk_warmup", "ret_warmup"],
              highlightEdges: ["e_in_push", "e_push_pr", "e_pr_pct", "e_pct_wu", "e_wu_no"],
              concept: "During the initial 30 seconds, closed samples build up in perclos_buffer_ while warmed_up == false prevents premature triggers.",
              codeRef: "src/EarDetector.cpp (L187-223)"
            }
          ]
        },
        {
          id: "perclos_trigger",
          label: "Trace: Warmed Up >= 15% -> DROWSY & Self-Wipe",
          steps: [
            {
              title: "High PERCLOS Fires Strong Cue & Resets Warmup",
              highlightNodes: ["calc_pct", "chk_warmup", "chk_high_perclos", "wipe_perclos", "term_drowsy"],
              highlightEdges: ["e_pct_wu", "e_wu_yes", "e_hi_yes", "e_wipe_al"],
              concept: "When PERCLOS >= 15% after 30 s warmup (while in_blink_cycle and >= 5 s since last cue), a Strong cue triggers DROWSY and clear_perclos() restarts the 30 s window.",
              codeRef: "src/EarDetector.cpp (L223-235)"
            }
          ]
        }
      ]
    },
    {
      id: "3C",
      label: "3C. Blink Duration (Cut-Off)",
      subtitle: "Calibration baseline, Face Lost wipe, 0.2 s yawn suppression, handoff to 3D & MovingCalibrated gate",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "3C: Blink Duration — System & FSM Cut-Off View",
        concept: "Shows how EarDetector's blink/micro-sleep state machine interacts with external subsystems: 1C (68 landmarks vs Face Lost), 1B (open/closed calibration baseline), 4B (>= 0.2 s yawn suppression), internal sequential continuation into 3D (PERCLOS), and 1A's MovingCalibrated gate into 6B.",
        note: "When parked (StationaryCalibrated), EarDetector::process() still updates adaptive EAR and last_blink_duration_ms for HUD display, while cues are only forwarded when SystemState == MovingCalibrated.",
        codeRef: "src/PipelineCoordinator.cpp (L491, L528-560)\nsrc/EarDetector.cpp (L42-182, L266-274)"
      },
      nodes: [
        {
          id: "from_calib_ear", kind: "portal_in", x: 20, y: 24, w: 230, h: 66,
          label: "FROM 1B: APPLY_EAR", sublabel: "ear_open, ear_closed ->\nT_desc (70%), floor (20%)",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_3c_ear_base" },
          details: {
            concept: "apply_baseline() pre-fills 5-med & 40-MA buffers with ear_open, sets floor_closure_ = thresh_closure_, and marks calibrated_ = true.",
            codeRef: "src/EarDetector.cpp (L42-83)"
          }
        },
        {
          id: "from_face_gate", kind: "portal_in", x: 20, y: 122, w: 230, h: 66,
          label: "FROM 1C: DLIB_GATE", sublabel: "68 Landmarks | Face Lost",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "out_landmarks" },
          details: {
            concept: "Routes to EarDetector::process() when landmarks exist, or EarDetector::on_face_lost() when missing or throttled.",
            codeRef: "src/PipelineCoordinator.cpp (L547-555)"
          }
        },
        {
          id: "from_yawn_sup", kind: "portal_in", x: 20, y: 254, w: 230, h: 64,
          label: "FROM 4B: YAWN_SUP", sublabel: "suppress_by_yawn >= 0.2s",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "sig_yawn_sup" },
          details: {
            concept: "Early >= 0.2 s yawn detection mutes eye closure / squint artifacts before a 500 ms false micro-sleep can trip.",
            codeRef: "src/PipelineCoordinator.cpp (L528-531, L550) | src/EarDetector.cpp (L132-136)"
          }
        },
        {
          id: "chk_face", kind: "decision", x: 295, y: 108, w: 215, h: 94,
          label: "Landmarks present?", sublabel: "1C output branch",
          details: {
            concept: "Determines whether to run EAR filtering & blink tracking or wipe unconfirmed closure streaks.",
            codeRef: "src/PipelineCoordinator.cpp (L547-555)"
          }
        },
        {
          id: "lost_wipe", kind: "process", x: 560, y: 24, w: 255, h: 72,
          label: "on_face_lost: Wipe Streak", sublabel: "in_blink=false, accum=0\nhas_last_time=false",
          details: {
            concept: "Immediately clears unconfirmed blink/micro-sleep streaks so an unobserved gap never bridges into a false micro-sleep.",
            codeRef: "src/EarDetector.cpp (L266-274)"
          }
        },
        {
          id: "to_3d_lost", kind: "portal_out", x: 860, y: 28, w: 235, h: 64,
          label: "TO 3D: PERCLOS_LOST", sublabel: "Prune 30s window only",
          target: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "from_blink_perclos" },
          details: {
            concept: "Continues into on_face_lost()'s PERCLOS pruning branch (no closed sample pushed).",
            codeRef: "src/EarDetector.cpp (L275-330)"
          }
        },
        {
          id: "ear_adapt_proc", kind: "process", x: 295, y: 246, w: 215, h: 74,
          label: "EAR + Adaptive Filter", sublabel: "compute_ear -> 5-Med ->\n40-MA (open eyes only)",
          details: {
            concept: "Computes EAR, dt, and updates T_desc / T_close when !in_blink_cycle && ear_raw > thresh_descent.",
            codeRef: "src/EarDetector.cpp (L92-129)"
          }
        },
        {
          id: "chk_yawn_sup", kind: "decision", x: 295, y: 356, w: 215, h: 92,
          label: "suppress_by_yawn\n== true?", sublabel: "Yawn >= 0.2 s mute",
          details: {
            concept: "Checks whether the full-frame Yawn CNN has already been yawning for >= 0.2 s.",
            codeRef: "src/EarDetector.cpp (L132)"
          }
        },
        {
          id: "yawn_wipe", kind: "process", x: 560, y: 366, w: 245, h: 72,
          label: "Mute & Wipe Blink State", sublabel: "in_blink=false, accum=0\nmicrosleep_active=false",
          details: {
            concept: "Cancels any in-progress eye-closure streak caused by squinting during a yawn.",
            codeRef: "src/EarDetector.cpp (L132-136)"
          }
        },
        {
          id: "blink_fsm_box", kind: "process", x: 295, y: 486, w: 215, h: 74,
          label: "Blink & >=500ms Check", sublabel: "If calibrated_ && >=500ms:\nCueEvent(BlinkDuration, Strong)",
          details: {
            concept: "Runs the 3A blink/micro-sleep state machine; guards Strong cue emission with if (calibrated_).",
            codeRef: "src/EarDetector.cpp (L137-182)"
          }
        },
        {
          id: "to_3d_perclos", kind: "portal_out", x: 560, y: 490, w: 245, h: 66,
          label: "TO 3D: ROLLING_PERCLOS", sublabel: "Pass EAR, T_close, in_blink",
          target: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "from_blink_perclos" },
          details: {
            concept: "Proceeds sequentially into Section 5 of EarDetector::process() to update the 30-second PERCLOS window.",
            codeRef: "src/EarDetector.cpp (L184-263)"
          }
        },
        {
          id: "read_mov_cal", kind: "decision", x: 855, y: 352, w: 240, h: 98,
          label: "READ 1A: SystemState\n== MovingCalibrated?", sublabel: "Click -> 1A FSM",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            concept: "After EarDetector::process() returns EarAssessment, PipelineCoordinator forwards cues to AlertManager only when MovingCalibrated.",
            codeRef: "src/PipelineCoordinator.cpp (L557-559)"
          }
        },
        {
          id: "goto_6b", kind: "portal_out", x: 865, y: 492, w: 225, h: 64,
          label: "GOTO 6B: CUE_FUSION", sublabel: "update_somnolence(ear...)",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
          details: {
            concept: "Forwards BlinkDuration (and PERCLOS) Strong cues into AlertManager::mutex_.",
            codeRef: "src/PipelineCoordinator.cpp (L558) | src/AlertManager.cpp (L289-294)"
          }
        },
        {
          id: "ret_hud_only", kind: "term_return", x: 865, y: 224, w: 225, h: 56,
          label: "RETURN (HUD Only)", sublabel: "Live EAR & blink telemetry",
          details: {
            concept: "Displays live EAR, thresholds, and last blink duration on DebugVisualizer without alerting.",
            codeRef: "src/PipelineCoordinator.cpp (L557-575)"
          }
        }
      ],
      edges: [
        { id: "e_in_chk", points: [[250, 155], [295, 155]], tone: "default" },
        { id: "e_fl_yes", points: [[402, 108], [402, 60], [560, 60]], label: "NO (Lost)", tone: "no", lx: 478, ly: 52 },
        { id: "e_fl_3d", points: [[815, 60], [860, 60]], tone: "portal" },
        { id: "e_ff_ad", points: [[402, 202], [402, 246]], label: "YES", tone: "yes" },
        { id: "e_ad_ys", points: [[402, 320], [402, 356]], tone: "default" },
        { id: "e_ys_in", points: [[250, 286], [272, 286], [272, 402], [295, 402]], tone: "portal" },
        { id: "e_ys_yes", points: [[510, 402], [560, 402]], label: "YES", tone: "yes" },
        { id: "e_ys_no", points: [[402, 448], [402, 486]], label: "NO", tone: "no" },
        { id: "e_yw_3d", points: [[682, 438], [682, 490]], tone: "portal" },
        { id: "e_bl_3d", points: [[510, 523], [560, 523]], tone: "portal" },
        { id: "e_3d_mov", points: [[805, 523], [830, 523], [830, 401], [855, 401]], tone: "default" },
        { id: "e_mov_no", points: [[975, 352], [975, 280]], label: "NO", tone: "no", lx: 992, ly: 316 },
        { id: "e_mov_yes", points: [[975, 450], [975, 492]], label: "YES", tone: "yes", lx: 994, ly: 471 }
      ],
      traces: [
        {
          id: "cut_microsleep",
          label: "Trace: MovingCalibrated Micro-Sleep -> 3D -> 6B",
          steps: [
            {
              title: "Step 1: Landmarks & Calibration Active (Not Yawning)",
              highlightNodes: ["from_calib_ear", "from_face_gate", "chk_face", "ear_adapt_proc", "chk_yawn_sup"],
              highlightEdges: ["e_in_chk", "e_ff_ad", "e_ad_ys"],
              concept: "68 landmarks arrive with calibrated thresholds applied and suppress_by_yawn == false.",
              codeRef: "src/EarDetector.cpp (L92-136)"
            },
            {
              title: "Step 2: >= 500 ms Cue -> Continue to PERCLOS (3D) & 6B",
              highlightNodes: ["chk_yawn_sup", "blink_fsm_box", "to_3d_perclos", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_ys_no", "e_bl_3d", "e_3d_mov", "e_mov_yes"],
              concept: "Calibrated micro-sleep emits CueEvent(BlinkDuration, Strong), passes through PERCLOS (3D), and enters AlertManager::update_somnolence (6B).",
              portalTarget: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "from_blink_perclos" },
              codeRef: "src/EarDetector.cpp (L153-185) | src/PipelineCoordinator.cpp (L557-558)"
            }
          ]
        },
        {
          id: "cut_yawn_mute",
          label: "Trace: Yawn >= 0.2 s Wipes Blink/Squint State",
          steps: [
            {
              title: "Yawn Suppressor Clears Closure Accumulator",
              highlightNodes: ["from_yawn_sup", "chk_yawn_sup", "yawn_wipe", "to_3d_perclos"],
              highlightEdges: ["e_ys_in", "e_ys_yes", "e_yw_3d"],
              concept: "Natural eye-squinting during yawning is muted immediately once yawn_accum >= 0.2 s, preventing false micro-sleep alerts.",
              codeRef: "src/EarDetector.cpp (L132-136)"
            }
          ]
        },
        {
          id: "cut_face_lost",
          label: "Trace: Face Lost Clears Unconfirmed Closure Streak",
          steps: [
            {
              title: "on_face_lost Prevents Gap Bridging",
              highlightNodes: ["from_face_gate", "chk_face", "lost_wipe", "to_3d_lost"],
              highlightEdges: ["e_in_chk", "e_fl_yes", "e_fl_3d"],
              concept: "Missing landmarks immediately clear in_blink_cycle_ and closure_accum_sec_ before branching to 3D's face-lost PERCLOS prune.",
              codeRef: "src/EarDetector.cpp (L266-274)"
            }
          ]
        }
      ]
    },
    {
      id: "3D",
      label: "3D. PERCLOS (Cut-Off)",
      subtitle: "MovingCalibrated + calibrated gate, park clear_perclos(), face-lost prune & disabled Medium tier",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "3D: PERCLOS — System & FSM Cut-Off View",
        concept: "Unlike EAR and head pitch, PERCLOS is strictly suppressed and wiped whenever the car is parked or uncalibrated. Stopping from MovingCalibrated also triggers ear_detector_.clear_perclos().",
        note: "Two MS6 specifics: (1) Medium PERCLOS (7.5%-15.0%) is defined in EarConfig (Types.hpp L231) but disabled in EarDetector.cpp; (2) clear_perclos() is invoked directly from Main (L340) and IMU (L668) without a mutex against somnolence_worker_loop.",
        codeRef: "src/PipelineCoordinator.cpp (L340, L515-522, L550, L668)\nsrc/EarDetector.cpp (L17-22, L184-331)\ninclude/Types.hpp (L230-232)"
      },
      nodes: [
        {
          id: "from_blink_perclos", kind: "portal_in", x: 20, y: 120, w: 235, h: 66,
          label: "FROM 3C: EAR / LOST", sublabel: "EAR, T_close, in_blink | Lost",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "to_3d_perclos" },
          details: {
            concept: "Continues directly from Blink Duration (3C) on either the Face Found or Face Lost branch.",
            codeRef: "src/EarDetector.cpp (L186, L275)"
          }
        },
        {
          id: "from_park_clear", kind: "portal_in", x: 20, y: 24, w: 235, h: 66,
          label: "FROM 1A: CLEAR_PERCLOS", sublabel: "On Park (Main / IMU thread)",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "goto_3d_clear" },
          details: {
            concept: "Main ('m', L340) and IMU (L668) call clear_perclos() when MovingCalibrated -> StationaryCalibrated.",
            note: "Concurrency caveat: mutates perclos_buffer_ without a mutex against somnolence_worker_loop (which also has a fallback clear on L520).",
            codeRef: "src/PipelineCoordinator.cpp (L340, L520, L668) | src/EarDetector.cpp (L17-22)"
          }
        },
        {
          id: "from_yawn_sup", kind: "portal_in", x: 20, y: 392, w: 235, h: 64,
          label: "FROM 4B: YAWN_SUP", sublabel: "suppress_by_yawn >= 0.2s",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "sig_yawn_sup" },
          details: {
            concept: "During yawning >= 0.2 s, closed samples record as 0 (not closed) and High PERCLOS cue emission is muted.",
            codeRef: "src/EarDetector.cpp (L187, L223)"
          }
        },
        {
          id: "chk_face_lost", kind: "decision", x: 300, y: 106, w: 225, h: 94,
          label: "Face Lost branch?", sublabel: "on_face_lost vs process",
          details: {
            concept: "Routes Face Lost frames to non-inserting window pruning rather than inserting fake open/closed samples.",
            codeRef: "src/EarDetector.cpp (L275-330)"
          }
        },
        {
          id: "lost_prune", kind: "process", x: 575, y: 24, w: 265, h: 76,
          label: "Face-Lost PERCLOS Prune", sublabel: "If !cal: clear; else prune <30s\n(no push; reset start if empty)",
          details: {
            concept: "Prunes stale samples older than 30 s when calibrated without adding new samples; never emits PERCLOS cues while face is missing.",
            codeRef: "src/EarDetector.cpp (L275-330)"
          }
        },
        {
          id: "ret_lost", kind: "term_return", x: 890, y: 34, w: 205, h: 56,
          label: "RETURN (No Cue)", sublabel: "Updated perclos_pct_",
          details: {
            concept: "Returns EarAssessment with empty cues vector.",
            codeRef: "src/EarDetector.cpp (L320-330)"
          }
        },
        {
          id: "gate_mov_cal", kind: "decision", x: 300, y: 246, w: 225, h: 100,
          label: "calibrated_ &&\nREAD 1A: MovingCal?", sublabel: "Click -> 1A FSM",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            concept: "record_perclos is set strictly to (SystemState == MovingCalibrated) in PipelineCoordinator.cpp L515 & L550.",
            codeRef: "src/PipelineCoordinator.cpp (L515, L550) | src/EarDetector.cpp (L186, L248-250)"
          }
        },
        {
          id: "parked_clear", kind: "process", x: 575, y: 260, w: 265, h: 72,
          label: "clear_perclos() Every Frame", sublabel: "PERCLOS = 0.0%; warmed_up=false\nwarmup_remaining = 30.0 s",
          details: {
            concept: "Whenever uncalibrated or stationary, PERCLOS history stays completely cleared and warmup stays reset to 30 s.",
            codeRef: "src/EarDetector.cpp (L248-262)"
          }
        },
        {
          id: "ret_parked", kind: "term_return", x: 890, y: 268, w: 205, h: 56,
          label: "RETURN (0% Paused)", sublabel: "PERCLOS disabled",
          details: {
            concept: "No PERCLOS tracking or cues occur while parked or uncalibrated.",
            codeRef: "src/EarDetector.cpp (L252-262)"
          }
        },
        {
          id: "active_perclos", kind: "process", x: 300, y: 386, w: 225, h: 78,
          label: "Update 30 s Window", sublabel: "closed=(!yawn && EAR<=T_close)\nPush, prune <30s, compute %",
          details: {
            concept: "Records closed sample only if !suppress_by_yawn and EAR <= T_close, prunes < 30 s, and updates perclos_pct_.",
            codeRef: "src/EarDetector.cpp (L187-218)"
          }
        },
        {
          id: "chk_high_gate", kind: "decision", x: 575, y: 374, w: 265, h: 102,
          label: "warmed && !yawn &&\nin_blink && 5s && >=15%?", sublabel: "High PERCLOS only",
          details: {
            concept: "Emits Strong PERCLOS cue when all five conditions hold (Medium 7.5% tier is disabled).",
            codeRef: "src/EarDetector.cpp (L223-235)"
          }
        },
        {
          id: "emit_and_wipe", kind: "process", x: 575, y: 522, w: 265, h: 72,
          label: "Cue(Perclos, Strong) + Wipe", sublabel: "last_cue = now\nclear_perclos() restarts 30 s",
          details: {
            concept: "Attaches CueEvent(Perclos, Strong) and calls clear_perclos() to wipe the 30-second window.",
            codeRef: "src/EarDetector.cpp (L226-234)"
          }
        },
        {
          id: "goto_6b", kind: "portal_out", x: 885, y: 452, w: 210, h: 66,
          label: "GOTO 6B: CUE_FUSION", sublabel: "update_somnolence(ear...)",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
          details: {
            concept: "Because MovingCalibrated == true, EarAssessment (with any BlinkDuration or Perclos cue) enters AlertManager::update_somnolence (6B).",
            codeRef: "src/PipelineCoordinator.cpp (L557-559) | src/AlertManager.cpp (L289-294)"
          }
        }
      ],
      edges: [
        { id: "e_in_chk", points: [[255, 153], [300, 153]], tone: "default" },
        { id: "e_fl_yes", points: [[412, 106], [412, 62], [575, 62]], label: "YES", tone: "yes", lx: 490, ly: 54 },
        { id: "e_fl_ret", points: [[840, 62], [890, 62]], tone: "default" },
        { id: "e_fl_no", points: [[412, 200], [412, 246]], label: "NO", tone: "no" },
        { id: "e_mov_no", points: [[525, 296], [575, 296]], label: "NO", tone: "no" },
        { id: "e_prk_ret", points: [[840, 296], [890, 296]], tone: "default" },
        { id: "e_mov_yes", points: [[412, 346], [412, 386]], label: "YES", tone: "yes" },
        { id: "e_yawn_in", points: [[255, 424], [300, 424]], tone: "portal" },
        { id: "e_act_chk", points: [[525, 425], [575, 425]], tone: "default" },
        { id: "e_hi_yes", points: [[707, 476], [707, 522]], label: "YES", tone: "yes" },
        { id: "e_hi_no", points: [[840, 425], [990, 425], [990, 452]], label: "NO", tone: "no", lx: 915, ly: 416 },
        { id: "e_wipe_6b", points: [[840, 558], [990, 558], [990, 518]], tone: "portal" }
      ],
      traces: [
        {
          id: "cut_perclos_fire",
          label: "Trace: MovingCalibrated >= 15% -> Wipe & GOTO 6B",
          steps: [
            {
              title: "Step 1: MovingCalibrated Gate Passed",
              highlightNodes: ["from_blink_perclos", "chk_face_lost", "gate_mov_cal", "active_perclos"],
              highlightEdges: ["e_in_chk", "e_fl_no", "e_mov_yes"],
              concept: "Because calibrated_ && SystemState == MovingCalibrated, PERCLOS records (!yawn && EAR <= T_close) and prunes > 30 s.",
              codeRef: "src/EarDetector.cpp (L186-218)"
            },
            {
              title: "Step 2: High PERCLOS Triggers Strong Cue & Restarts 30 s",
              highlightNodes: ["chk_high_gate", "emit_and_wipe", "goto_6b"],
              highlightEdges: ["e_act_chk", "e_hi_yes", "e_wipe_6b"],
              concept: "After 30 s warmup and >= 15.0% closure during a blink, PERCLOS emits a Strong cue, wipes its window via clear_perclos(), and enters 6B.",
              portalTarget: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
              codeRef: "src/EarDetector.cpp (L223-235) | src/PipelineCoordinator.cpp (L558)"
            }
          ]
        },
        {
          id: "cut_park_wipe",
          label: "Trace: Vehicle Parks (1A) -> Clear PERCLOS",
          steps: [
            {
              title: "Parking Immediately Wipes PERCLOS History",
              highlightNodes: ["from_park_clear", "gate_mov_cal", "parked_clear", "ret_parked"],
              highlightEdges: ["e_mov_no", "e_prk_ret"],
              concept: "Main/IMU park transitions call clear_perclos(), and every stationary frame continues clearing the window so old highway closures never carry over.",
              codeRef: "src/PipelineCoordinator.cpp (L340, L520, L668) | src/EarDetector.cpp (L248-262)"
            }
          ]
        },
        {
          id: "cut_lost_prune",
          label: "Trace: Face Lost -> Prune Without Fake Samples",
          steps: [
            {
              title: "Missing Face Ages Out Old Samples Cleanly",
              highlightNodes: ["from_blink_perclos", "chk_face_lost", "lost_prune", "ret_lost"],
              highlightEdges: ["e_in_chk", "e_fl_yes", "e_fl_ret"],
              concept: "During face loss or 500 ms Dlib throttling, on_face_lost() evicts entries older than 30 s without pushing false 0/1 samples.",
              codeRef: "src/EarDetector.cpp (L292-330)"
            }
          ]
        }
      ]
    }
  ]
});
