window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec3",
  title: "3. Blinking & PERCLOS",
  subviews: [
    {
      id: "3A",
      label: "3A. Blink Duration (Fused)",
      subtitle: "6-point eye ratio, cascaded 5-median + 40-MA adaptive thresholds (70% / 20%) & 500 ms micro-sleep",
      viewBox: "0 0 1080 580",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "Blink Duration & Micro-Sleep — Fused Algorithm View",
        concept: "Measures average eye aperture ratio from 6 landmarks per eye, adapts the open-eye baseline via a 5-frame median and 40-frame moving average whenever eyes stay open, and times closure below the 20% floor to flag a micro-sleep at 500 ms or longer.",
        code: "double dynamic_range = std::max(0.05, current_ear_open_ - ear_base_closed_);\nthresh_descent_ = ear_base_closed_ + 0.70 * dynamic_range;\nthresh_closure_ = std::max(floor_closure_, ear_base_closed_ + 0.20 * dynamic_range);\nif (closure_accum_sec_ >= 0.500 && !microsleep_alarm_active_) emit(BlinkDuration, Strong);",
        note: "Adaptive thresholds update only when not in a blink cycle and eye ratio stays above the 70% descent threshold, with at least 15 median samples collected.",
        codeRef: "src/FaceGeometry.cpp (L7-20)\nsrc/EarDetector.cpp (L85-182)\ninclude/Types.hpp (L222-233)"
      },
      nodes: [
        {
          id: "calc_ear", kind: "process", x: 20, y: 24, w: 220, h: 56,
          label: "6-Point Eye Ratio", sublabel: "Average both eyes & clamp step",
          details: {
            title: "Eye Aspect Ratio Calculation",
            concept: "Computes vertical eyelid separation divided by horizontal eye width for each eye and averages them, plus computes clamped frame delta time.",
            code: "double ear_raw = (ear_r + ear_l) * 0.5;\nconst double dt = MathUtils::compute_clamped_dt(now, last_time_, has_last_time_);",
            codeRef: "src/FaceGeometry.cpp (L7-20) | src/EarDetector.cpp (L92, L97)"
          }
        },
        {
          id: "chk_open_adapt", kind: "decision", x: 280, y: 12, w: 240, h: 82,
          label: "Eyes Open &\nAbove 70% Thresh?", sublabel: "Baseline adaptation gate",
          details: {
            title: "Open-Eye Adaptation Gate",
            concept: "Updates dynamic blink thresholds only while eyes are open and outside an active blink cycle so closing eyelids never pull thresholds down.",
            code: "if (!in_blink_cycle_ && ear_raw > thresh_descent_) { ... }",
            codeRef: "src/EarDetector.cpp (L100)"
          }
        },
        {
          id: "adapt_filter", kind: "process", x: 565, y: 24, w: 255, h: 56,
          label: "Cascaded Adaptive Filter", sublabel: "5-med -> 40-MA -> 70%/20% bounds",
          details: {
            title: "Cascaded Median & Moving Average Filter",
            concept: "Pushes open-eye samples into a 5-frame median deque and feeds medians into a 40-frame moving average (active once >= 15 medians exist) to update 70% descent and 20% closure thresholds.",
            code: "double ear_med = compute_median(buffer_raw_); // size 5\nbuffer_median_.push_back(ear_med); // cap 40, update if >= 15\ncurrent_ear_open_ = compute_mean(buffer_median_);\nthresh_closure_ = std::max(floor_closure_, ear_base_closed_ + 0.20 * range);",
            codeRef: "src/EarDetector.cpp (L101-126)"
          }
        },
        {
          id: "chk_descent", kind: "decision", x: 280, y: 132, w: 240, h: 82,
          label: "Eye Ratio Below\n70% Threshold?", sublabel: "Blink entry test",
          details: {
            title: "Blink Cycle Entry Check",
            concept: "Crossing below the 70% descent threshold begins a blink cycle; rising back at or above it finishes the blink.",
            code: "else if (ear_raw < thresh_descent_) { /* in blink cycle */ }",
            codeRef: "src/EarDetector.cpp (L137-182)"
          }
        },
        {
          id: "enter_blink", kind: "process", x: 20, y: 145, w: 220, h: 56,
          label: "Enter Blink Cycle", sublabel: "Start blink state if new",
          details: {
            title: "Initialize Blink Cycle",
            concept: "On the first frame below 70%, sets blink active and zeroes the closure timer.",
            code: "if (!in_blink_cycle_) {\n  in_blink_cycle_ = true; reached_effective_closure_ = false; closure_accum_sec_ = 0.0;\n}",
            codeRef: "src/EarDetector.cpp (L138-142)"
          }
        },
        {
          id: "chk_closure", kind: "decision", x: 20, y: 246, w: 220, h: 84,
          label: "At or Below\n20% Closure?", sublabel: "Effective eye shut",
          details: {
            title: "Effective Closure Check",
            concept: "Distinguishes true eyelid closure (at or below 20% aperture) from mid-blink hovering between 20% and 70%.",
            code: "if (ear_raw <= thresh_closure_) { ... }",
            codeRef: "src/EarDetector.cpp (L144)"
          }
        },
        {
          id: "acc_close", kind: "process", x: 20, y: 374, w: 220, h: 56,
          label: "Accumulate Closure", sublabel: "Add time step while shut",
          details: {
            title: "Closure Duration Timer",
            concept: "Starts the timer on the first closed frame and adds each subsequent frame time step while eyelids stay shut.",
            code: "if (!reached_effective_closure_) { reached_effective_closure_ = true; closure_accum_sec_ = dt; }\nelse { closure_accum_sec_ += dt; }",
            codeRef: "src/EarDetector.cpp (L145-150)"
          }
        },
        {
          id: "chk_microsleep", kind: "decision", x: 280, y: 360, w: 240, h: 84,
          label: "Shut >= 500 ms &\nNot Yet Active?", sublabel: "Strong micro-sleep",
          details: {
            title: "500 ms Micro-Sleep Threshold",
            concept: "Once uninterrupted eye closure reaches 500 ms, triggers an immediate Strong micro-sleep cue.",
            code: "if (closure_accum_sec_ >= 0.500 && !microsleep_alarm_active_) {\n  microsleep_alarm_active_ = true;\n  cues.push_back(CueEvent{CueType::BlinkDuration, CueStrength::Strong, now, ...});\n}",
            codeRef: "src/EarDetector.cpp (L152-167)"
          }
        },
        {
          id: "term_drowsy", kind: "term_alarm", x: 575, y: 376, w: 215, h: 52,
          label: "DROWSY", sublabel: "Strong: Micro-sleep >= 500 ms",
          details: {
            title: "Immediate Micro-Sleep Verdict",
            concept: "A single Strong cue immediately sets Driver State to Drowsy without waiting for any secondary cues.",
            code: "if (cue.strength == CueStrength::Strong) current_state_ = DriverState::Drowsy;",
            codeRef: "src/EarDetector.cpp (L155-165) | src/AlertManager.cpp (L213-219)"
          }
        },
        {
          id: "ret_closed_wait", kind: "term_return", x: 280, y: 492, w: 240, h: 50,
          label: "RETURN", sublabel: "Normal blink (< 500 ms)",
          details: {
            title: "Normal Blink in Progress",
            concept: "Returns cleanly while blink closure remains below 500 ms (or after micro-sleep has already latched).",
            code: "// Normal blink (< 500 ms) or already latched",
            codeRef: "src/EarDetector.cpp (L153-167)"
          }
        },
        {
          id: "hover_decay", kind: "process", x: 280, y: 260, w: 240, h: 56,
          label: "1.5x Mid-Lid Decay", sublabel: "Between 20% and 70% aperture",
          details: {
            title: "Partial Eyelid Hover Decay",
            concept: "When eyelids hover between 20% and 70% aperture, the timer drains smoothly at 1.5x speed rather than resetting abruptly.",
            code: "else {\n  closure_accum_sec_ = std::max(0.0, closure_accum_sec_ - 1.5 * dt);\n}",
            codeRef: "src/EarDetector.cpp (L168-171)"
          }
        },
        {
          id: "ret_hover", kind: "term_return", x: 575, y: 263, w: 195, h: 50,
          label: "RETURN", sublabel: "Partial closure decay",
          details: {
            title: "Partial Closure Return",
            concept: "Returns while keeping the blink cycle active until eyes reopen above 70%.",
            code: "return EarAssessment{..., in_blink_cycle_, cues};",
            codeRef: "src/EarDetector.cpp (L168-171)"
          }
        },
        {
          id: "blink_end", kind: "process", x: 575, y: 145, w: 245, h: 56,
          label: "Complete Blink Cycle", sublabel: "Save blink ms & clear flags",
          details: {
            title: "Finalize Blink Duration",
            concept: "When eye ratio climbs back above 70%, saves the completed blink duration in milliseconds and resets blink tracking flags.",
            code: "if (reached_effective_closure_) last_blink_duration_ms_ = closure_accum_sec_ * 1000.0;\nin_blink_cycle_ = false; microsleep_alarm_active_ = false; closure_accum_sec_ = 0.0;",
            codeRef: "src/EarDetector.cpp (L172-182)"
          }
        },
        {
          id: "ret_open", kind: "term_return", x: 865, y: 148, w: 185, h: 50,
          label: "RETURN", sublabel: "Eyes open",
          details: {
            title: "Open-Eye Return",
            concept: "Normal open-eye frame finishes.",
            code: "return EarAssessment{ear_raw, ..., cues};",
            codeRef: "src/EarDetector.cpp (L172-182)"
          }
        }
      ],
      edges: [
        { id: "e_ear_ad", points: [[240, 52], [280, 52]], tone: "default" },
        { id: "e_ad_yes", points: [[520, 52], [565, 52]], label: "YES", tone: "yes" },
        { id: "e_ad_no", points: [[400, 94], [400, 132]], label: "NO", tone: "no", lx: 416, ly: 113 },
        { id: "e_ad_desc", points: [[692, 80], [692, 108], [400, 108], [400, 132]], tone: "default" },
        { id: "e_desc_yes", points: [[280, 173], [240, 173]], label: "YES", tone: "yes" },
        { id: "e_desc_no", points: [[520, 173], [575, 173]], label: "NO", tone: "no" },
        { id: "e_end_ret", points: [[820, 173], [865, 173]], tone: "yes" },
        { id: "e_ent_cls", points: [[130, 201], [130, 246]], tone: "default" },
        { id: "e_cls_yes", points: [[130, 330], [130, 374]], label: "YES", tone: "yes", lx: 148, ly: 352 },
        { id: "e_cls_no", points: [[240, 288], [280, 288]], label: "NO", tone: "no" },
        { id: "e_hov_ret", points: [[520, 288], [575, 288]], tone: "default" },
        { id: "e_acc_mic", points: [[240, 402], [280, 402]], tone: "default" },
        { id: "e_mic_yes", points: [[520, 402], [575, 402]], label: "YES", tone: "yes" },
        { id: "e_mic_no", points: [[400, 444], [400, 492]], label: "NO", tone: "no", lx: 418, ly: 468 }
      ],
      traces: [
        {
          id: "open_adapt_trace",
          label: "Trace: Open Eye Drift -> Adaptive Filter",
          steps: [
            {
              title: "Step 1: Open Eyes Update Median & Moving Average",
              highlightNodes: ["calc_ear", "chk_open_adapt", "adapt_filter"],
              highlightEdges: ["e_ear_ad", "e_ad_yes"],
              concept: "While eyes remain open above 70%, the 5-sample median and 40-sample moving average adapt thresholds smoothly to natural baseline drift.",
              code: "if (!in_blink_cycle_ && ear_raw > thresh_descent_) { /* 5-med -> 40-MA */ }"
            },
            {
              title: "Step 2: Exit Open",
              highlightNodes: ["chk_descent", "blink_end", "ret_open"],
              highlightEdges: ["e_ad_desc", "e_desc_no", "e_end_ret"],
              concept: "Because eye ratio is above 70%, any completed blink duration is stored and the frame finishes cleanly.",
              code: "in_blink_cycle_ = false; closure_accum_sec_ = 0.0;"
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
              concept: "Eye ratio drops below 70% and 20%, accumulating 150 to 300 ms without crossing the 500 ms threshold.",
              code: "closure_accum_sec_ += dt; // < 0.500 s -> normal blink"
            },
            {
              title: "Step 2: Eyes Reopen & Save Duration",
              highlightNodes: ["chk_descent", "blink_end", "ret_open"],
              highlightEdges: ["e_desc_no", "e_end_ret"],
              concept: "Once eyes reopen past 70%, the completed blink duration is recorded in milliseconds and flags clear.",
              code: "last_blink_duration_ms_ = closure_accum_sec_ * 1000.0;"
            }
          ]
        },
        {
          id: "microsleep_trace",
          label: "Trace: Micro-Sleep (>= 500 ms) -> DROWSY",
          steps: [
            {
              title: "500 ms Closure Triggers Immediate Drowsiness",
              highlightNodes: ["enter_blink", "chk_closure", "acc_close", "chk_microsleep", "term_drowsy"],
              highlightEdges: ["e_ent_cls", "e_cls_yes", "e_acc_mic", "e_mic_yes"],
              concept: "Holding eyelids at or below 20% aperture for 500 ms or longer sets the micro-sleep latch and immediately flags Drowsy.",
              code: "if (closure_accum_sec_ >= 0.500 && !microsleep_alarm_active_) { /* Strong cue */ }"
            }
          ]
        },
        {
          id: "hover_trace",
          label: "Trace: Partial Lid Hover (1.5x Decay)",
          steps: [
            {
              title: "Mid-Aperture Hover Drains Timer Gradually",
              highlightNodes: ["enter_blink", "chk_closure", "hover_decay", "ret_hover"],
              highlightEdges: ["e_ent_cls", "e_cls_no", "e_hov_ret"],
              concept: "Between 20% and 70% aperture, the closure timer decays smoothly at 1.5x speed rather than resetting instantly.",
              code: "closure_accum_sec_ = std::max(0.0, closure_accum_sec_ - 1.5 * dt);"
            }
          ]
        }
      ]
    },
    {
      id: "3B",
      label: "3B. PERCLOS (Fused)",
      subtitle: "30-second rolling closure percentage, 30 s warmup, >= 15.0% High PERCLOS trigger & buffer wipe",
      viewBox: "0 0 1080 580",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "30-Second Rolling PERCLOS — Fused Algorithm View",
        concept: "Tracks the percentage of eye-closed frames across a rolling 30-second window. After a 30-second warmup, reaching 15% or more closure during an active blink (with a 5-second cooldown) flags High PERCLOS as a Strong Drowsy trigger and resets the window.",
        code: "int is_closed = (ear_raw <= thresh_closure_) ? 1 : 0;\nperclos_pct_ = (closed_samples_count_ / double(perclos_buffer_.size())) * 100.0;\nif (warmed_up && in_blink_cycle_ && cooldown >= 5s && perclos_pct_ >= 15.0) {\n  emit(Perclos, Strong); clear_perclos();\n}",
        note: "Only High PERCLOS (>= 15.0%) emits an alert in Morphed System 6; the Medium PERCLOS (7.5%-15.0%) tier is intentionally disabled.",
        codeRef: "src/EarDetector.cpp (L17-22, L184-247)\ninclude/Types.hpp (L230-232)"
      },
      nodes: [
        {
          id: "in_ear", kind: "process", x: 24, y: 34, w: 225, h: 56,
          label: "Eye Ratio & Threshold", sublabel: "From blink step + current time",
          details: {
            title: "PERCLOS Inputs",
            concept: "Consumes the current frame's eye aspect ratio, adaptive 20% closure floor, blink-active flag, and timestamp.",
            code: "// Inputs: ear_raw, thresh_closure_, in_blink_cycle_, now",
            codeRef: "src/EarDetector.cpp (L186-188)"
          }
        },
        {
          id: "push_sample", kind: "process", x: 290, y: 34, w: 240, h: 56,
          label: "Record Closure Sample", sublabel: "1 if shut <= 20%, else 0",
          details: {
            title: "Push Binary Closure Sample",
            concept: "Labels the frame closed or open against the 20% threshold, appends it to the deque, and updates the running closed counter in constant time.",
            code: "int is_closed = (ear_raw <= thresh_closure_) ? 1 : 0;\nperclos_buffer_.push_back({now, is_closed});\nif (is_closed == 1) ++closed_samples_count_;",
            codeRef: "src/EarDetector.cpp (L187-191)"
          }
        },
        {
          id: "prune_30s", kind: "process", x: 575, y: 34, w: 240, h: 56,
          label: "Prune Samples > 30 s Old", sublabel: "Maintain rolling 30 s window",
          details: {
            title: "Evict Expired Samples",
            concept: "Removes timestamps older than 30 seconds from the front of the window and decrements the running closed counter accordingly.",
            code: "auto cutoff = now - std::chrono::seconds(30);\nwhile (!perclos_buffer_.empty() && perclos_buffer_.front().first < cutoff) {\n  if (perclos_buffer_.front().second == 1) --closed_samples_count_;\n  perclos_buffer_.pop_front();\n}",
            codeRef: "src/EarDetector.cpp (L193-201)"
          }
        },
        {
          id: "calc_pct", kind: "process", x: 575, y: 132, w: 240, h: 56,
          label: "Compute Closure Percent", sublabel: "Closed count / total samples",
          details: {
            title: "PERCLOS Percentage & Warmup",
            concept: "Starts the warmup timer on the first recorded sample and computes the percentage of closed frames in the 30-second window.",
            code: "if (!has_perclos_start_) { perclos_start_time_ = now; has_perclos_start_ = true; }\nperclos_pct_ = (double(closed_samples_count_) / perclos_buffer_.size()) * 100.0;",
            codeRef: "src/EarDetector.cpp (L203-217)"
          }
        },
        {
          id: "chk_warmup", kind: "decision", x: 575, y: 230, w: 240, h: 84,
          label: "30-Second Warmup\nComplete?", sublabel: "Avoid early false spikes",
          details: {
            title: "30-Second Warmup Gate",
            concept: "Holds back PERCLOS alerts during the initial 30 seconds so a single blink cannot artificially dominate a short buffer.",
            code: "const bool perclos_warmed_up = (now - perclos_start_time_) >= std::chrono::seconds(30);",
            codeRef: "src/EarDetector.cpp (L208-211, L223)"
          }
        },
        {
          id: "ret_warmup", kind: "term_return", x: 865, y: 247, w: 190, h: 50,
          label: "RETURN", sublabel: "Warming up (< 30 s)",
          details: {
            title: "Warmup In Progress",
            concept: "Continues filling the 30-second window and reports remaining warmup seconds to the display.",
            code: "warmup_remaining_sec = duration<double>(30s - elapsed).count();",
            codeRef: "src/EarDetector.cpp (L210-211, L237-247)"
          }
        },
        {
          id: "chk_high_perclos", kind: "decision", x: 255, y: 226, w: 270, h: 92,
          label: "In Blink & >= 15%\n(5 s Cooldown)?", sublabel: "High PERCLOS check",
          details: {
            title: "High PERCLOS Trigger Gate",
            concept: "Fires when PERCLOS reaches 15% or higher while the driver is actively in a blink cycle and at least 5 seconds have elapsed since the previous PERCLOS cue.",
            code: "if (perclos_warmed_up && in_blink_cycle_ && elapsed_since_cue >= 5s && perclos_pct_ >= 15.0)",
            codeRef: "src/EarDetector.cpp (L223-225)"
          }
        },
        {
          id: "ret_normal", kind: "term_return", x: 24, y: 247, w: 185, h: 50,
          label: "RETURN", sublabel: "PERCLOS below 15%",
          details: {
            title: "Normal Closure Rate",
            concept: "Returns current closure percentage with no fatigue cue triggered.",
            code: "return EarAssessment{..., perclos_pct_, ..., {}}; // Empty cues",
            codeRef: "src/EarDetector.cpp (L237-247)"
          }
        },
        {
          id: "wipe_perclos", kind: "process", x: 255, y: 368, w: 270, h: 56,
          label: "Emit Strong Cue & Wipe", sublabel: "Clear buffer & restart 30 s",
          details: {
            title: "High PERCLOS Cue & Self-Wipe",
            concept: "Emits a Strong PERCLOS cue and immediately wipes the 30-second buffer, restarting warmup from scratch.",
            code: "last_perclos_cue_time_ = now;\ncues.push_back(CueEvent{CueType::Perclos, CueStrength::Strong, now, ...});\nclear_perclos();",
            codeRef: "src/EarDetector.cpp (L226-234)"
          }
        },
        {
          id: "term_drowsy", kind: "term_alarm", x: 585, y: 370, w: 225, h: 52,
          label: "DROWSY", sublabel: "Strong: High PERCLOS >= 15%",
          details: {
            title: "Drowsy Verdict",
            concept: "High PERCLOS is a Strong cue and raises Drowsy immediately.",
            code: "current_state_ = DriverState::Drowsy; // Strong cue immediate alarm",
            codeRef: "src/EarDetector.cpp (L227-232) | src/AlertManager.cpp (L213-219)"
          }
        }
      ],
      edges: [
        { id: "e_in_push", points: [[249, 62], [290, 62]], tone: "default" },
        { id: "e_push_pr", points: [[530, 62], [575, 62]], tone: "default" },
        { id: "e_pr_pct", points: [[695, 90], [695, 132]], tone: "default" },
        { id: "e_pct_wu", points: [[695, 188], [695, 230]], tone: "default" },
        { id: "e_wu_no", points: [[815, 272], [865, 272]], label: "NO", tone: "no" },
        { id: "e_wu_yes", points: [[575, 272], [525, 272]], label: "YES", tone: "yes" },
        { id: "e_hi_no", points: [[255, 272], [209, 272]], label: "NO", tone: "no" },
        { id: "e_hi_yes", points: [[390, 318], [390, 368]], label: "YES", tone: "yes" },
        { id: "e_wipe_al", points: [[525, 396], [585, 396]], tone: "yes" }
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
              concept: "During the initial 30 seconds, closed samples build in the rolling buffer while the warmup check withholds any alerts.",
              code: "const bool perclos_warmed_up = (elapsed >= std::chrono::seconds(30)); // false"
            }
          ]
        },
        {
          id: "perclos_trigger",
          label: "Trace: Warmed Up >= 15% -> DROWSY & Self-Wipe",
          steps: [
            {
              title: "High PERCLOS Fires Strong Cue & Restarts 30 s",
              highlightNodes: ["calc_pct", "chk_warmup", "chk_high_perclos", "wipe_perclos", "term_drowsy"],
              highlightEdges: ["e_pct_wu", "e_wu_yes", "e_hi_yes", "e_wipe_al"],
              concept: "After warmup, reaching 15% or more closure during a blink emits a Strong cue, wipes the buffer, and triggers Drowsy.",
              code: "cues.push_back(CueEvent{Perclos, Strong, ...}); clear_perclos();"
            }
          ]
        }
      ]
    },
    {
      id: "3C",
      label: "3C. Blink Duration (Cut-Off)",
      subtitle: "Calibration baseline, face-loss streak wipe, 0.2 s yawn suppression, handoff to 3D & moving gate",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "Blink Duration — System & FSM Cut-Off View",
        concept: "Connects blink and micro-sleep detection to surrounding runtime modules: 1C (landmarks vs. face-lost streak clearing), 1B (open/closed calibration thresholds), 4B (yawn suppression), internal continuation into PERCLOS (3D), and the moving-calibrated gate into cue fusion (6B).",
        code: "ear_res = landmarks ? ear_detector_.process(*landmarks, now, suppress_by_yawn, is_moving_cal)\n                    : ear_detector_.on_face_lost(now);\nif (is_moving_cal) alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
        note: "When parked and calibrated, live eye ratios and blink durations still update on screen while alerts remain gated off.",
        codeRef: "src/PipelineCoordinator.cpp (L491, L528-560)\nsrc/EarDetector.cpp (L42-182, L266-274)"
      },
      nodes: [
        {
          id: "from_calib_ear", kind: "portal_in", x: 20, y: 24, w: 220, h: 56,
          label: "FROM 1B: Apply Eye Base", sublabel: "Set 70% & 20% thresholds",
          target: { sectionId: "sec1", subviewId: "1B", highlightNodeId: "goto_3c_ear_base" },
          details: {
            title: "Calibration Eye Baseline",
            concept: "Seeds the median and moving-average buffers with the open-eye baseline, sets the closed-eye floor, and marks eye tracking calibrated.",
            code: "void EarDetector::apply_baseline(double ear_open, double ear_closed, time_point now) {\n  thresh_descent_ = ear_closed + 0.70 * range;\n  floor_closure_ = thresh_closure_ = ear_closed + 0.20 * range;\n  calibrated_ = true;\n}",
            codeRef: "src/EarDetector.cpp (L42-83)"
          }
        },
        {
          id: "from_face_gate", kind: "portal_in", x: 20, y: 110, w: 220, h: 56,
          label: "FROM 1C: Face Gate", sublabel: "68 landmarks or face lost",
          target: { sectionId: "sec1", subviewId: "1C", highlightNodeId: "out_landmarks" },
          details: {
            title: "Shared Landmark Input",
            concept: "Routes frames with verified landmarks to eye processing and missing-face frames to streak cleanup.",
            code: "if (landmarks_opt) ear_res = ear_detector_.process(...);\nelse ear_res = ear_detector_.on_face_lost(t_work_start);",
            codeRef: "src/PipelineCoordinator.cpp (L547-555)"
          }
        },
        {
          id: "from_yawn_sup", kind: "portal_in", x: 20, y: 240, w: 220, h: 56,
          label: "FROM 4B: Yawn Suppressor", sublabel: "Active when yawning >= 0.2 s",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "sig_yawn_eyes" },
          details: {
            title: "Yawn Suppression Signal",
            concept: "When yawning exceeds 0.2 seconds, squinting eyes are muted immediately before a 500 ms micro-sleep can trigger.",
            code: "const bool suppress_by_yawn = (yawn_res.yawn_duration_sec >= 0.2);",
            codeRef: "src/PipelineCoordinator.cpp (L528-531, L550) | src/EarDetector.cpp (L132-136)"
          }
        },
        {
          id: "chk_face", kind: "decision", x: 280, y: 96, w: 210, h: 84,
          label: "Landmarks\nPresent?", sublabel: "Face found vs lost",
          details: {
            title: "Face Presence Check",
            concept: "Branches between active eye tracking and missing-face streak cleanup.",
            code: "if (landmarks_opt.has_value()) { /* process */ } else { /* on_face_lost */ }",
            codeRef: "src/PipelineCoordinator.cpp (L547-555)"
          }
        },
        {
          id: "lost_wipe", kind: "process", x: 540, y: 24, w: 245, h: 56,
          label: "Wipe Closure Streak", sublabel: "Reset blink state on face loss",
          details: {
            title: "Face-Lost Streak Cleanup",
            concept: "Clears any partial blink or closure timer immediately so an unobserved missing-face gap can never bridge into a false micro-sleep.",
            code: "in_blink_cycle_ = false; reached_effective_closure_ = false;\nmicrosleep_alarm_active_ = false; closure_accum_sec_ = 0.0; has_last_time_ = false;",
            codeRef: "src/EarDetector.cpp (L266-274)"
          }
        },
        {
          id: "to_3d_lost", kind: "portal_out", x: 835, y: 24, w: 220, h: 56,
          label: "TO 3D: PERCLOS Lost", sublabel: "Prune old samples only",
          target: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "from_blink_perclos" },
          details: {
            title: "Continue to Face-Lost PERCLOS",
            concept: "Proceeds to PERCLOS's missing-face pruning branch without inserting fake closure frames.",
            code: "// Next inside on_face_lost(): prune < (now - 30s)",
            codeRef: "src/EarDetector.cpp (L275-330)"
          }
        },
        {
          id: "ear_adapt_proc", kind: "process", x: 280, y: 216, w: 210, h: 56,
          label: "Eye Ratio & Filter", sublabel: "5-med -> 40-MA adaptation",
          details: {
            title: "Adaptive Eye Ratio Filtering",
            concept: "Computes instantaneous eye aspect ratio and updates adaptive open-eye thresholds when eyes are open.",
            code: "double ear_raw = compute_ear(landmarks);\n// Update 5-median -> 40-MA if !in_blink_cycle_ && ear_raw > thresh_descent_",
            codeRef: "src/EarDetector.cpp (L92-129)"
          }
        },
        {
          id: "chk_yawn_sup", kind: "decision", x: 280, y: 312, w: 210, h: 84,
          label: "Yawn Suppressor\nActive?", sublabel: "Yawn >= 0.2 s mute",
          details: {
            title: "Yawn Suppression Gate",
            concept: "Checks whether the yawn detector has flagged suppression for squinting during yawns.",
            code: "if (suppress_closure) { /* reset blink & closure accum */ }",
            codeRef: "src/EarDetector.cpp (L132)"
          }
        },
        {
          id: "yawn_wipe", kind: "process", x: 540, y: 326, w: 235, h: 56,
          label: "Mute & Wipe Blink State", sublabel: "Clear squint timer & flags",
          details: {
            title: "Clear Blink State During Yawn",
            concept: "Cancels any in-progress eye-closure timer caused by squinting while yawning.",
            code: "in_blink_cycle_ = false; reached_effective_closure_ = false;\nmicrosleep_alarm_active_ = false; closure_accum_sec_ = 0.0;",
            codeRef: "src/EarDetector.cpp (L132-136)"
          }
        },
        {
          id: "blink_fsm_box", kind: "process", x: 280, y: 440, w: 210, h: 56,
          label: "Blink & 500 ms Micro-Sleep", sublabel: "Emit Strong cue if calibrated",
          details: {
            title: "Calibrated Micro-Sleep Evaluation",
            concept: "Runs the 500 ms closure timer and attaches a Strong Blink Duration cue only when calibration has completed.",
            code: "if (closure_accum_sec_ >= 0.500 && !microsleep_alarm_active_) {\n  microsleep_alarm_active_ = true;\n  if (calibrated_) cues.push_back(CueEvent{BlinkDuration, Strong, now, ...});\n}",
            codeRef: "src/EarDetector.cpp (L137-182)"
          }
        },
        {
          id: "to_3d_perclos", kind: "portal_out", x: 540, y: 440, w: 235, h: 56,
          label: "TO 3D: Rolling PERCLOS", sublabel: "Pass ratio, threshold & blink",
          target: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "from_blink_perclos" },
          details: {
            title: "Continue to Rolling PERCLOS",
            concept: "Continues sequentially into PERCLOS evaluation with the current eye ratio, 20% floor, and blink flag.",
            code: "// Next inside EarDetector::process(): section 5 (30 s PERCLOS)",
            codeRef: "src/EarDetector.cpp (L184-263)"
          }
        },
        {
          id: "read_mov_cal", kind: "decision", x: 825, y: 312, w: 230, h: 86,
          label: "READ 1A: Moving\n& Calibrated?", sublabel: "Lifecycle gate",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            title: "Vehicle Movement & Calibration Gate",
            concept: "The pipeline forwards eye assessments into cue fusion only while moving and calibrated.",
            code: "if (is_moving_calibrated) alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
            codeRef: "src/PipelineCoordinator.cpp (L557-559)"
          }
        },
        {
          id: "goto_6b", kind: "portal_out", x: 830, y: 440, w: 220, h: 56,
          label: "GOTO 6B: Cue Fusion", sublabel: "Send eye assessment",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
          details: {
            title: "Forward to Cue Fusion",
            concept: "Passes any Blink Duration or PERCLOS Strong cues into the mutex-protected Alert Coordinator.",
            code: "alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
            codeRef: "src/PipelineCoordinator.cpp (L558) | src/AlertManager.cpp (L289-294)"
          }
        },
        {
          id: "ret_hud_only", kind: "term_return", x: 830, y: 198, w: 220, h: 50,
          label: "RETURN (HUD Only)", sublabel: "Live eye & blink telemetry",
          details: {
            title: "Telemetry Only Return",
            concept: "Shows live eye ratios and last blink duration on screen while parked or uncalibrated without triggering alarms.",
            code: "debug_visualizer_.update_somnolence(..., ear_res, ...);",
            codeRef: "src/PipelineCoordinator.cpp (L557-575)"
          }
        }
      ],
      edges: [
        { id: "e_in_chk", points: [[240, 138], [280, 138]], tone: "default" },
        { id: "e_fl_yes", points: [[385, 96], [385, 52], [540, 52]], label: "NO", tone: "no", lx: 460, ly: 44 },
        { id: "e_fl_3d", points: [[785, 52], [835, 52]], tone: "portal" },
        { id: "e_ff_ad", points: [[385, 180], [385, 216]], label: "YES", tone: "yes" },
        { id: "e_ad_ys", points: [[385, 272], [385, 312]], tone: "default" },
        { id: "e_ys_in", points: [[240, 268], [260, 268], [260, 354], [280, 354]], tone: "portal" },
        { id: "e_ys_yes", points: [[490, 354], [540, 354]], label: "YES", tone: "yes" },
        { id: "e_ys_no", points: [[385, 396], [385, 440]], label: "NO", tone: "no" },
        { id: "e_yw_3d", points: [[657, 382], [657, 440]], tone: "portal" },
        { id: "e_bl_3d", points: [[490, 468], [540, 468]], tone: "portal" },
        { id: "e_3d_mov", points: [[775, 468], [800, 468], [800, 355], [825, 355]], tone: "default" },
        { id: "e_mov_no", points: [[940, 312], [940, 248]], label: "NO", tone: "no", lx: 958, ly: 280 },
        { id: "e_mov_yes", points: [[940, 398], [940, 440]], label: "YES", tone: "yes", lx: 960, ly: 419 }
      ],
      traces: [
        {
          id: "cut_microsleep",
          label: "Trace: Moving & Calibrated Micro-Sleep -> 3D -> 6B",
          steps: [
            {
              title: "Step 1: Landmarks & Thresholds Active",
              highlightNodes: ["from_calib_ear", "from_face_gate", "chk_face", "ear_adapt_proc", "chk_yawn_sup"],
              highlightEdges: ["e_in_chk", "e_ff_ad", "e_ad_ys"],
              concept: "Calibrated eye thresholds evaluate incoming landmarks while yawn suppression is inactive.",
              code: "ear_raw = compute_ear(landmarks); // suppress_by_yawn == false"
            },
            {
              title: "Step 2: 500 ms Cue -> PERCLOS (3D) -> Cue Fusion (6B)",
              highlightNodes: ["chk_yawn_sup", "blink_fsm_box", "to_3d_perclos", "read_mov_cal", "goto_6b"],
              highlightEdges: ["e_ys_no", "e_bl_3d", "e_3d_mov", "e_mov_yes"],
              concept: "A 500 ms closure emits a Strong Blink Duration cue, passes through PERCLOS (3D), and enters cue fusion (6B).",
              code: "cues.push_back(CueEvent{BlinkDuration, Strong, ...});\nif (is_moving_calibrated) alert_manager_.update_somnolence(...);",
              portalTarget: { sectionId: "sec3", subviewId: "3D", highlightNodeId: "from_blink_perclos" }
            }
          ]
        },
        {
          id: "cut_yawn_mute",
          label: "Trace: Yawn >= 0.2 s Wipes Blink/Squint State",
          steps: [
            {
              title: "Yawn Suppressor Clears Closure Timer",
              highlightNodes: ["from_yawn_sup", "chk_yawn_sup", "yawn_wipe", "to_3d_perclos"],
              highlightEdges: ["e_ys_in", "e_ys_yes", "e_yw_3d"],
              concept: "Natural squinting during yawning clears immediately once yawn duration reaches 0.2 s.",
              code: "if (suppress_closure) { in_blink_cycle_ = false; closure_accum_sec_ = 0.0; }"
            }
          ]
        },
        {
          id: "cut_face_lost",
          label: "Trace: Face Lost Clears Unconfirmed Closure Streak",
          steps: [
            {
              title: "Face Loss Resets Blink Streak Before PERCLOS",
              highlightNodes: ["from_face_gate", "chk_face", "lost_wipe", "to_3d_lost"],
              highlightEdges: ["e_in_chk", "e_fl_yes", "e_fl_3d"],
              concept: "Missing landmarks wipe unconfirmed blink streaks so an observation gap can never become a false micro-sleep.",
              code: "in_blink_cycle_ = false; reached_effective_closure_ = false; closure_accum_sec_ = 0.0;"
            }
          ]
        }
      ]
    },
    {
      id: "3D",
      label: "3D. PERCLOS (Cut-Off)",
      subtitle: "Moving & calibrated gate, park clear, face-lost pruning & disabled Medium tier",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "PERCLOS — System & FSM Cut-Off View",
        concept: "Unlike eye ratio and head posture, PERCLOS history is completely suppressed and cleared whenever the vehicle is parked or uncalibrated. Stopping from a moving calibrated state also wipes the rolling 30-second window.",
        code: "if (calibrated_ && record_perclos) { /* update 30 s window */ }\nelse { clear_perclos(); }",
        note: "Medium PERCLOS (7.5%-15.0%) is defined in configuration but disabled in code; only High PERCLOS (>= 15.0%) emits cues. Also note that parking clears PERCLOS directly from Main/IMU threads without a mutex.",
        codeRef: "src/PipelineCoordinator.cpp (L340, L515-522, L550, L668)\nsrc/EarDetector.cpp (L17-22, L184-331)\ninclude/Types.hpp (L230-232)"
      },
      nodes: [
        {
          id: "from_blink_perclos", kind: "portal_in", x: 20, y: 110, w: 225, h: 56,
          label: "FROM 3C: Blink Step", sublabel: "Eye ratio, floor or face lost",
          target: { sectionId: "sec3", subviewId: "3C", highlightNodeId: "to_3d_perclos" },
          details: {
            title: "Continuation from Blink Step",
            concept: "Enters directly after the blink and micro-sleep evaluation on either the face-found or face-lost branch.",
            code: "// Continues inside EarDetector::process() or EarDetector::on_face_lost()",
            codeRef: "src/EarDetector.cpp (L186, L275)"
          }
        },
        {
          id: "from_park_clear", kind: "portal_in", x: 20, y: 24, w: 225, h: 56,
          label: "FROM 1A: Clear PERCLOS", sublabel: "On park (Main / IMU)",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "goto_3d_clear" },
          details: {
            title: "Parking Cleanup Clear",
            concept: "When stopping from a moving calibrated state, Main and IMU threads clear the PERCLOS buffer.",
            code: "ear_detector_.clear_perclos();",
            note: "Concurrency caveat: called directly from Main (L340) and IMU (L668) without a mutex against the fatigue worker (which also has a fallback clear on L520).",
            codeRef: "src/PipelineCoordinator.cpp (L340, L520, L668) | src/EarDetector.cpp (L17-22)"
          }
        },
        {
          id: "from_yawn_sup", kind: "portal_in", x: 20, y: 342, w: 225, h: 56,
          label: "FROM 4B: Yawn Suppressor", sublabel: "Active when yawning >= 0.2 s",
          target: { sectionId: "sec4", subviewId: "4B", highlightNodeId: "sig_yawn_eyes" },
          details: {
            title: "Yawn Suppression Guard",
            concept: "During yawning (0.2 seconds or longer), frames do not count as closed and High PERCLOS cues are muted.",
            code: "int is_closed_sample = (!suppress_closure && ear_raw <= thresh_closure_) ? 1 : 0;",
            codeRef: "src/EarDetector.cpp (L187, L223)"
          }
        },
        {
          id: "chk_face_lost", kind: "decision", x: 285, y: 96, w: 215, h: 84,
          label: "Face Lost\nBranch?", sublabel: "Missing vs found",
          details: {
            title: "Face-Lost vs Active Branch",
            concept: "Routes face-lost frames to stale-sample pruning without adding artificial open or closed readings.",
            code: "// on_face_lost(now) vs process(landmarks, ...)",
            codeRef: "src/EarDetector.cpp (L275-330)"
          }
        },
        {
          id: "lost_prune", kind: "process", x: 550, y: 24, w: 250, h: 56,
          label: "Prune Samples > 30 s", sublabel: "No fake sample pushed",
          details: {
            title: "Face-Lost Window Pruning",
            concept: "If calibrated, evicts samples older than 30 seconds without adding new ones; resets the warmup clock if the window empties completely.",
            code: "while (!perclos_buffer_.empty() && perclos_buffer_.front().first < now - 30s) {\n  if (perclos_buffer_.front().second == 1) --closed_samples_count_;\n  perclos_buffer_.pop_front();\n}\nif (perclos_buffer_.empty()) has_perclos_start_ = false;",
            codeRef: "src/EarDetector.cpp (L275-330)"
          }
        },
        {
          id: "ret_lost", kind: "term_return", x: 850, y: 27, w: 200, h: 50,
          label: "RETURN (No Cue)", sublabel: "Updated closure %",
          details: {
            title: "Face-Lost Assessment Return",
            concept: "Returns the decayed PERCLOS percentage with no cues triggered.",
            code: "return EarAssessment{..., perclos_pct_, ..., {}}; // Empty cues",
            codeRef: "src/EarDetector.cpp (L320-330)"
          }
        },
        {
          id: "gate_mov_cal", kind: "decision", x: 285, y: 212, w: 215, h: 86,
          label: "Calibrated & READ 1A:\nMoving?", sublabel: "Lifecycle gate",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            title: "Moving & Calibrated Requirement",
            concept: "PERCLOS only records when both calibrated and actively moving; otherwise every frame clears the window.",
            code: "if (calibrated_ && record_perclos) { /* record_perclos == is_moving_calibrated */ }",
            codeRef: "src/PipelineCoordinator.cpp (L515, L550) | src/EarDetector.cpp (L186, L248-250)"
          }
        },
        {
          id: "parked_clear", kind: "process", x: 550, y: 227, w: 250, h: 56,
          label: "Clear PERCLOS Every Frame", sublabel: "Reset to 0% & 30 s warmup",
          details: {
            title: "Stationary / Uncalibrated Clear",
            concept: "Keeps the PERCLOS buffer empty, percentage at 0%, and warmup timer reset to 30 seconds whenever stopped.",
            code: "clear_perclos();\nreturn EarAssessment{..., 0.0, false, 30.0, ...};",
            codeRef: "src/EarDetector.cpp (L248-262)"
          }
        },
        {
          id: "ret_parked", kind: "term_return", x: 850, y: 230, w: 200, h: 50,
          label: "RETURN (0% Paused)", sublabel: "PERCLOS disabled",
          details: {
            title: "PERCLOS Disabled",
            concept: "Exits with zero PERCLOS accumulation while parked or uncalibrated.",
            code: "// Warmup remains 30.0 s while stationary",
            codeRef: "src/EarDetector.cpp (L252-262)"
          }
        },
        {
          id: "active_perclos", kind: "process", x: 285, y: 342, w: 215, h: 56,
          label: "Update 30 s Window", sublabel: "Count closed; prune > 30 s",
          details: {
            title: "Rolling 30-Second Update",
            concept: "Counts frame as closed only if not yawning and eye ratio is at or below 20%, prunes samples older than 30 seconds, and recalculates percentage.",
            code: "int is_closed = (!suppress_closure && ear_raw <= thresh_closure_) ? 1 : 0;\nperclos_pct_ = (double(closed_samples_count_) / perclos_buffer_.size()) * 100.0;",
            codeRef: "src/EarDetector.cpp (L187-218)"
          }
        },
        {
          id: "chk_high_gate", kind: "decision", x: 550, y: 326, w: 250, h: 88,
          label: "Warmed, In Blink,\n5 s Cool & >= 15%?", sublabel: "High PERCLOS trigger",
          details: {
            title: "High PERCLOS Trigger Check",
            concept: "Emits a Strong PERCLOS cue when all conditions hold (Medium 7.5% alerts remain disabled).",
            code: "if (perclos_warmed_up && !suppress_closure && in_blink_cycle_ &&\n    elapsed_since_cue >= 5s && perclos_pct_ >= 15.0)",
            note: "Medium PERCLOS (7.5%-15.0%) is defined in EarConfig but intentionally disabled in EarDetector.cpp.",
            codeRef: "src/EarDetector.cpp (L223-235)"
          }
        },
        {
          id: "emit_and_wipe", kind: "process", x: 550, y: 454, w: 250, h: 56,
          label: "Strong Cue & Self-Wipe", sublabel: "Clear buffer & restart 30 s",
          details: {
            title: "Emit Strong Cue & Restart Warmup",
            concept: "Attaches a Strong PERCLOS cue and immediately wipes the 30-second window.",
            code: "cues.push_back(CueEvent{CueType::Perclos, CueStrength::Strong, now, ...});\nclear_perclos();",
            codeRef: "src/EarDetector.cpp (L226-234)"
          }
        },
        {
          id: "goto_6b", kind: "portal_out", x: 850, y: 392, w: 200, h: 56,
          label: "GOTO 6B: Cue Fusion", sublabel: "Send eye assessment",
          target: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" },
          details: {
            title: "Forward to Cue Fusion",
            concept: "Passes the eye assessment (including any Blink Duration or PERCLOS Strong cues) into the Alert Coordinator.",
            code: "alert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
            codeRef: "src/PipelineCoordinator.cpp (L557-559) | src/AlertManager.cpp (L289-294)"
          }
        }
      ],
      edges: [
        { id: "e_in_chk", points: [[245, 138], [285, 138]], tone: "default" },
        { id: "e_fl_yes", points: [[392, 96], [392, 52], [550, 52]], label: "YES", tone: "yes", lx: 470, ly: 44 },
        { id: "e_fl_ret", points: [[800, 52], [850, 52]], tone: "default" },
        { id: "e_fl_no", points: [[392, 180], [392, 212]], label: "NO", tone: "no" },
        { id: "e_mov_no", points: [[500, 255], [550, 255]], label: "NO", tone: "no" },
        { id: "e_prk_ret", points: [[800, 255], [850, 255]], tone: "default" },
        { id: "e_mov_yes", points: [[392, 298], [392, 342]], label: "YES", tone: "yes" },
        { id: "e_yawn_in", points: [[245, 370], [285, 370]], tone: "portal" },
        { id: "e_act_chk", points: [[500, 370], [550, 370]], tone: "default" },
        { id: "e_hi_yes", points: [[675, 414], [675, 454]], label: "YES", tone: "yes" },
        { id: "e_hi_no", points: [[800, 370], [950, 370], [950, 392]], label: "NO", tone: "no", lx: 880, ly: 360 },
        { id: "e_wipe_6b", points: [[800, 482], [950, 482], [950, 448]], tone: "portal" }
      ],
      traces: [
        {
          id: "cut_perclos_fire",
          label: "Trace: Moving & Calibrated >= 15% -> Wipe & GOTO 6B",
          steps: [
            {
              title: "Step 1: Moving & Calibrated Gate Passed",
              highlightNodes: ["from_blink_perclos", "chk_face_lost", "gate_mov_cal", "active_perclos"],
              highlightEdges: ["e_in_chk", "e_fl_no", "e_mov_yes"],
              concept: "With calibration complete and vehicle moving, PERCLOS counts non-yawn closed samples and prunes entries older than 30 seconds.",
              code: "if (calibrated_ && is_moving_calibrated) { /* update 30 s window */ }"
            },
            {
              title: "Step 2: High PERCLOS Triggers Strong Cue & Restarts",
              highlightNodes: ["chk_high_gate", "emit_and_wipe", "goto_6b"],
              highlightEdges: ["e_act_chk", "e_hi_yes", "e_wipe_6b"],
              concept: "After 30 s warmup, hitting 15% or more closure during a blink emits a Strong cue, wipes the window, and forwards to cue fusion (6B).",
              code: "cues.push_back(CueEvent{Perclos, Strong, ...}); clear_perclos();\nalert_manager_.update_somnolence(ear_res, head_res, yawn_res);",
              portalTarget: { sectionId: "sec6", subviewId: "6B", highlightNodeId: "from_detectors" }
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
              concept: "Parking clears the window immediately, and every stationary frame keeps PERCLOS at zero so highway eye closures never carry over.",
              code: "ear_detector_.clear_perclos();"
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
              concept: "During missing-face intervals, old samples older than 30 seconds are evicted without pushing artificial open or closed readings.",
              code: "while (!perclos_buffer_.empty() && perclos_buffer_.front().first < now - 30s) { ... }"
            }
          ]
        }
      ]
    }
  ]
});
