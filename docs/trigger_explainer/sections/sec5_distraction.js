window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec5",
  title: "5. Distraction",
  subviews: [
    {
      id: "5A",
      label: "5A. Fused Algorithm",
      subtitle: "224x224 grayscale contrast & ImageNet norm -> ResNet-18 + 10-frame LSTM -> 54% vote -> 10f mode + 2.0 s timer",
      viewBox: "0 0 1080 580",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "Gaze Distraction — Fused Algorithm View",
        concept: "Normalizes each 224x224 frame with grayscale contrast enhancement and ImageNet scaling, extracts spatial features via ResNet-18 into a 10-frame circular buffer (20 frames under the unified preset), classifies road attention with an LSTM head (54% cutoff), and requires a 10-frame off-road majority sustained for 2.0 seconds to trigger Distracted.",
        code: "// 224x224 Gray CLAHE + ImageNet -> ResNet18 (512-d) -> LSTM -> Softmax\nGazeClass c = (p_looking >= 0.54f) ? LookingRoad : NotLookingRoad;\n// 2-stage filter: window.size() == 10 && not_looking > looking for >= 2000 ms",
        note: "Grayscale contrast enhancement converts the resized RGB frame to grayscale, applies CLAHE (clip 2.5, 8x8 tiles), and replicates it across 3 channels to match training.",
        codeRef: "src/Preprocessor.cpp (L33-84)\nsrc/GazeDetector.cpp (L99-190)\nsrc/AlertLogic.cpp (L5-49)\ninclude/Types.hpp (L57-61, L149-155, L172-179)"
      },
      nodes: [
        {
          id: "pre_clahe", kind: "process", x: 20, y: 26, w: 230, h: 58,
          label: "Preprocess 224x224 Frame", sublabel: "Grayscale CLAHE & ImageNet scale",
          details: {
            title: "Frame Preprocessing",
            concept: "Resizes the frame to 224x224, equalizes luminance contrast in grayscale, replicates across 3 channels, and applies ImageNet mean/std normalization.",
            code: "cv::resize(rgb, resized, Size(224, 224));\ncv::cvtColor(resized, gray, COLOR_RGB2GRAY); clahe_->apply(gray, clahe_gray);\n// Pack planar float32 (1 x 3 x 224 x 224): pixel * SCALE + SHIFT",
            codeRef: "src/Preprocessor.cpp (L33-84)"
          }
        },
        {
          id: "resnet_ring", kind: "process", x: 285, y: 26, w: 235, h: 58,
          label: "ResNet-18 + 10f Buffer", sublabel: "512-d feat; left-pad cold start",
          details: {
            title: "Spatial Backbone & Ring Buffer",
            concept: "Extracts a 512-dimensional feature vector per frame and writes it into a 10-frame circular buffer (20 frames for unified), padding empty startup slots with the earliest embedding.",
            code: "cv::Mat feat = net_backbone_.forward(); // 512-d\nmemcpy(ring_buffer_.data() + ring_head_ * 512, feat.ptr<float>(0), ...);\n// pack_sequence_blob(): pad leading slots with oldest_ptr when ring_count_ < seq_len",
            codeRef: "src/GazeDetector.cpp (L99-140)"
          }
        },
        {
          id: "lstm_softmax", kind: "process", x: 555, y: 26, w: 225, h: 58,
          label: "LSTM Head + Softmax", sublabel: "Score road-attention odds",
          details: {
            title: "Temporal LSTM Sequence Head",
            concept: "Feeds the 10-frame feature sequence into the LSTM head and applies numerically stable softmax to get road-looking vs. off-road probabilities.",
            code: "net_head_.setInput(seq_blob_);\ncv::Mat logits = net_head_.forward();\n// Softmax -> p_not_looking, p_looking",
            codeRef: "src/GazeDetector.cpp (L141-155)"
          }
        },
        {
          id: "chk_conf", kind: "decision", x: 820, y: 14, w: 235, h: 82,
          label: "Looking-Road Prob\n>= 54%?", sublabel: "Binarize policy",
          details: {
            title: "Per-Frame Confidence Policy",
            concept: "Under the default binarize policy, frames with road-looking probability of 54% or higher vote attentive; anything below votes off-road.",
            code: "assigned_class = (p_looking >= min_confidence) // 0.54f\n  ? GazeClass::LookingRoad : GazeClass::NotLookingRoad;",
            note: "Alternative CLI policies include Argmax (50/50 split) and FilterUncertain (ignoring ambiguous frames around 50%).",
            codeRef: "src/GazeDetector.cpp (L156-182)"
          }
        },
        {
          id: "push_mode", kind: "process", x: 820, y: 136, w: 235, h: 56,
          label: "Push to 10-Frame Window", sublabel: "Keep newest 10 votes (or 20)",
          details: {
            title: "Sliding Mode Window",
            concept: "Appends the single-frame vote to the sliding evaluation deque and discards the oldest vote beyond the window size.",
            code: "gaze_window_.push_back(detection);\nwhile (gaze_window_.size() > mode_frames) gaze_window_.pop_front();",
            codeRef: "src/AlertManager.cpp (L94-100)"
          }
        },
        {
          id: "chk_warmup", kind: "decision", x: 550, y: 123, w: 225, h: 82,
          label: "Window Has < 10\nFrames?", sublabel: "Mode warmup check",
          details: {
            title: "Sliding Window Warmup",
            concept: "Waits until the sliding window holds a full 10 votes (or 20 under unified) before running distraction decisions.",
            code: "if (gaze_window_.size() < mode_frames) return;",
            codeRef: "src/AlertManager.cpp (L102-105)"
          }
        },
        {
          id: "ret_warmup", kind: "term_return", x: 550, y: 244, w: 225, h: 50,
          label: "RETURN", sublabel: "Mode window warming up",
          details: {
            title: "Warmup Return",
            concept: "Exits early while the sliding vote buffer fills.",
            code: "return; // Waiting for mode_frames predictions",
            codeRef: "src/AlertManager.cpp (L103-105)"
          }
        },
        {
          id: "chk_stage1", kind: "decision", x: 265, y: 122, w: 240, h: 84,
          label: "Stage 1: Off-Road\n> On-Road Votes?", sublabel: "Sliding mode majority",
          details: {
            title: "Stage 1 Sliding Mode Majority",
            concept: "Requires strictly more off-road votes than road-looking votes across the 10-frame sliding window.",
            code: "assessment.is_distracted = (not_looking_count > looking_count);",
            codeRef: "src/AlertLogic.cpp (L5-49)"
          }
        },
        {
          id: "reset_timer", kind: "process", x: 20, y: 136, w: 210, h: 56,
          label: "Reset Sustained Timer", sublabel: "Return gaze to road",
          details: {
            title: "Reset Sustained Distraction Timer",
            concept: "Clears the continuous off-road timestamp immediately whenever road attention recovers or votes tie.",
            code: "distracted_since_.reset();\nif (current_state_ == DriverState::Distracted) current_state_ = DriverState::Safe;",
            codeRef: "src/AlertManager.cpp (L123-144)"
          }
        },
        {
          id: "ret_attentive", kind: "term_return", x: 20, y: 244, w: 210, h: 50,
          label: "RETURN", sublabel: "Looking at road (Safe)",
          details: {
            title: "Attentive Safe Return",
            concept: "Driver is attentive on the road; no distraction alarm triggers.",
            code: "// State remains Safe",
            codeRef: "src/AlertManager.cpp (L138-144)"
          }
        },
        {
          id: "hold_timer", kind: "process", x: 265, y: 244, w: 240, h: 56,
          label: "Start / Hold 2.0 s Timer", sublabel: "Track continuous off-road span",
          details: {
            title: "Track Continuous Distraction",
            concept: "Records the start timestamp on the first off-road majority frame and measures elapsed duration.",
            code: "if (!distracted_since_.has_value()) distracted_since_ = now;\nconst auto sustained = duration_cast<milliseconds>(now - *distracted_since_);",
            codeRef: "src/AlertManager.cpp (L130-133)"
          }
        },
        {
          id: "chk_stage2", kind: "decision", x: 265, y: 342, w: 240, h: 86,
          label: "Stage 2: Sustained\n>= 2.0 Seconds?", sublabel: "Continuous off-road",
          details: {
            title: "Stage 2 Continuous Duration Gate",
            concept: "Requires off-road mode to hold continuously for at least 2.0 seconds so quick mirror checks do not alarm.",
            code: "if (sustained >= config_.distraction_window) { // 2000 ms\n  current_state_ = DriverState::Distracted; trigger_distraction_alert = true;\n}",
            codeRef: "src/AlertManager.cpp (L134-137)"
          }
        },
        {
          id: "ret_counting", kind: "term_return", x: 20, y: 360, w: 210, h: 50,
          label: "RETURN", sublabel: "Brief glance (< 2.0 s)",
          details: {
            title: "Under 2-Second Return",
            concept: "Short off-road glances under 2.0 seconds return without raising an alarm.",
            code: "// sustained < 2000 ms: continue counting",
            codeRef: "src/AlertManager.cpp (L134-137)"
          }
        },
        {
          id: "term_distracted", kind: "term_alarm", x: 555, y: 359, w: 235, h: 52,
          label: "DISTRACTED", sublabel: "Wipes window & timer after beep",
          details: {
            title: "Distraction Verdict",
            concept: "Marks the driver Distracted, dispatches the warning, and resets the 10-frame window and timer once audio finishes.",
            code: "current_state_ = DriverState::Distracted;\n// After audio: gaze_window_.clear(); distracted_since_.reset();",
            codeRef: "src/AlertManager.cpp (L135-136, L187-192)"
          }
        }
      ],
      edges: [
        { id: "e_pre_res", points: [[250, 55], [285, 55]], tone: "default" },
        { id: "e_res_lst", points: [[520, 55], [555, 55]], tone: "default" },
        { id: "e_lst_cnf", points: [[780, 55], [820, 55]], tone: "default" },
        { id: "e_cnf_psh", points: [[937, 96], [937, 136]], label: "vote", tone: "default" },
        { id: "e_psh_wu", points: [[820, 164], [775, 164]], tone: "default" },
        { id: "e_wu_yes", points: [[662, 205], [662, 244]], label: "YES", tone: "yes" },
        { id: "e_wu_no", points: [[550, 164], [505, 164]], label: "NO", tone: "no" },
        { id: "e_s1_no", points: [[265, 164], [230, 164]], label: "NO", tone: "no" },
        { id: "e_rst_ret", points: [[125, 192], [125, 244]], tone: "default" },
        { id: "e_s1_yes", points: [[385, 206], [385, 244]], label: "YES", tone: "yes" },
        { id: "e_tmr_s2", points: [[385, 300], [385, 342]], tone: "default" },
        { id: "e_s2_no", points: [[265, 385], [230, 385]], label: "NO", tone: "no" },
        { id: "e_s2_yes", points: [[505, 385], [555, 385]], label: "YES", tone: "yes" }
      ],
      traces: [
        {
          id: "gaze_warmup",
          label: "Trace: Cold-Start Padding -> 10f Warmup -> Attentive",
          steps: [
            {
              title: "Step 1: Warmup Padding & 10-Frame Buffer Fill",
              highlightNodes: ["pre_clahe", "resnet_ring", "lstm_softmax", "chk_conf", "push_mode", "chk_warmup", "ret_warmup"],
              highlightEdges: ["e_pre_res", "e_res_lst", "e_lst_cnf", "e_cnf_psh", "e_psh_wu", "e_wu_yes"],
              concept: "During startup, ResNet-18 features left-pad the 10-slot sequence buffer, and the sliding vote window fills to 10 frames before Stage 1 runs.",
              code: "if (ring_count_ < seq_len) pad_leading_with(oldest);\nif (gaze_window_.size() < mode_frames) return;"
            },
            {
              title: "Step 2: On-Road Majority Keeps Driver Safe",
              highlightNodes: ["chk_stage1", "reset_timer", "ret_attentive"],
              highlightEdges: ["e_wu_no", "e_s1_no", "e_rst_ret"],
              concept: "With attentive votes at or above off-road votes, the sustained timer stays clear and driver state remains Safe.",
              code: "assessment.is_distracted = false; distracted_since_.reset();"
            }
          ]
        },
        {
          id: "gaze_sustained",
          label: "Trace: 10f Majority Held >= 2.0 s -> DISTRACTED",
          steps: [
            {
              title: "Stage 1 Majority Held 2.0 Seconds Triggers DISTRACTED",
              highlightNodes: ["chk_conf", "push_mode", "chk_warmup", "chk_stage1", "hold_timer", "chk_stage2", "term_distracted"],
              highlightEdges: ["e_cnf_psh", "e_psh_wu", "e_wu_no", "e_s1_yes", "e_tmr_s2", "e_s2_yes"],
              concept: "When off-road majority persists continuously for at least 2.0 seconds, the system declares Distracted.",
              code: "if (not_looking > looking && sustained >= 2000ms) current_state_ = DriverState::Distracted;"
            }
          ]
        },
        {
          id: "gaze_glance",
          label: "Trace: Brief Mirror Check (< 2.0 s) -> Timer Clears",
          steps: [
            {
              title: "Short Glance Resets Before 2.0 Seconds",
              highlightNodes: ["chk_stage1", "hold_timer", "chk_stage2", "ret_counting", "reset_timer", "ret_attentive"],
              highlightEdges: ["e_s1_yes", "e_tmr_s2", "e_s2_no", "e_s1_no", "e_rst_ret"],
              concept: "Looking away under 2.0 seconds does not alarm, and returning eyes to the road immediately clears the timer.",
              code: "// sustained < 2000 ms -> next attentive frame clears distracted_since_"
            }
          ]
        }
      ]
    },
    {
      id: "5B",
      label: "5B. System & FSM Cut-Off",
      subtitle: "Strict C++ cut-off at Gaze Detector: atomic reset, Moving-Calibrated gate & single vote -> GOTO 6A",
      viewBox: "0 0 1080 580",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "Distraction — System & FSM Cut-Off View",
        concept: "Captures the exact worker and class boundary on the Gaze Worker thread: consumes the atomic reset flag safely in-thread, skips all neural work when not moving and calibrated, runs ResNet-18 plus LSTM, and cuts off right after emitting a single frame vote into the Alert Coordinator (6A).",
        code: "if (gaze_reset_requested_.exchange(false, acq_rel)) gaze_detector_.reset();\nif (system_state_.load() != SystemState::MovingCalibrated) continue;\nauto res = gaze_detector_.infer_frame(blob, min_conf, policy);\nalert_manager_.update_gaze(res);",
        note: "Unlike the Fatigue Worker (which keeps HUD telemetry running when parked), the Gaze Worker performs zero neural inference whenever parked, uncalibrated, or calibrating.",
        codeRef: "src/PipelineCoordinator.cpp (L69-108, L331, L341, L399-457, L669, L677)\nsrc/GazeDetector.cpp (L94-190)"
      },
      nodes: [
        {
          id: "from_atomic_reset", kind: "portal_in", x: 24, y: 26, w: 225, h: 56,
          label: "FROM 1A: Gaze Reset Flag", sublabel: "Raised on park by Main / IMU",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "set_gaze_atomic" },
          details: {
            title: "Atomic Reset Signal",
            concept: "Main and IMU threads set an atomic flag on parking rather than touching Gaze Detector's circular buffer across threads.",
            code: "gaze_reset_requested_.store(true, std::memory_order_release);",
            codeRef: "src/PipelineCoordinator.cpp (L331, L341, L669, L677)"
          }
        },
        {
          id: "chk_atomic", kind: "decision", x: 290, y: 12, w: 245, h: 84,
          label: "Atomic Reset Flag\nRaised?", sublabel: "Checked pre & post wait",
          details: {
            title: "Atomic Reset Exchange",
            concept: "Atomically reads and clears the reset request both before and immediately after waiting on the latest camera frame.",
            code: "if (gaze_reset_requested_.exchange(false, std::memory_order_acq_rel))\n  gaze_detector_.reset();",
            codeRef: "src/PipelineCoordinator.cpp (L412-426)"
          }
        },
        {
          id: "wipe_ring", kind: "process", x: 585, y: 26, w: 230, h: 56,
          label: "Clear Sequence Buffer", sublabel: "Reset circular buffer in-thread",
          details: {
            title: "Thread-Confined Ring Wipe",
            concept: "Zeroes ring indices safely on the Gaze Worker's own thread with zero mutex overhead.",
            code: "void GazeDetector::reset() noexcept {\n  ring_head_ = 0; ring_count_ = 0;\n}",
            codeRef: "src/PipelineCoordinator.cpp (L412-426) | src/GazeDetector.cpp (L94-97)"
          }
        },
        {
          id: "gate_mov_cal", kind: "decision", x: 290, y: 138, w: 245, h: 86,
          label: "READ 1A: Moving\n& Calibrated?", sublabel: "Lifecycle inference gate",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            title: "Moving & Calibrated Gate",
            concept: "Discards incoming camera frames immediately unless the vehicle is both moving and calibrated.",
            code: "if (system_state_.load() != SystemState::MovingCalibrated) continue;",
            codeRef: "src/PipelineCoordinator.cpp (L428-431)"
          }
        },
        {
          id: "skip_frame", kind: "term_return", x: 585, y: 156, w: 230, h: 50,
          label: "SKIP FRAME", sublabel: "Zero neural compute",
          details: {
            title: "Idle Frame Skip",
            concept: "Skips preprocessing and both neural models completely when stationary, uncalibrated, or calibrating.",
            code: "continue; // Skip Preprocessor, ResNet-18, and LSTM",
            codeRef: "src/PipelineCoordinator.cpp (L429-431)"
          }
        },
        {
          id: "pre_cut", kind: "process", x: 290, y: 268, w: 245, h: 56,
          label: "Preprocess 224x224 Frame", sublabel: "Grayscale CLAHE & ImageNet",
          details: {
            title: "Frame Preprocessing",
            concept: "Prepares the normalized 224x224 tensor into a pre-allocated buffer.",
            code: "preprocessor_.process(frame, input_blob);",
            codeRef: "src/PipelineCoordinator.cpp (L434) | src/Preprocessor.cpp (L33-84)"
          }
        },
        {
          id: "net_cut", kind: "process", x: 290, y: 368, w: 245, h: 58,
          label: "Run ResNet-18 + LSTM", sublabel: "Update 10f/20f ring & softmax",
          details: {
            title: "Spatial-Temporal Inference",
            concept: "Extracts spatial features into the sequence ring buffer, runs the LSTM head, and computes attention probabilities.",
            code: "DistractionDetectionResult result = gaze_detector_.infer_frame(\n  input_blob, alert_config_.min_confidence, alert_config_.policy);",
            codeRef: "src/PipelineCoordinator.cpp (L436-440) | src/GazeDetector.cpp (L121-155)"
          }
        },
        {
          id: "policy_cut", kind: "process", x: 585, y: 368, w: 230, h: 58,
          label: "Assign Single-Frame Vote", sublabel: "Default 54% cutoff -> 1 vote",
          details: {
            title: "Per-Frame Vote Classification",
            concept: "Applies the confidence threshold policy to emit a single frame vote and probability.",
            code: "return DistractionDetectionResult{assigned_class, p_looking, now};",
            codeRef: "src/GazeDetector.cpp (L156-189)"
          }
        },
        {
          id: "goto_6a", kind: "portal_out", x: 855, y: 369, w: 200, h: 56,
          label: "GOTO 6A: Gaze Filter", sublabel: "Send 1 vote to mutex",
          target: { sectionId: "sec6", subviewId: "6A", highlightNodeId: "from_gaze_vote" },
          details: {
            title: "Handoff to Alert Manager",
            concept: "Strict boundary cut-off: Gaze Detector outputs one vote per frame and hands off to Alert Manager (6A) for the 10-frame sliding mode, 2-second timer, drowsiness preemption, and audio.",
            code: "alert_manager_.update_gaze(result);",
            codeRef: "src/PipelineCoordinator.cpp (L443) | src/AlertManager.cpp (L83-145)"
          }
        }
      ],
      edges: [
        { id: "e_at_in", points: [[249, 54], [290, 54]], tone: "portal" },
        { id: "e_at_yes", points: [[535, 54], [585, 54]], label: "YES", tone: "yes" },
        { id: "e_at_no", points: [[412, 96], [412, 138]], label: "NO", tone: "no" },
        { id: "e_wipe_gt", points: [[700, 82], [700, 114], [412, 114], [412, 138]], tone: "default" },
        { id: "e_mov_no", points: [[535, 181], [585, 181]], label: "NO", tone: "no" },
        { id: "e_mov_yes", points: [[412, 224], [412, 268]], label: "YES", tone: "yes" },
        { id: "e_pre_net", points: [[412, 324], [412, 368]], tone: "default" },
        { id: "e_net_pol", points: [[535, 397], [585, 397]], tone: "default" },
        { id: "e_pol_6a", points: [[815, 397], [855, 397]], tone: "portal" }
      ],
      traces: [
        {
          id: "gaze_cut_active",
          label: "Trace: Moving & Calibrated -> Single Vote -> GOTO 6A",
          steps: [
            {
              title: "Step 1: Reset Clear & Moving-Calibrated Passed",
              highlightNodes: ["chk_atomic", "gate_mov_cal", "pre_cut"],
              highlightEdges: ["e_at_no", "e_mov_yes"],
              concept: "With no reset pending and vehicle moving calibrated, the frame enters image preprocessing.",
              code: "preprocessor_.process(frame, input_blob);"
            },
            {
              title: "Step 2: Single-Frame Vote Enters Alert Manager (6A)",
              highlightNodes: ["pre_cut", "net_cut", "policy_cut", "goto_6a"],
              highlightEdges: ["e_pre_net", "e_net_pol", "e_pol_6a"],
              concept: "ResNet-18 plus LSTM produces one attention vote and hands it to the Alert Coordinator (6A).",
              code: "alert_manager_.update_gaze(result);",
              portalTarget: { sectionId: "sec6", subviewId: "6A", highlightNodeId: "from_gaze_vote" }
            }
          ]
        },
        {
          id: "gaze_cut_skip",
          label: "Trace: Parked / Uncalibrated -> Skip Frame",
          steps: [
            {
              title: "Zero Neural Compute When Not Driving Calibrated",
              highlightNodes: ["gate_mov_cal", "skip_frame"],
              highlightEdges: ["e_mov_no"],
              concept: "Whenever parked, uncalibrated, or calibrating, the Gaze Worker skips all image preprocessing and neural inference.",
              code: "if (system_state_.load() != SystemState::MovingCalibrated) continue;"
            }
          ]
        },
        {
          id: "gaze_cut_reset",
          label: "Trace: Park -> Atomic Reset Wipes Sequence Buffer",
          steps: [
            {
              title: "In-Thread Sequence Wipe on Next Wakeup",
              highlightNodes: ["from_atomic_reset", "chk_atomic", "wipe_ring", "gate_mov_cal", "skip_frame"],
              highlightEdges: ["e_at_in", "e_at_yes", "e_wipe_gt", "e_mov_no"],
              concept: "Stopping raises the atomic reset flag (1A); the Gaze Worker consumes it safely on its own thread, resets its ring buffer, and skips inference.",
              code: "if (gaze_reset_requested_.exchange(false, acq_rel)) gaze_detector_.reset();"
            }
          ]
        }
      ]
    }
  ]
});
