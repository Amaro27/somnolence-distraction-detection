window.TRIGGER_SECTIONS = window.TRIGGER_SECTIONS || [];
window.TRIGGER_SECTIONS.push({
  id: "sec5",
  title: "5. Distraction",
  subviews: [
    {
      id: "5A",
      label: "5A. Fused Algorithm",
      subtitle: "224x224 grayscale CLAHE + ImageNet -> ResNet-18 + 10f LSTM -> P>=0.54 -> 10f mode + 2.0 s timer",
      viewBox: "0 0 1120 660",
      legend: ["process", "decision", "return", "alarm", "yes", "no"],
      overview: {
        title: "5A: Gaze Distraction — Fused Algorithm View",
        concept: "End-to-end algorithmic flow from BGR frame through grayscale-replicated CLAHE and ImageNet normalization, ResNet-18 spatial embedding + 10-frame (or 20-frame) LSTM sequence inference, per-frame BinarizeThreshold classification (P(looking) >= 0.54), and the 2-stage temporal filter (10f majority mode + 2.0 s continuous duration).",
        note: "Preprocessor.cpp L40-43 converts resized RGB to 1-channel grayscale, runs CLAHE (clipLimit=2.5, 8x8 tiles), and converts GRAY2RGB before ImageNet channel affine scaling. Default preset is preopt (10 frames); --unified uses 20 frames.",
        codeRef: "src/Preprocessor.cpp (L33-84)\nsrc/GazeDetector.cpp (L99-190)\nsrc/AlertLogic.cpp (L5-49)\ninclude/Types.hpp (L57-61, L149-155, L172-179)"
      },
      nodes: [
        {
          id: "pre_clahe", kind: "process", x: 20, y: 26, w: 240, h: 78,
          label: "Preprocess 224x224 NCHW", sublabel: "BGR->RGB->224x224->Gray\nCLAHE(2.5, 8x8)->GRAY2RGB\nImageNet (x/255 - mean)/std",
          details: {
            concept: "Normalizes frame lighting via grayscale CLAHE replicated across 3 channels, then packs NCHW float32 tensor (1x3x224x224).",
            codeRef: "src/Preprocessor.cpp (L33-84)"
          }
        },
        {
          id: "resnet_ring", kind: "process", x: 295, y: 26, w: 255, h: 78,
          label: "ResNet-18 + Ring Buffer", sublabel: "512-d feat -> 10f ring buf\n(20f if --unified); left-pad\nwarmup with earliest feat",
          details: {
            concept: "Extracts a 512-d spatial feature vector and stores it in a pre-allocated ring buffer, padding unfilled leading slots during cold start with the oldest vector.",
            codeRef: "src/GazeDetector.cpp (L99-140)"
          }
        },
        {
          id: "lstm_softmax", kind: "process", x: 585, y: 26, w: 235, h: 78,
          label: "LSTM Head + Softmax", sublabel: "Forward (1x10x512)\nP_not_look, P_look = softmax",
          details: {
            concept: "Evaluates the temporal feature sequence with the LSTM ONNX head and applies numerically stable softmax.",
            codeRef: "src/GazeDetector.cpp (L141-155)"
          }
        },
        {
          id: "chk_conf", kind: "decision", x: 855, y: 18, w: 240, h: 94,
          label: "P(looking) >= 0.54?", sublabel: "Binarize policy",
          details: {
            concept: "Assigns per-frame LookingRoad if P(looking) >= 0.54, else NotLookingRoad (Argmax & FilterUncertain policies also supported via flags).",
            codeRef: "src/GazeDetector.cpp (L156-182)"
          }
        },
        {
          id: "push_mode", kind: "process", x: 855, y: 156, w: 240, h: 72,
          label: "Push Vote to 10f Window", sublabel: "gaze_window_.push_back()\nTrim oldest if > 10 (or 20)",
          details: {
            concept: "Maintains a sliding window of the latest mode_window_frames classifications (default matches seq_len = 10).",
            codeRef: "src/AlertManager.cpp (L94-100)"
          }
        },
        {
          id: "chk_warmup", kind: "decision", x: 580, y: 146, w: 235, h: 92,
          label: "window.size() < 10?", sublabel: "Sliding window warmup",
          details: {
            concept: "Waits until mode_window_frames predictions accumulate before evaluating distraction mode.",
            codeRef: "src/AlertManager.cpp (L102-105)"
          }
        },
        {
          id: "ret_warmup", kind: "term_return", x: 580, y: 282, w: 235, h: 54,
          label: "RETURN", sublabel: "Mode window warming up",
          details: {
            concept: "Returns without evaluating Stage 1 or Stage 2 until 10 frames fill the window.",
            codeRef: "src/AlertManager.cpp (L103-105)"
          }
        },
        {
          id: "chk_stage1", kind: "decision", x: 280, y: 144, w: 255, h: 96,
          label: "Stage 1: Off-Road Votes\n> On-Road Votes?", sublabel: "Sliding mode majority",
          details: {
            concept: "evaluate_distraction_mode() checks strict majority: not_looking_votes > looking_votes across the 10-frame window.",
            codeRef: "src/AlertLogic.cpp (L5-49)"
          }
        },
        {
          id: "reset_timer", kind: "process", x: 20, y: 156, w: 220, h: 72,
          label: "Reset Sustained Timer", sublabel: "distracted_since_.reset()\nReturn to attentive state",
          details: {
            concept: "When gaze returns to the road (or ties 5-5), the 2.0 s continuous distraction timer resets immediately.",
            codeRef: "src/AlertManager.cpp (L123-144)"
          }
        },
        {
          id: "ret_attentive", kind: "term_return", x: 20, y: 282, w: 220, h: 54,
          label: "RETURN", sublabel: "Looking at road (Safe)",
          details: {
            concept: "Driver is attentive; no distraction alarm emitted.",
            codeRef: "src/AlertManager.cpp (L138-144)"
          }
        },
        {
          id: "hold_timer", kind: "process", x: 280, y: 284, w: 255, h: 70,
          label: "Start / Hold 2.0 s Timer", sublabel: "If unset: distracted_since = now\nelapsed = now - distracted_since",
          details: {
            concept: "Starts the continuous off-road timer on the first NotLookingRoad majority window.",
            codeRef: "src/AlertManager.cpp (L130-133)"
          }
        },
        {
          id: "chk_stage2", kind: "decision", x: 280, y: 396, w: 255, h: 98,
          label: "Stage 2: elapsed\n>= 2000 ms?", sublabel: "Continuous distraction",
          details: {
            concept: "Requires the sliding mode to stay off-road continuously for 2.0 seconds before alarming.",
            codeRef: "src/AlertManager.cpp (L134-137)"
          }
        },
        {
          id: "ret_counting", kind: "term_return", x: 20, y: 418, w: 220, h: 54,
          label: "RETURN", sublabel: "Off-road (< 2.0 s)",
          details: {
            concept: "Brief mirror or dashboard check under 2.0 s does not trigger an alarm.",
            codeRef: "src/AlertManager.cpp (L134-137)"
          }
        },
        {
          id: "term_distracted", kind: "term_alarm", x: 585, y: 415, w: 240, h: 62,
          label: "DISTRACTED", sublabel: "Clear window + timer after beep",
          details: {
            concept: "Triggers distraction warning, then clears gaze_window_ and distracted_since_ after playback finishes.",
            codeRef: "src/AlertManager.cpp (L135-136, L187-192)"
          }
        }
      ],
      edges: [
        { id: "e_pre_res", points: [[260, 65], [295, 65]], tone: "default" },
        { id: "e_res_lst", points: [[550, 65], [585, 65]], tone: "default" },
        { id: "e_lst_cnf", points: [[820, 65], [855, 65]], tone: "default" },
        { id: "e_cnf_psh", points: [[975, 112], [975, 156]], label: "vote", tone: "default" },
        { id: "e_psh_wu", points: [[855, 192], [815, 192]], tone: "default" },
        { id: "e_wu_yes", points: [[697, 238], [697, 282]], label: "YES", tone: "yes" },
        { id: "e_wu_no", points: [[580, 192], [535, 192]], label: "NO", tone: "no" },
        { id: "e_s1_no", points: [[280, 192], [240, 192]], label: "NO", tone: "no" },
        { id: "e_rst_ret", points: [[130, 228], [130, 282]], tone: "default" },
        { id: "e_s1_yes", points: [[407, 240], [407, 284]], label: "YES", tone: "yes" },
        { id: "e_tmr_s2", points: [[407, 354], [407, 396]], tone: "default" },
        { id: "e_s2_no", points: [[280, 445], [240, 445]], label: "NO", tone: "no" },
        { id: "e_s2_yes", points: [[535, 445], [585, 445]], label: "YES", tone: "yes" }
      ],
      traces: [
        {
          id: "gaze_warmup",
          label: "Trace: Cold-Start Padding -> 10f Warmup -> Road Attentive",
          steps: [
            {
              title: "Step 1: Warmup Padding & 10-Frame Window Fill",
              highlightNodes: ["pre_clahe", "resnet_ring", "lstm_softmax", "chk_conf", "push_mode", "chk_warmup", "ret_warmup"],
              highlightEdges: ["e_pre_res", "e_res_lst", "e_lst_cnf", "e_cnf_psh", "e_psh_wu", "e_wu_yes"],
              concept: "During startup, ResNet-18 embeddings front-pad the 10-slot LSTM sequence with the earliest feature, while the 10-frame mode window fills before Stage 1 runs.",
              codeRef: "src/GazeDetector.cpp (L109-118) | src/AlertManager.cpp (L102-105)"
            },
            {
              title: "Step 2: On-Road Majority Keeps Safe",
              highlightNodes: ["chk_stage1", "reset_timer", "ret_attentive"],
              highlightEdges: ["e_wu_no", "e_s1_no", "e_rst_ret"],
              concept: "Once warmed up, if LookingRoad votes >= NotLookingRoad votes, distracted_since resets and driver remains Safe.",
              codeRef: "src/AlertLogic.cpp (L47) | src/AlertManager.cpp (L138-144)"
            }
          ]
        },
        {
          id: "gaze_sustained",
          label: "Trace: 10f Majority Held >= 2.0 s -> DISTRACTED",
          steps: [
            {
              title: "Stage 1 Majority + Stage 2 >= 2000 ms -> DISTRACTED",
              highlightNodes: ["chk_conf", "push_mode", "chk_warmup", "chk_stage1", "hold_timer", "chk_stage2", "term_distracted"],
              highlightEdges: ["e_cnf_psh", "e_psh_wu", "e_wu_no", "e_s1_yes", "e_tmr_s2", "e_s2_yes"],
              concept: "When > 50% of the 10-frame window stays NotLookingRoad continuously for >= 2.0 s, DISTRACTED triggers.",
              codeRef: "src/AlertManager.cpp (L129-137)"
            }
          ]
        },
        {
          id: "gaze_glance",
          label: "Trace: Brief Mirror Check (< 2.0 s) -> Timer Clears",
          steps: [
            {
              title: "Short Glance (< 2.0 s) Resets Before Trigger",
              highlightNodes: ["chk_stage1", "hold_timer", "chk_stage2", "ret_counting", "reset_timer", "ret_attentive"],
              highlightEdges: ["e_s1_yes", "e_tmr_s2", "e_s2_no", "e_s1_no", "e_rst_ret"],
              concept: "Looking away for under 2.0 s enters Stage 2's NO branch, and looking back resets distracted_since_ immediately.",
              codeRef: "src/AlertManager.cpp (L129-144)"
            }
          ]
        }
      ]
    },
    {
      id: "5B",
      label: "5B. System & FSM Cut-Off",
      subtitle: "Strict C++ cut-off at GazeDetector: atomic reset, MovingCalibrated gate & 1-frame vote -> GOTO 6A",
      viewBox: "0 0 1120 660",
      legend: ["portal_in", "process", "decision", "portal_out", "return", "yes", "no", "portal_edge"],
      overview: {
        title: "5B: Distraction — System & FSM Cut-Off View",
        concept: "Matches the exact C++ worker/class boundary on GazeWorker: consumes the atomic gaze_reset_requested_ flag, discards frames immediately if SystemState != MovingCalibrated (zero neural compute), runs Preprocessor + ResNet-18 + LSTM, and cuts off right after emitting 1 frame classification into AlertManager (6A).",
        note: "Unlike FatigueWorker (which keeps Yawn CNN and landmarks running while parked), GazeWorker skips ALL inference when parked, uncalibrated, or calibrating.",
        codeRef: "src/PipelineCoordinator.cpp (L69-108, L331, L341, L399-457, L669, L677)\nsrc/GazeDetector.cpp (L94-190)"
      },
      nodes: [
        {
          id: "from_atomic_reset", kind: "portal_in", x: 24, y: 34, w: 235, h: 66,
          label: "FROM 1A: ATOMIC RESET", sublabel: "gaze_reset_requested_ flag\n(set on Park by Main / IMU)",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "set_gaze_atomic" },
          details: {
            concept: "Main ('m') and IMU threads never mutate GazeDetector directly; they store gaze_reset_requested_ = true.",
            codeRef: "src/PipelineCoordinator.cpp (L331, L341, L669, L677)"
          }
        },
        {
          id: "chk_atomic", kind: "decision", x: 295, y: 20, w: 255, h: 96,
          label: "gaze_reset_requested_\n.exchange(false)?", sublabel: "Before & after mailbox",
          details: {
            concept: "Checked both before and right after mailbox_gaze_->get_latest() so a stop during frame wait still cleans up cleanly.",
            codeRef: "src/PipelineCoordinator.cpp (L412-426)"
          }
        },
        {
          id: "wipe_ring", kind: "process", x: 595, y: 32, w: 235, h: 72,
          label: "gaze_detector_.reset()", sublabel: "ring_head = 0; ring_count = 0\n(Thread-confined, lock-free)",
          details: {
            concept: "Resets circular buffer indices safely on GazeWorker's own thread with zero mutex overhead.",
            codeRef: "src/PipelineCoordinator.cpp (L412-426) | src/GazeDetector.cpp (L94-97)"
          }
        },
        {
          id: "gate_mov_cal", kind: "decision", x: 295, y: 162, w: 255, h: 100,
          label: "READ 1A: SystemState\n== MovingCalibrated?", sublabel: "Click -> 1A FSM",
          target: { sectionId: "sec1", subviewId: "1A", highlightNodeId: "st_mov_cal" },
          details: {
            concept: "Hard inference gate in GazeWorker: skips Preprocessor, ResNet-18, and LSTM whenever not MovingCalibrated.",
            codeRef: "src/PipelineCoordinator.cpp (L428-431)"
          }
        },
        {
          id: "skip_frame", kind: "term_return", x: 605, y: 184, w: 240, h: 58,
          label: "SKIP FRAME", sublabel: "Zero neural compute",
          details: {
            concept: "Discards the frame and loops back to wait on mailbox_gaze_ without running ResNet-18 or LSTM.",
            codeRef: "src/PipelineCoordinator.cpp (L429-431)"
          }
        },
        {
          id: "pre_cut", kind: "process", x: 295, y: 312, w: 255, h: 74,
          label: "Preprocessor::process", sublabel: "224x224 Grayscale CLAHE\n+ ImageNet NCHW blob",
          details: {
            concept: "Writes normalized NCHW tensor directly into pre-allocated input_blob.",
            codeRef: "src/PipelineCoordinator.cpp (L434) | src/Preprocessor.cpp (L33-84)"
          }
        },
        {
          id: "net_cut", kind: "process", x: 295, y: 426, w: 255, h: 80,
          label: "GazeDetector::infer_frame", sublabel: "ResNet-18 (512-d) -> 10f/20f\nring -> LSTM -> Softmax",
          details: {
            concept: "Runs spatial backbone + temporal LSTM head and computes softmax probabilities P(not_looking), P(looking).",
            codeRef: "src/PipelineCoordinator.cpp (L436-440) | src/GazeDetector.cpp (L121-155)"
          }
        },
        {
          id: "policy_cut", kind: "process", x: 595, y: 426, w: 255, h: 80,
          label: "Evaluate Frame Policy", sublabel: "Default: Binarize (>=0.54)\n(Opt: FilterUncertain / Argmax)\nEmit 1-frame GazeResult",
          details: {
            concept: "Maps probabilities to a single DistractionDetectionResult{gaze_class, looking_road_probability, timestamp}.",
            codeRef: "src/GazeDetector.cpp (L156-189)"
          }
        },
        {
          id: "goto_6a", kind: "portal_out", x: 890, y: 430, w: 205, h: 72,
          label: "GOTO 6A: GAZE_FILTER", sublabel: "update_gaze(result)\n(10f mode + 2s in mutex)",
          target: { sectionId: "sec6", subviewId: "6A", highlightNodeId: "from_gaze_vote" },
          details: {
            concept: "Strict C++/mutex cut-off: GazeDetector knows nothing about the 10-frame majority filter, 2.0 s timer, drowsiness preemption, or audio; AlertManager handles all of those inside mutex_ (6A).",
            codeRef: "src/PipelineCoordinator.cpp (L443) | src/AlertManager.cpp (L83-145)"
          }
        }
      ],
      edges: [
        { id: "e_at_in", points: [[259, 68], [295, 68]], tone: "portal" },
        { id: "e_at_yes", points: [[550, 68], [595, 68]], label: "YES", tone: "yes" },
        { id: "e_at_no", points: [[422, 116], [422, 162]], label: "NO", tone: "no" },
        { id: "e_wipe_gt", points: [[712, 104], [712, 134], [422, 134], [422, 162]], tone: "default" },
        { id: "e_mov_no", points: [[550, 212], [605, 212]], label: "NO", tone: "no" },
        { id: "e_mov_yes", points: [[422, 262], [422, 312]], label: "YES", tone: "yes" },
        { id: "e_pre_net", points: [[422, 386], [422, 426]], tone: "default" },
        { id: "e_net_pol", points: [[550, 466], [595, 466]], tone: "default" },
        { id: "e_pol_6a", points: [[850, 466], [890, 466]], tone: "portal" }
      ],
      traces: [
        {
          id: "gaze_cut_active",
          label: "Trace: MovingCalibrated -> Single Vote -> GOTO 6A",
          steps: [
            {
              title: "Step 1: Reset Clear & MovingCalibrated Passed",
              highlightNodes: ["chk_atomic", "gate_mov_cal", "pre_cut"],
              highlightEdges: ["e_at_no", "e_mov_yes"],
              concept: "No atomic reset pending and SystemState == MovingCalibrated, so frame proceeds to CLAHE + ImageNet preprocessing.",
              codeRef: "src/PipelineCoordinator.cpp (L412-434)"
            },
            {
              title: "Step 2: Single-Frame Vote Crosses into AlertManager (6A)",
              highlightNodes: ["pre_cut", "net_cut", "policy_cut", "goto_6a"],
              highlightEdges: ["e_pre_net", "e_net_pol", "e_pol_6a"],
              concept: "GazeDetector produces 1 vote (LookingRoad or NotLookingRoad) and immediately crosses into AlertManager::update_gaze() (6A).",
              portalTarget: { sectionId: "sec6", subviewId: "6A", highlightNodeId: "from_gaze_vote" },
              codeRef: "src/GazeDetector.cpp (L121-190) | src/PipelineCoordinator.cpp (L443)"
            }
          ]
        },
        {
          id: "gaze_cut_skip",
          label: "Trace: Parked / Uncalibrated -> Skip Frame",
          steps: [
            {
              title: "Zero Neural Compute When Not MovingCalibrated",
              highlightNodes: ["gate_mov_cal", "skip_frame"],
              highlightEdges: ["e_mov_no"],
              concept: "Whenever parked, uncalibrated, or calibrating, GazeWorker skips preprocessing, ResNet-18, and LSTM completely.",
              codeRef: "src/PipelineCoordinator.cpp (L428-431)"
            }
          ]
        },
        {
          id: "gaze_cut_reset",
          label: "Trace: Park -> Atomic Reset Wipes Ring Buffer",
          steps: [
            {
              title: "In-Band Thread-Confined Ring-Buffer Wipe",
              highlightNodes: ["from_atomic_reset", "chk_atomic", "wipe_ring", "gate_mov_cal", "skip_frame"],
              highlightEdges: ["e_at_in", "e_at_yes", "e_wipe_gt", "e_mov_no"],
              concept: "Stopping sets gaze_reset_requested_ (1A). GazeWorker consumes it via atomic exchange(false), resets ring_head_/ring_count_ on its own thread, and skips inference.",
              codeRef: "src/PipelineCoordinator.cpp (L412-431)"
            }
          ]
        }
      ]
    }
  ]
});
