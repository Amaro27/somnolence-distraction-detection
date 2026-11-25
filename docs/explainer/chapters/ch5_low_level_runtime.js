window.EXPLAINER_CHAPTERS = window.EXPLAINER_CHAPTERS || [];
window.EXPLAINER_CHAPTERS.push({
  id: "ch5",
  title: "5. Low-Level & Edge",
  subtitle: "48 kHz streaming audio, 500 ms Dlib throttle, zero-alloc edge tuning & build modes",
  viewBox: "0 0 960 540",
  legend: ["thread", "mailbox", "atomic", "sync", "debug"],
  elements: [
    // Top-Left Quadrant: Streaming Audio Engine
    { id: "audio_eng", kind: "lane", x: 20, y: 18, w: 446, h: 240, label: "Always-On Audio Mixer", sublabel: "Pre-decoded 48 kHz stereo PCM", tag: "TIME-CRITICAL", tone: "amber" },
    { id: "pcm_cache", kind: "box", x: 34, y: 68, w: 132, h: 86, label: "WAV RAM Cache", sublabel: "Preloaded + lazy\n48 kHz stereo" },
    { id: "ring_buf", kind: "mailbox", x: 246, y: 68, w: 204, h: 86, label: "6x10 ms Ring Buffer", slotLabel: "60 ms -> waveOut / aplay" },
    { id: "ticket_pill", kind: "pill", x: 42, y: 186, w: 200, h: 38, label: "Playback Ticket #N" },
    { id: "cond_wait", kind: "box", x: 278, y: 172, w: 172, h: 70, label: "Waiter Unblocks Fast", sublabel: "Unlock before wait;\nbump ID to cancel" },
    { id: "e_cache_ring", kind: "edge", points: [[166, 111], [246, 111]], label: "480f/10ms" },
    { id: "e_ring_wait", kind: "edge", points: [[364, 154], [364, 172]] },
    { id: "e_ticket_wait", kind: "edge", points: [[242, 205], [278, 205]] },

    // Top-Right Quadrant: 500 ms Face-Loss CPU Throttle
    { id: "throttle_lane", kind: "lane", x: 494, y: 18, w: 446, h: 240, label: "500 ms Face-Loss Throttle", sublabel: "Frees CPU for Gaze while Yawn CNN stays active", tag: "CPU SAVER" },
    { id: "probe_gate", kind: "box", x: 512, y: 80, w: 190, h: 146, label: "500 ms Probe Timer", sublabel: "Face missing?\nThrottle Dlib search\nto once every 500 ms" },
    { id: "cpu_saved", kind: "box", x: 734, y: 80, w: 190, h: 146, label: "Gaze + Yawn Keep Running", sublabel: "Skips heavy Dlib pass;\nfull-frame Yawn CNN &\nGaze Worker run smoothly" },
    { id: "e_probe_cpu", kind: "edge", points: [[702, 153], [734, 153]], label: "skip" },

    // Bottom-Left Quadrant: Zero-Allocation Hot Paths & Edge Presets
    { id: "mem_lane", kind: "lane", x: 20, y: 282, w: 446, h: 240, label: "Zero-Alloc Hot Paths & Edge Presets", sublabel: "Pre-allocated tensors, OS-guarded IMU & SIMD", tag: "ZERO ALLOC" },
    { id: "prealloc_box", kind: "box", x: 36, y: 344, w: 196, h: 156, label: "Reusable Frame Buffers", sublabel: "224x224 & 112x112 blobs,\nCLAHE & gray buffers\nallocated once at init" },
    { id: "pi_preset", kind: "box", x: 254, y: 344, w: 196, h: 156, label: "10f Pre-Opt, IMU & SIMD", sublabel: "Flat circular LSTM ring,\nLinux I2C / Win stub IMU\n+ ARM NEON / x86 AVX2" },
    { id: "e_prealloc_pi", kind: "edge", points: [[232, 422], [254, 422]] },

    // Bottom-Right Quadrant: Compile-Time Build Modes
    { id: "build_lane", kind: "lane", x: 494, y: 282, w: 446, h: 240, label: "Debug HUD & Recording vs. Release", sublabel: "Selected at compile time via debug views flag", tag: "COMPILE-TIME", tone: "purple" },
    { id: "dbg_mode", kind: "box", x: 512, y: 344, w: 194, h: 156, label: "Debug Build", sublabel: "3rd mailbox + OpenCV HUD\n+ 15 FPS MP4 recorder ('r')\n+ P95 latency summary", badge: "debug" },
    { id: "rel_mode", kind: "box", x: 730, y: 344, w: 194, h: 156, label: "Release Build", sublabel: "Zero-cost empty stubs\n+ 50 ms main sleep\n(MP4 recording disabled)" }
  ],
  walkthrough: [
    {
      title: "1. Always-On 48 kHz Streaming Audio Mixer",
      show: ["audio_eng", "pcm_cache", "ring_buf", "e_cache_ring"],
      highlight: ["audio_eng", "pcm_cache", "ring_buf"],
      edgeFlow: ["e_cache_ring"],
      concept: "Instead of reopening the sound card on every alert, startup decodes calibration audio into 48 kHz stereo memory and launches a time-critical mixer feeding a 60 ms ring buffer. Starting an alert just swaps a memory pointer with sub-10 ms response.",
      note: "Older README notes describe Windows PlaySoundA and spawning per-clip aplay commands, whereas the running Windows and Linux/Raspberry Pi engine streams raw PCM continuously through this custom 6x10 ms ring buffer. The two alert WAV files decode lazily on first play rather than in startup warmup.",
      codeRef: "morphed_system6/distraction/modular/src/SoundPlayer.cpp (L36-40 constants; L226-494 AudioStreamEngine; L503-519 warmup)"
    },
    {
      title: "2. Instant Generation-Ticket Cancellation",
      show: ["audio_eng", "pcm_cache", "ring_buf", "ticket_pill", "cond_wait", "e_cache_ring", "e_ring_wait", "e_ticket_wait"],
      highlight: ["ticket_pill", "cond_wait"],
      edgeFlow: ["e_ticket_wait", "e_ring_wait"],
      concept: "Every playback gets a rising integer ticket number. Stopping or preempting an alert clears the pointer, bumps the ticket, and wakes sleeping waiters immediately without holding the mixer lock across blocking waits.",
      codeRef: "morphed_system6/distraction/modular/src/SoundPlayer.cpp (L253-295 trigger / stop_playback)\nsrc/AudioManager.cpp (L20-30 play_blocking / stop)"
    },
    {
      title: "3. 500 ms Face-Loss CPU Throttling",
      show: ["throttle_lane", "probe_gate", "cpu_saved", "e_probe_cpu"],
      highlight: ["throttle_lane", "probe_gate", "cpu_saved"],
      edgeFlow: ["e_probe_cpu"],
      concept: "Full-image face detection is heavy on embedded CPUs. Once the driver's face disappears, the Fatigue Worker throttles landmark searches to once every 500 milliseconds, freeing CPU for the Gaze Worker while the fast full-frame yawning net keeps running.",
      codeRef: "morphed_system6/distraction/modular/src/PipelineCoordinator.cpp (L526-559)\ninclude/Types.hpp (L184-188 FaceLandmarkConfig)"
    },
    {
      title: "4. Zero-Allocation Hot Paths & Cross-Platform Edge Tuning",
      show: ["mem_lane", "prealloc_box", "pi_preset", "e_prealloc_pi"],
      highlight: ["mem_lane", "prealloc_box", "pi_preset"],
      edgeFlow: ["e_prealloc_pi"],
      concept: "Neural input blobs, grayscale buffers, and a flat circular sequence ring are pre-allocated once at startup to prevent per-frame memory allocation jitter. Default 10-frame pre-optimized models, SIMD flags, and OS-guarded IMU support target smooth cross-platform deployment.",
      note: "During the first 9 frames of a 10-frame sequence window, the circular buffer pads leading slots with the earliest captured feature vector so predictions start immediately.",
      codeRef: "morphed_system6/distraction/modular/include/GazeDetector.hpp (L70-76) | src/GazeDetector.cpp (L60-65, L99-119)\ninclude/YawnDetector.hpp (L58-61) | include/Preprocessor.hpp (L46-51) | include/MPU6050.hpp (L51-205) | CMakeLists.txt (L8-25)"
    },
    {
      title: "5. Debug HUD & Recording vs. Zero-Cost Release",
      show: ["build_lane", "dbg_mode", "rel_mode"],
      highlight: ["build_lane", "dbg_mode", "rel_mode"],
      edgeFlow: [],
      concept: "Compiling with debug views enabled adds a third mailbox for the live OpenCV overlay, an on-demand MP4 recorder, and end-of-run latency percentiles. In Release builds, all visualization and recording calls compile into empty stubs with a 50 ms main sleep.",
      note: "Because recording lives inside the debug visualizer, passing the record flag or pressing 'r' silently does nothing in Release builds unless debug views are compiled on.",
      codeRef: "morphed_system6/distraction/modular/CMakeLists.txt (L27-33, L66-70)\ninclude/DebugVisualizer.hpp (L17-179) | src/DebugVisualizer.cpp (L176-207, L563-598) | src/PipelineCoordinator.cpp (L218-229, L255-260, L298-309)"
    }
  ],
  traces: [
    {
      id: "ring_preempt",
      label: "Trace: 10 ms Ring-Buffer Preemption",
      steps: [
        {
          title: "Step 1: Distraction Streams on Ticket #4",
          show: ["audio_eng", "pcm_cache", "ring_buf", "ticket_pill", "cond_wait", "e_cache_ring", "e_ring_wait", "e_ticket_wait"],
          highlight: ["pcm_cache", "ring_buf", "ticket_pill"],
          edgeFlow: ["e_cache_ring", "e_ticket_wait"],
          slotText: { ring_buf: "Streaming: Distract #4" },
          concept: "The distraction clip starts with ticket #4. The Gaze Worker unlocks the audio state and sleeps on a condition variable waiting for ticket #4 to finish."
        },
        {
          title: "Step 2: 10 ms Blocks Feed the Audio Card",
          show: ["audio_eng", "pcm_cache", "ring_buf", "ticket_pill", "cond_wait", "e_cache_ring", "e_ring_wait"],
          highlight: ["ring_buf"],
          edgeFlow: ["e_cache_ring"],
          slotText: { ring_buf: "480 frames / 10 ms" },
          concept: "Every 10 milliseconds, the high-priority mixer copies 480 stereo samples from the cached clip into the next ring buffer slot for hardware output."
        },
        {
          title: "Step 3: Drowsiness Bumps Ticket to #5 -> #6",
          show: ["audio_eng", "pcm_cache", "ring_buf", "ticket_pill", "cond_wait", "e_cache_ring", "e_ring_wait", "e_ticket_wait"],
          highlight: ["ticket_pill", "ring_buf"],
          edgeFlow: ["e_ticket_wait"],
          slotText: { ring_buf: "Ticket #5 Stop -> #6 Beep" },
          concept: "Drowsiness preemption calls stop (clearing the active pointer and bumping ticket to #5) then play (pointing at the drowsiness beep under ticket #6)."
        },
        {
          title: "Step 4: Waiter Wakes & Next 10 ms Plays Beep",
          show: ["audio_eng", "pcm_cache", "ring_buf", "ticket_pill", "cond_wait", "e_cache_ring", "e_ring_wait", "e_ticket_wait"],
          highlight: ["ring_buf", "cond_wait"],
          edgeFlow: ["e_cache_ring", "e_ring_wait"],
          slotText: { ring_buf: "Streaming: Drowsy #6" },
          concept: "Because the active ticket no longer equals #4, the Gaze Worker wakes immediately, while the very next 10 ms audio block streams the drowsiness beep seamlessly."
        }
      ]
    }
  ]
});
