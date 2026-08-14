"""
Haar Cascade Classifier Tester — GUI
=====================================
A lightweight desktop tool for visually comparing OpenCV Haar cascade
classifiers in real time. Useful when deciding which cascade (frontal
face, profile face, eyes, etc.) performs best for a given use case
before committing to it in a larger project.

Layout
------
  Left panel  (~35% of window): cascade selector + live stats
  Right panel (~65% of window): webcam feed with detection boxes drawn

Source for cascade files:
  https://github.com/opencv/opencv/tree/master/data/haarcascades

─────────────────────────────────────────────────────────────────────
How to run
─────────────────────────────────────────────────────────────────────
  Requirements:
    pip install opencv-python Pillow

  Run:
    python haar_cascades_gui.py

  Controls:
    • Click any button on the left panel to switch cascade.
    • Scroll (mouse wheel) the button list if there are many cascades.
    • Close the window to exit — the capture thread shuts down cleanly.

Cascade XMLs are loaded automatically from the folder:
  <script_dir>/xmls_haar_cascades/*.xml
Any XML file dropped into that folder will appear as a button on the
next launch — no code changes required.

---> If interested in how it works, you continue reading. Else, just run it :) <----

---------------------------------------------------------------------------

─────────────────────────────────────────────────────────────────────
Architecture — two threads
─────────────────────────────────────────────────────────────────────
This program uses two threads to keep the GUI responsive while doing
heavy image processing:

  1. MAIN THREAD (tkinter event loop)
     Owns all UI widgets. Never blocks. Receives frames and stats from
     the capture thread via `self.after(0, ...)` callbacks, which
     schedule updates safely on the main thread.

  2. CAPTURE THREAD (_capture_loop)
     Runs in the background for the entire lifetime of the app.
     Does all the heavy lifting: reading webcam frames, running
     detectMultiScale, drawing bounding boxes, and computing FPS.

─────────────────────────────────────────────────────────────────────
Executed ONCE (at startup)
─────────────────────────────────────────────────────────────────────
  • glob scan of xmls_haar_cascades/ → builds the list of available
    cascades and creates one button per XML file found.
  • First cascade in the sorted list is loaded as the default.
  • cv2.VideoCapture(0) opens the webcam.
  • The capture thread is started (daemon=True so it dies with the app).
  • All tkinter widgets are built and laid out.
  • The scrollbar is conditionally shown only if > 6 cascades exist.

─────────────────────────────────────────────────────────────────────
Running CONTINUOUSLY (every frame, ~30 fps)
─────────────────────────────────────────────────────────────────────
  • cap.read()               — pulls the next raw BGR frame from webcam
  • cvtColor (BGR→GRAY)      — converts frame for cascade processing
  • detectMultiScale()       — slides the cascade window across the
                               frame at multiple scales to find objects
  • cv2.rectangle / putText  — draws bounding boxes and labels on frame
  • FPS calculation          — delta-time between consecutive frames
  • cvtColor (BGR→RGB) + PIL — converts frame for tkinter display
  • self.after(0, ...)       — posts the rendered frame + stats to the
                               main thread for display (thread-safe)

─────────────────────────────────────────────────────────────────────
Cascade switching (on demand, debounced)
─────────────────────────────────────────────────────────────────────
  When the user clicks a button, a short-lived worker thread loads the
  new CascadeClassifier from disk (which can take ~0.5 s for large
  XMLs) and then atomically replaces self.current_cascade. The capture
  loop reads this reference once per frame via a local variable, so
  the swap never causes a crash or a freeze — the old cascade simply
  finishes its current frame before the new one takes over.
  A debounce delay (150 ms) prevents rapid repeated clicks from
  spawning multiple loaders simultaneously.


"""

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import os
import glob

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CASCADES_DIR = os.path.join(SCRIPT_DIR, "xmls_haar_cascades")

# ─── Detection parameters ─────────────────────────────────────────────────────
SCALE_FACTOR  = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE      = (30, 30)
BOX_COLOR     = (50, 220, 100)   # BGR green
BOX_THICKNESS = 2

# ─── Pretty-print cascade names ───────────────────────────────────────────────
def friendly_name(xml_path: str) -> str:
    """Derive a human-readable label from any cascade filename automatically."""
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    return stem.replace("haarcascade_", "").replace("_", " ").title()


# ─── Main App ─────────────────────────────────────────────────────────────────
class HaarGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Haar Cascade Tester")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        # Window geometry
        self.geometry("1100x680")
        self.minsize(800, 500)

        # ── State ──────────────────────────────────────────────────────────────
        self.cascade_paths   = sorted(glob.glob(os.path.join(CASCADES_DIR, "*.xml")))
        self.current_cascade = cv2.CascadeClassifier(self.cascade_paths[0])
        self.active_path     = self.cascade_paths[0]
        self._pending_path   = None          # cascade switch requested
        self._switching      = False         # debounce flag

        self.detections      = 0
        self.fps             = 0.0
        self._running        = True

        # ── Build UI ───────────────────────────────────────────────────────────
        self._build_ui()

        # ── Start capture thread ───────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._set_status("⚠  Could not open webcam", error=True)

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=35, minsize=260)
        self.grid_columnconfigure(1, weight=65)
        self.grid_rowconfigure(0, weight=1)

        # ── Left panel ────────────────────────────────────────────────────────
        left = tk.Frame(self, bg="#16213e", padx=16, pady=16)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_rowconfigure(3, weight=1)

        tk.Label(
            left, text="Haar Cascade\nTester",
            font=("Segoe UI", 17, "bold"),
            fg="#e2e8f0", bg="#16213e", justify="left"
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        tk.Label(
            left, text="Select a cascade to apply:",
            font=("Segoe UI", 9),
            fg="#94a3b8", bg="#16213e", justify="left"
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        # Separator
        sep = tk.Frame(left, bg="#334155", height=1)
        sep.grid(row=2, column=0, sticky="ew", pady=(0, 14))

        # ── Scrollable button list ─────────────────────────────────────────
        scroll_wrapper = tk.Frame(left, bg="#16213e")
        scroll_wrapper.grid(row=3, column=0, sticky="nsew")
        scroll_wrapper.grid_rowconfigure(0, weight=1)
        scroll_wrapper.grid_columnconfigure(0, weight=1)

        btn_canvas = tk.Canvas(
            scroll_wrapper, bg="#16213e",
            highlightthickness=0, bd=0
        )
        btn_canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            scroll_wrapper, orient="vertical", command=btn_canvas.yview
        )
        # Only show scrollbar when there are enough cascades to need it
        if len(self.cascade_paths) > 6:
            scrollbar.grid(row=0, column=1, sticky="ns")
            btn_canvas.configure(yscrollcommand=scrollbar.set)

        btn_inner = tk.Frame(btn_canvas, bg="#16213e")
        btn_canvas_window = btn_canvas.create_window(
            (0, 0), window=btn_inner, anchor="nw"
        )

        def _on_inner_configure(event):
            btn_canvas.configure(scrollregion=btn_canvas.bbox("all"))

        def _on_canvas_configure(event):
            btn_canvas.itemconfig(btn_canvas_window, width=event.width)

        btn_inner.bind("<Configure>", _on_inner_configure)
        btn_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse-wheel scrolling (Windows & Linux)
        def _on_mousewheel(event):
            btn_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        btn_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._btn_refs = {}
        for path in self.cascade_paths:
            name = friendly_name(path)
            btn = tk.Button(
                btn_inner,
                text=name,
                font=("Segoe UI", 10),
                anchor="w",
                padx=12, pady=10,
                relief="flat",
                cursor="hand2",
                bg="#1e3a5f", fg="#cbd5e1",
                activebackground="#2563eb",
                activeforeground="#ffffff",
                command=lambda p=path: self._request_cascade(p),
            )
            btn.pack(fill="x", pady=3)
            self._btn_refs[path] = btn

        # Highlight first button as active
        self._highlight_button(self.active_path)

        # Stats frame at bottom of left panel
        stats_frame = tk.Frame(left, bg="#0f172a", bd=0)
        stats_frame.grid(row=4, column=0, sticky="ew", pady=(18, 0))

        self._stat_cascade = self._stat_row(stats_frame, "Cascade",     0)
        self._stat_detect  = self._stat_row(stats_frame, "Detections",  1)
        self._stat_fps     = self._stat_row(stats_frame, "FPS",         2)
        self._status_var   = tk.StringVar(value="Running…")
        self._stat_status  = self._stat_row(stats_frame, "Status",      3, var=self._status_var)

        self._stat_cascade.set(friendly_name(self.active_path))

        # ── Right panel (camera) ───────────────────────────────────────────────
        right = tk.Frame(self, bg="#0f172a")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Label(right, bg="#0f172a", text="Starting camera…",
                                fg="#475569", font=("Segoe UI", 13))
        self._canvas.grid(row=0, column=0, sticky="nsew")

    def _stat_row(self, parent, label, row, var=None):
        tk.Label(
            parent, text=label + ":", font=("Segoe UI", 8, "bold"),
            fg="#64748b", bg="#0f172a", width=10, anchor="w"
        ).grid(row=row, column=0, sticky="w", padx=(10, 4), pady=3)
        var = var or tk.StringVar(value="—")
        tk.Label(
            parent, textvariable=var, font=("Segoe UI", 8),
            fg="#94a3b8", bg="#0f172a", anchor="w", wraplength=160
        ).grid(row=row, column=1, sticky="w", padx=(0, 10), pady=3)
        return var

    def _highlight_button(self, active_path):
        for path, btn in self._btn_refs.items():
            if path == active_path:
                btn.configure(bg="#2563eb", fg="#ffffff",
                              font=("Segoe UI", 10, "bold"))
            else:
                btn.configure(bg="#1e3a5f", fg="#cbd5e1",
                              font=("Segoe UI", 10))

    # ──────────────────────────────────────────────────────────────────────────
    # Cascade switching (debounced, done on background thread)
    # ──────────────────────────────────────────────────────────────────────────
    def _request_cascade(self, path):
        if path == self.active_path:
            return
        self._pending_path = path
        if not self._switching:
            self._switching = True
            threading.Thread(target=self._do_switch, daemon=True).start()

    def _do_switch(self):
        """Loads the new cascade on a worker thread to avoid freezing the UI."""
        time.sleep(0.15)                          # small debounce
        path = self._pending_path
        self._set_status("Loading…")
        new_cascade = cv2.CascadeClassifier(path)
        # Atomic swap
        self.current_cascade = new_cascade
        self.active_path = path
        self._switching = False
        # Update UI on main thread
        self.after(0, self._highlight_button, path)
        self.after(0, self._stat_cascade.set, friendly_name(path))
        self._set_status("Running…")

    # ──────────────────────────────────────────────────────────────────────────
    # Capture / detection loop  (background thread)
    # ──────────────────────────────────────────────────────────────────────────
    def _capture_loop(self):
        prev_time = time.time()
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                self._set_status("⚠  Frame read failed", error=True)
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = self.current_cascade          # local ref (thread-safe read)

            detections = cascade.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=MIN_SIZE,
            )

            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
                cv2.putText(
                    frame,
                    friendly_name(self.active_path),
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, BOX_COLOR, 1, cv2.LINE_AA,
                )

            # FPS overlay
            now      = time.time()
            fps      = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                frame, f"FPS: {fps:.1f}  |  Detections: {len(detections)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA,
            )

            # Convert BGR → RGB → PIL → ImageTk
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            # Schedule UI update on main thread
            self.after(0, self._update_canvas, image, len(detections), fps)

        self.cap.release()

    def _update_canvas(self, image: Image.Image, n_detections: int, fps: float):
        if not self._running:
            return
        # Fit the image to the canvas label size
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w > 1 and h > 1:
            image = image.resize((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._canvas.configure(image=photo, text="")
        self._canvas.image = photo          # hold reference

        self._stat_detect.set(str(n_detections))
        self._stat_fps.set(f"{fps:.1f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _set_status(self, msg: str, error: bool = False):
        color = "#f87171" if error else "#94a3b8"
        self.after(0, lambda: (
            self._status_var.set(msg),
        ))

    def _on_close(self):
        self._running = False
        time.sleep(0.1)
        self.destroy()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        raise SystemExit(
            "Pillow is required: run  pip install Pillow  and try again."
        )
    app = HaarGUI()
    app.mainloop()
