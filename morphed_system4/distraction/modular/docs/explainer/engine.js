(() => {
  const NS = "http://www.w3.org/2000/svg";
  const chapters = window.EXPLAINER_CHAPTERS || [];
  if (!chapters.length) return;

  const state = { chIdx: 0, mode: "walkthrough", stepIdx: 0 };
  const $ = (id) => document.getElementById(id);

  const LEGEND_DEFS = {
    thread:  { label: "Persistent Thread", css: "border:2px solid #64748b;background:#f1f5f9;" },
    mailbox: { label: "Drop-Oldest 1-Slot Mailbox", css: "border:3px double #0284c7;background:#e0f2fe;" },
    mutex:   { label: "Mutex Lock", css: "border:2px solid #d97706;background:#fef3c7;" },
    atomic:  { label: "Atomic Flag / State", css: "border:2px solid #059669;background:#d1fae5;border-radius:9px;" },
    sync:    { label: "Frame / Direct Flow", css: "border-top:3px solid #2563eb;height:3px;border-radius:0;" },
    async:   { label: "Detached / Async", css: "border-top:2px dashed #7c3aed;height:2px;border-radius:0;" }
  };

  function svgEl(tag, attrs = {}, text = null) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    }
    if (text !== null) node.textContent = text;
    return node;
  }

  function addWrappedText(parent, x, y, text, cls, size = 11.5, fill = "#334155") {
    const lines = String(text).split("\n");
    lines.forEach((line, idx) => {
      parent.appendChild(svgEl("text", {
        x, y: y + idx * (size + 3), "text-anchor": "middle",
        "font-size": size, fill, class: cls
      }, line));
    });
  }

  function activeSteps(ch) {
    if (state.mode === "walkthrough") return ch.walkthrough || [];
    const tr = (ch.traces || []).find((t) => t.id === state.mode);
    return tr ? tr.steps : (ch.walkthrough || []);
  }

  function drawStage(ch) {
    const stage = $("stage");
    stage.setAttribute("viewBox", ch.viewBox || "0 0 960 540");
    stage.innerHTML = "";

    const defs = svgEl("defs");
    [["arrow-muted", "#64748b"], ["arrow-active", "#2563eb"], ["arrow-async", "#7c3aed"]].forEach(([id, col]) => {
      const m = svgEl("marker", { id, markerWidth: 8, markerHeight: 8, refX: 7, refY: 3.5, orient: "auto" });
      m.appendChild(svgEl("polygon", { points: "0 0.5, 7 3.5, 0 6.5", fill: col }));
      defs.appendChild(m);
    });
    stage.appendChild(defs);

    for (const el of ch.elements) {
      const g = svgEl("g", { id: `el-${el.id}`, class: "el el-hidden" });
      if (el.kind === "lane") {
        const fill = el.tone === "amber" ? "#fffbeb" : el.tone === "purple" ? "#f5f3ff" : "#f1f5f9";
        const stroke = el.tone === "amber" ? "#d97706" : el.tone === "purple" ? "#7c3aed" : "#64748b";
        g.appendChild(svgEl("rect", {
          x: el.x, y: el.y, width: el.w, height: el.h, rx: 12,
          fill, stroke, "stroke-width": 2, class: "card-bg"
        }));
        g.appendChild(svgEl("text", {
          x: el.x + 12, y: el.y + 20, "font-size": 13, "font-weight": 700, fill: "#0f172a"
        }, el.label));
        if (el.tag) {
          const tw = el.tag.length * 6.5 + 12;
          g.appendChild(svgEl("rect", {
            x: el.x + el.w - tw - 10, y: el.y + 7, width: tw, height: 17, rx: 5, fill: "#e2e8f0"
          }));
          g.appendChild(svgEl("text", {
            x: el.x + el.w - tw / 2 - 10, y: el.y + 19, "text-anchor": "middle",
            "font-size": 9.5, "font-weight": 700, fill: "#334155"
          }, el.tag));
        }
        if (el.sublabel) {
          g.appendChild(svgEl("text", {
            x: el.x + 12, y: el.y + 36, "font-size": 10.5, fill: "#475569"
          }, el.sublabel));
        }
      } else if (el.kind === "box") {
        const isAsync = el.badge === "async";
        const isMutex = el.badge === "mutex";
        const stroke = isAsync ? "#7c3aed" : isMutex ? "#d97706" : "#3b82f6";
        const fill = isAsync ? "#faf5ff" : isMutex ? "#fffbeb" : "#ffffff";
        g.appendChild(svgEl("rect", {
          x: el.x, y: el.y, width: el.w, height: el.h, rx: 8,
          fill, stroke, "stroke-width": 1.8,
          "stroke-dasharray": isAsync ? "6 4" : null, class: "card-bg"
        }));
        const cx = el.x + el.w / 2;
        const baseY = el.pythonOrigin ? el.y + el.h * 0.36 : el.y + el.h * 0.44;
        g.appendChild(svgEl("text", {
          x: cx, y: baseY, "text-anchor": "middle", "font-size": 12.2, "font-weight": 700, fill: "#0f172a"
        }, el.label));
        if (el.sublabel) {
          addWrappedText(g, cx, baseY + 16, el.sublabel, "box-sub", 10.5, "#475569");
        }
        if (el.pythonOrigin) {
          g.appendChild(svgEl("text", {
            x: cx, y: el.y + el.h - 9, "text-anchor": "middle",
            "font-size": 9.5, "font-weight": 600, fill: "#0369a1"
          }, `Py: ${el.pythonOrigin}`));
        }
        if (isMutex) {
          g.appendChild(svgEl("rect", {
            x: el.x + el.w - 58, y: el.y + 6, width: 52, height: 17, rx: 4, fill: "#fef3c7", stroke: "#d97706"
          }));
          g.appendChild(svgEl("text", {
            x: el.x + el.w - 32, y: el.y + 18, "text-anchor": "middle",
            "font-size": 9.5, "font-weight": 700, fill: "#92400e"
          }, "MUTEX"));
        } else if (isAsync) {
          g.appendChild(svgEl("rect", {
            x: el.x + el.w - 54, y: el.y + 6, width: 48, height: 17, rx: 4, fill: "#ede9fe"
          }));
          g.appendChild(svgEl("text", {
            x: el.x + el.w - 30, y: el.y + 18, "text-anchor": "middle",
            "font-size": 9.5, "font-weight": 700, fill: "#5b21b6"
          }, "ASYNC"));
        }
      } else if (el.kind === "mailbox") {
        g.appendChild(svgEl("rect", {
          x: el.x, y: el.y, width: el.w, height: el.h, rx: 7,
          fill: "#e0f2fe", stroke: "#0284c7", "stroke-width": 2, class: "card-bg"
        }));
        g.appendChild(svgEl("rect", {
          x: el.x + 4, y: el.y + 4, width: el.w - 8, height: el.h - 8, rx: 4,
          fill: "none", stroke: "#0284c7", "stroke-width": 1
        }));
        g.appendChild(svgEl("text", {
          x: el.x + el.w / 2, y: el.y + el.h * 0.43, "text-anchor": "middle",
          "font-size": 11.5, "font-weight": 700, fill: "#0c4a6e"
        }, el.label));
        g.appendChild(svgEl("text", {
          id: `slot-${el.id}`, x: el.x + el.w / 2, y: el.y + el.h * 0.72, "text-anchor": "middle",
          "font-size": 10.5, fill: "#0369a1"
        }, el.slotLabel || "1-slot"));
      } else if (el.kind === "pill") {
        g.appendChild(svgEl("rect", {
          x: el.x, y: el.y, width: el.w, height: el.h, rx: el.h / 2,
          fill: "#ecfdf5", stroke: "#059669", "stroke-width": 2, class: "card-bg"
        }));
        g.appendChild(svgEl("text", {
          x: el.x + el.w / 2, y: el.y + el.h / 2 + 4, "text-anchor": "middle",
          "font-size": 11.5, "font-weight": 700, fill: "#065f46"
        }, el.label));
      } else if (el.kind === "edge") {
        const pts = el.points.map((p) => p.join(",")).join(" ");
        const marker = el.dashed ? "url(#arrow-async)" : "url(#arrow-muted)";
        g.appendChild(svgEl("polyline", {
          points: pts, fill: "none", stroke: el.dashed ? "#7c3aed" : "#64748b",
          "stroke-width": 2, "stroke-dasharray": el.dashed ? "6 4" : null,
          "marker-end": marker, class: "edge-line"
        }));
        if (el.label && el.points.length >= 2) {
          const mid = el.points[Math.floor((el.points.length - 1) / 2)];
          const nxt = el.points[Math.floor((el.points.length - 1) / 2) + 1];
          const lx = el.lx ?? (mid[0] + nxt[0]) / 2;
          const ly = el.ly ?? (mid[1] + nxt[1]) / 2 - 6;
          g.appendChild(svgEl("text", {
            x: lx, y: ly, "text-anchor": "middle", "font-size": 10, "font-weight": 600, fill: "#475569"
          }, el.label));
        }
      }
      stage.appendChild(g);
    }
  }

  function renderTabs(ch) {
    const chTabs = $("chapter-tabs");
    chTabs.innerHTML = "";
    chapters.forEach((c, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `tab-btn${idx === state.chIdx ? " active" : ""}`;
      btn.textContent = c.title;
      btn.onclick = () => {
        state.chIdx = idx;
        state.mode = "walkthrough";
        state.stepIdx = 0;
        renderChapter();
      };
      chTabs.appendChild(btn);
    });

    const mTabs = $("mode-tabs");
    mTabs.innerHTML = "";
    const modes = [{ id: "walkthrough", label: "Walkthrough" }, ...(ch.traces || []).map((t) => ({ id: t.id, label: t.label }))];
    if (modes.length > 1) {
      modes.forEach((m) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `tab-btn${state.mode === m.id ? " active" : ""}`;
        btn.textContent = m.label;
        btn.onclick = () => { state.mode = m.id; state.stepIdx = 0; renderStep(); renderModeState(); };
        mTabs.appendChild(btn);
      });
    }
  }

  function renderModeState() {
    document.querySelectorAll("#mode-tabs .tab-btn").forEach((b, i) => {
      const ch = chapters[state.chIdx];
      const modes = ["walkthrough", ...(ch.traces || []).map((t) => t.id)];
      b.classList.toggle("active", modes[i] === state.mode);
    });
  }

  function renderLegend(ch) {
    const box = $("legend");
    box.innerHTML = "";
    (ch.legend || []).forEach((key) => {
      const def = LEGEND_DEFS[key];
      if (!def) return;
      const item = document.createElement("span");
      item.className = "legend-item";
      item.innerHTML = `<span class="legend-swatch" style="${def.css}"></span>${def.label}`;
      box.appendChild(item);
    });
  }

  function renderStep() {
    const ch = chapters[state.chIdx];
    const steps = activeSteps(ch);
    const step = steps[state.stepIdx];
    if (!step) return;

    const showSet = new Set(step.show || []);
    const hiSet = new Set(step.highlight || []);
    const flowSet = new Set(step.edgeFlow || []);

    for (const el of ch.elements) {
      const node = $(`el-${el.id}`);
      if (!node) continue;
      const shown = showSet.has(el.id);
      const active = hiSet.has(el.id) || flowSet.has(el.id);
      node.setAttribute("class", `el ${!shown ? "el-hidden" : active ? "el-active" : "el-dim"}${flowSet.has(el.id) ? " edge-flow" : ""}`);
      if (el.kind === "edge") {
        const poly = node.querySelector("polyline");
        if (poly) poly.setAttribute("marker-end", flowSet.has(el.id) || active ? "url(#arrow-active)" : el.dashed ? "url(#arrow-async)" : "url(#arrow-muted)");
      }
      if (el.kind === "mailbox") {
        const slot = $(`slot-${el.id}`);
        if (slot) slot.textContent = (step.slotText && step.slotText[el.id]) || el.slotLabel || "1-slot";
      }
    }

    $("step-counter").textContent = `Step ${state.stepIdx + 1} of ${steps.length}`;
    $("step-title").textContent = step.title;
    $("step-concept").textContent = step.concept;

    const t2 = $("tier2-box");
    const t3 = $("tier3-box");
    t2.open = false;
    t3.open = false;
    t2.hidden = !step.note;
    if (step.note) $("step-note").textContent = step.note;
    t3.hidden = !step.codeRef;
    if (step.codeRef) $("step-coderef").textContent = step.codeRef;

    $("prev-btn").disabled = state.stepIdx === 0;
    $("next-btn").disabled = state.stepIdx >= steps.length - 1;

    const dots = $("step-dots");
    dots.innerHTML = "";
    steps.forEach((_, idx) => {
      const d = document.createElement("button");
      d.type = "button";
      d.className = `dot${idx === state.stepIdx ? " active" : ""}`;
      d.title = `Go to step ${idx + 1}`;
      d.onclick = () => { state.stepIdx = idx; renderStep(); };
      dots.appendChild(d);
    });
  }

  function renderChapter() {
    const ch = chapters[state.chIdx];
    $("chapter-subtitle").textContent = ch.subtitle || "";
    renderTabs(ch);
    drawStage(ch);
    renderLegend(ch);
    renderStep();
  }

  function exportCleanSvg() {
    const ch = chapters[state.chIdx];
    const clone = $("stage").cloneNode(true);
    clone.setAttribute("font-family", "system-ui, -apple-system, Segoe UI, sans-serif");
    clone.querySelectorAll(".el-hidden").forEach((n) => n.remove());
    clone.querySelectorAll(".el-dim").forEach((n) => n.setAttribute("opacity", "0.22"));
    clone.querySelectorAll(".el-active").forEach((n) => n.setAttribute("opacity", "1"));
    clone.querySelectorAll(".el-active .edge-line").forEach((p) => {
      p.setAttribute("stroke", "#2563eb");
      p.setAttribute("stroke-width", "2.5");
    });
    const xml = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${ch.id}-step-${state.stepIdx + 1}.svg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  $("prev-btn").onclick = () => { if (state.stepIdx > 0) { state.stepIdx--; renderStep(); } };
  $("next-btn").onclick = () => {
    const steps = activeSteps(chapters[state.chIdx]);
    if (state.stepIdx < steps.length - 1) { state.stepIdx++; renderStep(); }
  };
  $("export-svg-btn").onclick = exportCleanSvg;

  document.addEventListener("keydown", (e) => {
    if (e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.key === "ArrowLeft") $("prev-btn").click();
    if (e.key === "ArrowRight") $("next-btn").click();
  });

  renderChapter();
})();
