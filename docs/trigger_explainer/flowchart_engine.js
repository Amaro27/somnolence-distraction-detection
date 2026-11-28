(() => {
  const NS = "http://www.w3.org/2000/svg";
  const sections = window.TRIGGER_SECTIONS || [];
  if (!sections.length) return;

  const state = {
    secIdx: 0,
    subIdx: 0,
    mode: "full", // "full" or trace id
    stepIdx: 0,
    selectedNodeId: null,
    navHistory: []
  };

  const $ = (id) => document.getElementById(id);

  const LEGEND_DEFS = {
    process:     { label: "Process / Formula", css: "border:2px solid #2563eb;background:#ffffff;" },
    decision:    { label: "Decision Diamond", css: "border:2px solid #4f46e5;background:#eef2ff;transform:rotate(45deg);width:12px;height:12px;margin:0 3px;" },
    return:      { label: "RETURN / Skip Exit", css: "border:2px solid #16a34a;background:#dcfce7;border-radius:9px;" },
    alarm:       { label: "DROWSY / DISTRACTED", css: "border:2px solid #e11d48;background:#ffe4e6;border-radius:9px;" },
    portal_out:  { label: "GOTO / Signal Portal (Click)", css: "border:2px solid #0284c7;background:#e0f2fe;" },
    portal_in:   { label: "FROM / FSM Read / Atomic", css: "border:2px solid #d97706;border-left:5px solid #d97706;background:#fef3c7;" },
    mutex:       { label: "Mutex Region", css: "border:2px dashed #d97706;background:#fffbeb;" },
    state:       { label: "SystemState FSM", css: "border:2px solid #059669;background:#ecfdf5;" },
    yes:         { label: "YES Branch", css: "border-top:3px solid #16a34a;height:3px;border-radius:0;" },
    no:          { label: "NO Branch", css: "border-top:3px solid #dc2626;height:3px;border-radius:0;" },
    portal_edge: { label: "Portal / Cross-Call", css: "border-top:2px dashed #0284c7;height:2px;border-radius:0;" }
  };

  const DEFAULT_LEGEND = ["process", "decision", "return", "alarm", "portal_out", "portal_in", "yes", "no"];

  function svgEl(tag, attrs = {}, text = null) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    }
    if (text !== null) node.textContent = text;
    return node;
  }

  function currentSection() {
    return sections[state.secIdx];
  }

  function currentSubview() {
    const sec = currentSection();
    return (sec.subviews || [])[state.subIdx] || sec.subviews[0];
  }

  function activeTrace(sub) {
    if (state.mode === "full") return null;
    return (sub.traces || []).find((t) => t.id === state.mode) || null;
  }

  function findLocation(sectionId, subviewId) {
    const sIdx = sections.findIndex((s) => s.id === sectionId);
    if (sIdx < 0) return null;
    const subIdx = (sections[sIdx].subviews || []).findIndex((sv) => sv.id === subviewId);
    if (subIdx < 0) return null;
    return { sIdx, subIdx };
  }

  function renderCenteredLines(parent, cx, cy, label, sublabel, opts = {}) {
    const labelColor = opts.labelColor || "#0f172a";
    const subColor = opts.subColor || "#475569";
    const labelSize = opts.labelSize || 11.8;
    const subSize = opts.subSize || 10.2;
    const labelLineH = labelSize + 2.6;
    const subLineH = subSize + 2.6;

    const labelLines = label ? String(label).split("\n") : [];
    const subLines = sublabel ? String(sublabel).split("\n") : [];
    const gap = labelLines.length && subLines.length ? 3 : 0;
    const totalH = labelLines.length * labelLineH + gap + subLines.length * subLineH;

    let yCursor = cy - totalH / 2 + labelLineH * 0.78;
    labelLines.forEach((line) => {
      parent.appendChild(svgEl("text", {
        x: cx,
        y: yCursor.toFixed(1),
        "text-anchor": "middle",
        "font-size": labelSize,
        "font-weight": 700,
        fill: labelColor
      }, line));
      yCursor += labelLineH;
    });

    if (gap) yCursor += gap - (labelLineH - subLineH) * 0.3;
    subLines.forEach((line) => {
      parent.appendChild(svgEl("text", {
        x: cx,
        y: yCursor.toFixed(1),
        "text-anchor": "middle",
        "font-size": subSize,
        "font-weight": 500,
        fill: subColor
      }, line));
      yCursor += subLineH;
    });
  }

  function edgeStroke(tone) {
    if (tone === "yes") return "#16a34a";
    if (tone === "no") return "#dc2626";
    if (tone === "portal") return "#0284c7";
    if (tone === "async") return "#7c3aed";
    return "#64748b";
  }

  function edgeMarker(tone, isFlow) {
    if (isFlow) return "url(#arrow-active)";
    if (tone === "yes") return "url(#arrow-yes)";
    if (tone === "no") return "url(#arrow-no)";
    if (tone === "portal") return "url(#arrow-portal)";
    if (tone === "async") return "url(#arrow-async)";
    return "url(#arrow-default)";
  }

  function drawStage(sub) {
    const stage = $("stage");
    stage.setAttribute("viewBox", sub.viewBox || "0 0 1080 580");
    stage.innerHTML = "";

    const defs = svgEl("defs");
    [
      ["arrow-default", "#64748b"],
      ["arrow-yes", "#16a34a"],
      ["arrow-no", "#dc2626"],
      ["arrow-portal", "#0284c7"],
      ["arrow-async", "#7c3aed"],
      ["arrow-active", "#2563eb"]
    ].forEach(([id, col]) => {
      const m = svgEl("marker", {
        id,
        markerWidth: 8,
        markerHeight: 8,
        refX: 7,
        refY: 3.5,
        orient: "auto"
      });
      m.appendChild(svgEl("polygon", { points: "0 0.5, 7 3.5, 0 6.5", fill: col }));
      defs.appendChild(m);
    });
    stage.appendChild(defs);

    const groupLayer = svgEl("g", { id: "layer-groups" });
    const edgeLayer = svgEl("g", { id: "layer-edges" });
    const nodeLayer = svgEl("g", { id: "layer-nodes" });
    stage.appendChild(groupLayer);
    stage.appendChild(edgeLayer);
    stage.appendChild(nodeLayer);

    const nodes = sub.nodes || [];
    for (const n of nodes) {
      if (n.kind !== "group") continue;
      const g = svgEl("g", { id: `el-${n.id}`, class: "el el-normal node-clickable" });
      const isMutex = n.tone === "mutex";
      g.appendChild(svgEl("rect", {
        x: n.x,
        y: n.y,
        width: n.w,
        height: n.h,
        rx: 12,
        fill: isMutex ? "#fffbeb" : "#f1f5f9",
        stroke: isMutex ? "#d97706" : "#94a3b8",
        "stroke-width": 1.8,
        "stroke-dasharray": "6 4",
        class: "shape-main"
      }));
      g.appendChild(svgEl("text", {
        x: n.x + 12,
        y: n.y + 19,
        "font-size": 12.2,
        "font-weight": 700,
        fill: isMutex ? "#92400e" : "#0f172a"
      }, n.label || ""));
      if (n.sublabel) {
        g.appendChild(svgEl("text", {
          x: n.x + 12,
          y: n.y + 34,
          "font-size": 10.2,
          fill: isMutex ? "#b45309" : "#475569"
        }, n.sublabel));
      }
      if (n.badge || isMutex) {
        const badgeText = n.badge || "MUTEX";
        const bw = badgeText.length * 6.5 + 14;
        g.appendChild(svgEl("rect", {
          x: n.x + n.w - bw - 10,
          y: n.y + 7,
          width: bw,
          height: 17,
          rx: 4,
          fill: isMutex ? "#fef3c7" : "#e2e8f0",
          stroke: isMutex ? "#d97706" : "#cbd5e1"
        }));
        g.appendChild(svgEl("text", {
          x: n.x + n.w - bw / 2 - 10,
          y: n.y + 18.5,
          "text-anchor": "middle",
          "font-size": 9.3,
          "font-weight": 700,
          fill: isMutex ? "#92400e" : "#334155"
        }, badgeText));
      }
      g.onclick = () => onNodeClick(n);
      groupLayer.appendChild(g);
    }

    for (const e of (sub.edges || [])) {
      const g = svgEl("g", { id: `el-${e.id}`, class: "el el-normal" });
      const pts = (e.points || []).map((p) => p.join(",")).join(" ");
      const stroke = edgeStroke(e.tone);
      const isDashed = Boolean(e.dashed || e.tone === "portal" || e.tone === "async");
      g.appendChild(svgEl("polyline", {
        points: pts,
        fill: "none",
        stroke,
        "stroke-width": 2,
        "stroke-dasharray": isDashed ? "6 4" : null,
        "marker-end": edgeMarker(e.tone, false),
        class: "edge-line"
      }));

      if (e.label && e.points && e.points.length >= 2) {
        const midIdx = Math.floor((e.points.length - 1) / 2);
        const p1 = e.points[midIdx];
        const p2 = e.points[midIdx + 1];
        const mx = e.lx !== undefined ? e.lx : (p1[0] + p2[0]) / 2;
        const my = e.ly !== undefined ? e.ly : (p1[1] + p2[1]) / 2 - 5;
        const textStr = String(e.label);
        const lw = Math.max(24, textStr.length * 6.4 + 10);
        const txtCol = e.tone === "yes" ? "#15803d"
          : e.tone === "no" ? "#b91c1c"
          : e.tone === "portal" ? "#0369a1"
          : "#334155";
        g.appendChild(svgEl("rect", {
          x: (mx - lw / 2).toFixed(1),
          y: (my - 10.5).toFixed(1),
          width: lw.toFixed(1),
          height: 14,
          rx: 3,
          fill: "#ffffff",
          "fill-opacity": "0.94"
        }));
        g.appendChild(svgEl("text", {
          x: mx.toFixed(1),
          y: my.toFixed(1),
          "text-anchor": "middle",
          "font-size": 10,
          "font-weight": 700,
          fill: txtCol
        }, textStr));
      }
      edgeLayer.appendChild(g);
    }

    for (const n of nodes) {
      if (n.kind === "group") continue;
      const hasTarget = Boolean(n.target);
      const cls = `el el-normal ${hasTarget ? "portal-clickable" : "node-clickable"}`;
      const g = svgEl("g", { id: `el-${n.id}`, class: cls });
      const cx = n.x + n.w / 2;
      const cy = n.y + n.h / 2;

      if (n.kind === "process") {
        const isNeutral = n.tone === "neutral";
        const isAsync = n.tone === "async";
        g.appendChild(svgEl("rect", {
          x: n.x,
          y: n.y,
          width: n.w,
          height: n.h,
          rx: 8,
          fill: isAsync ? "#faf5ff" : "#ffffff",
          stroke: isAsync ? "#7c3aed" : isNeutral ? "#475569" : "#2563eb",
          "stroke-width": 1.9,
          "stroke-dasharray": isAsync ? "6 4" : null,
          class: "shape-main"
        }));
        renderCenteredLines(g, cx, cy, n.label, n.sublabel, {
          labelColor: "#0f172a",
          subColor: "#475569"
        });
      } else if (n.kind === "decision") {
        const pts = [
          [cx, n.y],
          [n.x + n.w, cy],
          [cx, n.y + n.h],
          [n.x, cy]
        ].map((p) => p.join(",")).join(" ");
        g.appendChild(svgEl("polygon", {
          points: pts,
          fill: "#eef2ff",
          stroke: "#4f46e5",
          "stroke-width": 2,
          class: "shape-main"
        }));
        renderCenteredLines(g, cx, cy, n.label, n.sublabel, {
          labelColor: "#1e1b4b",
          subColor: "#4338ca",
          labelSize: 11.2,
          subSize: 9.8
        });
      } else if (n.kind === "term_return") {
        g.appendChild(svgEl("rect", {
          x: n.x,
          y: n.y,
          width: n.w,
          height: n.h,
          rx: n.h / 2,
          fill: "#dcfce7",
          stroke: "#16a34a",
          "stroke-width": 2.2,
          class: "shape-main"
        }));
        renderCenteredLines(g, cx, cy, n.label, n.sublabel, {
          labelColor: "#14532d",
          subColor: "#166534"
        });
      } else if (n.kind === "term_alarm") {
        g.appendChild(svgEl("rect", {
          x: n.x,
          y: n.y,
          width: n.w,
          height: n.h,
          rx: n.h / 2,
          fill: "#ffe4e6",
          stroke: "#e11d48",
          "stroke-width": 2.2,
          class: "shape-main"
        }));
        renderCenteredLines(g, cx, cy, n.label, n.sublabel, {
          labelColor: "#881337",
          subColor: "#9f1239"
        });
      } else if (n.kind === "portal_out") {
        const tip = 13;
        const pts = [
          [n.x, n.y],
          [n.x + n.w - tip, n.y],
          [n.x + n.w, cy],
          [n.x + n.w - tip, n.y + n.h],
          [n.x, n.y + n.h]
        ].map((p) => p.join(",")).join(" ");
        g.appendChild(svgEl("polygon", {
          points: pts,
          fill: "#e0f2fe",
          stroke: "#0284c7",
          "stroke-width": 2,
          class: "shape-main"
        }));
        renderCenteredLines(g, n.x + (n.w - tip * 0.6) / 2, cy, n.label, n.sublabel, {
          labelColor: "#0c4a6e",
          subColor: "#0369a1",
          labelSize: 11.2,
          subSize: 9.8
        });
      } else if (n.kind === "portal_in") {
        g.appendChild(svgEl("rect", {
          x: n.x,
          y: n.y,
          width: n.w,
          height: n.h,
          rx: 6,
          fill: "#fef3c7",
          stroke: "#d97706",
          "stroke-width": 2,
          class: "shape-main"
        }));
        g.appendChild(svgEl("rect", {
          x: n.x,
          y: n.y,
          width: 6,
          height: n.h,
          rx: 2,
          fill: "#d97706"
        }));
        renderCenteredLines(g, n.x + 3 + (n.w - 3) / 2, cy, n.label, n.sublabel, {
          labelColor: "#78350f",
          subColor: "#92400e",
          labelSize: 11.2,
          subSize: 9.8
        });
      } else if (n.kind === "state") {
        const isActiveState = n.tone === "active";
        const isCalib = n.tone === "calib";
        const fill = isActiveState ? "#ecfdf5" : isCalib ? "#fef3c7" : "#ffffff";
        const stroke = isActiveState ? "#059669" : isCalib ? "#d97706" : "#334155";
        g.appendChild(svgEl("rect", {
          x: n.x,
          y: n.y,
          width: n.w,
          height: n.h,
          rx: 12,
          fill,
          stroke,
          "stroke-width": 2.3,
          class: "shape-main"
        }));
        if (n.doubleBorder) {
          g.appendChild(svgEl("rect", {
            x: n.x + 4,
            y: n.y + 4,
            width: n.w - 8,
            height: n.h - 8,
            rx: 8,
            fill: "none",
            stroke,
            "stroke-width": 1
          }));
        }
        renderCenteredLines(g, cx, cy, n.label, n.sublabel, {
          labelColor: isActiveState ? "#065f46" : isCalib ? "#78350f" : "#0f172a",
          subColor: isActiveState ? "#047857" : "#475569",
          labelSize: 11.8,
          subSize: 10.0
        });
      }

      g.onclick = () => onNodeClick(n);
      nodeLayer.appendChild(g);
    }
  }

  function onNodeClick(node) {
    if (node.target) {
      jumpToPortal(node.target);
      return;
    }
    state.selectedNodeId = (state.selectedNodeId === node.id && state.mode === "full") ? null : node.id;
    renderViewState();
  }

  function jumpToPortal(target) {
    const loc = findLocation(target.sectionId, target.subviewId);
    if (!loc) return;
    const callerSec = currentSection();
    const callerSub = currentSubview();
    state.navHistory.push({
      secIdx: state.secIdx,
      subIdx: state.subIdx,
      mode: state.mode,
      stepIdx: state.stepIdx,
      selectedNodeId: state.selectedNodeId,
      callerLabel: callerSub.label || callerSec.title
    });
    state.secIdx = loc.sIdx;
    state.subIdx = loc.subIdx;
    state.mode = "full";
    state.stepIdx = 0;
    state.selectedNodeId = target.highlightNodeId || null;
    renderSection();
  }

  function renderNavTabs() {
    const secTabs = $("section-tabs");
    secTabs.innerHTML = "";
    sections.forEach((sec, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `tab-btn${idx === state.secIdx ? " active" : ""}`;
      btn.textContent = sec.title;
      btn.onclick = () => {
        state.secIdx = idx;
        state.subIdx = 0;
        state.mode = "full";
        state.stepIdx = 0;
        state.selectedNodeId = null;
        state.navHistory = [];
        renderSection();
      };
      secTabs.appendChild(btn);
    });

    const sec = currentSection();
    const svTabs = $("subview-tabs");
    svTabs.innerHTML = "";
    (sec.subviews || []).forEach((sv, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `tab-btn${idx === state.subIdx ? " active" : ""}`;
      btn.textContent = sv.label;
      btn.onclick = () => {
        state.subIdx = idx;
        state.mode = "full";
        state.stepIdx = 0;
        state.selectedNodeId = null;
        renderSubview();
      };
      svTabs.appendChild(btn);
    });

    const sub = currentSubview();
    const mTabs = $("mode-tabs");
    mTabs.innerHTML = "";
    const modes = [{ id: "full", label: "Full Flowchart" }, ...(sub.traces || []).map((t) => ({ id: t.id, label: t.label }))];
    modes.forEach((m) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `tab-btn${state.mode === m.id ? " active" : ""}`;
      btn.textContent = m.label;
      btn.onclick = () => {
        state.mode = m.id;
        state.stepIdx = 0;
        state.selectedNodeId = null;
        renderModeButtons(sub);
        renderViewState();
      };
      mTabs.appendChild(btn);
    });
  }

  function renderModeButtons(sub) {
    const modes = ["full", ...(sub.traces || []).map((t) => t.id)];
    document.querySelectorAll("#mode-tabs .tab-btn").forEach((b, i) => {
      b.classList.toggle("active", modes[i] === state.mode);
    });
  }

  function renderLegend(sub) {
    const box = $("legend");
    box.innerHTML = "";
    const keys = sub.legend || DEFAULT_LEGEND;
    keys.forEach((key) => {
      const def = LEGEND_DEFS[key];
      if (!def) return;
      const item = document.createElement("span");
      item.className = "legend-item";
      item.innerHTML = `<span class="legend-swatch" style="${def.css}"></span>${def.label}`;
      box.appendChild(item);
    });
  }

  function setDisclosures(title, concept, code, note, codeRef, badgeText, jumpTarget) {
    $("info-badge").textContent = badgeText;
    $("step-title").textContent = title || "";
    $("step-concept").textContent = concept || "";

    const codeBox = $("code-box");
    const t2 = $("tier2-box");
    const t3 = $("tier3-box");
    codeBox.open = false;
    t2.open = false;
    t3.open = false;
    codeBox.hidden = !code;
    if (code) $("step-code").textContent = code;
    t2.hidden = !note;
    if (note) $("step-note").textContent = note;
    t3.hidden = !codeRef;
    if (codeRef) $("step-coderef").textContent = codeRef;

    const jumpBtn = $("portal-jump-btn");
    if (jumpTarget && findLocation(jumpTarget.sectionId, jumpTarget.subviewId)) {
      const loc = findLocation(jumpTarget.sectionId, jumpTarget.subviewId);
      const targetSub = sections[loc.sIdx].subviews[loc.subIdx];
      jumpBtn.textContent = `Jump to ${targetSub.label} ->`;
      jumpBtn.hidden = false;
      jumpBtn.onclick = () => jumpToPortal(jumpTarget);
    } else {
      jumpBtn.hidden = true;
      jumpBtn.onclick = null;
    }
  }

  function renderViewState() {
    const sub = currentSubview();
    const trace = activeTrace(sub);
    const traceNav = $("trace-nav");
    const kbdHint = $("kbd-hint");
    const clearBtn = $("clear-sel-btn");
    const backPortalBtn = $("back-portal-btn");

    if (state.navHistory.length > 0) {
      const prev = state.navHistory[state.navHistory.length - 1];
      backPortalBtn.textContent = `<- Back to ${prev.callerLabel}`;
      backPortalBtn.hidden = false;
    } else {
      backPortalBtn.hidden = true;
    }

    const nodes = sub.nodes || [];
    const edges = sub.edges || [];

    if (trace) {
      const steps = trace.steps || [];
      const step = steps[state.stepIdx] || steps[0];
      const hiNodes = new Set((step && step.highlightNodes) || []);
      const hiEdges = new Set((step && step.highlightEdges) || []);

      for (const n of nodes) {
        const el = $(`el-${n.id}`);
        if (!el) continue;
        const active = hiNodes.has(n.id);
        const sel = state.selectedNodeId === n.id;
        const base = n.target ? "portal-clickable" : "node-clickable";
        el.setAttribute("class", `el ${base} ${active ? "el-active" : "el-dim"}${sel ? " el-selected" : ""}`);
      }
      for (const e of edges) {
        const el = $(`el-${e.id}`);
        if (!el) continue;
        const active = hiEdges.has(e.id);
        el.setAttribute("class", `el ${active ? "el-active edge-flow" : "el-dim"}`);
        const poly = el.querySelector("polyline");
        if (poly) poly.setAttribute("marker-end", edgeMarker(e.tone, active));
      }

      traceNav.hidden = false;
      kbdHint.hidden = false;
      $("step-counter").textContent = `Step ${state.stepIdx + 1} of ${steps.length}`;
      $("prev-btn").disabled = state.stepIdx === 0;
      $("next-btn").disabled = state.stepIdx >= steps.length - 1;

      const dots = $("step-dots");
      dots.innerHTML = "";
      steps.forEach((_, idx) => {
        const d = document.createElement("button");
        d.type = "button";
        d.className = `dot${idx === state.stepIdx ? " active" : ""}`;
        d.title = `Go to step ${idx + 1}`;
        d.onclick = () => {
          state.selectedNodeId = null;
          state.stepIdx = idx;
          renderViewState();
        };
        dots.appendChild(d);
      });

      const selNode = state.selectedNodeId ? nodes.find((n) => n.id === state.selectedNodeId) : null;
      if (selNode && selNode.details) {
        clearBtn.hidden = false;
        clearBtn.textContent = "Back to Trace Step";
        setDisclosures(
          selNode.details.title || selNode.label.replace(/\n/g, " "),
          selNode.details.concept,
          selNode.details.code,
          selNode.details.note,
          selNode.details.codeRef,
          "Selected Node",
          selNode.target
        );
      } else if (step) {
        clearBtn.hidden = true;
        setDisclosures(
          step.title,
          step.concept,
          step.code,
          step.note,
          step.codeRef,
          `Trace Step ${state.stepIdx + 1}/${steps.length}`,
          step.portalTarget || null
        );
      }
    } else {
      for (const n of nodes) {
        const el = $(`el-${n.id}`);
        if (!el) continue;
        const sel = state.selectedNodeId === n.id;
        const base = n.target ? "portal-clickable" : "node-clickable";
        el.setAttribute("class", `el ${base} el-normal${sel ? " el-selected" : ""}`);
      }
      for (const e of edges) {
        const el = $(`el-${e.id}`);
        if (!el) continue;
        el.setAttribute("class", "el el-normal");
        const poly = el.querySelector("polyline");
        if (poly) poly.setAttribute("marker-end", edgeMarker(e.tone, false));
      }

      traceNav.hidden = true;
      kbdHint.hidden = false;
      $("step-counter").textContent = `${sub.id} · Full Flowchart`;

      const selNode = state.selectedNodeId ? nodes.find((n) => n.id === state.selectedNodeId) : null;
      if (selNode) {
        clearBtn.hidden = false;
        clearBtn.textContent = "Subview Overview";
        const d = selNode.details || {};
        const fallbackConcept = sub.overview ? sub.overview.concept : "";
        setDisclosures(
          d.title || selNode.label.replace(/\n/g, " "),
          d.concept || fallbackConcept,
          d.code,
          d.note,
          d.codeRef,
          selNode.target ? "Portal Node" : "Selected Node",
          selNode.target
        );
      } else {
        clearBtn.hidden = true;
        const ov = sub.overview || {};
        setDisclosures(
          ov.title || `${sub.id}: ${sub.label}`,
          ov.concept || sub.subtitle || "",
          ov.code,
          ov.note,
          ov.codeRef,
          "Subview Overview",
          null
        );
      }
    }
  }

  function renderSubview() {
    const sub = currentSubview();
    $("subview-subtitle").textContent = sub.subtitle || "";
    renderNavTabs();
    drawStage(sub);
    renderLegend(sub);
    renderViewState();
  }

  function renderSection() {
    renderSubview();
  }

  function exportCleanSvg() {
    const sub = currentSubview();
    const stage = $("stage");
    const clone = stage.cloneNode(true);
    clone.setAttribute("font-family", "system-ui, -apple-system, Segoe UI, sans-serif");

    const vb = (sub.viewBox || "0 0 1080 580").split(/\s+/);
    const bg = document.createElementNS(NS, "rect");
    bg.setAttribute("x", vb[0] || "0");
    bg.setAttribute("y", vb[1] || "0");
    bg.setAttribute("width", vb[2] || "1080");
    bg.setAttribute("height", vb[3] || "580");
    bg.setAttribute("fill", "#ffffff");
    clone.insertBefore(bg, clone.firstChild);

    clone.querySelectorAll(".el-dim").forEach((n) => n.setAttribute("opacity", "0.22"));
    clone.querySelectorAll(".el-active, .el-normal").forEach((n) => n.setAttribute("opacity", "1"));
    clone.querySelectorAll(".el-active .edge-line").forEach((p) => {
      p.setAttribute("stroke-width", "2.6");
    });

    const xml = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${sub.id}-${state.mode}.svg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  $("prev-btn").onclick = () => {
    if (state.stepIdx > 0) {
      state.selectedNodeId = null;
      state.stepIdx--;
      renderViewState();
    }
  };

  $("next-btn").onclick = () => {
    const trace = activeTrace(currentSubview());
    if (!trace) return;
    if (state.stepIdx < trace.steps.length - 1) {
      state.selectedNodeId = null;
      state.stepIdx++;
      renderViewState();
    }
  };

  $("clear-sel-btn").onclick = () => {
    state.selectedNodeId = null;
    renderViewState();
  };

  $("back-portal-btn").onclick = () => {
    const prev = state.navHistory.pop();
    if (!prev) return;
    state.secIdx = prev.secIdx;
    state.subIdx = prev.subIdx;
    state.mode = prev.mode;
    state.stepIdx = prev.stepIdx;
    state.selectedNodeId = prev.selectedNodeId;
    renderSection();
  };

  $("export-svg-btn").onclick = exportCleanSvg;

  document.addEventListener("keydown", (e) => {
    if (e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
    const trace = activeTrace(currentSubview());
    if (!trace) return;
    if (e.key === "ArrowLeft") $("prev-btn").click();
    if (e.key === "ArrowRight") $("next-btn").click();
  });

  renderSection();
})();
