/* ============================================================
   Sorbed — vanilla frontend (no deps, offline, strict CSP safe)
   ============================================================ */
(function () {
  "use strict";

  /* ---------- Constants: colors & labels ---------- */
  var STAGE_COLORS = {
    stage_1: [255, 214, 0],
    stage_2: [255, 138, 101],
    stage_3: [255, 111, 0],
    stage_4: [233, 30, 99],
    unstageable: [0, 200, 83],
    deep_tissue_injury: [213, 0, 0],
    mucosal_not_stageable: [124, 77, 255],
    not_pressure_injury: [120, 144, 156],
    indeterminate: [158, 158, 158]
  };

  var TISSUE_COLORS = {
    epithelial: [245, 176, 200],
    granulation: [214, 40, 57],
    slough: [240, 200, 70],
    eschar: [40, 40, 45],
    adipose: [250, 232, 150],
    muscle: [140, 30, 60],
    tendon_bone: [238, 238, 220],
    intact_skin: [150, 190, 235],
    unknown: [130, 130, 130]
  };

  var TRAJ_COLORS = {
    healing: [0, 178, 80],
    stalled: [230, 160, 30],
    deteriorating: [213, 40, 40],
    indeterminate: [158, 158, 158]
  };

  var NICE_LABELS = {
    tendon_bone: "Tendon / Bone",
    intact_skin: "Intact skin",
    deep_tissue_injury: "Deep Tissue Injury",
    mucosal_not_stageable: "Mucosal (not stageable)",
    not_pressure_injury: "Not a pressure injury",
    fitzpatrick_i_iii: "Fitzpatrick I–III",
    fitzpatrick_iv_vi: "Fitzpatrick IV–VI",
    unknown: "Unknown"
  };

  /* ---------- Small helpers ---------- */
  function $(sel, root) { return (root || document).querySelector(sel); }
  function rgb(arr) { return "rgb(" + arr[0] + "," + arr[1] + "," + arr[2] + ")"; }
  function rgba(arr, a) { return "rgba(" + arr[0] + "," + arr[1] + "," + arr[2] + "," + a + ")"; }

  function prettyLabel(key) {
    if (key == null) return "—";
    if (NICE_LABELS[key]) return NICE_LABELS[key];
    return String(key)
      .split("_")
      .map(function (w) { return w.charAt(0).toUpperCase() + w.slice(1); })
      .join(" ");
  }

  function stageColor(stage) { return STAGE_COLORS[stage] || STAGE_COLORS.indeterminate; }
  function tissueColor(t) { return TISSUE_COLORS[t] || TISSUE_COLORS.unknown; }

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function fmtNum(v, digits) {
    if (v == null || isNaN(v)) return "—";
    var d = digits == null ? 2 : digits;
    return Number(v).toFixed(d);
  }
  function fmtPct(frac, digits) {
    if (frac == null || isNaN(frac)) return "—";
    return (frac * 100).toFixed(digits == null ? 1 : digits) + "%";
  }

  /* DOM builders (safe: text via textContent) */
  function el(tag, opts) {
    var node = document.createElement(tag);
    opts = opts || {};
    if (opts.className) node.className = opts.className;
    if (opts.text != null) node.textContent = String(opts.text);
    if (opts.attrs) {
      for (var k in opts.attrs) {
        if (Object.prototype.hasOwnProperty.call(opts.attrs, k)) node.setAttribute(k, opts.attrs[k]);
      }
    }
    if (opts.html != null) node.innerHTML = opts.html; // only used with static, non-user strings
    if (opts.children) {
      opts.children.forEach(function (c) { if (c) node.appendChild(c); });
    }
    return node;
  }
  var SVGNS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVGNS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) node.setAttribute(k, attrs[k]);
      }
    }
    return node;
  }

  /* ---------- Theme toggle ---------- */
  function initTheme() {
    var btn = $("#themeToggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var root = document.documentElement;
      var cur = root.getAttribute("data-theme");
      var isDark;
      if (cur === "dark") isDark = true;
      else if (cur === "light") isDark = false;
      else isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.setAttribute("data-theme", isDark ? "light" : "dark");
    });
  }

  /* ---------- Disclaimer (dismissible but returns on tab switch) ---------- */
  function initDisclaimer() {
    var strip = $("#disclaimer");
    var close = $("#disclaimerClose");
    if (close) {
      close.addEventListener("click", function () { strip.hidden = true; });
    }
    return { restore: function () { if (strip) strip.hidden = false; } };
  }

  /* ---------- Tabs (ARIA, roving tabindex, arrow keys) ---------- */
  function initTabs(disclaimer) {
    var tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"]'));
    function select(tab) {
      tabs.forEach(function (t) {
        var selected = t === tab;
        t.setAttribute("aria-selected", selected ? "true" : "false");
        t.tabIndex = selected ? 0 : -1;
        var panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !selected;
      });
      disclaimer.restore();
    }
    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () { select(tab); });
      tab.addEventListener("keydown", function (e) {
        var idx = null;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") idx = (i + 1) % tabs.length;
        else if (e.key === "ArrowLeft" || e.key === "ArrowUp") idx = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") idx = 0;
        else if (e.key === "End") idx = tabs.length - 1;
        if (idx != null) {
          e.preventDefault();
          tabs[idx].focus();
          select(tabs[idx]);
        }
      });
    });
  }

  /* ---------- Lightbox ---------- */
  var lightbox = {
    dialog: null, img: null, cap: null, lastFocus: null,
    init: function () {
      this.dialog = $("#lightbox");
      this.img = $("#lightboxImg");
      this.cap = $("#lightboxCap");
      var self = this;
      var closeBtn = $("#lightboxClose");
      if (closeBtn) closeBtn.addEventListener("click", function () { self.close(); });
      if (this.dialog) {
        this.dialog.addEventListener("cancel", function (e) { e.preventDefault(); self.close(); });
        this.dialog.addEventListener("click", function (e) {
          if (e.target === self.dialog) self.close();
        });
      }
    },
    open: function (src, alt, caption) {
      if (!this.dialog) return;
      this.lastFocus = document.activeElement;
      this.img.src = src;
      this.img.alt = alt || "";
      this.cap.textContent = caption || "";
      if (typeof this.dialog.showModal === "function") this.dialog.showModal();
      else this.dialog.setAttribute("open", "");
      var closeBtn = $("#lightboxClose");
      if (closeBtn) closeBtn.focus();
    },
    close: function () {
      if (!this.dialog) return;
      if (typeof this.dialog.close === "function") this.dialog.close();
      else this.dialog.removeAttribute("open");
      if (this.lastFocus && this.lastFocus.focus) this.lastFocus.focus();
    }
  };

  /* ---------- Chart tooltip (shared) ---------- */
  var tip = {
    node: null,
    init: function () { this.node = $("#chartTip"); },
    show: function (html, x, y) {
      if (!this.node) return;
      this.node.innerHTML = html; // built from escaped text below
      this.node.hidden = false;
      var w = this.node.offsetWidth, h = this.node.offsetHeight;
      var px = x + 14, py = y + 14;
      if (px + w > window.innerWidth - 8) px = x - w - 14;
      if (py + h > window.innerHeight - 8) py = y - h - 14;
      this.node.style.left = px + "px";
      this.node.style.top = py + "px";
    },
    hide: function () { if (this.node) this.node.hidden = true; }
  };
  function tipEscape(s) {
    var d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }

  /* ============================================================
     TISSUE DONUT (SVG)
     ============================================================ */
  function buildTissueChart(fractions) {
    var entries = [];
    if (fractions) {
      Object.keys(fractions).forEach(function (k) {
        var v = fractions[k];
        if (v != null && v > 0.0001) entries.push({ key: k, val: v });
      });
    }
    entries.sort(function (a, b) { return b.val - a.val; });

    var wrap = el("div", { className: "chart-card" });

    if (!entries.length) {
      wrap.appendChild(el("p", { className: "empty-hint", text: "No tissue composition available." }));
      return wrap;
    }

    var total = entries.reduce(function (s, e) { return s + e.val; }, 0) || 1;
    var size = 220, cx = size / 2, cy = size / 2, r = 82, sw = 34;
    var svg = svgEl("svg", {
      viewBox: "0 0 " + size + " " + size, class: "chart-svg",
      role: "img", "aria-label": "Tissue composition donut chart"
    });
    svg.style.maxWidth = "240px";

    var circumference = 2 * Math.PI * r;
    var offset = 0;
    var reduce = prefersReducedMotion();

    entries.forEach(function (e, i) {
      var frac = e.val / total;
      var seg = svgEl("circle", {
        cx: cx, cy: cy, r: r, fill: "none",
        stroke: rgb(tissueColor(e.key)),
        "stroke-width": sw,
        "stroke-dasharray": (frac * circumference) + " " + circumference,
        "stroke-dashoffset": -offset * circumference,
        transform: "rotate(-90 " + cx + " " + cy + ")",
        tabindex: "0",
        role: "listitem",
        "aria-label": prettyLabel(e.key) + " " + fmtPct(frac)
      });
      seg.classList.add("data-pt");
      if (!reduce) {
        seg.style.strokeDasharray = "0 " + circumference;
        // animate after insertion
        (function (s, target) {
          requestAnimationFrame(function () {
            s.style.transition = "stroke-dasharray .9s cubic-bezier(.2,.7,.2,1)";
            requestAnimationFrame(function () { s.style.strokeDasharray = target; });
          });
        })(seg, (frac * circumference) + " " + circumference);
      }
      function showTip(ev) {
        var pt = ev.touches ? ev.touches[0] : ev;
        tip.show("<span class='tip-title'>" + tipEscape(prettyLabel(e.key)) + "</span><br>" + tipEscape(fmtPct(frac)),
          pt.clientX, pt.clientY);
      }
      seg.addEventListener("mousemove", showTip);
      seg.addEventListener("mouseleave", function () { tip.hide(); });
      seg.addEventListener("focus", function () {
        var b = seg.getBoundingClientRect();
        tip.show("<span class='tip-title'>" + tipEscape(prettyLabel(e.key)) + "</span><br>" + tipEscape(fmtPct(frac)),
          b.left + b.width / 2, b.top + b.height / 2);
      });
      seg.addEventListener("blur", function () { tip.hide(); });
      svg.appendChild(seg);
      offset += frac;
    });

    // center label
    var domKey = entries[0].key;
    var center = el("div");
    var svgWrap = el("div", { attrs: { style: "position:relative; display:inline-block;" } });
    svgWrap.appendChild(svg);
    var centerLabel = el("div", {
      attrs: { style: "position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; pointer-events:none;" }
    });
    centerLabel.appendChild(el("span", { text: fmtPct(entries[0].val / total, 0), attrs: { style: "font-size:24px; font-weight:600;" } }));
    centerLabel.appendChild(el("span", { text: prettyLabel(domKey), className: "muted", attrs: { style: "font-size:12px;" } }));
    svgWrap.appendChild(centerLabel);
    center.appendChild(svgWrap);
    wrap.appendChild(center);

    // legend
    var legend = el("div", { className: "legend", attrs: { role: "list" } });
    entries.forEach(function (e) {
      var item = el("div", { className: "legend-item", attrs: { role: "listitem" } });
      item.appendChild(el("span", { className: "legend-dot", attrs: { style: "background:" + rgb(tissueColor(e.key)) } }));
      item.appendChild(el("span", { className: "legend-name", text: prettyLabel(e.key) }));
      item.appendChild(el("span", { className: "legend-val", text: fmtPct(e.val / total) }));
      legend.appendChild(item);
    });
    wrap.appendChild(legend);
    return wrap;
  }

  /* ============================================================
     GRADE BANNER
     ============================================================ */
  function needsReview(decision) {
    if (!decision) return false;
    if (decision.abstained) return true;
    if (typeof decision.confidence === "number" && decision.confidence < 0.5) return true;
    var depthDependent = ["stage_3", "stage_4", "unstageable", "deep_tissue_injury"];
    return depthDependent.indexOf(decision.stage) !== -1;
  }

  function buildGradeBanner(analysis) {
    var d = analysis.decision || {};
    var color = stageColor(d.stage);
    var banner = el("div", { className: "grade-banner" });

    var pill = el("span", { className: "stage-pill", attrs: { style: "background:" + rgb(color) } });
    pill.appendChild(el("span", { className: "dot" }));
    pill.appendChild(document.createTextNode(prettyLabel(d.stage)));
    banner.appendChild(pill);

    var conf = typeof d.confidence === "number" ? d.confidence : 0;
    var confWrap = el("div", { className: "conf-wrap" });
    var top = el("div", { className: "conf-top" });
    top.appendChild(el("span", { text: "Confidence" }));
    top.appendChild(el("span", { text: fmtPct(conf, 0) }));
    confWrap.appendChild(top);
    var track = el("div", { className: "conf-track" });
    var fill = el("div", { className: "conf-fill", attrs: { style: "background:" + rgb(color) } });
    track.appendChild(fill);
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(Math.round(conf * 100)));
    track.setAttribute("aria-label", "Confidence");
    confWrap.appendChild(track);
    banner.appendChild(confWrap);

    if (prefersReducedMotion()) fill.style.width = (conf * 100) + "%";
    else requestAnimationFrame(function () { requestAnimationFrame(function () { fill.style.width = (conf * 100) + "%"; }); });

    if (needsReview(d)) {
      banner.appendChild(el("span", { className: "review-chip", text: "Clinician review required" }));
    }
    return banner;
  }

  /* ============================================================
     GALLERY
     ============================================================ */
  var IMG_META = {
    input: { label: "Input", note: "" },
    mask: { label: "Wound mask", note: "" },
    overlay: { label: "Tissue overlay", note: "" },
    depth: { label: "Depth", note: "relative shading cue, not a measurement" },
    detection: { label: "Detection", note: "" },
    schematic: { label: "Schematic", note: "" }
  };
  function buildGallery(images) {
    var grid = el("div", { className: "gallery" });
    var order = ["input", "mask", "overlay", "depth", "detection", "schematic"];
    var any = false;
    order.forEach(function (key) {
      var src = images && images[key];
      if (!src) return;
      any = true;
      var meta = IMG_META[key] || { label: prettyLabel(key), note: "" };
      var fig = el("figure");
      var btn = el("button", { className: "thumb", attrs: { type: "button", "aria-label": "Enlarge " + meta.label + " image" } });
      var img = el("img", { attrs: { src: src, alt: meta.label + " visualization of the wound", loading: "lazy" } });
      btn.appendChild(img);
      btn.addEventListener("click", function () {
        lightbox.open(src, meta.label + " visualization", meta.label + (meta.note ? " — " + meta.note : ""));
      });
      fig.appendChild(btn);
      var cap = el("figcaption", { text: meta.label });
      if (meta.note) cap.appendChild(el("span", { className: "cap-note", text: meta.note }));
      fig.appendChild(cap);
      grid.appendChild(fig);
    });
    if (!any) grid.appendChild(el("p", { className: "empty-hint", text: "No images returned." }));
    return grid;
  }

  /* ============================================================
     METRICS GRID
     ============================================================ */
  function metricCard(label, value, sub) {
    var m = el("div", { className: "metric" });
    m.appendChild(el("div", { className: "m-label", text: label }));
    m.appendChild(el("div", { className: "m-value", text: value == null ? "—" : value }));
    if (sub) m.appendChild(el("div", { className: "m-sub", text: sub }));
    return m;
  }

  function buildMetrics(analysis) {
    var m = analysis.metrics || {};
    var g = m.geometry || {};
    var grid = el("div", { className: "metrics-grid" });

    // Area
    if (g.area_cm2 != null) {
      grid.appendChild(metricCard("Area", fmtNum(g.area_cm2, 2) + " cm²", g.area_px != null ? fmtNum(g.area_px, 0) + " px" : ""));
    } else {
      grid.appendChild(metricCard("Area", g.area_px != null ? fmtNum(g.area_px, 0) + " px" : "—", "uncalibrated"));
    }

    // Length x Width
    var lw = "—";
    if (g.length_mm != null || g.width_mm != null) {
      lw = fmtNum(g.length_mm, 1) + " × " + fmtNum(g.width_mm, 1) + " mm";
    }
    grid.appendChild(metricCard("Length × Width", lw));

    grid.appendChild(metricCard("Perimeter", g.perimeter_px != null ? fmtNum(g.perimeter_px, 0) + " px" : "—"));
    grid.appendChild(metricCard("Circularity", g.circularity != null ? fmtNum(g.circularity, 2) : "—",
      g.solidity != null ? "solidity " + fmtNum(g.solidity, 2) : ""));
    grid.appendChild(metricCard("Wound / image", g.wound_fraction_of_image != null ? fmtPct(g.wound_fraction_of_image, 1) : "—"));

    // Skin tone
    grid.appendChild(metricCard("Skin tone", prettyLabel(analysis.skin_tone_band),
      m.skin_intact != null ? (m.skin_intact ? "skin intact" : "skin broken") : ""));

    // Calibration
    var cal = analysis.calibration || {};
    grid.appendChild(metricCard("Calibration", prettyLabel(cal.status),
      cal.mm_per_px != null ? fmtNum(cal.mm_per_px, 4) + " mm/px" +
        (cal.uncertainty_pct != null ? " ±" + fmtNum(cal.uncertainty_pct, 0) + "%" : "") : "uncalibrated"));

    // Healing scores
    var hs = m.healing_scores;
    if (hs) {
      if (hs.push_partial_total != null || hs.push_size_subscore != null || hs.push_tissue_subscore != null) {
        var pushSub = [];
        if (hs.push_size_subscore != null) pushSub.push("size " + fmtNum(hs.push_size_subscore, 0));
        if (hs.push_tissue_subscore != null) pushSub.push("tissue " + fmtNum(hs.push_tissue_subscore, 0));
        grid.appendChild(metricCard("PUSH (partial)",
          hs.push_partial_total != null ? fmtNum(hs.push_partial_total, 0) : "—", pushSub.join(" · ")));
      }
      if (hs.design_r_size_subscore != null) {
        grid.appendChild(metricCard("DESIGN-R size", fmtNum(hs.design_r_size_subscore, 0)));
      }
      if (hs.granulation_percent != null) {
        grid.appendChild(metricCard("Granulation", fmtPct(hs.granulation_percent > 1 ? hs.granulation_percent / 100 : hs.granulation_percent, 0)));
      }
    }

    // Depth proxy
    var dp = m.depth_proxy;
    if (dp && dp.relative_depth_index != null) {
      grid.appendChild(metricCard("Depth (relative)", fmtNum(dp.relative_depth_index, 2),
        (dp.method ? dp.method + " · " : "") + "not physical"));
    }

    // Periwound
    var pw = m.periwound;
    if (pw && (pw.erythema_index != null || pw.maceration_suspected != null)) {
      grid.appendChild(metricCard("Periwound erythema",
        pw.erythema_index != null ? fmtNum(pw.erythema_index, 2) : "—",
        pw.maceration_suspected ? "maceration suspected" : ""));
    }

    // Color cues
    var cc = m.color_cues;
    if (cc) {
      grid.appendChild(metricCard("Open bed", cc.open_bed_fraction != null ? fmtPct(cc.open_bed_fraction, 0) : "—",
        cc.erythema_fraction != null ? "erythema " + fmtPct(cc.erythema_fraction, 0) : ""));
    }

    return grid;
  }

  /* ============================================================
     EVIDENCE + CAVEATS
     ============================================================ */
  function buildEvidence(analysis) {
    var d = analysis.decision || {};
    var wrap = el("div");
    if (d.narrative) {
      wrap.appendChild(el("p", { className: "narrative", text: d.narrative }));
    }
    var ev = d.evidence || [];
    if (ev.length) {
      var list = el("ul", { className: "evidence-list" });
      ev.forEach(function (item) {
        var li = el("li");
        var tag = (item.source ? item.source : "") + (item.code ? ":" + item.code : "");
        li.appendChild(el("span", { className: "ev-tag", text: tag || "note" }));
        li.appendChild(el("span", { className: "ev-desc", text: item.description || "" }));
        if (item.direction) li.appendChild(el("span", { className: "ev-dir", text: item.direction }));
        list.appendChild(li);
      });
      wrap.appendChild(list);
    }
    return wrap;
  }

  function buildCaveats(analysis) {
    var d = analysis.decision || {};
    var caveats = d.caveats || [];
    if (!caveats.length) return null;
    var icons = { info: "i", warning: "!", critical: "×" };
    var list = el("ul", { className: "caveats" });
    caveats.forEach(function (c) {
      var sev = (c.severity || "info").toLowerCase();
      if (["info", "warning", "critical"].indexOf(sev) === -1) sev = "info";
      var li = el("li", { className: "caveat " + sev });
      li.appendChild(el("span", { className: "cav-ico", text: icons[sev], attrs: { "aria-hidden": "true" } }));
      var body = el("span");
      body.appendChild(el("span", { className: "cav-sev", text: sev + ": " }));
      body.appendChild(document.createTextNode(c.message || ""));
      li.appendChild(body);
      list.appendChild(li);
    });
    return list;
  }

  function buildProvenance(analysis) {
    var p = analysis.provenance || {};
    var timings = analysis.stage_timings_ms || {};
    var total = 0;
    Object.keys(timings).forEach(function (k) { var v = timings[k]; if (typeof v === "number") total += v; });
    var line = el("div", { className: "provenance" });
    function part(label, val) {
      var s = el("span");
      s.appendChild(document.createTextNode(label + " "));
      s.appendChild(el("code", { text: val || "—" }));
      return s;
    }
    line.appendChild(part("seg:", p.segmentation_backend));
    line.appendChild(part("tissue:", p.tissue_backend));
    line.appendChild(part("staging:", p.staging_backend));
    if (total > 0) line.appendChild(el("span", { text: "total " + Math.round(total) + " ms" }));
    return line;
  }

  /* ---------- Section wrapper ---------- */
  function section(title, subEl, bodyNode) {
    var card = el("div", { className: "card" });
    if (title) {
      var h = el("h2", { className: "section-title", text: title });
      if (subEl) h.appendChild(subEl);
      card.appendChild(h);
    }
    if (bodyNode) card.appendChild(bodyNode);
    return card;
  }

  /* ============================================================
     RENDER ANALYSIS
     ============================================================ */
  function renderAnalysis(data) {
    var container = $("#analyzeResults");
    container.textContent = "";
    var analysis = data.analysis || {};

    // Grade banner card
    var bannerCard = el("div", { className: "card" });
    bannerCard.appendChild(buildGradeBanner(analysis));
    container.appendChild(bannerCard);

    // Gallery
    container.appendChild(section("Visualizations", null, buildGallery(data.images)));

    // Tissue composition
    var tsub = el("span", { className: "section-sub", text: analysis.metrics && analysis.metrics.tissue ? "  dominant: " + prettyLabel(analysis.metrics.tissue.dominant) : "" });
    container.appendChild(section("Tissue composition", tsub,
      buildTissueChart(analysis.metrics && analysis.metrics.tissue ? analysis.metrics.tissue.fractions : null)));

    // Metrics
    container.appendChild(section("Measurements", null, buildMetrics(analysis)));

    // Why this grade
    var whyCard = section("Why this grade", null, buildEvidence(analysis));
    var caveats = buildCaveats(analysis);
    if (caveats) {
      whyCard.appendChild(el("h3", { className: "section-title", text: "Caveats", attrs: { style: "margin-top:16px; font-size:14px;" } }));
      whyCard.appendChild(caveats);
    }
    container.appendChild(whyCard);

    // Disclaimer + provenance footer
    var footCard = el("div", { className: "card" });
    if (analysis.disclaimer) footCard.appendChild(el("p", { className: "muted", text: analysis.disclaimer, attrs: { style: "font-size:12.5px; margin:0 0 8px;" } }));
    footCard.appendChild(buildProvenance(analysis));
    container.appendChild(footCard);

    // announce
    container.setAttribute("aria-busy", "false");
  }

  /* ============================================================
     ANALYZE TAB LOGIC
     ============================================================ */
  function initAnalyzeTab() {
    var dropzone = $("#dropzone");
    var fileInput = $("#fileInput");
    var preview = $("#dzPreview");
    var previewImg = $("#previewImg");
    var fileNameEl = $("#fileName");
    var clearBtn = $("#clearFile");
    var analyzeBtn = $("#analyzeBtn");
    var errorBox = $("#analyzeError");
    var results = $("#analyzeResults");
    var dzInner = $(".dz-inner", dropzone);

    var currentFile = null;
    var previewUrl = null;
    var previewNote = $("#previewNote");

    // Browsers cannot render some clinical formats (TIFF/HEIC/DICOM/RAW): fall
    // back to a file glyph + note instead of a broken image. The server still
    // decodes and analyzes them.
    previewImg.addEventListener("load", function () {
      dropzone.classList.remove("preview-unavailable");
      if (previewNote) previewNote.hidden = true;
    });
    previewImg.addEventListener("error", function () {
      dropzone.classList.add("preview-unavailable");
      if (previewNote) previewNote.hidden = false;
    });

    function setFile(file) {
      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
      currentFile = file || null;
      dropzone.classList.remove("preview-unavailable");
      if (previewNote) previewNote.hidden = true;
      if (currentFile) {
        previewUrl = URL.createObjectURL(currentFile);
        previewImg.src = previewUrl;
        fileNameEl.textContent = currentFile.name;
        preview.hidden = false;
        dropzone.classList.add("has-file");
        if (dzInner) dzInner.style.display = "none";
        analyzeBtn.disabled = false;
        dropzone.setAttribute("aria-label", "Selected file: " + currentFile.name + ". Press Enter to choose a different file.");
      } else {
        previewImg.removeAttribute("src");
        fileNameEl.textContent = "";
        preview.hidden = true;
        dropzone.classList.remove("has-file");
        if (dzInner) dzInner.style.display = "";
        analyzeBtn.disabled = true;
        fileInput.value = "";
      }
    }

    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
    });

    dropzone.addEventListener("click", function (e) {
      if (e.target.closest(".link-btn") || e.target.closest(".dz-preview")) return;
      fileInput.click();
    });
    dropzone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    });
    ["dragenter", "dragover"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.add("dragover"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.remove("dragover"); });
    });
    dropzone.addEventListener("drop", function (e) {
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files[0]) setFile(files[0]);
    });
    clearBtn.addEventListener("click", function (e) { e.stopPropagation(); setFile(null); results.textContent = ""; errorBox.hidden = true; });

    analyzeBtn.addEventListener("click", function () {
      if (!currentFile) return;
      errorBox.hidden = true;
      results.textContent = "";
      setBusy(analyzeBtn, results, true);

      var fd = new FormData();
      fd.append("file", currentFile);
      appendIfNum(fd, "mm_per_px", $("#mmPerPx").value);
      appendIfNum(fd, "marker_mm", $("#markerMm").value);
      appendIfNum(fd, "coin_mm", $("#coinMm").value);
      fd.append("include_images", "true");

      fetch("/v1/analyze", { method: "POST", body: fd })
        .then(handleJson)
        .then(function (data) {
          renderAnalysis(data);
        })
        .catch(function (err) {
          showError(errorBox, err);
        })
        .then(function () {
          setBusy(analyzeBtn, results, false);
        });
    });
  }

  /* ============================================================
     HEALING TREND TAB
     ============================================================ */
  function initTrendTab() {
    var rowsWrap = $("#visitRows");
    var addBtn = $("#addVisitBtn");
    var multiInput = $("#multiFileInput");
    var compareBtn = $("#compareBtn");
    var errorBox = $("#trendError");
    var results = $("#trendResults");

    var visits = []; // {id, file, day}
    var seq = 0;

    function defaultDay(index) { return index * 14; }

    function refreshCompareEnabled() {
      var withFiles = visits.filter(function (v) { return v.file; });
      compareBtn.disabled = withFiles.length < 2;
    }

    function renderRows() {
      rowsWrap.textContent = "";
      visits.forEach(function (v, i) {
        var row = el("div", { className: "visit-row" });
        row.appendChild(el("span", { className: "visit-num", text: String(i + 1) }));

        var fileCell = el("div", { className: "visit-file" });
        if (v.file) {
          fileCell.appendChild(el("span", { className: "vf-name", text: v.file.name }));
        } else {
          var fi = el("input", { attrs: { type: "file", accept: "image/*", "aria-label": "Image for visit " + (i + 1) } });
          fi.addEventListener("change", function () {
            if (fi.files && fi.files[0]) { v.file = fi.files[0]; renderRows(); refreshCompareEnabled(); }
          });
          fileCell.appendChild(fi);
        }
        row.appendChild(fileCell);

        var dayCell = el("div", { className: "visit-day" });
        var dl = el("label", { text: "Day", attrs: { for: "day-" + v.id } });
        var di = el("input", { attrs: { type: "number", id: "day-" + v.id, min: "0", step: "1", value: v.day } });
        di.addEventListener("input", function () { v.day = di.value; });
        dayCell.appendChild(dl);
        dayCell.appendChild(di);
        row.appendChild(dayCell);

        var rm = el("button", { className: "remove-visit", text: "×", attrs: { type: "button", "aria-label": "Remove visit " + (i + 1) } });
        rm.addEventListener("click", function () {
          visits = visits.filter(function (x) { return x.id !== v.id; });
          renderRows(); refreshCompareEnabled();
        });
        row.appendChild(rm);

        rowsWrap.appendChild(row);
      });
    }

    function addVisit(file) {
      var idx = visits.length;
      visits.push({ id: ++seq, file: file || null, day: String(defaultDay(idx)) });
      renderRows(); refreshCompareEnabled();
    }

    addBtn.addEventListener("click", function () { addVisit(null); });
    multiInput.addEventListener("change", function () {
      var files = multiInput.files;
      if (files) {
        for (var i = 0; i < files.length; i++) addVisit(files[i]);
      }
      multiInput.value = "";
    });

    // start with two empty rows
    addVisit(null);
    addVisit(null);

    compareBtn.addEventListener("click", function () {
      var withFiles = visits.filter(function (v) { return v.file; });
      if (withFiles.length < 2) return;
      errorBox.hidden = true;
      results.textContent = "";
      setBusy(compareBtn, results, true);

      var fd = new FormData();
      withFiles.forEach(function (v) { fd.append("files", v.file); });
      var days = withFiles.map(function (v) { return (v.day === "" || v.day == null) ? "" : v.day; }).join(",");
      if (days.replace(/,/g, "").length) fd.append("days", days);
      appendIfNum(fd, "mm_per_px", $("#trendMmPerPx").value);
      var patient = $("#patientLabel").value;
      if (patient && patient.trim()) fd.append("patient", patient.trim());

      fetch("/v1/compare", { method: "POST", body: fd })
        .then(handleJson)
        .then(function (data) { renderTrend(data); })
        .catch(function (err) { showError(errorBox, err); })
        .then(function () { setBusy(compareBtn, results, false); });
    });
  }

  /* ============================================================
     RENDER TREND
     ============================================================ */
  function renderTrend(data) {
    var container = $("#trendResults");
    container.textContent = "";
    var trend = data.trend || {};
    var thumbs = data.thumbnails || [];
    var points = trend.points || [];
    var unit = trend.unit || "px";

    // Trajectory banner
    var bannerCard = el("div", { className: "card" });
    var banner = el("div", { className: "traj-banner" });
    var traj = trend.trajectory || "indeterminate";
    var tcolor = TRAJ_COLORS[traj] || TRAJ_COLORS.indeterminate;
    var pill = el("span", { className: "traj-pill", attrs: { style: "background:" + rgb(tcolor) } });
    pill.appendChild(el("span", { className: "dot" }));
    pill.appendChild(document.createTextNode(prettyLabel(traj)));
    banner.appendChild(pill);
    if (trend.percent_area_reduction != null) {
      var par = trend.percent_area_reduction;
      var sign = par >= 0 ? "−" : "+"; // positive PAR = shrinking
      banner.appendChild(el("span", { className: "traj-delta", text: sign + fmtNum(Math.abs(par), 1) + "% area" }));
    }
    if (trend.patient_ref) banner.appendChild(el("span", { className: "traj-patient", text: trend.patient_ref }));
    bannerCard.appendChild(banner);
    container.appendChild(bannerCard);

    // Filmstrip
    container.appendChild(section("Visits", null, buildFilmstrip(points, thumbs, unit)));

    // Area over time chart
    var areaSub = el("span", { className: "section-sub", text: "  area in " + (unit === "cm2" ? "cm²" : "px") });
    container.appendChild(section("Area over time", areaSub, buildAreaChart(trend)));

    // Tissue over time
    container.appendChild(section("Tissue composition over time", null, buildTissueTrendChart(points)));

    // Stat cards
    container.appendChild(section("Summary", null, buildTrendStats(trend)));

    // Notes
    if (trend.notes && trend.notes.length) {
      var notes = el("ul", { className: "notes-list" });
      trend.notes.forEach(function (n) { notes.appendChild(el("li", { text: n })); });
      container.appendChild(section("Notes", null, notes));
    }

    container.setAttribute("aria-busy", "false");
  }

  function buildFilmstrip(points, thumbs, unit) {
    var strip = el("div", { className: "filmstrip" });
    if (!points.length) { strip.appendChild(el("p", { className: "empty-hint", text: "No visits." })); return strip; }
    points.forEach(function (p, i) {
      var fig = el("figure");
      var src = thumbs[i];
      if (src) {
        var btn = el("button", { className: "thumb", attrs: { type: "button", "aria-label": "Enlarge visit " + (p.label || i + 1) } });
        var img = el("img", { attrs: { src: src, alt: "Visit " + (p.label || i + 1) + " wound thumbnail", loading: "lazy" } });
        btn.appendChild(img);
        btn.addEventListener("click", function () { lightbox.open(src, "Visit " + (p.label || i + 1), captionFor(p, unit)); });
        fig.appendChild(btn);
      }
      fig.appendChild(el("figcaption", { text: captionFor(p, unit) }));
      strip.appendChild(fig);
    });
    return strip;
  }
  function captionFor(p, unit) {
    var area = unit === "cm2"
      ? (p.area_cm2 != null ? fmtNum(p.area_cm2, 2) + " cm²" : "—")
      : (p.area_px != null ? fmtNum(p.area_px, 0) + " px" : "—");
    var dayTxt = p.day != null ? "Day " + p.day : (p.label || "");
    return dayTxt + " · " + prettyLabel(p.stage) + " · " + area;
  }

  /* ---------- Area-over-time line chart ---------- */
  function buildAreaChart(trend) {
    var points = (trend.points || []).slice();
    var unit = trend.unit || "px";
    var wrap = el("div");
    if (points.length < 2) {
      wrap.appendChild(el("p", { className: "empty-hint", text: "Need at least two visits to plot a trend." }));
      return wrap;
    }
    var getArea = function (p) { return unit === "cm2" ? p.area_cm2 : p.area_px; };
    // fallback to px if cm2 missing
    if (unit === "cm2" && points.some(function (p) { return p.area_cm2 == null; })) {
      unit = "px";
      getArea = function (p) { return p.area_px; };
    }

    var days = points.map(function (p, i) { return p.day != null ? p.day : i; });
    var areas = points.map(getArea);

    var W = 640, H = 300, padL = 54, padR = 24, padT = 20, padB = 42;
    var plotW = W - padL - padR, plotH = H - padT - padB;

    var maxDay = Math.max.apply(null, days);
    var projDays = trend.projected_days_to_closure;
    if (projDays != null && isFinite(projDays)) maxDay = Math.max(maxDay, days[days.length - 1] + Math.min(projDays, 400));
    if (maxDay <= 0) maxDay = 1;
    var maxArea = Math.max.apply(null, areas.filter(function (a) { return a != null; }));
    if (!(maxArea > 0)) maxArea = 1;
    maxArea = maxArea * 1.12;

    function sx(d) { return padL + (d / maxDay) * plotW; }
    function sy(a) { return padT + plotH - (a / maxArea) * plotH; }

    var svg = svgEl("svg", {
      viewBox: "0 0 " + W + " " + H, class: "chart-svg",
      role: "img", "aria-label": "Wound area over time line chart"
    });

    // gridlines + y axis labels
    var ticks = 4;
    for (var t = 0; t <= ticks; t++) {
      var val = (maxArea / ticks) * t;
      var y = sy(val);
      svg.appendChild(svgEl("line", { x1: padL, y1: y, x2: W - padR, y2: y, class: "grid-line" }));
      var lbl = svgEl("text", { x: padL - 8, y: y + 4, "text-anchor": "end", class: "axis-label" });
      lbl.textContent = fmtNum(val, unit === "cm2" ? 1 : 0);
      svg.appendChild(lbl);
    }
    // x axis labels
    days.forEach(function (d) {
      var x = sx(d);
      var lbl = svgEl("text", { x: x, y: H - padB + 20, "text-anchor": "middle", class: "axis-label" });
      lbl.textContent = "D" + d;
      svg.appendChild(lbl);
    });
    var axisTitle = svgEl("text", { x: padL + plotW / 2, y: H - 4, "text-anchor": "middle", class: "axis-label" });
    axisTitle.textContent = "Day";
    svg.appendChild(axisTitle);

    var color = stageColor(points[points.length - 1].stage);

    // line path
    var dPath = "";
    points.forEach(function (p, i) {
      var a = getArea(p);
      if (a == null) return;
      dPath += (dPath ? " L" : "M") + sx(days[i]) + " " + sy(a);
    });
    var line = svgEl("path", { d: dPath, fill: "none", stroke: rgb(color), "stroke-width": "2.5", "stroke-linejoin": "round", "stroke-linecap": "round" });
    svg.appendChild(line);

    // reduced-motion safe draw animation
    if (!prefersReducedMotion()) {
      try {
        var len = line.getTotalLength ? line.getTotalLength() : 0;
        if (len) {
          line.style.strokeDasharray = len;
          line.style.strokeDashoffset = len;
          requestAnimationFrame(function () {
            line.style.transition = "stroke-dashoffset 1s ease";
            requestAnimationFrame(function () { line.style.strokeDashoffset = "0"; });
          });
        }
      } catch (e) { /* getTotalLength may be unavailable */ }
    }

    // projected closure dashed segment
    if (projDays != null && isFinite(projDays) && projDays > 0) {
      var lastX = sx(days[days.length - 1]);
      var lastY = sy(getArea(points[points.length - 1]));
      var closeX = sx(days[days.length - 1] + projDays);
      var proj = svgEl("path", {
        d: "M" + lastX + " " + lastY + " L" + closeX + " " + sy(0),
        fill: "none", stroke: rgb(color), "stroke-width": "2",
        "stroke-dasharray": "5 5", opacity: "0.6"
      });
      svg.appendChild(proj);
      var projLbl = svgEl("text", { x: closeX, y: sy(0) - 8, "text-anchor": "end", class: "axis-label" });
      projLbl.textContent = "≈closure";
      svg.appendChild(projLbl);
    }

    // hover guide line (hidden)
    var guide = svgEl("line", { x1: 0, y1: padT, x2: 0, y2: padT + plotH, stroke: rgb(color), "stroke-width": "1", opacity: "0", "stroke-dasharray": "3 3" });
    svg.appendChild(guide);

    var baseArea = trend.baseline_area != null ? trend.baseline_area : areas[0];

    // data points
    points.forEach(function (p, i) {
      var a = getArea(p);
      if (a == null) return;
      var cx = sx(days[i]), cy = sy(a);
      var parVs = baseArea ? ((baseArea - a) / baseArea * 100) : null;
      var areaTxt = unit === "cm2" ? fmtNum(a, 2) + " cm²" : fmtNum(a, 0) + " px";
      var tipHtml = "<span class='tip-title'>Day " + tipEscape(days[i]) + "</span><br>" +
        tipEscape(areaTxt) + (parVs != null ? "<br>PAR " + tipEscape(fmtNum(parVs, 1)) + "%" : "");
      var dot = svgEl("circle", {
        cx: cx, cy: cy, r: 5, fill: rgb(color), stroke: "var(--surface)", "stroke-width": "2",
        tabindex: "0", role: "img",
        "aria-label": "Day " + days[i] + ", area " + areaTxt + (parVs != null ? ", PAR " + fmtNum(parVs, 0) + "%" : "")
      });
      dot.classList.add("data-pt");
      function enter(ev) {
        var pt = ev.touches ? ev.touches[0] : ev;
        tip.show(tipHtml, pt.clientX, pt.clientY);
        guide.setAttribute("x1", cx); guide.setAttribute("x2", cx); guide.setAttribute("opacity", "0.5");
      }
      dot.addEventListener("mousemove", enter);
      dot.addEventListener("mouseenter", enter);
      dot.addEventListener("mouseleave", function () { tip.hide(); guide.setAttribute("opacity", "0"); });
      dot.addEventListener("focus", function () {
        var b = dot.getBoundingClientRect();
        tip.show(tipHtml, b.left + b.width / 2, b.top);
        guide.setAttribute("x1", cx); guide.setAttribute("x2", cx); guide.setAttribute("opacity", "0.5");
      });
      dot.addEventListener("blur", function () { tip.hide(); guide.setAttribute("opacity", "0"); });
      svg.appendChild(dot);
    });

    wrap.appendChild(svg);
    return wrap;
  }

  /* ---------- Tissue over time stacked bars ---------- */
  function buildTissueTrendChart(points) {
    var wrap = el("div", { className: "chart-card" });
    if (!points.length) { wrap.appendChild(el("p", { className: "empty-hint", text: "No visits." })); return wrap; }

    // collect tissue keys present
    var keySet = {};
    points.forEach(function (p) {
      var f = p.tissue_fractions || {};
      Object.keys(f).forEach(function (k) { if (f[k] > 0.0001) keySet[k] = true; });
    });
    var keys = Object.keys(keySet);
    if (!keys.length) { wrap.appendChild(el("p", { className: "empty-hint", text: "No tissue data across visits." })); return wrap; }

    var n = points.length;
    var W = 480, H = 260, padL = 34, padR = 12, padT = 12, padB = 34;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var bw = Math.min(56, (plotW / n) * 0.62);
    var gap = plotW / n;

    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "chart-svg", role: "img", "aria-label": "Stacked tissue composition per visit" });

    // y gridlines (0-100%)
    [0, 0.25, 0.5, 0.75, 1].forEach(function (frac) {
      var y = padT + plotH - frac * plotH;
      svg.appendChild(svgEl("line", { x1: padL, y1: y, x2: W - padR, y2: y, class: "grid-line" }));
      var lbl = svgEl("text", { x: padL - 6, y: y + 4, "text-anchor": "end", class: "axis-label" });
      lbl.textContent = (frac * 100) + "%";
      svg.appendChild(lbl);
    });

    points.forEach(function (p, i) {
      var f = p.tissue_fractions || {};
      var total = 0;
      keys.forEach(function (k) { total += (f[k] || 0); });
      if (total <= 0) total = 1;
      var x = padL + gap * i + (gap - bw) / 2;
      var yCursor = padT + plotH;
      keys.forEach(function (k) {
        var frac = (f[k] || 0) / total;
        if (frac <= 0) return;
        var segH = frac * plotH;
        yCursor -= segH;
        var rect = svgEl("rect", {
          x: x, y: yCursor, width: bw, height: segH, fill: rgb(tissueColor(k)),
          tabindex: "0", role: "img",
          "aria-label": (p.day != null ? "Day " + p.day : "Visit " + (i + 1)) + ": " + prettyLabel(k) + " " + fmtPct(frac)
        });
        rect.classList.add("data-pt");
        var th = "<span class='tip-title'>" + tipEscape(prettyLabel(k)) + "</span><br>" +
          tipEscape((p.day != null ? "Day " + p.day : "Visit " + (i + 1))) + " · " + tipEscape(fmtPct(frac));
        rect.addEventListener("mousemove", function (ev) { tip.show(th, ev.clientX, ev.clientY); });
        rect.addEventListener("mouseleave", function () { tip.hide(); });
        rect.addEventListener("focus", function () { var b = rect.getBoundingClientRect(); tip.show(th, b.left + b.width / 2, b.top); });
        rect.addEventListener("blur", function () { tip.hide(); });
        svg.appendChild(rect);
      });
      var lbl = svgEl("text", { x: x + bw / 2, y: H - padB + 18, "text-anchor": "middle", class: "axis-label" });
      lbl.textContent = p.day != null ? "D" + p.day : "V" + (i + 1);
      svg.appendChild(lbl);
    });

    wrap.appendChild(svg);

    // legend
    var legend = el("div", { className: "legend", attrs: { role: "list" } });
    keys.forEach(function (k) {
      var item = el("div", { className: "legend-item", attrs: { role: "listitem" } });
      item.appendChild(el("span", { className: "legend-dot", attrs: { style: "background:" + rgb(tissueColor(k)) } }));
      item.appendChild(el("span", { className: "legend-name", text: prettyLabel(k) }));
      legend.appendChild(item);
    });
    wrap.appendChild(legend);
    return wrap;
  }

  /* ---------- Trend stat cards ---------- */
  function buildTrendStats(trend) {
    var unit = trend.unit === "cm2" ? "cm²" : "px";
    var grid = el("div", { className: "stat-row" });

    function stat(label, value, note, noteClass) {
      var s = el("div", { className: "stat" });
      s.appendChild(el("div", { className: "s-label", text: label }));
      s.appendChild(el("div", { className: "s-value", text: value }));
      if (note) s.appendChild(el("div", { className: "s-note" + (noteClass ? " " + noteClass : ""), text: note }));
      return s;
    }

    grid.appendChild(stat("Area reduction",
      trend.percent_area_reduction != null ? fmtNum(trend.percent_area_reduction, 1) + "%" : "—",
      trend.percent_area_reduction != null ? (trend.percent_area_reduction >= 0 ? "shrinking" : "growing") : "",
      trend.percent_area_reduction != null ? (trend.percent_area_reduction >= 0 ? "good" : "bad") : ""));

    grid.appendChild(stat("Healing rate",
      trend.healing_rate_per_week != null ? fmtNum(trend.healing_rate_per_week, 2) + " " + unit + "/wk" : "—",
      trend.healing_rate_pct_per_week != null ? fmtNum(trend.healing_rate_pct_per_week, 1) + "%/wk" : "",
      trend.healing_rate_per_week != null ? (trend.healing_rate_per_week <= 0 ? "good" : "bad") : ""));

    var parNote = "", parClass = "";
    if (trend.par_at_4_weeks != null) {
      var onTrack = trend.par_at_4_weeks >= 40;
      parNote = onTrack ? "on track" : "below 40%";
      parClass = onTrack ? "good" : "bad";
    }
    grid.appendChild(stat("4-week PAR",
      trend.par_at_4_weeks != null ? fmtNum(trend.par_at_4_weeks, 1) + "%" : "—", parNote, parClass));

    grid.appendChild(stat("Projected closure",
      (trend.projected_days_to_closure != null && isFinite(trend.projected_days_to_closure)) ? "≈ " + Math.round(trend.projected_days_to_closure) + " d" : "—",
      trend.likely_to_heal != null ? (trend.likely_to_heal ? "likely to heal" : "unlikely") : "",
      trend.likely_to_heal != null ? (trend.likely_to_heal ? "good" : "bad") : ""));

    if (trend.push_trend) {
      grid.appendChild(stat("PUSH trend", prettyLabel(trend.push_trend), "",
        trend.push_trend === "improving" ? "good" : (trend.push_trend === "worsening" ? "bad" : "")));
    }

    grid.appendChild(stat("Baseline → latest",
      fmtNum(trend.baseline_area, unit === "cm²" ? 2 : 0) + " → " + fmtNum(trend.latest_area, unit === "cm²" ? 2 : 0),
      unit));

    return grid;
  }

  /* ============================================================
     Shared network helpers
     ============================================================ */
  function appendIfNum(fd, key, val) {
    if (val == null) return;
    var s = String(val).trim();
    if (s === "") return;
    var n = Number(s);
    if (!isNaN(n)) fd.append(key, s);
  }

  function handleJson(resp) {
    return resp.text().then(function (text) {
      var data = null;
      try { data = text ? JSON.parse(text) : {}; } catch (e) { data = null; }
      if (!resp.ok) {
        var msg = (data && data.error) ? data.error : ("Request failed (" + resp.status + ")");
        throw new Error(msg);
      }
      if (data && data.error) throw new Error(data.error);
      if (!data) throw new Error("Unexpected response from server.");
      return data;
    });
  }

  function showError(box, err) {
    box.textContent = err && err.message ? err.message : "Something went wrong.";
    box.hidden = false;
  }

  function setBusy(btn, resultsRegion, busy) {
    btn.setAttribute("aria-busy", busy ? "true" : "false");
    btn.disabled = busy;
    if (resultsRegion) resultsRegion.setAttribute("aria-busy", busy ? "true" : "false");
    if (!busy) btn.disabled = false;
  }

  /* ============================================================
     Boot
     ============================================================ */
  function boot() {
    lightbox.init();
    tip.init();
    var disclaimer = initDisclaimer();
    initTheme();
    initTabs(disclaimer);
    initAnalyzeTab();
    initTrendTab();

    // global Escape closes tooltip
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") tip.hide(); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
