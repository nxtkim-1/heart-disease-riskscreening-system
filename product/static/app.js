(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // Escape any server/API-provided text before it is interpolated into innerHTML.
  const esc = (v) =>
    String(v === null || v === undefined ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  // Tab navigation
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      $("tab-" + btn.dataset.tab).classList.add("active");
    });
  });

  // Unknown checkboxes disable numeric fields
  document.querySelectorAll("[data-unknown-for]").forEach((cb) => {
    cb.addEventListener("change", () => {
      const target = $(cb.dataset.unknownFor);
      if (target) target.disabled = cb.checked;
    });
  });

  function fillPatientForm(features) {
    if (!features) return;
    $("age").value = features.age ?? 55;
    $("sex").value = features.sex ?? "Male";
    $("cp").value = features.cp ?? "asymptomatic";

    ["trestbps", "chol", "thalch", "oldpeak"].forEach((id) => {
      const cb = document.querySelector(`[data-unknown-for="${id}"]`);
      const input = $(id);
      const val = features[id];
      if (val === null || val === undefined) {
        if (cb) cb.checked = true;
        input.disabled = true;
        input.value = "";
      } else {
        if (cb) cb.checked = false;
        input.disabled = false;
        input.value = String(val);
      }
    });

    ["fbs", "exang"].forEach((id) => {
      const v = features[id];
      $(id).value = v === true ? "true" : v === false ? "false" : "";
    });
    ["slope", "restecg"].forEach((id) => {
      $(id).value = features[id] ?? "";
    });
  }

  function readPatientForm() {
    const optionalNum = (id) => {
      const cb = document.querySelector(`[data-unknown-for="${id}"]`);
      if (cb && cb.checked) return null;
      const val = $(id).value.trim();
      return val === "" ? null : Number(val);
    };
    const optionalBool = (id) => {
      const val = $(id).value;
      if (val === "true") return true;
      if (val === "false") return false;
      return null;
    };
    const optionalStr = (id) => {
      const val = $(id).value.trim();
      return val === "" ? null : val;
    };
    return {
      age: Number($("age").value),
      sex: $("sex").value,
      cp: $("cp").value,
      trestbps: optionalNum("trestbps"),
      chol: optionalNum("chol"),
      fbs: optionalBool("fbs"),
      restecg: optionalStr("restecg"),
      thalch: optionalNum("thalch"),
      exang: optionalBool("exang"),
      oldpeak: optionalNum("oldpeak"),
      slope: optionalStr("slope"),
    };
  }

  function showFormError(msg) {
    const el = $("form-error");
    el.textContent = msg;
    el.classList.remove("hidden");
  }

  function clearFormError() {
    $("form-error").classList.add("hidden");
    $("form-error").textContent = "";
  }

  function pct(v) {
    return v === null || v === undefined ? "N/A" : `${(v * 100).toFixed(1)}%`;
  }

  function drawRiskArc(prob, ciLow, ciHigh, threshold) {
    const svg = $("risk-arc-svg");
    if (!svg) return;
    const cx = 160;
    const cy = 152;
    const r = 114;
    const clamp01 = (v) => Math.max(0, Math.min(1, v));

    const pt = (p) => {
      const ang = Math.PI * (1 - clamp01(p));
      return [cx + r * Math.cos(ang), cy - r * Math.sin(ang)];
    };

    const arcPath = (p1, p2) => {
      const a = clamp01(p1);
      const b = clamp01(p2);
      if (b <= a) return "";
      const [x1, y1] = pt(a);
      const [x2, y2] = pt(b);
      return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
    };

    const [txInX, txInY] = (() => {
      const ang = Math.PI * (1 - clamp01(threshold));
      return [cx + (r - 16) * Math.cos(ang), cy - (r - 16) * Math.sin(ang)];
    })();
    const [txOutX, txOutY] = (() => {
      const ang = Math.PI * (1 - clamp01(threshold));
      return [cx + (r + 16) * Math.cos(ang), cy - (r + 16) * Math.sin(ang)];
    })();
    const [px, py] = pt(prob);
    const activeColor = prob >= threshold ? "#B8423B" : "#256B56";

    const parts = [
      `<path d="${arcPath(0, 1)}" fill="none" stroke="#EAE2D7" stroke-width="16" stroke-linecap="round" />`,
      `<path d="${arcPath(0, threshold)}" fill="none" stroke="#D3EAE0" stroke-width="16" stroke-linecap="round" />`,
    ];

    if (ciLow !== null && ciHigh !== null && ciHigh > ciLow) {
      parts.push(
        `<path d="${arcPath(ciLow, ciHigh)}" fill="none" stroke="${activeColor}" stroke-opacity="0.28" stroke-width="24" stroke-linecap="round" />`
      );
    }

    if (prob > 0.005) {
      parts.push(
        `<path d="${arcPath(0, prob)}" fill="none" stroke="${activeColor}" stroke-width="10" stroke-linecap="round" />`
      );
    }

    parts.push(
      `<line x1="${txInX.toFixed(1)}" y1="${txInY.toFixed(1)}" x2="${txOutX.toFixed(1)}" y2="${txOutY.toFixed(1)}" stroke="#1B4D54" stroke-width="2.5" stroke-linecap="round" />`,
      `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="8" fill="${activeColor}" stroke="#FFFFFF" stroke-width="3" />`,
      `<text x="36" y="168" font-size="10" font-family="IBM Plex Mono, monospace" fill="#5C6773">0%</text>`,
      `<text x="272" y="168" font-size="10" font-family="IBM Plex Mono, monospace" fill="#5C6773">100%</text>`
    );

    svg.innerHTML = parts.join("");
  }

  function renderPrediction(data) {
    $("result-placeholder").classList.add("hidden");
    $("result-body").classList.remove("hidden");

    const banner = $("result-banner");
    const isBorderline = data.risk_band === "borderline_near_threshold";
    banner.className = "alert " + (data.flagged ? (isBorderline ? "alert-warn" : "alert-danger") : "alert-ok");
    banner.innerHTML = data.flagged
      ? `<strong>Above ${pct(data.threshold)} Sensitivity Cutoff:</strong> Estimated score ${pct(data.probability)}. ${esc(data.uncertainty.threshold_proximity_note)}`
      : `<strong>Below ${pct(data.threshold)} Sensitivity Cutoff:</strong> Estimated score ${pct(data.probability)}. ${esc(data.uncertainty.threshold_proximity_note)}`;

    $("res-prob").textContent = pct(data.probability);
    $("res-band-label").textContent =
      data.risk_band === "borderline_near_threshold"
        ? "Borderline near 39% cutoff"
        : data.flagged
        ? "Elevated above 39% cutoff"
        : "Below 39% screening cutoff";

    const u = data.uncertainty;
    drawRiskArc(data.probability, u.bootstrap_ci_low, u.bootstrap_ci_high, data.threshold);

    if (u.bootstrap_ci_low !== null && u.bootstrap_ci_high !== null) {
      $("res-ci").textContent = `${pct(u.bootstrap_ci_low)} – ${pct(u.bootstrap_ci_high)}`;
      $("res-ci-sub").textContent = `B=${u.bootstrap_samples} train resamples`;
      $("gauge-ci").style.left = `${(u.bootstrap_ci_low * 100).toFixed(1)}%`;
      $("gauge-ci").style.width = `${Math.max(1, (u.bootstrap_ci_high - u.bootstrap_ci_low) * 100).toFixed(1)}%`;
    } else {
      $("res-ci").textContent = "N/A";
      $("gauge-ci").style.width = "0%";
    }
    $("gauge-point").style.left = `${(data.probability * 100).toFixed(1)}%`;
    $("gauge-threshold").style.left = `${(data.threshold * 100).toFixed(1)}%`;
    $("res-cal").textContent = pct(data.calibrated_reference_probability);

    const impWarn = $("res-imputation-warn");
    if (u.imputation_warning) {
      impWarn.textContent = u.imputation_warning;
      impWarn.classList.remove("hidden");
    } else {
      impWarn.classList.add("hidden");
    }
    $("res-shift-warn").textContent = `${u.population_shift_warning} ${u.calibration_note}`;

    const contribs = $("res-contribs");
    contribs.innerHTML = "";
    const maxAbs = Math.max(0.5, ...data.top_contributions.map((c) => Math.abs(c.log_odds_contribution)));
    data.top_contributions.forEach((c) => {
      const row = document.createElement("div");
      row.className = "contrib-row";
      const halfWidth = Math.min(50, (Math.abs(c.log_odds_contribution) / maxAbs) * 50);
      const cls = c.direction === "increases_risk" ? "contrib-up" : "contrib-dn";
      const sign = c.log_odds_contribution > 0 ? "+" : "";
      row.innerHTML = `
        <span>${esc(c.label)}</span>
        <div class="contrib-diverge">
          <div class="contrib-center-line"></div>
          <div class="contrib-bar-fill ${cls}" style="width:${halfWidth.toFixed(1)}%"></div>
        </div>
        <span class="contrib-val">${sign}${c.log_odds_contribution.toFixed(2)}</span>
      `;
      contribs.appendChild(row);
    });
  }

  $("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFormError();
    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readPatientForm()),
      });
      const body = await resp.json();
      if (!resp.ok) {
        const detail = body.details ? body.details.map((d) => `${d.field}: ${d.message}`).join(" | ") : body.message;
        showFormError(detail || "Prediction request failed.");
        return;
      }
      renderPrediction(body);
    } catch (err) {
      showFormError("Could not reach the screening API.");
    }
  });

  $("local-explain-btn").addEventListener("click", async () => {
    clearFormError();
    try {
      const resp = await fetch("/api/assistant/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          use_external_meta_api: false,
          patient: readPatientForm(),
          question: "Explain this screening result, its main drivers, and its uncertainty in plain language.",
        }),
      });
      const body = await resp.json();
      if (!resp.ok) {
        showFormError(body.message || "Local explanation failed.");
        return;
      }
      renderPrediction(body.prediction);
      $("local-explain-card").classList.remove("hidden");
      $("local-explain-text").textContent = body.explanation;
      const ul = $("local-explain-takeaways");
      ul.innerHTML = "";
      body.key_takeaways.forEach((t) => {
        const li = document.createElement("li");
        li.textContent = t;
        ul.appendChild(li);
      });
    } catch (err) {
      showFormError("Could not reach the local explanation endpoint.");
    }
  });

  $("clear-session-btn").addEventListener("click", async () => {
    $("predict-form").reset();
    document.querySelectorAll(".qf-pill").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll("[data-unknown-for]").forEach((cb) => {
      cb.checked = false;
      const target = $(cb.dataset.unknownFor);
      if (target) target.disabled = false;
    });
    clearFormError();
    $("result-body").classList.add("hidden");
    $("result-placeholder").classList.remove("hidden");
    $("local-explain-card").classList.add("hidden");
    $("assistant-output").classList.add("hidden");
    $("assistant-placeholder").classList.remove("hidden");
    $("assistant-error").classList.add("hidden");
    try {
      await fetch("/api/session/clear", { method: "POST" });
    } catch (_) {}
  });

  function drawCalibrationSvg(oofBins, lohoBins, histBins) {
    const svg = $("cal-svg");
    const padL = 46, padR = 16, padT = 16, padB = 38;
    const W = 460 - padL - padR;
    const H = 320 - padT - padB;
    const sx = (v) => padL + v * W;
    const sy = (v) => padT + (1 - v) * H;

    const parts = [];
    for (let t = 0; t <= 1.001; t += 0.2) {
      const x = sx(t).toFixed(1), y = sy(t).toFixed(1);
      parts.push(`<line x1="${padL}" y1="${y}" x2="${padL + W}" y2="${y}" stroke="#EFE7DC" />`);
      parts.push(`<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + H}" stroke="#EFE7DC" />`);
      parts.push(`<text x="${padL - 8}" y="${Number(y) + 4}" font-size="10" text-anchor="end" fill="#5C6773">${Math.round(t * 100)}%</text>`);
      parts.push(`<text x="${x}" y="${padT + H + 15}" font-size="10" text-anchor="middle" fill="#5C6773">${Math.round(t * 100)}%</text>`);
    }
    parts.push(`<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="#9E9386" stroke-dasharray="4,4" />`);

    const addSeries = (bins, color, label) => {
      if (!bins || !bins.length) return;
      const pts = bins.map((b) => `${sx(b.mean_predicted).toFixed(1)},${sy(b.observed_rate).toFixed(1)}`).join(" ");
      parts.push(`<polyline fill="none" stroke="${color}" stroke-width="2.3" points="${pts}" />`);
      bins.forEach((b) => {
        const cx = sx(b.mean_predicted).toFixed(1);
        const cy = sy(b.observed_rate).toFixed(1);
        if (b.observed_ci_low !== null && b.observed_ci_high !== null) {
          parts.push(`<line x1="${cx}" y1="${sy(b.observed_ci_low).toFixed(1)}" x2="${cx}" y2="${sy(b.observed_ci_high).toFixed(1)}" stroke="${color}" stroke-opacity="0.45" stroke-width="1.6" />`);
        }
        parts.push(
          `<circle cx="${cx}" cy="${cy}" r="3.8" fill="${color}" stroke="#fff" stroke-width="1"><title>${esc(label)}: predicted ${pct(b.mean_predicted)}, observed ${pct(b.observed_rate)} (n=${b.n})</title></circle>`
        );
      });
    };

    addSeries(oofBins, "#1B4D54", "OOF Train CV");
    addSeries(lohoBins, "#B8423B", "LOHO (Unseen Site)");
    addSeries(histBins, "#9A5B13", "Historical Baseline");

    parts.push(`<text x="${padL + W / 2}" y="312" font-size="11" text-anchor="middle" fill="#1F2933">Mean Predicted Probability</text>`);
    parts.push(`<rect x="${padL + 8}" y="${padT + 6}" width="10" height="3" fill="#1B4D54" /><text x="${padL + 22}" y="${padT + 10}" font-size="10" fill="#1F2933">OOF Train CV (n=736)</text>`);
    parts.push(`<rect x="${padL + 8}" y="${padT + 20}" width="10" height="3" fill="#B8423B" /><text x="${padL + 22}" y="${padT + 24}" font-size="10" fill="#1F2933">LOHO Unseen Hospital (n=920)</text>`);
    parts.push(`<rect x="${padL + 8}" y="${padT + 34}" width="10" height="3" fill="#9A5B13" /><text x="${padL + 22}" y="${padT + 38}" font-size="10" fill="#1F2933">Frozen Baseline Hist (n=184)</text>`);

    svg.innerHTML = parts.join("");
  }

  function renderSafeguards(health, modelInfo) {
    const grid = $("safeguards-grid");
    if (!grid) return;
    const shaShort = modelInfo && modelInfo.baseline && modelInfo.baseline.sha256
      ? modelInfo.baseline.sha256.slice(0, 12) + "…"
      : "verified";
    const cards = [
      {
        title: "Immutable Baseline Seal",
        badge: health.baseline_sha256_valid ? "Verified" : "Error",
        cls: health.baseline_sha256_valid ? "pill-ok" : "pill-danger",
        desc: `Frozen logistic regression (threshold 0.39) verified by SHA-256 (${shaShort}) on startup.`,
      },
      {
        title: "Pinned Runtime Stack",
        badge: `sklearn ${health.sklearn_version || "?"}`,
        cls: health.sklearn_pinned_1_7_2 ? "pill-ok" : "pill-warn",
        desc: "Exact scikit-learn 1.7.2 pin guarantees artifact serialization fidelity.",
      },
      {
        title: "Zero Hospital Leakage",
        badge: "Site ID Dropped",
        cls: "pill-ok",
        desc: "Hospital source column excluded so predictions rely on physiology, not site prevalence.",
      },
      {
        title: "Zero Patient Persistence",
        badge: "No-Store Ephemeral",
        cls: "pill-ok",
        desc: "Patient inputs are scored strictly in memory; zero disk logs, cookies, or database writes.",
      },
      {
        title: "Synthetic-Only Meta Guard",
        badge: health.assistant && health.assistant.synthetic_only_enforced ? "Enforced" : "Check",
        cls: "pill-info",
        desc: "External API calls allow only registered synthetic archetypes + canonical question templates.",
      },
      {
        title: "Isolated Billing Boundary",
        badge: health.assistant && health.assistant.gemini_touched === false ? "Gemini Untouched" : "Check",
        cls: "pill-ok",
        desc: "Zero Gemini imports or credentials; Meta adapter runs in replay mode until operator clearance.",
      },
    ];
    grid.innerHTML = cards
      .map(
        (c) => `
      <div class="safeguard-tile">
        <div class="safeguard-tile-head">
          <h4>${esc(c.title)}</h4>
          <span class="pill ${c.cls}">${esc(c.badge)}</span>
        </div>
        <p>${esc(c.desc)}</p>
      </div>`
      )
      .join("");
  }

  let syntheticProfiles = [];

  async function loadInitialData() {
    try {
      const [hResp, mResp, sResp, pResp, lResp] = await Promise.all([
        fetch("/api/health"),
        fetch("/api/model/info"),
        fetch("/api/statistics"),
        fetch("/api/assistant/profiles"),
        fetch("/api/assistant/ledger"),
      ]);

      const health = await hResp.json();
      $("health-json").textContent = JSON.stringify(health, null, 2);
      const pill = $("health-pill");
      pill.textContent = health.ready ? "Ready · Baseline SHA-256 OK" : "Degraded · Check Health";
      pill.className = "pill " + (health.ready ? "pill-ok" : "pill-danger");

      let modelInfo = null;
      if (mResp.ok) {
        modelInfo = await mResp.json();
        $("model-json").textContent = JSON.stringify(modelInfo, null, 2);
      }
      renderSafeguards(health, modelInfo);

      if (sResp.ok) {
        const stats = await sResp.json();
        const hb = stats.frozen_baseline_historical_audit_n184;
        const hp = hb.point_metrics;
        const hci = hb.bootstrap_95ci;
        const lohoLr = stats.loho_zero_site_overlap_n920.model_families["Logistic Regression"];
        const oofLr = stats.unspent_oof_train_cv_n736.models["Logistic Regression"];

        $("stats-meta-note").textContent = stats.metadata.evaluation_discipline_note;
        $("stats-kpis").innerHTML = `
          <div class="kpi">
            <div class="label">Historical Baseline Recall (n=184)</div>
            <div class="val">${pct(hp.recall)}</div>
            <div class="sub">95% Boot CI: ${pct(hci.recall.ci_low)}–${pct(hci.recall.ci_high)}</div>
          </div>
          <div class="kpi">
            <div class="label">OOF Train CV ECE (n=736)</div>
            <div class="val">${oofLr.calibration.ece_quantile.toFixed(3)}</div>
            <div class="sub">Brier ${oofLr.calibration.brier_score.toFixed(3)} · Slope ${oofLr.calibration.calibration_slope.toFixed(2)}</div>
          </div>
          <div class="kpi">
            <div class="label">LOHO Unseen-Site Macro AUC</div>
            <div class="val">${lohoLr.macro_avg.roc_auc.toFixed(3)}</div>
            <div class="sub">Zero hospital overlap (4 sites)</div>
          </div>
          <div class="kpi">
            <div class="label">LOHO Unseen-Site Macro ECE</div>
            <div class="val">${lohoLr.macro_avg.ece_quantile.toFixed(3)}</div>
            <div class="sub">Brier ${lohoLr.macro_avg.brier_score.toFixed(3)}</div>
          </div>
        `;

        drawCalibrationSvg(
          oofLr.calibration.bins_quantile,
          lohoLr.pooled_calibration.bins_quantile,
          hb.calibration.bins_quantile
        );

        const lohoBody = $("loho-table").querySelector("tbody");
        lohoBody.innerHTML = lohoLr.per_hospital
          .map(
            (r) => `
          <tr>
            <td><strong>${esc(r.held_out_hospital)}</strong></td>
            <td>${r.at_cv_threshold.n} (${pct(r.at_cv_threshold.prevalence)})</td>
            <td>${pct(r.at_cv_threshold.recall)} [${pct(r.bootstrap_95ci_at_cv_threshold.recall[0])}, ${pct(r.bootstrap_95ci_at_cv_threshold.recall[1])}]</td>
            <td>${r.at_cv_threshold.roc_auc.toFixed(3)}</td>
            <td>${(r.calibration.mean_prob_bias * 100) > 0 ? "+" : ""}${(r.calibration.mean_prob_bias * 100).toFixed(1)} pp</td>
            <td>${r.calibration.ece_quantile.toFixed(3)}</td>
          </tr>
        `
          )
          .join("");

        const harmBody = $("harm-table").querySelector("tbody");
        harmBody.innerHTML = stats.harmonization_comparison
          .map(
            (s) => `
          <tr>
            <td>${esc(s.label)}</td>
            <td>${s.inference_hospital_id_free ? "Yes" : "No (Bound)"}</td>
            <td>${s.nested_train_cv_n736.roc_auc.toFixed(4)}</td>
            <td>${s.nested_train_cv_n736.ece_quantile.toFixed(4)}</td>
            <td>${s.loho_zero_site_overlap_n920.macro_roc_auc.toFixed(4)}</td>
            <td>${s.loho_zero_site_overlap_n920.macro_ece_quantile.toFixed(4)}</td>
            <td>${s.loho_zero_site_overlap_n920.macro_brier.toFixed(4)}</td>
          </tr>
        `
          )
          .join("");

        const subBody = $("subgroup-table").querySelector("tbody");
        const subRows = [];
        for (const [dim, items] of Object.entries(stats.loho_zero_site_overlap_n920.lr_subgroups_at_039)) {
          for (const it of items) {
            subRows.push(`
              <tr>
                <td>${esc(dim)}</td>
                <td>${esc(it.subgroup)}</td>
                <td>${it.n} (${it.n_pos})</td>
                <td>${pct(it.recall)} [${pct(it.bootstrap_95ci.recall[0])}, ${pct(it.bootstrap_95ci.recall[1])}]</td>
                <td>${pct(it.precision)}</td>
                <td>${it.roc_auc === null ? "N/A" : it.roc_auc.toFixed(3)}</td>
                <td>${it.ece_quantile.toFixed(3)}</td>
                <td>${it.small_sample_warning ? '<span class="pill pill-warn">Small n</span>' : ""}</td>
              </tr>
            `);
          }
        }
        subBody.innerHTML = subRows.join("");
      }

      if (pResp.ok) {
        const pData = await pResp.json();
        syntheticProfiles = pData.profiles || [];

        // Quick-Fill Pills in Tab 1
        const qfBox = $("quick-fill-pills");
        if (qfBox) {
          qfBox.innerHTML = syntheticProfiles
            .map(
              (p) =>
                `<button type="button" class="qf-pill" data-qf-id="${esc(p.profile_id)}" title="${esc(p.title)}">${esc(p.profile_id)} · ${esc(p.title.split(" with ")[0])}</button>`
            )
            .join("");
          qfBox.querySelectorAll(".qf-pill").forEach((btn) => {
            btn.addEventListener("click", () => {
              const prof = syntheticProfiles.find((p) => p.profile_id === btn.dataset.qfId);
              if (!prof) return;
              qfBox.querySelectorAll(".qf-pill").forEach((b) => b.classList.remove("active"));
              btn.classList.add("active");
              fillPatientForm(prof.features);
              $("predict-form").requestSubmit();
            });
          });
        }

        // Synthetic Archetype Selector & Cards in Tab 3
        const sel = $("synth-select");
        sel.innerHTML = syntheticProfiles
          .map((p) => `<option value="${esc(p.profile_id)}">${esc(p.profile_id)} — ${esc(p.title)}</option>`)
          .join("");

        const cardsBox = $("synth-cards");
        if (cardsBox) {
          cardsBox.innerHTML = syntheticProfiles
            .map(
              (p, idx) => `
            <button type="button" class="synth-card ${idx === 0 ? "active" : ""}" data-synth-id="${esc(p.profile_id)}">
              <div class="synth-card-id">${esc(p.profile_id)}</div>
              <div class="synth-card-title">${esc(p.title)}</div>
            </button>`
            )
            .join("");
          cardsBox.querySelectorAll(".synth-card").forEach((card) => {
            card.addEventListener("click", () => {
              sel.value = card.dataset.synthId;
              cardsBox.querySelectorAll(".synth-card").forEach((c) => c.classList.remove("active"));
              card.classList.add("active");
              updateScenario();
            });
          });
        }

        const tSel = $("assistant-template");
        tSel.innerHTML = (pData.external_question_templates || [])
          .map((t) => `<option value="${esc(t.question)}">${esc(t.question)}</option>`)
          .join("");

        const updateScenario = () => {
          const cur = syntheticProfiles.find((p) => p.profile_id === sel.value);
          $("synth-scenario").textContent = cur
            ? `${cur.clinical_scenario} (SHA-256: ${cur.digest_sha256.slice(0, 12)}…)`
            : "";
        };
        sel.addEventListener("change", updateScenario);
        updateScenario();
      }

      if (lResp.ok) {
        renderLedger(await lResp.json());
      }
    } catch (err) {
      $("health-pill").textContent = "API Error";
      $("health-pill").className = "pill pill-danger";
    }
  }

  function renderLedger(data) {
    const s = data.ledger_summary;
    $("ledger-kpis").innerHTML = `
      <div class="kpi">
        <div class="label">Meta Spend Ledger</div>
        <div class="val">$${s.total_spent_usd.toFixed(4)}</div>
        <div class="sub">Global cap $${s.global_cap_usd.toFixed(2)} (${s.entries} calls)</div>
      </div>
      <div class="kpi">
        <div class="label">Live Clearance Gate</div>
        <div class="val">${data.live_cleared && data.api_key_configured ? "Cleared" : "Replay Only"}</div>
        <div class="sub">Synthetic profiles only</div>
      </div>
    `;
  }

  async function runAssistant(useExternal) {
    $("assistant-error").classList.add("hidden");
    const profileId = $("synth-select").value;
    // The Meta adapter only accepts a canonical template; free text is sent to the local path only.
    const question = useExternal ? $("assistant-template").value : $("assistant-question").value.trim();
    try {
      const resp = await fetch("/api/assistant/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          use_external_meta_api: useExternal,
          synthetic_profile_id: profileId,
          question,
        }),
      });
      const body = await resp.json();
      if (!resp.ok) {
        $("assistant-error").textContent = body.message || "Assistant request refused.";
        $("assistant-error").classList.remove("hidden");
        return;
      }
      $("assistant-placeholder").classList.add("hidden");
      $("assistant-output").classList.remove("hidden");
      $("asst-mode-pill").textContent = `Mode: ${body.mode} · Spend: $${body.spend_usd.toFixed(4)}`;
      $("asst-text").textContent = body.explanation;
      $("asst-takeaways").innerHTML = body.key_takeaways.map((t) => `<li>${esc(t)}</li>`).join("");
      $("asst-uncertainty").innerHTML = body.uncertainty_reminders.map((t) => `<li>${esc(t)}</li>`).join("");
      const lResp = await fetch("/api/assistant/ledger");
      if (lResp.ok) renderLedger(await lResp.json());
    } catch (err) {
      $("assistant-error").textContent = "Could not contact assistant API.";
      $("assistant-error").classList.remove("hidden");
    }
  }

  $("ask-meta-btn").addEventListener("click", () => runAssistant(true));
  $("ask-local-synth-btn").addEventListener("click", () => runAssistant(false));

  loadInitialData();
})();
