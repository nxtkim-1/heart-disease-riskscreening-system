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

  const pct = (v, digits = 1) =>
    v === null || v === undefined || Number.isNaN(v) ? "N/A" : `${(v * 100).toFixed(digits)}%`;

  const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const COLORS = {
    navy: "#173B69",
    navyDeep: "#0E2442",
    pulse: "#C23A4D",
    steelLight: "#6F9BC6",
    mist: "#D3DFEF",
    track: "#E6ECF4",
    grid: "#E8EDF4",
    diagonal: "#9AA8BC",
  };

  // ---------------------------------------------------------------- Tabs

  const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));

  function activateTab(btn, focus) {
    tabButtons.forEach((b) => {
      const on = b === btn;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
      $("tab-" + b.dataset.tab).classList.toggle("active", on);
    });
    if (focus) btn.focus();
  }

  tabButtons.forEach((btn, i) => {
    btn.addEventListener("click", () => activateTab(btn, false));
    btn.addEventListener("keydown", (e) => {
      let next = null;
      if (e.key === "ArrowRight") next = tabButtons[(i + 1) % tabButtons.length];
      if (e.key === "ArrowLeft") next = tabButtons[(i - 1 + tabButtons.length) % tabButtons.length];
      if (e.key === "Home") next = tabButtons[0];
      if (e.key === "End") next = tabButtons[tabButtons.length - 1];
      if (next) {
        e.preventDefault();
        activateTab(next, true);
      }
    });
  });

  // ---------------------------------------------------------------- Intake form

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
      const unknown = val === null || val === undefined;
      if (cb) cb.checked = unknown;
      input.disabled = unknown;
      input.value = unknown ? "" : String(val);
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

  // ---------------------------------------------------------------- Risk line

  let dotAnimation = null;
  let lastRiskArgs = null;

  // The viewBox matches the rendered width, so labels stay 12px on phones and wide embeds alike.
  function drawRiskLine(prob, lo, hi, threshold, flagged, animate = true) {
    lastRiskArgs = [prob, lo, hi, threshold, flagged];
    const svg = $("risk-line");
    const width = Math.max(260, Math.round(svg.getBoundingClientRect().width) || 560);
    svg.setAttribute("viewBox", `0 0 ${width} 96`);
    const x0 = 14;
    const x1 = width - 14;
    const y = 52;
    const clamp = (v) => Math.max(0, Math.min(1, v));
    const sx = (v) => x0 + clamp(v) * (x1 - x0);
    const dotColor = flagged ? COLORS.pulse : COLORS.navy;
    const tx = sx(threshold);

    const parts = [
      `<rect x="${x0}" y="${y - 3}" width="${x1 - x0}" height="6" rx="3" fill="${COLORS.track}" />`,
    ];

    if (lo !== null && hi !== null && hi > lo) {
      const bx = sx(lo);
      const bw = Math.max(6, sx(hi) - bx);
      parts.push(`<rect x="${bx.toFixed(1)}" y="${y - 12}" width="${bw.toFixed(1)}" height="24" rx="12" fill="${COLORS.mist}" fill-opacity="0.9" />`);
    }

    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      const x = sx(t).toFixed(1);
      const anchor = t === 0 ? "start" : t === 1 ? "end" : "middle";
      parts.push(`<text x="${x}" y="91" text-anchor="${anchor}">${Math.round(t * 100)}%</text>`);
    });

    parts.push(
      `<line x1="${tx.toFixed(1)}" y1="30" x2="${tx.toFixed(1)}" y2="74" stroke="${COLORS.navyDeep}" stroke-width="2" stroke-linecap="round" />`,
      `<text class="rl-cut" x="${tx.toFixed(1)}" y="21" text-anchor="middle">${Math.round(threshold * 100)}% cutoff</text>`,
      `<circle id="rl-halo" cx="${x0}" cy="${y}" r="15" fill="${dotColor}" fill-opacity="0.14" />`,
      `<circle id="rl-dot" cx="${x0}" cy="${y}" r="9" fill="${dotColor}" stroke="#FFFFFF" stroke-width="3" />`
    );
    svg.innerHTML = parts.join("");

    const intervalText = lo !== null && hi !== null ? `, 95% interval ${pct(lo)} to ${pct(hi)}` : "";
    svg.setAttribute(
      "aria-label",
      `Estimated ${pct(prob)}${intervalText}, ${flagged ? "above" : "below"} the ${Math.round(threshold * 100)}% screening cutoff`
    );

    // One orchestrated moment: the dot eases from 0% to the estimate.
    const target = sx(prob);
    const dot = $("rl-dot");
    const halo = $("rl-halo");
    const place = (x) => {
      dot.setAttribute("cx", x.toFixed(1));
      halo.setAttribute("cx", x.toFixed(1));
    };
    if (dotAnimation) cancelAnimationFrame(dotAnimation);
    if (!animate || prefersReducedMotion()) {
      place(target);
      return;
    }
    const start = performance.now();
    const duration = 650;
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      place(x0 + (target - x0) * eased);
      if (t < 1) dotAnimation = requestAnimationFrame(step);
    };
    dotAnimation = requestAnimationFrame(step);
  }

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (lastRiskArgs && !$("result-body").classList.contains("hidden")) drawRiskLine(...lastRiskArgs, false);
    }, 120);
  });

  // ---------------------------------------------------------------- Result rendering

  let lastPatient = null;
  let explainedFor = null;

  function resetExplanation() {
    explainedFor = null;
    $("explain-details").open = false;
    $("local-explain-text").textContent = "";
    $("local-explain-takeaways").innerHTML = "";
    $("local-explain-status").textContent = "Written on this server. No outside service is used.";
  }

  function renderPrediction(data) {
    $("result-placeholder").classList.add("hidden");
    $("result-body").classList.remove("hidden");

    const u = data.uncertainty;
    const lo = u.bootstrap_ci_low;
    const hi = u.bootstrap_ci_high;
    const cut = `${Math.round(data.threshold * 100)}%`;

    $("res-prob").textContent = pct(data.probability);
    drawRiskLine(data.probability, lo, hi, data.threshold, data.flagged);

    const verdict = $("result-banner");
    verdict.dataset.flagged = data.flagged ? "true" : "false";
    verdict.textContent = data.flagged
      ? `Above the ${cut} screening cutoff, so this profile would be flagged for follow-up review.`
      : `Below the ${cut} screening cutoff. Not being flagged does not rule out heart disease.`;

    let note = "";
    if (lo !== null && hi !== null && lo < data.threshold && hi > data.threshold) {
      note = "The 95% interval crosses the cutoff, so which side this estimate falls on is itself uncertain.";
    } else if (data.risk_band === "borderline_near_threshold") {
      note = "This estimate sits close to the cutoff.";
    }
    $("res-band-label").textContent = note;
    $("res-band-label").classList.toggle("hidden", !note);

    $("res-ci").textContent = lo !== null && hi !== null ? `${(lo * 100).toFixed(1)}–${pct(hi)}` : "N/A";
    $("res-cal").textContent = pct(data.calibrated_reference_probability);
    $("res-missing").textContent = u.missing_field_count === 0 ? "None" : `${u.missing_field_count} of 8`;

    const imp = $("res-imputation-warn");
    imp.textContent = u.imputation_warning || "";
    imp.classList.toggle("hidden", !u.imputation_warning);

    $("res-shift-warn").textContent = u.population_shift_warning || "";
    $("res-cal-note").textContent = u.calibration_note || "";

    const rows = (data.top_contributions || []).slice(0, 6);
    const maxAbs = Math.max(0.5, ...rows.map((c) => Math.abs(c.log_odds_contribution)));
    $("res-contribs").innerHTML = rows
      .map((c) => {
        const v = c.log_odds_contribution;
        const width = Math.min(50, (Math.abs(v) / maxAbs) * 50).toFixed(1);
        const dir = c.direction === "increases_risk" ? "up" : "dn";
        const sign = v > 0 ? "+" : "";
        return `
          <div class="driver-row">
            <span class="driver-label" title="${esc(c.label)}">${esc(c.label)}</span>
            <span class="driver-track" aria-hidden="true"><span class="driver-bar ${dir}" style="width:${width}%"></span></span>
            <span class="driver-val">${sign}${v.toFixed(2)}</span>
          </div>`;
      })
      .join("");
  }

  $("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFormError();
    const btn = $("predict-btn");
    const patient = readPatientForm();
    btn.disabled = true;
    btn.textContent = "Estimating…";
    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
      });
      const body = await resp.json();
      if (!resp.ok) {
        const detail = body.details
          ? body.details.map((d) => `${d.field}: ${d.message}`).join(" | ")
          : body.message;
        showFormError(detail || "The estimate could not be computed. Check the highlighted values and try again.");
        return;
      }
      if (JSON.stringify(patient) !== JSON.stringify(lastPatient)) resetExplanation();
      lastPatient = patient;
      renderPrediction(body);
    } catch (err) {
      showFormError("The screening server could not be reached. Check your connection and try again.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Estimate risk";
    }
  });

  // Plain-language explanation loads only when someone opens it.
  $("explain-details").addEventListener("toggle", async () => {
    const details = $("explain-details");
    if (!details.open || !lastPatient) return;
    const key = JSON.stringify(lastPatient);
    if (explainedFor === key) return;
    const status = $("local-explain-status");
    status.textContent = "Writing the explanation…";
    try {
      const resp = await fetch("/api/assistant/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          use_external_meta_api: false,
          patient: lastPatient,
          question: "Explain this screening result, its main drivers, and its uncertainty in plain language.",
        }),
      });
      const body = await resp.json();
      if (!resp.ok) {
        status.textContent = body.message || "The explanation could not be written for these inputs.";
        return;
      }
      explainedFor = key;
      status.textContent = "Written on this server. No outside service was used.";
      $("local-explain-text").textContent = body.explanation;
      const ul = $("local-explain-takeaways");
      ul.innerHTML = "";
      body.key_takeaways.forEach((t) => {
        const li = document.createElement("li");
        li.textContent = t;
        ul.appendChild(li);
      });
    } catch (err) {
      status.textContent = "The screening server could not be reached. Close and reopen this section to retry.";
    }
  });

  // ---------------------------------------------------------------- Reset

  $("clear-session-btn").addEventListener("click", async () => {
    $("predict-form").reset();
    document.querySelectorAll("[data-unknown-for]").forEach((cb) => {
      cb.checked = false;
      const target = $(cb.dataset.unknownFor);
      if (target) target.disabled = false;
    });
    document.querySelectorAll(".preset-btn").forEach((b) => b.setAttribute("aria-pressed", "false"));
    clearFormError();
    lastPatient = null;
    resetExplanation();
    $("shift-details").open = false;
    $("result-body").classList.add("hidden");
    $("result-placeholder").classList.remove("hidden");
    $("assistant-output").classList.add("hidden");
    $("assistant-placeholder").classList.remove("hidden");
    $("assistant-error").classList.add("hidden");
    try {
      await fetch("/api/session/clear", { method: "POST" });
    } catch (_) {}
  });

  // ---------------------------------------------------------------- Calibration chart

  function drawCalibrationSvg(series) {
    const svg = $("cal-svg");
    const padL = 44, padR = 14, padT = 14, padB = 46;
    const W = 460 - padL - padR;
    const H = 330 - padT - padB;
    const sx = (v) => padL + v * W;
    const sy = (v) => padT + (1 - v) * H;
    const parts = [];

    for (let i = 0; i <= 5; i++) {
      const t = i / 5;
      const x = sx(t).toFixed(1);
      const y = sy(t).toFixed(1);
      parts.push(`<line x1="${padL}" y1="${y}" x2="${padL + W}" y2="${y}" stroke="${COLORS.grid}" />`);
      parts.push(`<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + H}" stroke="${COLORS.grid}" />`);
      parts.push(`<text x="${padL - 8}" y="${(Number(y) + 4).toFixed(1)}" text-anchor="end">${Math.round(t * 100)}%</text>`);
      parts.push(`<text x="${x}" y="${padT + H + 16}" text-anchor="middle">${Math.round(t * 100)}%</text>`);
    }
    parts.push(`<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="${COLORS.diagonal}" stroke-dasharray="4 4" />`);
    parts.push(`<text class="axis-title" x="${padL + W / 2}" y="${padT + H + 38}" text-anchor="middle">Predicted probability</text>`);
    parts.push(`<text class="axis-title" transform="translate(12 ${padT + H / 2}) rotate(-90)" text-anchor="middle">Observed disease rate</text>`);

    series.forEach(({ bins, color, label }) => {
      if (!bins || !bins.length) return;
      const pts = bins.map((b) => `${sx(b.mean_predicted).toFixed(1)},${sy(b.observed_rate).toFixed(1)}`).join(" ");
      parts.push(`<polyline fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" points="${pts}" />`);
      bins.forEach((b) => {
        const cx = sx(b.mean_predicted).toFixed(1);
        const cy = sy(b.observed_rate).toFixed(1);
        if (b.observed_ci_low !== null && b.observed_ci_high !== null) {
          parts.push(`<line x1="${cx}" y1="${sy(b.observed_ci_low).toFixed(1)}" x2="${cx}" y2="${sy(b.observed_ci_high).toFixed(1)}" stroke="${color}" stroke-opacity="0.35" stroke-width="1.5" />`);
        }
        const tip = `${label}: predicted ${pct(b.mean_predicted)}, observed ${pct(b.observed_rate)} (${b.n} patients)`;
        parts.push(
          `<circle class="cal-pt" cx="${cx}" cy="${cy}" r="4" fill="${color}" stroke="#FFFFFF" stroke-width="1.5" tabindex="0" role="img" aria-label="${esc(tip)}" data-tip="${esc(tip)}" />`
        );
      });
    });

    svg.innerHTML = parts.join("");
  }

  (function wireChartTooltip() {
    const svg = $("cal-svg");
    const tip = $("cal-tip");
    const wrap = svg.parentElement;
    const show = (el) => {
      if (!el || !el.classList || !el.classList.contains("cal-pt")) return;
      tip.textContent = el.getAttribute("data-tip");
      const r = el.getBoundingClientRect();
      const w = wrap.getBoundingClientRect();
      tip.style.left = `${r.left + r.width / 2 - w.left}px`;
      tip.style.top = `${r.top - w.top}px`;
      tip.classList.remove("hidden");
    };
    const hide = () => tip.classList.add("hidden");
    svg.addEventListener("pointerover", (e) => show(e.target));
    svg.addEventListener("pointerout", (e) => {
      if (e.target.classList && e.target.classList.contains("cal-pt")) hide();
    });
    svg.addEventListener("focusin", (e) => show(e.target));
    svg.addEventListener("focusout", hide);
  })();

  // ---------------------------------------------------------------- Statistics tables

  const SUBGROUP_NAMES = {
    hospital: "Hospital",
    sex: "Sex",
    age_band: "Age",
    missingness: "Unknown fields",
    chest_pain: "Chest pain",
  };

  function renderStatistics(stats) {
    const hb = stats.frozen_baseline_historical_audit_n184;
    const hp = hb.point_metrics;
    const hci = hb.bootstrap_95ci;
    const lohoLr = stats.loho_zero_site_overlap_n920.model_families["Logistic Regression"];
    const oofLr = stats.unspent_oof_train_cv_n736.models["Logistic Regression"];

    $("stats-meta-note").textContent = stats.metadata.evaluation_discipline_note;

    const figures = [
      ["Recall on the historical test set", pct(hp.recall), `95% CI ${pct(hci.recall.ci_low)} to ${pct(hci.recall.ci_high)}, 184 patients`],
      ["Calibration error in cross-validation", oofLr.calibration.ece_quantile.toFixed(3), `Brier score ${oofLr.calibration.brier_score.toFixed(3)}, 736 patients`],
      ["AUC on unseen hospitals", lohoLr.macro_avg.roc_auc.toFixed(3), "Averaged over four held-out sites"],
      ["Calibration error on unseen hospitals", lohoLr.macro_avg.ece_quantile.toFixed(3), `Brier score ${lohoLr.macro_avg.brier_score.toFixed(3)}`],
    ];
    $("stats-kpis").innerHTML = figures
      .map(([label, val, sub]) => `<div><dt>${esc(label)}</dt><dd class="fig-val">${esc(val)}</dd><dd class="fig-sub">${esc(sub)}</dd></div>`)
      .join("");

    drawCalibrationSvg([
      { bins: oofLr.calibration.bins_quantile, color: COLORS.navy, label: "Cross-validation" },
      { bins: lohoLr.pooled_calibration.bins_quantile, color: COLORS.pulse, label: "Unseen hospital" },
      { bins: hb.calibration.bins_quantile, color: COLORS.steelLight, label: "Historical test" },
    ]);

    $("loho-table").querySelector("tbody").innerHTML = lohoLr.per_hospital
      .map((r) => {
        const m = r.at_cv_threshold;
        const ci = r.bootstrap_95ci_at_cv_threshold.recall;
        const bias = r.calibration.mean_prob_bias * 100;
        return `
          <tr>
            <td class="group-name">${esc(r.held_out_hospital)}</td>
            <td class="num">${m.n}<span class="ci">${pct(m.prevalence, 0)} with disease</span></td>
            <td class="num${m.recall < 0.8 ? " low" : ""}">${pct(m.recall)}<span class="ci">${pct(ci[0])} to ${pct(ci[1])}</span></td>
            <td class="num">${m.roc_auc.toFixed(3)}</td>
            <td class="num">${bias > 0 ? "+" : ""}${bias.toFixed(1)} pts</td>
          </tr>`;
      })
      .join("");

    $("harm-table").querySelector("tbody").innerHTML = stats.harmonization_comparison
      .map(
        (s) => `
          <tr>
            <td>${esc(s.label)}</td>
            <td>${s.inference_hospital_id_free ? "No" : "Yes"}</td>
            <td class="num">${s.nested_train_cv_n736.roc_auc.toFixed(3)}</td>
            <td class="num">${s.nested_train_cv_n736.ece_quantile.toFixed(3)}</td>
            <td class="num">${s.loho_zero_site_overlap_n920.macro_roc_auc.toFixed(3)}</td>
            <td class="num">${s.loho_zero_site_overlap_n920.macro_ece_quantile.toFixed(3)}</td>
            <td class="num">${s.loho_zero_site_overlap_n920.macro_brier.toFixed(3)}</td>
          </tr>`
      )
      .join("");

    const subRows = [];
    for (const [dim, items] of Object.entries(stats.loho_zero_site_overlap_n920.lr_subgroups_at_039)) {
      items.forEach((it, idx) => {
        const ci = it.bootstrap_95ci.recall;
        subRows.push(`
          <tr class="${idx === 0 ? "group-start" : ""}">
            <td class="group-name">${idx === 0 ? esc(SUBGROUP_NAMES[dim] || dim) : ""}</td>
            <td>${esc(it.subgroup)}${it.small_sample_warning ? '<span class="small-n">Small sample</span>' : ""}</td>
            <td class="num">${it.n} (${it.n_pos})</td>
            <td class="num${it.recall !== null && it.recall < 0.8 ? " low" : ""}">${pct(it.recall)}<span class="ci">${pct(ci[0])} to ${pct(ci[1])}</span></td>
            <td class="num">${pct(it.precision)}</td>
            <td class="num">${it.roc_auc === null ? "N/A" : it.roc_auc.toFixed(3)}</td>
            <td class="num">${it.ece_quantile.toFixed(3)}</td>
          </tr>`);
      });
    }
    $("subgroup-table").querySelector("tbody").innerHTML = subRows.join("");
  }

  // ---------------------------------------------------------------- Safeguards

  function renderSafeguards(health, modelInfo) {
    const sha = modelInfo && modelInfo.baseline && modelInfo.baseline.sha256
      ? modelInfo.baseline.sha256.slice(0, 12)
      : null;
    const assistant = health.assistant || {};
    const items = [
      {
        ok: !!health.baseline_sha256_valid,
        title: "Frozen baseline model is unchanged",
        desc: `The logistic regression (cutoff 0.39) is checked against its SHA-256 fingerprint${sha ? ` (${sha}…)` : ""} at startup.`,
        state: health.baseline_sha256_valid ? "Verified" : "Mismatch",
      },
      {
        ok: !!health.sklearn_pinned_1_7_2,
        title: "Pinned scikit-learn version",
        desc: "scikit-learn 1.7.2 is pinned so the saved model loads exactly as it was trained.",
        state: `scikit-learn ${health.sklearn_version || "unknown"}`,
      },
      {
        ok: true,
        title: "No hospital shortcut",
        desc: "The hospital column is dropped, so estimates rest on measurements rather than each site's disease rate.",
        state: "Dropped",
      },
      {
        ok: !(health.privacy && (health.privacy.patient_inputs_logged || health.privacy.patient_inputs_persisted)),
        title: "Nothing you enter is kept",
        desc: "Inputs are scored in memory. No logs, cookies, or database writes, and API responses are marked no-store.",
        state: "In memory only",
      },
      {
        ok: !!assistant.synthetic_only_enforced,
        title: "Outside assistant sees synthetic cases only",
        desc: "External calls accept only the six registered synthetic profiles and fixed questions. Free text stays on this server.",
        state: assistant.synthetic_only_enforced ? "Enforced" : "Check",
      },
      {
        ok: assistant.live_cleared !== true,
        title: "Spend stays at zero until approved",
        desc: "The external assistant replays recorded answers until an operator approves live calls, with a capped, hash-chained spend ledger.",
        state: assistant.live_cleared ? "Live approved" : "Replay only",
      },
    ];
    const checkIcon = '<svg viewBox="0 0 12 12" width="12" height="12" fill="none" aria-hidden="true"><path d="M2.5 6.3L5 8.6L9.5 3.6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    const warnIcon = '<svg viewBox="0 0 12 12" width="12" height="12" fill="none" aria-hidden="true"><path d="M6 3V6.6M6 8.8V9" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
    $("safeguards-grid").innerHTML = items
      .map(
        (it) => `
        <li>
          <span class="check${it.ok ? "" : " is-warn"}">${it.ok ? checkIcon : warnIcon}</span>
          <div>
            <div class="check-title">${esc(it.title)}</div>
            <p class="check-desc">${esc(it.desc)}</p>
          </div>
          <span class="check-state${it.ok ? "" : " is-warn"}">${esc(it.state)}</span>
        </li>`
      )
      .join("");
  }

  // ---------------------------------------------------------------- Synthetic profiles

  const PRESET_RULES = [
    [/low-risk/i, "Low risk"],
    [/borderline/i, "Borderline"],
    [/high-risk/i, "High risk"],
    [/missing/i, "Missing labs"],
    [/asymptomatic/i, "Silent symptoms"],
    [/healthy|reference/i, "Healthy 30s"],
  ];

  function shortLabel(profile) {
    const title = profile.title || profile.profile_id;
    for (const [re, label] of PRESET_RULES) {
      if (re.test(title)) return label;
    }
    return title.split(/\s+/).slice(0, 2).join(" ");
  }

  let syntheticProfiles = [];

  function renderPresets() {
    const row = $("quick-fill-pills");
    row.innerHTML = syntheticProfiles
      .map(
        (p) =>
          `<button type="button" class="preset-btn" aria-pressed="false" data-id="${esc(p.profile_id)}" title="${esc(p.profile_id)}: ${esc(p.title)}">${esc(shortLabel(p))}</button>`
      )
      .join("");
    row.querySelectorAll(".preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const prof = syntheticProfiles.find((p) => p.profile_id === btn.dataset.id);
        if (!prof) return;
        row.querySelectorAll(".preset-btn").forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
        fillPatientForm(prof.features);
        $("predict-form").requestSubmit();
      });
    });
  }

  function renderChoices() {
    const sel = $("synth-select");
    sel.innerHTML = syntheticProfiles
      .map((p) => `<option value="${esc(p.profile_id)}">${esc(p.profile_id)}</option>`)
      .join("");

    const list = $("synth-cards");
    list.innerHTML = syntheticProfiles
      .map(
        (p, i) => `
        <button type="button" class="choice" role="radio" aria-checked="${i === 0 ? "true" : "false"}" tabindex="${i === 0 ? "0" : "-1"}" data-id="${esc(p.profile_id)}">
          <span class="choice-name">${esc(shortLabel(p))}</span>
          <span class="choice-id">${esc(p.profile_id)}</span>
        </button>`
      )
      .join("");

    const choices = Array.from(list.querySelectorAll(".choice"));
    const select = (btn, focus) => {
      choices.forEach((c) => {
        const on = c === btn;
        c.setAttribute("aria-checked", on ? "true" : "false");
        c.tabIndex = on ? 0 : -1;
      });
      sel.value = btn.dataset.id;
      const cur = syntheticProfiles.find((p) => p.profile_id === btn.dataset.id);
      $("synth-scenario").textContent = cur ? `${cur.title}. ${cur.clinical_scenario}` : "";
      if (focus) btn.focus();
    };
    choices.forEach((btn, i) => {
      btn.addEventListener("click", () => select(btn, false));
      btn.addEventListener("keydown", (e) => {
        let next = null;
        if (e.key === "ArrowDown" || e.key === "ArrowRight") next = choices[(i + 1) % choices.length];
        if (e.key === "ArrowUp" || e.key === "ArrowLeft") next = choices[(i - 1 + choices.length) % choices.length];
        if (next) {
          e.preventDefault();
          select(next, true);
        }
      });
    });
    if (choices.length) select(choices[0], false);
  }

  // ---------------------------------------------------------------- Initial load

  function setStatus(state, text) {
    $("health-pill").dataset.state = state;
    $("health-text").textContent = text;
  }

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
      setStatus(health.ready ? "ok" : "error", health.ready ? "Model verified" : "Model check failed");

      let modelInfo = null;
      if (mResp.ok) {
        modelInfo = await mResp.json();
        $("model-json").textContent = JSON.stringify(modelInfo, null, 2);
      }
      renderSafeguards(health, modelInfo);

      if (sResp.ok) {
        renderStatistics(await sResp.json());
      } else {
        $("stats-meta-note").textContent = "The statistical report has not been generated on this server yet.";
      }

      if (pResp.ok) {
        const pData = await pResp.json();
        syntheticProfiles = pData.profiles || [];
        renderPresets();
        renderChoices();
        $("assistant-template").innerHTML = (pData.external_question_templates || [])
          .map((t) => `<option value="${esc(t.question)}">${esc(t.question)}</option>`)
          .join("");
      }

      if (lResp.ok) renderLedger(await lResp.json());
    } catch (err) {
      setStatus("error", "Server unreachable");
    }
  }

  function renderLedger(data) {
    const s = data.ledger_summary;
    const calls = s.entries === 1 ? "1 external call" : `${s.entries} external calls`;
    const mode = data.live_cleared && data.api_key_configured
      ? "Live calls are approved."
      : "Replay only, so no live calls are made.";
    $("ledger-kpis").textContent = `Spent $${s.total_spent_usd.toFixed(4)} of the $${s.global_cap_usd.toFixed(2)} cap across ${calls}. ${mode}`;
  }

  // ---------------------------------------------------------------- Assistant

  const MODE_LABELS = {
    local_offline: "Answered offline on this server",
    meta_replay: "External assistant, recorded reply (no live call)",
    meta_record: "External assistant, live call recorded",
    meta_live: "External assistant, live call",
  };

  async function runAssistant(useExternal) {
    $("assistant-error").classList.add("hidden");
    const profileId = $("synth-select").value;
    // The external adapter only accepts a canonical template; free text goes to the offline path only.
    const question = useExternal ? $("assistant-template").value : $("assistant-question").value.trim();
    const buttons = [$("ask-meta-btn"), $("ask-local-synth-btn")];
    buttons.forEach((b) => (b.disabled = true));
    try {
      const resp = await fetch("/api/assistant/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_external_meta_api: useExternal, synthetic_profile_id: profileId, question }),
      });
      const body = await resp.json();
      if (!resp.ok) {
        $("assistant-error").textContent = body.message || "The assistant refused this request.";
        $("assistant-error").classList.remove("hidden");
        return;
      }
      $("assistant-placeholder").classList.add("hidden");
      $("assistant-output").classList.remove("hidden");
      $("asst-mode-pill").textContent = `${MODE_LABELS[body.mode] || body.mode}. Cost $${body.spend_usd.toFixed(4)}.`;
      $("asst-text").textContent = body.explanation;
      $("asst-takeaways").innerHTML = body.key_takeaways.map((t) => `<li>${esc(t)}</li>`).join("");
      $("asst-uncertainty").innerHTML = body.uncertainty_reminders.map((t) => `<li>${esc(t)}</li>`).join("");
      const lResp = await fetch("/api/assistant/ledger");
      if (lResp.ok) renderLedger(await lResp.json());
    } catch (err) {
      $("assistant-error").textContent = "The assistant could not be reached. Check your connection and try again.";
      $("assistant-error").classList.remove("hidden");
    } finally {
      buttons.forEach((b) => (b.disabled = false));
    }
  }

  $("ask-meta-btn").addEventListener("click", () => runAssistant(true));
  $("ask-local-synth-btn").addEventListener("click", () => runAssistant(false));

  loadInitialData();
})();
