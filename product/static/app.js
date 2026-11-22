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
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
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

  function renderPrediction(data) {
    $("result-placeholder").classList.add("hidden");
    $("result-body").classList.remove("hidden");

    const banner = $("result-banner");
    banner.className = "alert " + (data.flagged ? "alert-danger" : "alert-ok");
    banner.innerHTML = data.flagged
      ? `<strong>Flagged (Above ${pct(data.threshold)} Cutoff):</strong> Estimated probability ${pct(data.probability)}. ${esc(data.uncertainty.threshold_proximity_note)}`
      : `<strong>Not Flagged (Below ${pct(data.threshold)} Cutoff):</strong> Estimated probability ${pct(data.probability)}. ${esc(data.uncertainty.threshold_proximity_note)}`;

    $("res-prob").textContent = pct(data.probability);
    const u = data.uncertainty;
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
      const width = Math.min(100, (Math.abs(c.log_odds_contribution) / maxAbs) * 100);
      const cls = c.direction === "increases_risk" ? "contrib-up" : "contrib-dn";
      const sign = c.log_odds_contribution > 0 ? "+" : "";
      row.innerHTML = `
        <span>${esc(c.label)}</span>
        <div class="contrib-bar-track"><div class="contrib-bar-fill ${cls}" style="width:${width.toFixed(0)}%"></div></div>
        <span class="mono" style="text-align:right;">${sign}${c.log_odds_contribution.toFixed(2)}</span>
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
      showFormError("Could not reach the local screening API.");
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

    let parts = [];
    // Grid & diagonal
    for (let t = 0; t <= 1.001; t += 0.2) {
      const x = sx(t).toFixed(1), y = sy(t).toFixed(1);
      parts.push(`<line x1="${padL}" y1="${y}" x2="${padL + W}" y2="${y}" stroke="#e7ebf0" />`);
      parts.push(`<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + H}" stroke="#e7ebf0" />`);
      parts.push(`<text x="${padL - 8}" y="${Number(y) + 4}" font-size="10" text-anchor="end" fill="#5a6578">${Math.round(t * 100)}%</text>`);
      parts.push(`<text x="${x}" y="${padT + H + 15}" font-size="10" text-anchor="middle" fill="#5a6578">${Math.round(t * 100)}%</text>`);
    }
    parts.push(`<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="#8892a0" stroke-dasharray="4,4" />`);

    const addSeries = (bins, color, label) => {
      if (!bins || !bins.length) return;
      const pts = bins.map((b) => `${sx(b.mean_predicted).toFixed(1)},${sy(b.observed_rate).toFixed(1)}`).join(" ");
      parts.push(`<polyline fill="none" stroke="${color}" stroke-width="2.2" points="${pts}" />`);
      bins.forEach((b) => {
        const cx = sx(b.mean_predicted).toFixed(1);
        const cy = sy(b.observed_rate).toFixed(1);
        if (b.observed_ci_low !== null && b.observed_ci_high !== null) {
          parts.push(`<line x1="${cx}" y1="${sy(b.observed_ci_low).toFixed(1)}" x2="${cx}" y2="${sy(b.observed_ci_high).toFixed(1)}" stroke="${color}" stroke-opacity="0.45" stroke-width="1.5" />`);
        }
        parts.push(`<circle cx="${cx}" cy="${cy}" r="3.6" fill="${color}" />`);
      });
    };

    addSeries(oofBins, "#22548a", "OOF Train CV");
    addSeries(lohoBins, "#b3261e", "LOHO (Unseen Site)");
    addSeries(histBins, "#7d5ba6", "Historical Baseline");

    parts.push(`<text x="${padL + W / 2}" y="312" font-size="11" text-anchor="middle" fill="#1a2230">Mean Predicted Probability</text>`);
    parts.push(`<rect x="${padL + 8}" y="${padT + 6}" width="10" height="3" fill="#22548a" /><text x="${padL + 22}" y="${padT + 10}" font-size="10" fill="#1a2230">OOF Train CV (n=736)</text>`);
    parts.push(`<rect x="${padL + 8}" y="${padT + 20}" width="10" height="3" fill="#b3261e" /><text x="${padL + 22}" y="${padT + 24}" font-size="10" fill="#1a2230">LOHO Unseen Hospital (n=920)</text>`);
    parts.push(`<rect x="${padL + 8}" y="${padT + 34}" width="10" height="3" fill="#7d5ba6" /><text x="${padL + 22}" y="${padT + 38}" font-size="10" fill="#1a2230">Frozen Baseline Hist (n=184)</text>`);

    svg.innerHTML = parts.join("");
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

      if (mResp.ok) {
        $("model-json").textContent = JSON.stringify(await mResp.json(), null, 2);
      }

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
        lohoBody.innerHTML = lohoLr.per_hospital.map((r) => `
          <tr>
            <td><strong>${esc(r.held_out_hospital)}</strong></td>
            <td>${r.at_cv_threshold.n} (${pct(r.at_cv_threshold.prevalence)})</td>
            <td>${pct(r.at_cv_threshold.recall)} [${pct(r.bootstrap_95ci_at_cv_threshold.recall[0])}, ${pct(r.bootstrap_95ci_at_cv_threshold.recall[1])}]</td>
            <td>${r.at_cv_threshold.roc_auc.toFixed(3)}</td>
            <td>${(r.calibration.mean_prob_bias * 100).toFixed(1)} pp</td>
            <td>${r.calibration.ece_quantile.toFixed(3)}</td>
          </tr>
        `).join("");

        const harmBody = $("harm-table").querySelector("tbody");
        harmBody.innerHTML = stats.harmonization_comparison.map((s) => `
          <tr>
            <td>${esc(s.label)}</td>
            <td>${s.inference_hospital_id_free ? "Yes" : "No (Bound)"}</td>
            <td>${s.nested_train_cv_n736.roc_auc.toFixed(4)}</td>
            <td>${s.nested_train_cv_n736.ece_quantile.toFixed(4)}</td>
            <td>${s.loho_zero_site_overlap_n920.macro_roc_auc.toFixed(4)}</td>
            <td>${s.loho_zero_site_overlap_n920.macro_ece_quantile.toFixed(4)}</td>
            <td>${s.loho_zero_site_overlap_n920.macro_brier.toFixed(4)}</td>
          </tr>
        `).join("");

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
        const sel = $("synth-select");
        sel.innerHTML = syntheticProfiles.map((p) => `<option value="${esc(p.profile_id)}">${esc(p.profile_id)} — ${esc(p.title)}</option>`).join("");
        // External path: only canonical templates are selectable (free text stays local).
        const tSel = $("assistant-template");
        tSel.innerHTML = (pData.external_question_templates || [])
          .map((t) => `<option value="${esc(t.question)}">${esc(t.question)}</option>`)
          .join("");
        const updateScenario = () => {
          const cur = syntheticProfiles.find((p) => p.profile_id === sel.value);
          $("synth-scenario").textContent = cur ? `${cur.clinical_scenario} (SHA-256: ${cur.digest_sha256.slice(0, 12)}…)` : "";
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
