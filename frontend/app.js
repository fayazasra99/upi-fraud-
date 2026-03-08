const TOKEN_KEY = "fraud_shield_token";

const state = {
  token: localStorage.getItem(TOKEN_KEY),
  user: null,
  merchants: [],
  selectedMerchant: "",
  decisionFilter: "",
  statusFilter: "",
};

const flash = document.getElementById("flash");

const authPanel = document.getElementById("auth-panel");
const appShell = document.getElementById("app-shell");
const userRow = document.getElementById("user-row");
const userChip = document.getElementById("user-chip");

const loginForm = document.getElementById("login-form");
const logoutBtn = document.getElementById("logout-btn");

const merchantSelect = document.getElementById("merchant-select");
const decisionFilter = document.getElementById("decision-filter");
const statusFilter = document.getElementById("status-filter");
const seedBtn = document.getElementById("seed-btn");
const refreshBtn = document.getElementById("refresh-btn");

const scoreForm = document.getElementById("score-form");
const scoreResult = document.getElementById("score-result");
const uploadForm = document.getElementById("upload-form");
const uploadSummary = document.getElementById("upload-summary");
const ruleForm = document.getElementById("rule-form");
const ruleList = document.getElementById("rule-list");
const reloadModelBtn = document.getElementById("reload-model-btn");
const trainModelBtn = document.getElementById("train-model-btn");
const modelStatus = document.getElementById("model-status");
const trainSummary = document.getElementById("train-summary");

const metricTotal = document.getElementById("metric-total");
const metricAllow = document.getElementById("metric-allow");
const metricReview = document.getElementById("metric-review");
const metricBlock = document.getElementById("metric-block");
const metricRisk = document.getElementById("metric-risk");

const merchantBody = document.getElementById("merchant-body");
const queueBody = document.getElementById("queue-body");
const txnBody = document.getElementById("txn-body");

loginForm.addEventListener("submit", onLoginSubmit);
logoutBtn.addEventListener("click", onLogoutClick);
refreshBtn.addEventListener("click", loadDashboard);
seedBtn.addEventListener("click", onSeedClick);
scoreForm.addEventListener("submit", onScoreSubmit);
uploadForm.addEventListener("submit", onUploadSubmit);
ruleForm.addEventListener("submit", onRuleSubmit);
ruleList.addEventListener("click", onRuleToggleClick);
queueBody.addEventListener("click", onQueueActionClick);
merchantSelect.addEventListener("change", onMerchantChange);
decisionFilter.addEventListener("change", onFilterChange);
statusFilter.addEventListener("change", onFilterChange);
reloadModelBtn.addEventListener("click", onReloadModel);
trainModelBtn.addEventListener("click", onTrainFromCases);

init();

async function init() {
  if (!state.token) {
    showLoggedOutView();
    return;
  }

  try {
    await bootstrapSession();
  } catch (error) {
    clearSession();
    showLoggedOutView();
    setFlash(error.message, true);
  }
}

async function bootstrapSession() {
  state.user = await request("/api/auth/me");
  applyRoleView();
  await loadMerchants();
  showLoggedInView();
  await loadDashboard();
}

async function onLoginSubmit(event) {
  event.preventDefault();

  const formData = new FormData(loginForm);
  const payload = {
    username: String(formData.get("username") || "").trim(),
    password: String(formData.get("password") || ""),
  };

  try {
    setFlash("Signing in...");
    const response = await request(
      "/api/auth/login",
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
      { auth: false }
    );

    state.token = response.access_token;
    localStorage.setItem(TOKEN_KEY, state.token);
    await bootstrapSession();
    setFlash("Signed in successfully.");
  } catch (error) {
    clearSession();
    showLoggedOutView();
    setFlash(error.message, true);
  }
}

function onLogoutClick() {
  clearSession();
  showLoggedOutView();
  setFlash("Logged out.");
}

function clearSession() {
  state.token = null;
  state.user = null;
  state.merchants = [];
  state.selectedMerchant = "";
  localStorage.removeItem(TOKEN_KEY);
}

function showLoggedInView() {
  authPanel.classList.add("hidden");
  appShell.classList.remove("hidden");
  userRow.classList.remove("hidden");

  const merchantText = state.user.merchant_id ? ` (${state.user.merchant_id})` : "";
  userChip.textContent = `${state.user.username} - ${state.user.role}${merchantText}`;
}

function showLoggedOutView() {
  authPanel.classList.remove("hidden");
  appShell.classList.add("hidden");
  userRow.classList.add("hidden");
}

function applyRoleView() {
  const isAnalyst = state.user?.role === "analyst";
  reloadModelBtn.classList.toggle("hidden", !isAnalyst);
  trainModelBtn.classList.toggle("hidden", !isAnalyst);

  const ruleMerchantInput = ruleForm.querySelector('input[name="merchant_id"]');
  const scoreMerchantInput = scoreForm.querySelector('input[name="merchant_id"]');

  if (!ruleMerchantInput || !scoreMerchantInput) return;

  if (isAnalyst) {
    ruleMerchantInput.readOnly = false;
    scoreMerchantInput.readOnly = false;
    trainSummary.textContent = "";
  } else {
    const merchant = state.user?.merchant_id || "";
    ruleMerchantInput.value = merchant;
    scoreMerchantInput.value = merchant;
    ruleMerchantInput.readOnly = true;
    scoreMerchantInput.readOnly = true;
    trainSummary.textContent = "Retraining is analyst-only.";
  }
}

async function loadMerchants() {
  const merchants = await request("/api/merchants");
  state.merchants = merchants;

  const options = [];
  if (state.user.role === "analyst") {
    options.push('<option value="">All merchants</option>');
  }

  for (const merchant of merchants) {
    options.push(`<option value="${escapeHtml(merchant)}">${escapeHtml(merchant)}</option>`);
  }

  merchantSelect.innerHTML = options.join("");

  if (state.user.role === "merchant_admin") {
    state.selectedMerchant = state.user.merchant_id || "";
    merchantSelect.value = state.selectedMerchant;
    merchantSelect.disabled = true;
  } else {
    merchantSelect.disabled = false;
    state.selectedMerchant = merchantSelect.value || "";
  }
}

async function loadDashboard() {
  if (!state.user) return;

  setFlash("Loading dashboard...");

  const query = new URLSearchParams();
  if (state.selectedMerchant) query.set("merchant_id", state.selectedMerchant);
  if (state.decisionFilter) query.set("decision", state.decisionFilter);
  if (state.statusFilter) query.set("status", state.statusFilter);

  const queueQuery = new URLSearchParams();
  if (state.selectedMerchant) queueQuery.set("merchant_id", state.selectedMerchant);

  try {
    const [metrics, transactions, rules, summary, queue, model] = await Promise.all([
      request(`/api/metrics?${query.toString()}`),
      request(`/api/transactions?limit=40&${query.toString()}`),
      request(`/api/rules${state.selectedMerchant ? `?merchant_id=${encodeURIComponent(state.selectedMerchant)}` : ""}`),
      request("/api/merchants/summary"),
      request(`/api/transactions/review-queue?limit=30&${queueQuery.toString()}`),
      request("/api/model/status"),
    ]);

    renderMetrics(metrics);
    renderTransactions(transactions);
    renderRules(rules);
    renderMerchantSummary(summary);
    renderReviewQueue(queue);
    renderModelStatus(model);

    setFlash("Dashboard updated.");
  } catch (error) {
    if (isAuthError(error)) {
      clearSession();
      showLoggedOutView();
    }
    setFlash(error.message, true);
  }
}

async function onSeedClick() {
  seedBtn.disabled = true;
  setFlash("Generating synthetic transactions...");

  const query = new URLSearchParams({ count: "60" });
  if (state.selectedMerchant) query.set("merchant_id", state.selectedMerchant);

  try {
    const response = await request(`/api/transactions/seed?${query.toString()}`, {
      method: "POST",
    });

    await loadDashboard();
    setFlash(`Inserted ${response.inserted} transactions.`);
  } catch (error) {
    setFlash(error.message, true);
  } finally {
    seedBtn.disabled = false;
  }
}

async function onScoreSubmit(event) {
  event.preventDefault();
  const payload = formToTransactionPayload(scoreForm);

  if (state.selectedMerchant) {
    payload.merchant_id = state.selectedMerchant;
  }

  try {
    const result = await request("/api/transactions/score", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    renderScoreResult(result);
    await loadDashboard();
    setFlash(`Transaction scored: ${result.decision.toUpperCase()} (${result.risk_score})`);
  } catch (error) {
    setFlash(error.message, true);
  }
}

async function onUploadSubmit(event) {
  event.preventDefault();

  const formData = new FormData(uploadForm);
  const file = formData.get("file");

  if (!(file instanceof File) || !file.name) {
    setFlash("Choose a CSV file first.", true);
    return;
  }

  const query = new URLSearchParams();
  if (state.selectedMerchant) {
    query.set("merchant_id", state.selectedMerchant);
  }

  try {
    setFlash("Uploading CSV and scoring transactions...");
    const result = await request(
      `/api/transactions/upload-csv${query.toString() ? `?${query.toString()}` : ""}`,
      {
        method: "POST",
        body: formData,
      },
      { json: false }
    );

    uploadSummary.textContent = formatUploadSummary(result);
    uploadForm.reset();
    await loadDashboard();
    setFlash(`CSV processed. Inserted ${result.inserted} / ${result.total_rows}.`);
  } catch (error) {
    setFlash(error.message, true);
  }
}

async function onRuleSubmit(event) {
  event.preventDefault();
  const payload = formToRulePayload(ruleForm);

  if (state.user.role === "merchant_admin") {
    payload.merchant_id = state.user.merchant_id;
  }

  try {
    await request("/api/rules", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    ruleForm.reset();
    if (state.user.role === "merchant_admin") {
      const input = ruleForm.querySelector('input[name="merchant_id"]');
      if (input) input.value = state.user.merchant_id || "";
    }

    await loadDashboard();
    setFlash("Rule created.");
  } catch (error) {
    setFlash(error.message, true);
  }
}

async function onRuleToggleClick(event) {
  const button = event.target.closest("button[data-rule-id]");
  if (!button) return;

  const ruleId = button.dataset.ruleId;
  const enabled = button.dataset.enabled !== "true";

  try {
    await request(`/api/rules/${ruleId}/enabled`, {
      method: "PATCH",
      body: JSON.stringify({ enabled }),
    });

    await loadDashboard();
    setFlash(`Rule ${enabled ? "enabled" : "disabled"}.`);
  } catch (error) {
    setFlash(error.message, true);
  }
}

async function onQueueActionClick(event) {
  const button = event.target.closest("button[data-txn-id]");
  if (!button) return;

  const txnId = button.dataset.txnId;
  const statusEl = document.querySelector(`select[data-status-for="${txnId}"]`);
  const noteEl = document.querySelector(`input[data-note-for="${txnId}"]`);

  if (!statusEl) return;

  try {
    await request(`/api/transactions/${txnId}/status`, {
      method: "PATCH",
      body: JSON.stringify({
        status: statusEl.value,
        note: noteEl ? noteEl.value.trim() || null : null,
      }),
    });

    await loadDashboard();
    setFlash("Case updated.");
  } catch (error) {
    setFlash(error.message, true);
  }
}

async function onReloadModel() {
  try {
    setFlash("Reloading model...");
    await request("/api/model/reload", { method: "POST" });
    await loadDashboard();
    setFlash("Model reloaded.");
  } catch (error) {
    setFlash(error.message, true);
  }
}

async function onTrainFromCases() {
  if (state.user?.role !== "analyst") {
    setFlash("Only analysts can train the model.", true);
    return;
  }

  try {
    setFlash("Training model from confirmed cases...");

    const payload = {
      merchant_id: state.selectedMerchant || null,
      min_samples: 300,
      max_cases: 20000,
      allow_synthetic: true,
    };

    const result = await request("/api/model/train-from-cases", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    trainSummary.textContent = `Trained with ${result.trained_samples} samples (real: ${result.confirmed_cases_used}, synthetic: ${result.synthetic_added}). ROC-AUC: ${result.metrics.roc_auc}, F1: ${result.metrics.f1}`;

    await loadDashboard();
    setFlash("Model retrained from cases.");
  } catch (error) {
    setFlash(error.message, true);
  }
}

function onMerchantChange() {
  state.selectedMerchant = merchantSelect.value || "";

  const scoreMerchantInput = scoreForm.querySelector('input[name="merchant_id"]');
  if (scoreMerchantInput && state.selectedMerchant) {
    scoreMerchantInput.value = state.selectedMerchant;
  }

  loadDashboard();
}

function onFilterChange() {
  state.decisionFilter = decisionFilter.value;
  state.statusFilter = statusFilter.value;
  loadDashboard();
}

function renderMetrics(metrics) {
  metricTotal.textContent = String(metrics.total_24h || 0);
  metricAllow.textContent = String(metrics.allow_24h || 0);
  metricReview.textContent = String(metrics.review_24h || 0);
  metricBlock.textContent = String(metrics.block_24h || 0);
  metricRisk.textContent = Number(metrics.avg_risk_score_24h || 0).toFixed(2);
}

function renderTransactions(rows) {
  if (!rows.length) {
    txnBody.innerHTML = '<tr><td colspan="8">No transactions found.</td></tr>';
    return;
  }

  txnBody.innerHTML = rows
    .map((txn) => {
      const reasons = (txn.reasons || []).slice(0, 2).join(" ");
      return `
        <tr>
          <td>${formatDate(txn.timestamp)}</td>
          <td>${escapeHtml(txn.merchant_id)}</td>
          <td>${escapeHtml(txn.payer_id)}</td>
          <td>${formatCurrency(txn.amount, txn.currency)}</td>
          <td>${Number(txn.risk_score || 0).toFixed(2)}</td>
          <td><span class="decision ${txn.decision}">${txn.decision.toUpperCase()}</span></td>
          <td><span class="decision ${txn.status}">${escapeHtml(txn.status.toUpperCase())}</span></td>
          <td class="reason-text">${escapeHtml(reasons || "-")}</td>
        </tr>
      `;
    })
    .join("");
}

function renderRules(rules) {
  if (!rules.length) {
    ruleList.innerHTML = '<li class="rule-item">No rules yet.</li>';
    return;
  }

  ruleList.innerHTML = rules
    .map((rule) => {
      const merchantTag = rule.merchant_id ? rule.merchant_id : "GLOBAL";
      return `
        <li class="rule-item">
          <strong>${escapeHtml(rule.name)}</strong>
          <p>${escapeHtml(merchantTag)} - Action: ${rule.action.toUpperCase()}</p>
          <div class="row">
            <span class="decision ${rule.enabled ? "allow" : "block"}">${rule.enabled ? "ENABLED" : "DISABLED"}</span>
            <button type="button" data-rule-id="${rule.id}" data-enabled="${rule.enabled}">
              ${rule.enabled ? "Disable" : "Enable"}
            </button>
          </div>
        </li>
      `;
    })
    .join("");
}

function renderMerchantSummary(rows) {
  if (!rows.length) {
    merchantBody.innerHTML = '<tr><td colspan="7">No merchant data yet.</td></tr>';
    return;
  }

  merchantBody.innerHTML = rows
    .map((row) => `
      <tr>
        <td>${escapeHtml(row.merchant_id)}</td>
        <td>${row.total_24h}</td>
        <td>${row.review_24h}</td>
        <td>${row.block_24h}</td>
        <td>${Number(row.avg_risk_score_24h || 0).toFixed(2)}</td>
        <td>${Number(row.review_rate_pct || 0).toFixed(2)}%</td>
        <td>${Number(row.block_rate_pct || 0).toFixed(2)}%</td>
      </tr>
    `)
    .join("");
}

function renderReviewQueue(rows) {
  if (!rows.length) {
    queueBody.innerHTML = '<tr><td colspan="6">Review queue is empty.</td></tr>';
    return;
  }

  const statusOptions = [
    "review",
    "investigating",
    "resolved_legit",
    "confirmed_fraud",
    "block",
  ];

  queueBody.innerHTML = rows
    .map((txn) => {
      const optionMarkup = statusOptions
        .map((status) => `<option value="${status}" ${txn.status === status ? "selected" : ""}>${status}</option>`)
        .join("");

      return `
        <tr>
          <td>${formatDate(txn.timestamp)}</td>
          <td>${escapeHtml(txn.merchant_id)}</td>
          <td>${escapeHtml(txn.payer_id)}</td>
          <td>${Number(txn.risk_score || 0).toFixed(2)}</td>
          <td><span class="decision ${txn.status}">${escapeHtml(txn.status.toUpperCase())}</span></td>
          <td>
            <div class="action-row">
              <select data-status-for="${txn.id}">${optionMarkup}</select>
              <input type="text" data-note-for="${txn.id}" placeholder="Case note" value="${escapeHtml(txn.case_note || "")}">
              <button type="button" data-txn-id="${txn.id}">Save</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}

function renderModelStatus(model) {
  if (!model) {
    modelStatus.textContent = "Model status unavailable.";
    return;
  }

  const loaded = model.loaded ? "Loaded" : "Not loaded";
  const auc = model.metrics?.roc_auc;
  const f1 = model.metrics?.f1;
  const trainedAt = model.trained_at ? ` | Trained: ${formatDate(model.trained_at)}` : "";

  modelStatus.textContent = `${loaded} | Path: ${model.path}${auc !== undefined ? ` | ROC-AUC: ${auc}` : ""}${f1 !== undefined ? ` | F1: ${f1}` : ""}${trainedAt}`;
}

function renderScoreResult(txn) {
  const reasons = (txn.reasons || []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join("");
  const modelProb = txn.model_probability == null ? "N/A" : `${(txn.model_probability * 100).toFixed(2)}%`;

  scoreResult.innerHTML = `
    <h3>Last Decision</h3>
    <p><strong>Transaction:</strong> ${escapeHtml(txn.id)}</p>
    <p><strong>Risk Score:</strong> ${Number(txn.risk_score || 0).toFixed(2)}</p>
    <p><strong>Heuristic Score:</strong> ${Number(txn.heuristic_score || 0).toFixed(2)}</p>
    <p><strong>Model Probability:</strong> ${modelProb}</p>
    <p><span class="decision ${txn.decision}">${txn.decision.toUpperCase()}</span></p>
    <ul>${reasons}</ul>
  `;
}

function formatUploadSummary(result) {
  const lines = [
    `Rows: ${result.total_rows}`,
    `Inserted: ${result.inserted}`,
    `Failed: ${result.failed}`,
  ];

  if (Array.isArray(result.errors) && result.errors.length) {
    lines.push("Errors:");
    for (const item of result.errors.slice(0, 10)) {
      lines.push(`- row ${item.row}: ${item.message}`);
    }
    if (result.errors.length > 10) {
      lines.push(`... and ${result.errors.length - 10} more`);
    }
  }

  return lines.join("\n");
}

function formToTransactionPayload(form) {
  const formData = new FormData(form);

  return {
    merchant_id: String(formData.get("merchant_id") || "").trim(),
    payer_id: String(formData.get("payer_id") || "").trim(),
    amount: Number(formData.get("amount")),
    currency: "INR",
    device_id: String(formData.get("device_id") || "").trim(),
    lat: Number(formData.get("lat")),
    lon: Number(formData.get("lon")),
    payer_account_age_days: Number(formData.get("payer_account_age_days")),
  };
}

function formToRulePayload(form) {
  const formData = new FormData(form);

  const payload = {
    name: String(formData.get("name") || "").trim(),
    merchant_id: normalizeOptionalString(formData.get("merchant_id")),
    action: String(formData.get("action") || "review"),
    enabled: true,
    condition: {
      require_new_device: formData.get("require_new_device") === "on",
      night_hours_only: formData.get("night_hours_only") === "on",
    },
  };

  const minAmount = normalizeOptionalNumber(formData.get("min_amount"));
  if (minAmount !== null) payload.condition.min_amount = minAmount;

  const velocity = normalizeOptionalNumber(formData.get("velocity_10m_gt"));
  if (velocity !== null) payload.condition.velocity_10m_gt = Math.floor(velocity);

  const amountSpike = normalizeOptionalNumber(formData.get("amount_spike_gt"));
  if (amountSpike !== null) payload.condition.amount_spike_gt = amountSpike;

  return payload;
}

async function request(url, options = {}, config = { auth: true, json: true }) {
  const headers = {
    ...(options.headers || {}),
  };

  const shouldAttachJsonHeader =
    config.json !== false && options.body !== undefined && !(options.body instanceof FormData);

  if (shouldAttachJsonHeader) {
    headers["Content-Type"] = "application/json";
  }

  if (config.auth !== false && state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  const isJson = response.headers.get("content-type")?.includes("application/json");
  const data = isJson ? await response.json() : null;

  if (!response.ok) {
    const message = data?.detail || `Request failed: ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }

  return data;
}

function isAuthError(error) {
  return Number(error?.status) === 401;
}

function setFlash(message, isError = false) {
  flash.textContent = message;
  flash.style.color = isError ? "#b93f33" : "#546377";
}

function normalizeOptionalNumber(value) {
  if (value === null) return null;
  const text = String(value).trim();
  if (!text) return null;

  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeOptionalString(value) {
  if (value === null) return null;
  const text = String(value).trim();
  return text ? text : null;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";

  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatCurrency(amount, currency) {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(amount);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
