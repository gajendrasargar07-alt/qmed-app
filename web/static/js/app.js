(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const MAX_MB = Number(document.body.dataset.maxMb) || 15;
  const STORE_KEY = "qmed.recent";
  const MAX_RECENT = 8;
  const TYPES = [
    ["breast", "Breast"], ["lung", "Lung"], ["prostate", "Prostate"],
    ["colon", "Colon"], ["cervical", "Cervical"],
  ];
  const BAND_LABEL = { low: "Low risk", moderate: "Moderate risk", high: "High risk" };

  const screens = {
    home: $("#screen-home"),
    working: $("#screen-working"),
    result: $("#screen-result"),
  };
  const resultBody = $("#resultBody");

  let lastInput = null;      // what was last analysed, so the type can be changed
  let controller = null;     // aborts an in-flight request
  let stepTimer = null;

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ------------------------------------------------------------ navigation */
  function show(name) {
    for (const [key, el] of Object.entries(screens)) el.hidden = key !== name;
    window.scrollTo(0, 0);
    const heading = screens[name].querySelector("h1");
    if (heading) heading.focus({ preventScroll: true });
  }

  function goHome() {
    if (history.state && history.state.view === "result") history.back();
    else show("home");
  }

  window.addEventListener("popstate", () => {
    if (controller) controller.abort();
    show("home");
  });

  /* ------------------------------------------------------------ file intake */
  const ACCEPT = /\.(pdf|txt|png|jpe?g|tiff?|webp|bmp)$/i;

  function takeFile(file) {
    if (!file) return;
    if (file.size > MAX_MB * 1024 * 1024) {
      return showMessage({ status: "error", title: "That file is too large",
        message: `Choose a file under ${MAX_MB} MB.` });
    }
    const okType = ACCEPT.test(file.name) || file.type.startsWith("image/")
      || file.type === "application/pdf" || file.type === "text/plain";
    if (!okType) {
      return showMessage({ status: "error", title: "We can't open that kind of file",
        message: "Upload a PDF, a photo of the report, or a text file." });
    }
    analyse({ file });
  }

  const fileInput = $("#fileInput");
  const cameraInput = $("#cameraInput");
  $("#pickBtn").addEventListener("click", () => fileInput.click());
  $("#cameraBtn").addEventListener("click", () => cameraInput.click());
  for (const input of [fileInput, cameraInput]) {
    input.addEventListener("change", () => {
      const f = input.files && input.files[0];
      input.value = "";
      takeFile(f);
    });
  }

  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove("over");
  }));
  drop.addEventListener("drop", (e) => takeFile(e.dataTransfer.files[0]));
  // a file dropped outside the zone should not navigate away from the app
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  const pasteToggle = $("#pasteToggle");
  const pasteBox = $("#pasteBox");
  pasteToggle.addEventListener("click", () => {
    const open = pasteBox.hidden;
    pasteBox.hidden = !open;
    pasteToggle.setAttribute("aria-expanded", String(open));
    pasteToggle.textContent = open ? "Hide pasted text" : "Paste text instead";
    if (open) $("#pasteText").focus();
  });
  $("#pasteGo").addEventListener("click", () => {
    const text = $("#pasteText").value.trim();
    if (!text) return $("#pasteText").focus();
    analyse({ text });
  });

  /* ------------------------------------------------------------ analysing */
  function startSteps() {
    const items = [...document.querySelectorAll("#steps li")];
    items.forEach((li) => li.className = "");
    let i = 0;
    items[0].classList.add("active");
    clearInterval(stepTimer);
    stepTimer = setInterval(() => {
      if (i < items.length - 1) {
        items[i].className = "done";
        i += 1;
        items[i].className = "active";
      }
    }, 650);
  }

  function stopSteps() {
    clearInterval(stepTimer);
    document.querySelectorAll("#steps li").forEach((li) => li.className = "done");
  }

  async function analyse(input, type) {
    lastInput = input;
    show("working");
    startSteps();
    controller = new AbortController();
    const body = new FormData();
    if (input.file) body.append("file", input.file);
    else body.append("text", input.text);
    if (type) body.append("type", type);

    let data;
    try {
      const [res] = await Promise.all([
        fetch("/api/analyze", { method: "POST", body, signal: controller.signal }),
        sleep(2000),   // keep the progress readable even when the server is fast
      ]);
      data = await res.json();
    } catch (err) {
      if (err.name === "AbortError") return;
      data = { status: "error", title: "Can't reach QMed",
        message: "Check your connection and try again." };
    }
    controller = null;
    stopSteps();
    data.status === "ok" ? showResult(data, true) : showMessage(data);
  }

  $("#cancelBtn").addEventListener("click", () => {
    if (controller) controller.abort();
    clearInterval(stepTimer);
    show("home");
  });

  /* ------------------------------------------------------------ rendering */
  function enter() {
    if (!(history.state && history.state.view === "result")) {
      history.pushState({ view: "result" }, "");
    }
    show("result");
  }

  function showMessage(vm) {
    const missing = (vm.missing || []).length
      ? `<p class="fine">These are needed for a reliable result:</p>
         <ul class="missing">${vm.missing.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>`
      : "";
    const canRetype = lastInput && vm.status !== "error" && vm.types;
    resultBody.innerHTML = `
      <div class="message">
        ${vm.report ? `<div class="type-row"><span>${esc(vm.report.label)} report</span></div>` : ""}
        <h1 tabindex="-1">${esc(vm.title)}</h1>
        <p>${esc(vm.message || "")}</p>
        ${missing}
        <div class="stack">
          <button class="btn" data-act="new" type="button">Try another report</button>
          ${canRetype && vm.status === "incomplete"
            ? `<button class="btn secondary" data-act="retype" type="button">This is a different type of report</button>` : ""}
        </div>
      </div>`;
    wireResult(vm);
    $("#saveBtn").hidden = true;
    enter();
  }

  function valueRow(v) {
    const unit = v.unit ? `<span class="unit">${esc(v.unit)}</span>` : "";
    return `<div class="row"><dt>${esc(v.label)}</dt><dd>${esc(v.value)}${unit}</dd></div>`;
  }

  function showResult(vm, fresh) {
    const v = vm.verdict;
    const values = vm.values || [];
    const shown = values.slice(0, 8);
    const extra = values.slice(8);

    const stageRow = vm.stage
      ? `<div class="row"><dt>Estimated stage</dt><dd>${esc(vm.stage.label)}</dd></div>` : "";
    const votes = vm.agreement.models.map((m) =>
      `<span class="vote ${m.flagged ? "on" : ""}"></span>`).join("");

    const notices = [];
    if (vm.quality.low_ocr) {
      notices.push("Parts of this scan were hard to read. Check the values below against your report.");
    }
    if (vm.quality.used_averages) {
      notices.push(`We read ${vm.quality.values_read} of ${vm.quality.values_expected} values. `
        + "The rest were filled in with typical values, so treat this result as less certain.");
    }

    resultBody.innerHTML = `
      <div class="type-row">
        <span>${esc(vm.report.label)} report</span>
        <button class="link-btn" data-act="retype" type="button">${vm.report.auto_detected ? "Not right?" : "Change type"}</button>
      </div>
      <h1 class="verdict ${v.band}" tabindex="-1">${esc(v.headline)}</h1>

      <div class="spectrum" role="img" aria-label="Score ${v.score} out of 100, ${BAND_LABEL[v.band].toLowerCase()}">
        <div class="track">
          <div class="zone low"></div><div class="zone moderate"></div><div class="zone high"></div>
          <div class="marker" id="marker"></div>
        </div>
        <div class="zone-labels"><span>Low</span><span>Moderate</span><span>High</span></div>
      </div>
      <p class="score-line">Score ${v.score} out of 100. A combined model score, not a probability.</p>

      <p class="summary">${esc(v.summary)}</p>

      ${notices.map((n) => `<p class="notice">${esc(n)}</p>`).join("")}

      <section class="next ${v.band}">
        <h2>What to do next</h2>
        <p class="action">${esc(v.action)}</p>
        <ul>${v.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>
      </section>

      <section class="block-section">
        <h2>More from the analysis</h2>
        <dl class="rows">
          <div class="row"><dt>Early-risk signal</dt>
            <dd class="${vm.early_risk.level}">${BAND_LABEL[vm.early_risk.level]}</dd></div>
          ${stageRow}
          <div class="row"><dt>Models that flagged it</dt>
            <dd><span class="votes" aria-hidden="true">${votes}</span>${vm.agreement.flagged} of ${vm.agreement.total}</dd></div>
        </dl>
        <details class="models">
          <summary>See each model</summary>
          <ul class="model-list">
            ${vm.agreement.models.map((m) => `
              <li><span class="vote ${m.flagged ? "on" : ""}"></span>
                <span>${esc(m.name)}</span>
                <span class="tag">${m.kind === "quantum" ? "Quantum" : "Classical"}</span></li>`).join("")}
          </ul>
        </details>
      </section>

      <section class="block-section">
        <h2>Values we read</h2>
        <dl class="rows">${shown.map(valueRow).join("")}</dl>
        ${extra.length ? `
          <dl class="rows extra-rows" id="extraRows" hidden>${extra.map(valueRow).join("")}</dl>
          <button class="link-btn more-btn" id="moreBtn" type="button" aria-expanded="false">Show ${extra.length} more</button>` : ""}
      </section>

      <p class="disclaimer">QMed gives information, not a diagnosis or medical advice. Only a doctor can diagnose a condition. If you have symptoms or worries, see a doctor whatever this result says.</p>

      <div class="stack">
        <button class="btn" data-act="new" type="button">Check another report</button>
      </div>`;

    wireResult(vm);
    $("#saveBtn").hidden = false;

    const more = $("#moreBtn");
    if (more) more.addEventListener("click", () => {
      const rows = $("#extraRows");
      const open = rows.hidden;
      rows.hidden = !open;
      more.setAttribute("aria-expanded", String(open));
      more.textContent = open ? "Show fewer" : `Show ${extra.length} more`;
    });

    enter();
    // single reveal: the marker travels to the score once the screen is up
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const m = $("#marker");
      if (m) m.style.setProperty("--pos", Math.min(Math.max(v.score, 1), 99));
    }));

    if (fresh) remember(vm);
  }

  function wireResult(vm) {
    resultBody.querySelectorAll("[data-act='new']").forEach((b) =>
      b.addEventListener("click", goHome));
    resultBody.querySelectorAll("[data-act='retype']").forEach((b) =>
      b.addEventListener("click", () => openTypeSheet(vm)));
    // results reopened from the recent list have no file to re-run
    if (!lastInput) {
      resultBody.querySelectorAll("[data-act='retype']").forEach((b) => b.hidden = true);
    }
  }

  $("#backBtn").addEventListener("click", goHome);
  $("#saveBtn").addEventListener("click", () => window.print());
  window.addEventListener("beforeprint", () => {
    document.querySelectorAll("details.models").forEach((d) => d.open = true);
    const rows = $("#extraRows");
    if (rows) rows.hidden = false;
  });

  /* ------------------------------------------------------------ type sheet */
  const typeSheet = $("#typeSheet");
  function openTypeSheet(vm) {
    const current = vm.report && vm.report.type;
    $("#typeList").innerHTML = TYPES.map(([id, label]) =>
      `<button class="type-btn" data-type="${id}" type="button"
        ${id === current ? 'aria-current="true"' : ""}>${label}</button>`).join("");
    typeSheet.showModal();
  }
  $("#typeList").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-type]");
    if (!btn || !lastInput) return;
    typeSheet.close();
    analyse(lastInput, btn.dataset.type);
  });
  $("#typeClose").addEventListener("click", () => typeSheet.close());

  /* ------------------------------------------------------------ about */
  const about = $("#about");
  $("#aboutBtn").addEventListener("click", () => about.showModal());
  $("#aboutClose").addEventListener("click", () => about.close());
  for (const dlg of [about, typeSheet]) {
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  }

  /* ------------------------------------------------------------ recent results */
  function readStore() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; }
    catch { return []; }
  }
  function writeStore(list) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(list)); } catch { /* storage unavailable */ }
  }

  function remember(vm) {
    const list = readStore();
    list.unshift({ id: Date.now(), ts: Date.now(), vm });
    writeStore(list.slice(0, MAX_RECENT));
    renderRecent();
  }

  function when(ts) {
    const days = Math.floor((new Date().setHours(0, 0, 0, 0) - new Date(ts).setHours(0, 0, 0, 0)) / 864e5);
    if (days <= 0) return "Today";
    if (days === 1) return "Yesterday";
    if (days < 7) return `${days} days ago`;
    return new Date(ts).toLocaleDateString(undefined, { day: "numeric", month: "short" });
  }

  function renderRecent() {
    const list = readStore();
    $("#recent").hidden = list.length === 0;
    $("#recentList").innerHTML = list.map((item) => `
      <li><button class="recent-item" data-id="${item.id}" type="button">
        <span class="dot ${item.vm.verdict.band}"></span>
        <span class="recent-main">${esc(item.vm.report.label)} report, ${BAND_LABEL[item.vm.verdict.band].toLowerCase()}</span>
        <span class="recent-meta">${when(item.ts)}</span>
      </button></li>`).join("");
  }

  $("#recentList").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-id]");
    if (!btn) return;
    const item = readStore().find((x) => String(x.id) === btn.dataset.id);
    if (!item) return;
    lastInput = null;
    showResult(item.vm, false);
  });

  $("#clearRecent").addEventListener("click", () => {
    writeStore([]);
    renderRecent();
  });

  renderRecent();
})();
