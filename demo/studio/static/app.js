/* ------------------------------------------------------------------ *
 * Anonymiser Studio
 *
 * One document, four views of it. The page is built once per analysis and
 * the entity elements are reused across modes, so switching treatment
 * changes the words in place rather than swapping one block of text for
 * another — the difference between the modes is the thing worth seeing.
 * ------------------------------------------------------------------ */
const MODES = {
  source:       { label: "Source",        blurb: "The document as it arrived." },
  redact:       { label: "Redact",        blurb: "Direct identifiers removed." },
  anonymise:    { label: "Anonymise",     blurb: "Direct removed, quasi-identifiers broadened until the document stops being unique." },
  pseudonymise: { label: "Pseudonymise",  blurb: "Coded references, reversible from the vault." },
};

const TYPE_MEANING = {
  PERSON: "names of people", ORG: "organisations", CODE: "case and file numbers",
  DATETIME: "dates and times", LOC: "places", DEM: "demographic descriptions",
  QUANTITY: "ages, amounts, sentence lengths", MISC: "anything else identifying",
};

const state = {
  data: null,
  mode: "source",
  selected: null,
  proPseudo: false,      // pseudonymise on top of the Anonymise pipeline
  keepTypes: [],         // entity kinds the reader has chosen to leave in the clear
  kTarget: 5,
  maxIters: 3,
  health: null,
  running: false,
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------------------------------------------------------------- *
 * Which mode's payload backs the current tab
 * ---------------------------------------------------------------- */
function payloadKey(mode) {
  if (mode === "pseudonymise") return state.proPseudo ? "pseudonymise_pro" : "pseudonymise";
  if (mode === "source") return "redact";     // same span layout, untouched text
  return mode;
}

function actionOf(seg, mode) {
  if (mode === "source" || !seg.changed) return "kept";
  if (/^\[[A-Z]+_[A-Z]+\]$/.test(seg.text)) return "pseudonym";
  if (/^\[[A-Z]+\]$/.test(seg.text)) return "redacted";
  return "generalised";
}

/* ---------------------------------------------------------------- *
 * Building the page
 * ---------------------------------------------------------------- */
function renderDoc() {
  const doc = $("doc");
  doc.textContent = "";
  const segments = state.data.modes.redact.segments;   // structure, not content
  const entityById = Object.fromEntries(state.data.entities.map((e) => [e.id, e]));

  for (const seg of segments) {
    if (seg.kind !== "span") {
      doc.appendChild(document.createTextNode(seg.text));
      continue;
    }
    const el = document.createElement("span");
    el.className = "ent";
    el.dataset.id = seg.id;
    el.dataset.type = seg.type;
    el.dataset.role = seg.role;
    el.dataset.source = entityById[seg.id]?.source || seg.source;
    el.tabIndex = 0;
    el.setAttribute("role", "button");
    el.addEventListener("click", () => selectEntity(seg.id));
    el.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); selectEntity(seg.id); }
    });
    doc.appendChild(el);
  }
  applyMode();
}

function applyMode() {
  const mode = state.mode;
  const segs = state.data.modes[payloadKey(mode)].segments;
  const byId = Object.fromEntries(segs.filter((s) => s.kind === "span").map((s) => [s.id, s]));

  for (const el of $("doc").querySelectorAll(".ent")) {
    const seg = byId[el.dataset.id];
    if (!seg) continue;
    const act = actionOf(seg, mode);
    el.textContent = mode === "source" ? seg.before : seg.text;
    el.dataset.act = act;
    el.dataset.view = mode === "source" ? "source" : "treated";
    el.title = mode === "source"
      ? `${seg.type} · ${seg.role === "DIRECT" ? "must be removed" : "quasi-identifier"}`
      : `${seg.before} → ${seg.text}`;
  }

  for (const tab of document.querySelectorAll(".tab")) {
    const on = tab.dataset.mode === mode;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
  }
  $("sheet-status").textContent = MODES[mode].blurb;
  renderPanel();
  renderLedger();
}

/* ---------------------------------------------------------------- *
 * Side panel — different question per mode
 * ---------------------------------------------------------------- */
// The same choice, in a form that fits under an output. Being able to keep a
// kind of identifier only from the Source tab would mean stepping away from
// the thing the choice changes.
function keepControls(d) {
  const types = Object.keys(d.counts || {}).sort();
  if (!types.length) return "";
  const chips = types.map((type) => {
    const kept = state.keepTypes.includes(type);
    return `<label class="chip ${kept ? "is-kept" : ""}">
      <input type="checkbox" class="keep-box" data-type="${esc(type)}" ${kept ? "" : "checked"}>
      <span>${esc(type)}</span></label>`;
  }).join("");
  return `<h3>What to redact</h3>
    <div class="chips">${chips}</div>
    <p class="note">Untick to keep that kind in the document. Kept identifiers still
    count towards the risk above.</p>`;
}

function renderPanel() {
  if (state.selected) return renderInspector();
  const panel = $("panel");
  const d = state.data;

  if (state.mode === "source") {
    const counts = d.counts || {};
    const direct = d.entities.filter((e) => e.role === "DIRECT").length;
    const quasi = d.entities.filter((e) => e.role === "QUASI").length;
    const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([type, n]) => {
      const role = d.roles[type] || "DIRECT";
      const kept = state.keepTypes.includes(type);
      return `<li class="${kept ? "is-kept" : ""}">
        <span class="swatch ${role === "DIRECT" ? "swatch-direct" : "swatch-quasi"}"></span>
        <span><strong>${esc(type)}</strong><br><span class="role">${esc(TYPE_MEANING[type] || "")}</span></span>
        <label class="keep" title="Untick to leave this kind of identifier in the document">
          <input type="checkbox" class="keep-box" data-type="${esc(type)}" ${kept ? "" : "checked"}>
          <span class="n">${n}</span>
        </label></li>`;
    }).join("");

    const coref = d.coref.added.length
      ? `<h3>Caught by the shorthand pass</h3>
         <p>${d.coref.added.length} mention${d.coref.added.length === 1 ? "" : "s"} the model
         missed after the first full name: ${d.coref.added.slice(0, 6).map((c) => `<strong>${esc(c.text)}</strong>`).join(", ")}.</p>
         <p class="note">Measured against the full benchmark this pass is worth +0.0001 macro recall —
         a null result, kept because it costs nothing and occasionally catches exactly this.</p>`
      : `<h3>Shorthand pass</h3><p>Found nothing the detector had already missed on this document.</p>`;

    panel.innerHTML = `
      <h2>${direct + quasi} identifiers found</h2>
      <p class="lede">${direct} must be removed. ${quasi} are quasi-identifiers —
      harmless alone, identifying in combination.</p>
      <h3>By kind</h3>
      <p class="note" style="margin-top:-4px">Untick a kind to keep it in the document. It
      is still counted in the re-identification risk, so the number you are shown is the
      risk for the document you would actually send.</p>
      <ul class="tally">${rows}</ul>
      ${state.keepTypes.length ? `<p class="keep-summary">Keeping
        ${state.keepTypes.map((t) => `<strong>${esc(t)}</strong>`).join(", ")} in the clear.</p>` : ""}
      ${coref}
      <p class="note">Click any marked phrase to see what each mode does to it.</p>`;

    return;
  }

  if (state.mode === "redact") {
    const removed = d.entities.filter((e) => e.fates.redact?.action === "redact").length;
    const left = d.entities.filter((e) => e.fates.redact?.action === "leave").length;
    panel.innerHTML = `
      <h2>${removed} removed</h2>
      <p class="lede">Names, organisations, case numbers. The fast mode, and the right
      one when the worry is that a third party logs your prompts.</p>
      <h3>What it deliberately leaves</h3>
      <p>${left} quasi-identifiers stay in the text — dates, places, ages, demographics.
      Individually none of them names anyone.</p>
      <p class="note">That is also the weakness. Switch to Anonymise to see how many
      documents in the benchmark corpus share this document's remaining fingerprint.</p>
      ${keepControls(d)}`;
    return;
  }

  if (state.mode === "anonymise") {
    const m = d.modes.anonymise.mosaic;
    const safe = m.final >= state.kTarget;
    const unavailable = d.haystack !== "tab";
    const ladder = d.entities
      .filter((e) => e.fates.anonymise?.action === "generalize")
      .map((e) => {
        const steps = e.fates.anonymise.steps || [];
        const path = steps.map((s, i) =>
          `<span class="${i === steps.length - 1 ? "to" : "mid"}">${esc(s.replacement)}</span>`
        ).join(" → ");
        return `<li><span class="from">${esc(e.text)}</span> → ${path || "—"}</li>`;
      }).join("");

    const kPath = (d.k_path || []).length
      ? `<h3>What broadening bought</h3>
         <p class="kpath">k = ${m.initial}${(d.k_path || []).map((s) => ` → ${s.k}`).join("")}</p>
         <p>${(d.k_path || []).slice(0, -1).every((s) => s.k < state.kTarget)
              ? "Every round of broadening left the document exactly as identifiable as it started. The number only moves at the end, when the last quasi-identifier is removed and there is no fingerprint left to match."
              : "Broadening moved the document out of uniqueness before suppression was needed."}</p>`
      : "";

    panel.innerHTML = `
      <h2>Re-identification risk</h2>
      ${unavailable
        ? `<p class="lede">The benchmark corpus is not loaded, so re-identification risk cannot be scored this session.</p>`
        : m.converged
          ? `<div class="stamp" data-safe="true">
               k = ${m.final}
               <small>reached by broadening</small>
             </div>
             <p class="lede">${m.initial === 1
               ? "No other document in the corpus shared this document's fingerprint — it was unique."
               : `${m.initial} documents shared this fingerprint.`}
               After ${m.iterations} round${m.iterations === 1 ? "" : "s"} of broadening,
               ${m.final} share it.</p>`
          : `<div class="stamp" data-safe="false">
               suppressed
               <small>broadening was not enough</small>
             </div>
             <p class="lede">${m.initial === 1
               ? "No other document in the corpus shared this document's fingerprint — it was unique."
               : `Only ${m.initial} documents shared this fingerprint.`}
               Broadening ran out after ${m.iterations} round${m.iterations === 1 ? "" : "s"} without
               reaching ${state.kTarget}, so every quasi-identifier was removed outright.</p>
             <p class="note">The fingerprint is now empty, which trivially satisfies any target —
               so k is not a meaningful score here. The pipeline reports this as a failure to
               converge rather than a success, and this panel follows it.</p>`}
      ${keepControls(d)}
      ${kPath}
      ${ladder ? `<h3>${m.converged ? "What was broadened" : "How far each one was broadened before it was dropped"}</h3>
                  <ul class="ladder">${ladder}</ul>` : ""}
      <h3>Target</h3>
      <label class="control">
        <span>Stop when at least <span class="val" id="k-val">${state.kTarget}</span> documents share the fingerprint</span>
        <input type="range" id="k-range" min="2" max="10" step="1" value="${state.kTarget}">
      </label>
      <label class="control">
        <span>Give up after <span class="val" id="i-val">${state.maxIters}</span> rounds</span>
        <input type="range" id="i-range" min="1" max="5" step="1" value="${state.maxIters}">
      </label>
      <p class="note">The corpus here is the benchmark's 555 test documents, standing in
      for a firm's own archive.</p>`;

    const bind = (rangeId, valId, key) => {
      const r = $(rangeId);
      r.addEventListener("input", () => { $(valId).textContent = r.value; });
      r.addEventListener("change", () => { state[key] = Number(r.value); run(); });
    };
    bind("k-range", "k-val", "kTarget");
    bind("i-range", "i-val", "maxIters");
    return;
  }

  // Pseudonymise
  const vault = d.modes[payloadKey("pseudonymise")].vault || {};
  const entries = Object.entries(vault);
  const rows = entries.map(([tok, orig]) =>
    `<tr><td><code>${esc(tok)}</code></td><td>${esc(orig)}</td></tr>`).join("");

  panel.innerHTML = `
    <h2>${entries.length} coded reference${entries.length === 1 ? "" : "s"}</h2>
    <p class="lede">Plain redaction destroys who-did-what-to-whom. Codes keep the
    structure, so the text is still worth sending — and the mapping never leaves.</p>
    <label class="check" style="margin-bottom:14px">
      <input type="checkbox" id="pro-pseudo" ${state.proPseudo ? "checked" : ""}>
      <span>Broaden quasi-identifiers too</span>
    </label>
    <h3>Vault</h3>
    ${entries.length
      ? `<table class="vault"><thead><tr><th>Code</th><th>Held locally</th></tr></thead><tbody>${rows}</tbody></table>`
      : `<p>No direct identifiers in this document, so there is nothing to code.</p>`}
    ${entries.length ? `
      <h3>Bring an answer back</h3>
      <div class="roundtrip">
        <textarea id="llm-answer">${esc(sampleAnswer(entries))}</textarea>
        <button class="btn" id="do-restore" type="button" style="margin-top:8px">Restore names</button>
        <div class="out" id="restored" hidden></div>
      </div>
      <p class="note">This is the point of the mode: the codes go out, the answer comes
      back, and the names are re-attached on this machine.</p>` : ""}
    ${linkLedger(d.modes[payloadKey("pseudonymise")].links)}
    ${keepControls(d)}`;

  $("pro-pseudo")?.addEventListener("change", (ev) => {
    state.proPseudo = ev.target.checked;
    applyMode();
  });
  $("do-restore")?.addEventListener("click", async () => {
    const res = await fetch("/api/restore", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("llm-answer").value, vault }),
    }).then((r) => r.json());
    const out = $("restored");
    let html = esc(res.restored);
    for (const orig of Object.values(vault)) {
      html = html.split(esc(orig)).join(`<mark>${esc(orig)}</mark>`);
    }
    out.innerHTML = html;
    out.hidden = false;
  });
}

// Which mentions were folded together, and which were deliberately not. A
// wrong merge is the expensive error here: two people behind one code means
// restore() writes one name over both.
function linkLedger(links) {
  const rows = Object.values(links || {});
  const merged = rows.filter((r) => r.merged && r.evidence !== "exact");
  const refused = rows.filter((r) => !r.merged && r.refusal && r.refusal !== "type_not_linkable");
  if (!merged.length && !refused.length) return "";

  const line = (r) => r.merged
    ? `<li><span class="ll-form">${esc(r.matched_form)}</span> and its shorthand share
         <code>${esc(r.token)}</code> — ${esc(EVIDENCE_PROSE[r.evidence] || r.evidence)},
         <span class="${confidenceClass(r.confidence)}">${Math.round(r.confidence * 100)}%</span></li>`
    : `<li class="is-refused">${esc(r.rationale)}</li>`;

  return `
    <h3>How the codes were decided</h3>
    <p class="note">Every code is a claim that two mentions are the same entity. The
    percentage is how often a link made on that evidence is right, measured against
    44,614 gold-annotated links.</p>
    <ul class="linkledger">
      ${merged.map(line).join("")}
      ${refused.map(line).join("")}
    </ul>`;
}

function sampleAnswer(entries) {
  const person = entries.filter(([t]) => t.startsWith("[PERSON")).map(([t]) => t);
  const org = entries.filter(([t]) => t.startsWith("[ORG")).map(([t]) => t);
  if (person.length >= 2) {
    return `The complaint turns on what ${person[0]} was told before ${person[1]} signed. ` +
           `On the facts as summarised, ${person[0]} has the stronger position.`;
  }
  if (person.length === 1 && org.length >= 1) {
    return `${person[0]} appears to have a claim against ${org[0]}. The limitation ` +
           `point should be checked before ${person[0]} is advised further.`;
  }
  return `The material question concerns ${entries[0][0]}, and the answer depends on ` +
         `what ${entries[0][0]} was told at the time.`;
}

/* ---------------------------------------------------------------- *
 * Inspector — one entity, every mode, and why
 * ---------------------------------------------------------------- */
function selectEntity(id) {
  state.selected = state.selected === id ? null : id;
  for (const el of $("doc").querySelectorAll(".ent")) {
    el.classList.toggle("is-selected", el.dataset.id === state.selected);
  }
  renderPanel();
}

// A pseudonym is a claim that two mentions are the same entity. Show which
// claim was made, on what evidence, and what that evidence is measured to be
// worth — including when the claim was refused.
const EVIDENCE_PROSE = {
  exact:                 "the identical phrase had already appeared",
  exact_casefold:        "the same phrase in different case",
  honorific_only:        "the same name with the title dropped",
  initialism:            "its initials",
  initialism_loose:      "its initials, with a word dropped",
  initialism_reverse:    "the full name it abbreviates",
  short_in_long_1tok:    "a shorter form of it",
  short_in_long_multi:   "a shorter form of it",
  long_over_short_1tok:  "a longer phrase containing it",
  long_over_short_multi: "a longer phrase containing it",
};

function confidenceClass(c) {
  if (c === null || c === undefined) return "";
  return c >= 0.8 ? "is-strong" : c >= 0.5 ? "is-fair" : "is-weak";
}

function linkNote(link) {
  if (!link) return "";
  if (link.merged) {
    const pct = Math.round(link.confidence * 100);
    return `<div class="link ${confidenceClass(link.confidence)}">
      <span class="link-tag">same entity as</span>
      <span class="link-form">${esc(link.matched_form)}</span>
      <span class="link-why">— matched by ${esc(EVIDENCE_PROSE[link.evidence] || link.evidence)}</span>
      <span class="link-conf">${pct}% of such links are right</span>
    </div>`;
  }
  const reason = {
    ambiguous: `${link.tied} entities matched it equally well, so neither was chosen`,
    below_threshold: "the only match was too weak to act on",
    type_not_linkable: "shorthand is not resolved for this kind of identifier",
  }[link.refusal];
  if (!reason) return "";
  return `<div class="link is-refused">
    <span class="link-tag">kept separate</span>
    <span class="link-why">— ${esc(reason)}</span>
  </div>`;
}

function renderInspector() {
  const e = state.data.entities.find((x) => x.id === state.selected);
  if (!e) { state.selected = null; return renderPanel(); }

  const order = [["redact", "Redact"], ["anonymise", "Anonymise"],
                 [payloadKey("pseudonymise"), "Pseudonymise"]];
  const fates = order.map(([key, label]) => {
    const f = e.fates[key];
    if (!f) return "";
    const verb = { redact: "removed", generalize: "broadened", leave: "left in place" }[f.action] || f.action;
    const trail = (f.steps && f.steps.length > 1)
      ? `<div class="trail">${e.text}${f.steps.map((s) => ` → ${esc(s.replacement)}`).join("")}</div>`
      : "";
    return `<div class="fate">
      <div class="fate-head"><span class="fate-mode">${label}</span>
        <span class="fate-act">${verb}</span></div>
      <div class="fate-val">${esc(f.replacement ?? e.text)}</div>
      ${trail}
      <p class="fate-why">${esc(f.rationale)}</p>
      ${linkNote(f.link)}
    </div>`;
  }).join("");

  $("panel").innerHTML = `
    <button class="inspector-back" id="unselect" type="button">← All identifiers</button>
    <h2>${esc(e.text)}</h2>
    <p class="lede">${esc(e.type)} — ${esc(TYPE_MEANING[e.type] || "")}.
    Classified ${e.role === "DIRECT" ? "a direct identifier, so it always goes" : "a quasi-identifier, so it is only broadened when the document is too unique"}.
    Found by ${e.source === "coref" ? "the shorthand pass, not the model"
              : e.source === "regex" ? "the pattern pass" : "the model"}.</p>
    ${fates}`;
  $("unselect").addEventListener("click", () => selectEntity(state.selected));
}

/* ---------------------------------------------------------------- *
 * Decision record
 * ---------------------------------------------------------------- */
function renderLedger() {
  const rows = state.data.modes[payloadKey(state.mode === "source" ? "redact" : state.mode)].audit;
  $("ledger-count").textContent =
    `Decision record — ${rows.length} entries, every one with its reason`;
  $("ledger-body").innerHTML = `
    <table class="record">
      <thead><tr><th>Phrase</th><th>Kind</th><th>Found by</th><th>Action</th><th>Became</th><th>Why</th></tr></thead>
      <tbody>${rows.map((r) => `
        <tr>
          <td>${esc(r.text)}</td>
          <td>${esc(r.type)} · ${esc(r.role)}</td>
          <td>${esc(r.source)}</td>
          <td>${esc(r.action)}</td>
          <td><code>${esc(r.replacement ?? "—")}</code></td>
          <td class="why">${esc(r.rationale)}</td>
        </tr>`).join("")}
      </tbody></table>`;
}

/* ---------------------------------------------------------------- *
 * Talking to the server
 * ---------------------------------------------------------------- */
async function run() {
  if (state.running) return;
  state.running = true;
  $("run").disabled = true;
  $("sheet-status").textContent = "Reading the document…";
  try {
    const res = await fetch("/api/analyse", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: $("input").value,
        k_target: state.kTarget,
        max_iterations: state.maxIters,
        coref: $("coref").checked,
        keep_types: state.keepTypes,
      }),
    });
    if (!res.ok) {
      $("sheet-status").textContent = "That document came back empty. Paste some text and try again.";
      return;
    }
    state.data = await res.json();
    state.selected = null;
    renderDoc();
    applyMode();          // stay on the tab the reader was looking at
  } finally {
    state.running = false;
    $("run").disabled = false;
  }
}

async function loadHealth() {
  const h = await fetch("/api/health").then((r) => r.json());
  state.health = h;
  const dot = document.querySelector(".detector-dot");
  const text = $("detector-text");
  dot.dataset.state = h.is_finetuned ? "real" : "fallback";
  text.innerHTML = h.is_finetuned
    ? `<strong>${esc(h.model_name)}</strong> · F1 ${h.f1} · ${esc(h.device)} · window ${esc(h.window)}`
    : `<strong>${esc(h.model_name)}</strong> · off-the-shelf fallback, weaker than the reported model`;
}

async function loadSamples() {
  const samples = await fetch("/api/samples").then((r) => r.json());
  const sel = $("samples");
  sel.innerHTML = samples.map((s, i) => `<option value="${i}">${esc(s.label)}</option>`).join("");
  sel.addEventListener("change", () => {
    $("input").value = samples[sel.value].text;
    run();
  });
  if (samples.length) $("input").value = samples[0].text;
}

/* ---------------------------------------------------------------- *
 * Wiring
 * ---------------------------------------------------------------- */
function openDrawer(open) {
  $("drawer").hidden = !open;
  $("scrim").hidden = !open;
  if (open) $("input").focus();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    state.mode = tab.dataset.mode;
    state.selected = null;
    for (const el of $("doc").querySelectorAll(".ent")) el.classList.remove("is-selected");
    applyMode();
  });
});

$("open-doc").addEventListener("click", () => openDrawer(true));
$("close-doc").addEventListener("click", () => openDrawer(false));
$("scrim").addEventListener("click", () => openDrawer(false));
$("run").addEventListener("click", () => { openDrawer(false); run(); });
$("coref").addEventListener("change", () => run());
$("ledger-toggle").addEventListener("click", () => {
  const body = $("ledger-body");
  body.hidden = !body.hidden;
  $("ledger-toggle").setAttribute("aria-expanded", String(!body.hidden));
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") { openDrawer(false); if (state.selected) selectEntity(state.selected); }
});

// Delegated once: the toggles are re-rendered on every panel, and rebinding
// them each time is how a listener ends up attached four deep.
document.getElementById("panel").addEventListener("change", (ev) => {
  const box = ev.target.closest(".keep-box");
  if (!box) return;
  const type = box.dataset.type;
  state.keepTypes = box.checked
    ? state.keepTypes.filter((t) => t !== type)
    : [...state.keepTypes, type];
  run();
});

(async function start() {
  await Promise.all([loadHealth(), loadSamples()]);
  await run();
})();
