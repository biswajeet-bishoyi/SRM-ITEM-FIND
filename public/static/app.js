// SRM CampusFind Frontend App JS — v5

const CAMPUS_LOCATIONS = [
    { name: "Central Library",            coords: [12.8231, 80.0442] },
    { name: "Tech Park — Java Canteen",   coords: [12.8245, 80.0455] },
    { name: "Tech Park — Labs",           coords: [12.8246, 80.0456] },
    { name: "University Building (UB)",   coords: [12.8220, 80.0430] },
    { name: "Sports Complex",             coords: [12.8260, 80.0470] },
    { name: "MTP Canteen",                coords: [12.8210, 80.0415] },
    { name: "Architecture Block",         coords: [12.8251, 80.0421] },
    { name: "Abode Valley",               coords: [12.8205, 80.0390] }
];

const PRESETS = [
    {
        desc: "I misplaced my black wireless Airdopes earbuds near the central library floor 2.",
        cat: "electronics", color: "black", location: 0, time: "2026-10-04T10:30"
    },
    {
        desc: "Lost my dark blue Nike backpack with laptop inside near Tech Park Java Canteen.",
        cat: "bags", color: "navy blue", location: 1, time: "2026-10-04T13:00"
    },
    {
        desc: "Dropped my student ID card with blue lanyard around UB building food court.",
        cat: "documents", color: "blue", location: 3, time: "2026-10-03T16:45"
    },
    {
        desc: "Misplaced an Apple Watch Series 7 with black silicone strap near sports complex.",
        cat: "electronics", color: "black", location: 4, time: "2026-10-05T08:00"
    }
];

// SVG icons for categories (used as thumbnail fallback)
const CATEGORY_SVG = {
    electronics: `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>`,
    bags:        `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/><line x1="3" y1="6" x2="21" y2="6"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>`,
    documents:   `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>`,
    accessories: `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>`
};

const TAB_HEADERS = {
    "tab-matching":      ["Intelligent Multimodal Matching Engine", "Semantic NLP + Vision Encoder + Geo-Temporal Fusion + GenAI Explanations"],
    "tab-report":        ["Report a Lost or Found Item", "New reports are embedded and indexed in the vector database instantly"],
    "tab-notifications": ["Match Notifications", "Alerts raised when a match reaches the 0.50 confidence threshold"],
    "tab-settings":      ["Settings", "Choose an interface color theme"]
};

const AVAILABLE_THEMES = new Set(["spider", "formal", "white", "cyberpunk", "aot", "tsushima"]);

// Resized photo data URLs, keyed by file input id
const selectedPhotos = {};

// In-memory notification store for dismiss support
let localNotifications = [];

document.addEventListener("DOMContentLoaded", () => {
    setupBurgerMenu();
    setupTabNavigation();
    setupThemePicker();
    setupLocationSelects();
    setupDropzones();
    fetchDatasetInfo();
    fetchNotifications();
});

function setupThemePicker() {
    let theme = "formal";
    try {
        const savedTheme = localStorage.getItem("campusfind-theme");
        if (AVAILABLE_THEMES.has(savedTheme)) theme = savedTheme;
    } catch (error) {
        console.warn("Theme preference could not be loaded:", error);
    }

    setTheme(theme);
    document.querySelectorAll(".theme-option").forEach(button => {
        button.addEventListener("click", () => setTheme(button.dataset.theme));
    });
}

function setTheme(theme) {
    if (!AVAILABLE_THEMES.has(theme)) return;
    document.documentElement.dataset.theme = theme;
    try {
        localStorage.setItem("campusfind-theme", theme);
    } catch (error) {
        console.warn("Theme preference could not be saved:", error);
    }
    document.querySelectorAll(".theme-option").forEach(button => {
        button.setAttribute("aria-pressed", String(button.dataset.theme === theme));
    });
}

// Escape user-provided text before inserting it into HTML
function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[ch]);
}

/* ---------- Navigation ---------- */

function setupBurgerMenu() {
    document.getElementById("burger-btn").addEventListener("click", () => {
        toggleMenu(!document.body.classList.contains("menu-open"));
    });
    document.getElementById("sidebar-overlay").addEventListener("click", () => toggleMenu(false));
    document.addEventListener("keydown", e => {
        if (e.key === "Escape") toggleMenu(false);
    });
}

function toggleMenu(open) {
    const burger = document.getElementById("burger-btn");
    document.body.classList.toggle("menu-open", open);
    document.getElementById("sidebar").setAttribute("aria-hidden", String(!open));
    burger.setAttribute("aria-expanded", String(open));
    burger.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
}

function setupTabNavigation() {
    document.querySelectorAll(".nav-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            openTab(btn.getAttribute("data-tab"));
            toggleMenu(false);
        });
    });
}

function openTab(targetId) {
    document.querySelectorAll(".nav-btn").forEach(b => {
        b.classList.toggle("active", b.getAttribute("data-tab") === targetId);
    });
    document.querySelectorAll(".tab-content").forEach(t => {
        t.classList.toggle("active", t.id === targetId);
    });

    const headers = TAB_HEADERS[targetId];
    if (headers) {
        document.getElementById("page-title").innerText = headers[0];
        document.getElementById("page-subtitle").innerText = headers[1];
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ---------- Form helpers ---------- */

function setupLocationSelects() {
    const options = CAMPUS_LOCATIONS.map((loc, i) => `<option value="${i}">${escapeHtml(loc.name)}</option>`).join("");
    document.querySelectorAll(".location-select").forEach(sel => {
        sel.innerHTML = options;
    });
}

function setupDropzones() {
    document.querySelectorAll(".dropzone").forEach(zone => {
        const input   = zone.querySelector(".dropzone-input");
        const empty   = zone.querySelector(".dropzone-empty");
        const preview = zone.querySelector(".dropzone-preview");
        const img     = preview.querySelector("img");

        const clear = () => {
            delete selectedPhotos[input.id];
            input.value = "";
            img.removeAttribute("src");
            preview.classList.add("hidden");
            empty.classList.remove("hidden");
            input.classList.remove("hidden");
        };
        zone.clearPhoto = clear;

        input.addEventListener("change", async () => {
            const file = input.files[0];
            if (!file) return;
            try {
                const dataUrl = await resizeImage(file, 640);
                selectedPhotos[input.id] = dataUrl;
                img.src = dataUrl;
                preview.classList.remove("hidden");
                empty.classList.add("hidden");
                input.classList.add("hidden");
            } catch (e) {
                clear();
            }
        });

        zone.querySelector(".dropzone-remove").addEventListener("click", clear);
        zone.addEventListener("dragenter", () => zone.classList.add("dragover"));
        zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
        zone.addEventListener("drop", () => zone.classList.remove("dragover"));
    });
}

// Downscale a photo in the browser so uploads stay small
function resizeImage(file, maxSize) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = reject;
        reader.onload = () => {
            const image = new Image();
            image.onerror = reject;
            image.onload = () => {
                const scale = Math.min(1, maxSize / Math.max(image.width, image.height));
                const canvas = document.createElement("canvas");
                canvas.width  = Math.round(image.width  * scale);
                canvas.height = Math.round(image.height * scale);
                canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
                resolve(canvas.toDataURL("image/jpeg", 0.85));
            };
            image.src = reader.result;
        };
        reader.readAsDataURL(file);
    });
}

function localTimestamp(date = new Date()) {
    const offsetMs = date.getTimezoneOffset() * 60000;
    return new Date(date.getTime() - offsetMs).toISOString().slice(0, 19);
}

function loadPreset(idx) {
    const p = PRESETS[idx];
    document.getElementById("query-desc").value     = p.desc;
    document.getElementById("query-cat").value      = p.cat;
    document.getElementById("query-color").value    = p.color;
    document.getElementById("query-location").value = p.location;
    document.getElementById("query-time").value     = p.time;
}

/* ---------- Data loading ---------- */

async function fetchDatasetInfo() {
    try {
        const res  = await fetch("/api/dataset");
        const data = await res.json();
        const records = data.records || [];
        const lost = records.filter(r => r.status === "LOST").length;

        document.getElementById("stat-records").innerText  = `${data.total_records} Indexed Embeddings`;
        document.getElementById("nav-status-text").innerText = `Engine online · ${data.total_records} reports`;
        document.getElementById("stat-total").innerText    = data.total_records;
        document.getElementById("stat-lost").innerText     = lost;
        document.getElementById("stat-found").innerText    = records.length - lost;
    } catch (e) {
        console.error("Error fetching dataset info:", e);
    }
}

/* ---------- Matching ---------- */

async function executeSearch(event) {
    event.preventDefault();

    const loc  = CAMPUS_LOCATIONS[document.getElementById("query-location").value];
    const time = document.getElementById("query-time").value;

    const payload = {
        query_description: document.getElementById("query-desc").value,
        target_status:     "FOUND",
        category:          document.getElementById("query-cat").value,
        color:             document.getElementById("query-color").value,
        location_name:     loc.name,
        coordinates:       loc.coords,
        timestamp:         time ? `${time}:00` : localTimestamp(),
        image_data:        selectedPhotos["query-photo"] || null
    };

    const container = document.getElementById("results-container");
    const submitBtn = event.target.querySelector("button[type=submit]");
    submitBtn.disabled = true;
    document.getElementById("results-count").innerText = "";
    container.innerHTML = `<div class="loader"><div class="spinner"></div>Fusing text, vision &amp; geo-temporal embeddings…</div>`;

    try {
        const res = await fetch("/api/search/matches", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`server returned ${res.status}`);

        const data = await res.json();
        renderMatches(data.matches);
        fetchNotifications();
    } catch (e) {
        container.innerHTML = `<div class="genai-box error">Error searching matches: ${escapeHtml(e.message)}</div>`;
    } finally {
        submitBtn.disabled = false;
    }
}

function subscoreBar(label, value) {
    if (value === null || value === undefined) {
        return `
            <div class="bar-item na">
                <div class="bar-label"><span>${label}</span><b>No photo</b></div>
                <div class="progress-track"></div>
            </div>`;
    }
    const pct = Math.round(value * 100);
    return `
        <div class="bar-item">
            <div class="bar-label"><span>${label}</span><b>${pct}%</b></div>
            <div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>
        </div>`;
}

function renderMatches(matches) {
    const container = document.getElementById("results-container");
    if (!matches || matches.length === 0) {
        container.innerHTML = `
            <div class="placeholder-state">
                <span class="placeholder-icon">
                    <svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
                </span>
                <p>No matches found above 40%.</p>
            </div>`;
        return;
    }

    document.getElementById("results-count").innerText = `${matches.length} matches above 40%`;

    const pinIcon = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:3px"><path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>`;
    const aiIcon  = `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:5px"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6"/><path d="M9 12h6"/><path d="M9 15h4"/></svg>`;

    container.innerHTML = matches.map((m, i) => {
        const cand     = m.candidate_record;
        const scorePct = Math.round(m.multimodal_score * 100);
        let tagClass = "tag-high", ringColor = "var(--success)";
        if (m.classification.includes("POSSIBLE")) { tagClass = "tag-possible"; ringColor = "var(--warning)"; }
        if (m.classification.includes("LOW"))      { tagClass = "tag-low";      ringColor = "var(--danger)"; }

        const thumb = cand.image_url
            ? `<img src="${escapeHtml(cand.image_url)}" alt="Photo of found item">`
            : (CATEGORY_SVG[cand.category] || CATEGORY_SVG.accessories);

        const sub = m.subscores;
        return `
            <div class="match-card ${i === 0 ? "best" : ""}" style="animation-delay:${i * 60}ms">
                <div class="match-top">
                    <div class="match-thumb">${thumb}</div>
                    <div class="match-info">
                        <div class="match-title"><span class="match-rank">#${i + 1}</span>Found Item ${escapeHtml(cand.id)}</div>
                        <div class="match-location">${pinIcon} ${escapeHtml(cand.location_name)}</div>
                    </div>
                    <div class="score-ring" style="--pct:${scorePct}; --ring-color:${ringColor}" title="Multimodal confidence">
                        <span>${scorePct}%</span>
                    </div>
                </div>

                <p class="match-desc">"${escapeHtml(cand.description)}"</p>
                <span class="confidence-tag ${tagClass}">${escapeHtml(m.classification)}</span>

                <div class="subscore-bars">
                    ${subscoreBar("Text similarity",  sub.text_similarity)}
                    ${subscoreBar("Vision similarity", sub.image_similarity)}
                    ${subscoreBar("Geo-proximity",    sub.location_similarity)}
                    ${subscoreBar("Time proximity",   sub.time_similarity)}
                    ${subscoreBar("Attribute match",  sub.attribute_similarity)}
                </div>

                <div class="genai-box">
                    <strong>${aiIcon} GenAI Grounded Explanation</strong><br>
                    ${escapeHtml(m.genai_explanation)}
                </div>
            </div>
        `;
    }).join("");
}

/* ---------- Reporting ---------- */

async function submitReport(event) {
    event.preventDefault();
    const form   = event.target;
    const status = form.querySelector("input[name=rep-status]:checked").value;
    const loc    = CAMPUS_LOCATIONS[document.getElementById("rep-location").value];
    const detail = document.getElementById("rep-location-detail").value.trim();

    const payload = {
        status:        status,
        description:   document.getElementById("rep-desc").value,
        category:      document.getElementById("rep-category").value,
        brand:         document.getElementById("rep-brand").value  || "unknown",
        color:         document.getElementById("rep-color").value  || "unspecified",
        location_name: detail ? `${loc.name}, ${detail}` : loc.name,
        coordinates:   loc.coords,
        timestamp:     localTimestamp(),
        image_data:    selectedPhotos["rep-photo"] || null
    };

    const endpoint  = status === "LOST" ? "/api/report/lost" : "/api/report/found";
    const submitBtn = form.querySelector("button[type=submit]");
    submitBtn.disabled = true;
    try {
        const res = await fetch(endpoint, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`server returned ${res.status}`);
        const data      = await res.json();
        const photoNote = data.record.image_url ? " with photo" : "";
        showReportStatus(`Report submitted — indexed as ${data.record.id}${photoNote}`, false);
        form.reset();
        form.querySelector(".dropzone").clearPhoto();
        fetchDatasetInfo();
    } catch (e) {
        showReportStatus("Error submitting report: " + e.message, true);
    } finally {
        submitBtn.disabled = false;
    }
}

function showReportStatus(message, isError) {
    const box = document.getElementById("report-status");
    box.innerText = message;
    box.classList.toggle("error", isError);
    box.classList.remove("hidden");
}

/* ---------- Notifications ---------- */

async function fetchNotifications() {
    try {
        const res  = await fetch("/api/notifications");
        const data = await res.json();
        localNotifications = (data.notifications || []).slice().reverse();
        renderNotifications();
    } catch (e) {
        console.error("Error fetching notifications:", e);
    }
}

function renderNotifications() {
    const list  = document.getElementById("notif-list");
    const badge = document.getElementById("notif-badge");
    const dot   = document.getElementById("burger-dot");
    const clearBtn = document.getElementById("clear-notifs-btn");

    badge.innerText = localNotifications.length;
    document.getElementById("stat-alerts").innerText = localNotifications.length;
    dot.classList.toggle("hidden", localNotifications.length === 0);

    if (localNotifications.length === 0) {
        clearBtn.style.display = "none";
        list.innerHTML = `
            <div class="placeholder-state">
                <span class="placeholder-icon">
                    <svg viewBox="0 0 24 24" width="44" height="44" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" opacity="0.4"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/><line x1="2" y1="2" x2="22" y2="22"/></svg>
                </span>
                <p>No notifications triggered yet. Run a match search to generate alerts.</p>
            </div>`;
        return;
    }

    clearBtn.style.display = "";

    list.innerHTML = localNotifications.map((n, idx) => {
        const time  = n.timestamp ? new Date(n.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
        const date  = n.timestamp ? new Date(n.timestamp).toLocaleDateString([], { month: "short", day: "numeric" }) : "";
        const score = n.confidence_score !== undefined ? Math.round(n.confidence_score * 100) : null;
        const scoreBar = score !== null
            ? `<div class="notif-score-row">
                   <div class="notif-score-track"><div class="notif-score-fill" style="width:${score}%"></div></div>
                   <span class="notif-score-label">${score}% confidence</span>
               </div>`
            : "";
        const lostId  = escapeHtml(n.lost_item_id  || "—");
        const foundId = escapeHtml(n.found_item_id || "—");

        return `
            <div class="notif-card" id="notif-${idx}">
                <div class="notif-card-top">
                    <div class="notif-icon-wrap">
                        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
                    </div>
                    <div class="notif-meta">
                        <strong class="notif-title">${escapeHtml(n.recipient_alert?.title || "Match Alert")}</strong>
                        <span class="notif-time">${date} · ${time}</span>
                    </div>
                    <button class="notif-dismiss" aria-label="Dismiss notification" onclick="dismissNotification(${idx})">
                        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
                </div>

                ${scoreBar}

                <p class="notif-message">${escapeHtml(n.recipient_alert?.message || "")}</p>

                <div class="notif-ids">
                    <span class="notif-id-tag lost-tag">
                        <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:3px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/></svg>
                        Lost: ${lostId}
                    </span>
                    <span class="notif-id-tag found-tag">
                        <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:3px"><polyline points="20 6 9 17 4 12"/></svg>
                        Found: ${foundId}
                    </span>
                    <span class="notif-channel">${escapeHtml(n.delivery_channel || "IN_APP")}</span>
                </div>
            </div>
        `;
    }).join("");
}

function dismissNotification(idx) {
    localNotifications.splice(idx, 1);
    renderNotifications();
}

function clearNotifications() {
    localNotifications = [];
    renderNotifications();
}
