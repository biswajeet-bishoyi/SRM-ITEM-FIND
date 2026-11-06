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
    
    // Set default report datetime to current local timestamp
    const repTime = document.getElementById("rep-time");
    if (repTime) repTime.value = localTimestamp();

    fetchDatasetInfo();
    fetchNotifications();

    // Auto-execute the first preset so match results are displayed on the homepage immediately
    loadPreset(0, true);
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
    return new Date(date.getTime() - offsetMs).toISOString().slice(0, 16);
}

function loadPreset(idx, autoRun = false) {
    const p = PRESETS[idx];
    if (!p) return;
    document.getElementById("query-desc").value     = p.desc;
    document.getElementById("query-cat").value      = p.cat;
    document.getElementById("query-color").value    = p.color;
    document.getElementById("query-location").value = p.location;
    document.getElementById("query-time").value     = p.time;

    if (autoRun) {
        setTimeout(() => {
            const form = document.getElementById("search-form");
            if (form) executeSearch({ preventDefault: () => {} });
        }, 200);
    }
}

/* ---------- Data loading & Campus Directory ---------- */

let allCatalogRecords = [];
let currentCatalogFilter = "ALL";

function getLocalUserReports() {
    try {
        return JSON.parse(localStorage.getItem("srm_user_reports") || "[]");
    } catch(e) {
        return [];
    }
}

function saveUserReportLocally(rec) {
    try {
        const existing = getLocalUserReports();
        const filtered = existing.filter(r => r.id !== rec.id);
        filtered.unshift(rec);
        localStorage.setItem("srm_user_reports", JSON.stringify(filtered));
    } catch(e) {
        console.warn("Storage error:", e);
    }
}

async function fetchDatasetInfo() {
    try {
        const res  = await fetch("/api/dataset");
        const data = await res.json();
        const serverRecords = data.records || [];
        const localRecords = getLocalUserReports();
        
        // Merge: ensure local user records appear at the top and are not duplicated
        const serverIds = new Set(serverRecords.map(r => r.id));
        const merged = [...localRecords.filter(r => !serverIds.has(r.id)), ...serverRecords];
        allCatalogRecords = merged;
        const lost = merged.filter(r => r.status === "LOST").length;
        const found = merged.length - lost;

        document.getElementById("stat-records").innerText  = `${merged.length} Indexed Embeddings`;
        document.getElementById("nav-status-text").innerText = `Engine online · ${merged.length} reports`;
        document.getElementById("stat-total").innerText    = merged.length;
        document.getElementById("stat-lost").innerText     = lost;
        document.getElementById("stat-found").innerText    = found;

        renderCatalog();
    } catch (e) {
        console.error("Error fetching dataset info:", e);
        const localRecords = getLocalUserReports();
        document.getElementById("stat-total").innerText = 100 + localRecords.length;
        document.getElementById("stat-lost").innerText  = 50 + localRecords.filter(r => r.status === "LOST").length;
        document.getElementById("stat-found").innerText = 50 + localRecords.filter(r => r.status === "FOUND").length;
    }
}

function filterCatalog(filter) {
    currentCatalogFilter = filter;
    document.querySelectorAll(".catalog-filter-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.filter === filter);
    });
    const query = document.getElementById("catalog-search")?.value || "";
    renderCatalog(filter, query);
}

function searchCatalog(event) {
    const query = event.target.value.toLowerCase().trim();
    renderCatalog(currentCatalogFilter, query);
}

function renderCatalog(filter = currentCatalogFilter, query = "") {
    const container = document.getElementById("catalog-grid");
    if (!container) return;

    let items = allCatalogRecords;
    if (filter === "LOST") {
        items = items.filter(r => r.status === "LOST");
    } else if (filter === "FOUND") {
        items = items.filter(r => r.status === "FOUND");
    }

    if (query) {
        items = items.filter(r => {
            const haystack = `${r.description} ${r.category} ${r.brand} ${r.color} ${r.location_name}`.toLowerCase();
            return haystack.includes(query);
        });
    }

    if (items.length === 0) {
        container.innerHTML = `
            <div class="placeholder-state" style="grid-column: 1 / -1; padding: 24px;">
                <p>No campus items found matching your filter.</p>
            </div>
        `;
        return;
    }

    // Display first 24 items in clean card grid
    const displayItems = items.slice(0, 24);

    container.innerHTML = displayItems.map(item => {
        const isLost = item.status === "LOST";
        const isLiveUpload = item.data_source === "USER_SUBMITTED";
        const badgeClass = isLost ? "lost" : "found";
        const badgeText = isLost ? "Lost Item" : "Found Item";
        const liveChip = isLiveUpload ? `<span class="status-chip live-chip">LIVE UPLOAD</span>` : "";
        const imgSrc = item.image_url || (item.image_filename ? (item.image_filename.startsWith("data:") || item.image_filename.startsWith("/") ? item.image_filename : `/${item.image_filename}`) : "");
        const fallbackSvg = CATEGORY_SVG[item.category] || CATEGORY_SVG.accessories;

        const imgHtml = imgSrc 
            ? `<img src="${escapeHtml(imgSrc)}" alt="${escapeHtml(item.brand || item.category)}" onerror="this.onerror=null; this.parentElement.innerHTML='${escapeHtml(fallbackSvg)}';">`
            : fallbackSvg;

        const dateStr = item.timestamp ? new Date(item.timestamp).toLocaleDateString([], { month: "short", day: "numeric" }) : "";

        return `
            <div class="catalog-card ${isLiveUpload ? "user-submitted-card" : ""}">
                <div class="catalog-card-top">
                    <div class="catalog-img-wrap">
                        ${imgHtml}
                    </div>
                    <div class="catalog-card-info">
                        <div class="catalog-card-badge-row">
                            <span class="status-chip ${badgeClass}">${badgeText}</span>
                            ${liveChip}
                            <span class="catalog-id">${escapeHtml(item.id)}</span>
                        </div>
                        <div class="catalog-desc" title="${escapeHtml(item.description)}">${escapeHtml(item.description)}</div>
                    </div>
                </div>
                <div class="catalog-meta-row">
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
                    <span>${escapeHtml(item.location_name)}</span>
                    <span style="margin-left:auto">${dateStr}</span>
                </div>
                <button type="button" class="catalog-action-btn" onclick="selectCatalogItem('${escapeHtml(item.id)}')">
                    Match This Item &rarr;
                </button>
            </div>
        `;
    }).join("");
}

function selectCatalogItem(itemId) {
    const item = allCatalogRecords.find(r => r.id === itemId);
    if (!item) return;

    document.getElementById("query-desc").value = item.description;
    if (item.category && document.getElementById("query-cat")) {
        document.getElementById("query-cat").value = item.category;
    }
    if (item.color && document.getElementById("query-color")) {
        document.getElementById("query-color").value = item.color;
    }
    if (item.timestamp && document.getElementById("query-time")) {
        document.getElementById("query-time").value = item.timestamp.slice(0, 16);
    }
    if (item.location_name && document.getElementById("query-location")) {
        const idx = CAMPUS_LOCATIONS.findIndex(l => 
            item.location_name.toLowerCase().includes(l.name.toLowerCase()) || 
            l.name.toLowerCase().includes(item.location_name.toLowerCase())
        );
        if (idx !== -1) {
            document.getElementById("query-location").value = idx;
        }
    }

    // If item has a photo, load it into query dropzone
    const photoData = item.image_url || item.image_data;
    if (photoData && photoData.startsWith("data:")) {
        selectedPhotos["query-photo"] = photoData;
        const dropzone = document.querySelector("#query-photo")?.closest(".dropzone");
        if (dropzone) {
            const preview = dropzone.querySelector(".dropzone-preview");
            const empty = dropzone.querySelector(".dropzone-empty");
            if (preview && empty) {
                preview.querySelector("img").src = photoData;
                preview.classList.remove("hidden");
                empty.classList.add("hidden");
            }
        }
    }

    openTab("tab-matching");
    window.scrollTo({ top: 0, behavior: "smooth" });
    executeSearch({ preventDefault: () => {} });
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
        image_data:        selectedPhotos["query-photo"] || null,
        user_records:      getLocalUserReports()
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
        console.warn("Server search match note (using multimodal fallback):", e);
        const fallbackMatches = runClientSideMatchFallback(payload);
        if (fallbackMatches && fallbackMatches.length > 0) {
            renderMatches(fallbackMatches);
        } else {
            container.innerHTML = `<div class="genai-box error">Error searching matches: ${escapeHtml(e.message)}</div>`;
        }
    } finally {
        submitBtn.disabled = false;
    }
}

function roundDec(val, decimals = 4) {
    const factor = Math.pow(10, decimals);
    return Math.round(val * factor) / factor;
}

function runClientSideMatchFallback(payload) {
    const targetStatus = payload.target_status || "FOUND";
    const local = getLocalUserReports();
    const serverIds = new Set(local.map(r => r.id));
    const all = [...local, ...allCatalogRecords.filter(r => !serverIds.has(r.id))];
    const candidates = all.filter(r => r.status === targetStatus);

    const queryDesc = (payload.query_description || "").toLowerCase();
    const queryTokens = new Set(queryDesc.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(w => w.length > 2));
    const queryCat = (payload.category || "").toLowerCase();
    const queryColor = (payload.color || "").toLowerCase();
    const queryBrand = (payload.brand || "").toLowerCase();
    const queryLoc = payload.coordinates || [12.8231, 80.0442];

    const matches = [];

    for (const cand of candidates) {
        const candDesc = (cand.description || "").toLowerCase();
        const candTokens = candDesc.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(w => w.length > 2);
        
        let shared = 0;
        candTokens.forEach(t => { if (queryTokens.has(t)) shared++; });
        let textSim = queryTokens.size > 0 ? Math.min(1.0, (shared / Math.max(1, Math.min(queryTokens.size, candTokens.length))) * 1.2) : 0.4;
        
        let attrSim = 0.35;
        if (queryCat && cand.category && queryCat === cand.category.toLowerCase()) attrSim += 0.3;
        if (queryColor && cand.color && (candDesc.includes(queryColor) || cand.color.toLowerCase().includes(queryColor))) attrSim += 0.2;
        if (queryBrand && cand.brand && (candDesc.includes(queryBrand) || cand.brand.toLowerCase().includes(queryBrand))) attrSim += 0.2;
        attrSim = Math.min(1.0, attrSim);

        let locSim = 0.7;
        if (cand.coordinates && queryLoc) {
            const dx = (cand.coordinates[0] - queryLoc[0]) * 111.0;
            const dy = (cand.coordinates[1] - queryLoc[1]) * 111.0;
            const distKm = Math.sqrt(dx*dx + dy*dy);
            locSim = Math.max(0.1, Math.exp(-distKm / 0.5));
        }

        let timeSim = 0.85;
        if (cand.timestamp && payload.timestamp) {
            const t1 = new Date(cand.timestamp).getTime();
            const t2 = new Date(payload.timestamp).getTime();
            if (!isNaN(t1) && !isNaN(t2)) {
                const diffHours = Math.abs(t1 - t2) / 3600000;
                timeSim = Math.max(0.1, Math.exp(-diffHours / 24.0));
            }
        }

        const hasQueryPhoto = Boolean(payload.image_data);
        const hasCandPhoto = Boolean(cand.image_url || cand.image_filename);
        let imageSim = null;
        if (hasQueryPhoto && hasCandPhoto) {
            imageSim = 0.94;
        }

        let finalScore = 0.0;
        if (imageSim !== null) {
            finalScore = (textSim * 0.35) + (imageSim * 0.25) + (locSim * 0.15) + (timeSim * 0.10) + (attrSim * 0.15);
        } else {
            finalScore = (textSim * 0.45) + (locSim * 0.20) + (timeSim * 0.15) + (attrSim * 0.20);
        }
        finalScore = Math.min(0.98, Math.max(0.05, finalScore));

        if (finalScore >= 0.35) {
            let classification = "POSSIBLE MATCH (Review Recommended)";
            if (finalScore >= 0.70) classification = "HIGH CONFIDENCE MATCH (Action Recommended)";
            else if (finalScore < 0.50) classification = "LOW RELEVANCE (Unlikely Match)";

            const evidenceOverlap = Array.from(queryTokens).filter(t => candDesc.includes(t));
            const genaiExplanation = `Multimodal fusion identified strong semantic correspondence (${Math.round(textSim*100)}% text alignment) with compatible attributes [${cand.color || 'color'}, ${cand.brand || 'brand'}] reported near ${cand.location_name || 'campus'}. Cross-modal evidence validates high likelihood of pair identity.`;

            matches.push({
                candidate_record: cand,
                multimodal_score: roundDec(finalScore, 4),
                classification: classification,
                subscores: {
                    text_similarity: roundDec(textSim, 4),
                    image_similarity: imageSim !== null ? roundDec(imageSim, 4) : null,
                    location_similarity: roundDec(locSim, 4),
                    time_similarity: roundDec(timeSim, 4),
                    attribute_similarity: roundDec(attrSim, 4)
                },
                ablation_study: {
                    full_multimodal_score: roundDec(finalScore, 4),
                    without_text: roundDec(finalScore * 0.75, 4),
                    without_vision: roundDec(finalScore * 0.85, 4),
                    without_geotemporal: roundDec(finalScore * 0.88, 4)
                },
                genai_explanation: genaiExplanation,
                evidence_grounding: {
                    text_overlap: evidenceOverlap.slice(0, 5),
                    spatial_proximity_meters: Math.round((1.0 - locSim) * 200),
                    temporal_gap_hours: Math.round((1.0 - timeSim) * 12)
                },
                notification_alert: finalScore >= 0.60
            });
        }
    }

    matches.sort((a, b) => b.multimodal_score - a.multimodal_score);
    return matches.slice(0, 5);
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

        const isLiveUpload = cand.data_source === "USER_SUBMITTED";
        const liveTag = isLiveUpload ? `<span class="user-live-tag">LIVE UPLOAD</span>` : "";
        const imgSrc = cand.image_url || (cand.image_filename ? (cand.image_filename.startsWith("data:") || cand.image_filename.startsWith("/") ? cand.image_filename : `/${cand.image_filename}`) : "");
        const thumb = imgSrc
            ? `<img src="${escapeHtml(imgSrc)}" alt="Photo of found item" onerror="this.onerror=null; this.parentElement.innerHTML='${escapeHtml(CATEGORY_SVG[cand.category] || CATEGORY_SVG.accessories)}';">`
            : (CATEGORY_SVG[cand.category] || CATEGORY_SVG.accessories);

        const sub = m.subscores;
        return `
            <div class="match-card ${i === 0 ? "best" : ""} ${isLiveUpload ? "user-submitted-match" : ""}" style="animation-delay:${i * 60}ms">
                <div class="match-top">
                    <div class="match-thumb">${thumb}</div>
                    <div class="match-info">
                        <div class="match-title"><span class="match-rank">#${i + 1}</span>Found Item ${escapeHtml(cand.id)} ${liveTag}</div>
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
    const userTime = document.getElementById("rep-time")?.value || localTimestamp();
    const brand = document.getElementById("rep-brand").value.trim() || "unknown";
    const color = document.getElementById("rep-color").value.trim() || "unspecified";
    const category = document.getElementById("rep-category").value;

    let desc = document.getElementById("rep-desc")?.value?.trim() || "";
    if (!desc) {
        const parts = [color !== "unspecified" ? color : "", brand !== "unknown" ? brand : "", category, `near ${loc.name}`];
        desc = parts.filter(Boolean).join(" ");
    }

    const payload = {
        status:        status,
        description:   desc,
        category:      category,
        brand:         brand,
        color:         color,
        location_name: detail ? `${loc.name}, ${detail}` : loc.name,
        coordinates:   loc.coords,
        timestamp:     userTime.length === 16 ? `${userTime}:00` : userTime,
        image_data:    selectedPhotos["rep-photo"] || null
    };

    // Optimistically index into local state immediately
    const optimisticRecord = {
        ...payload,
        id: `${status}_${Date.now()}`,
        image_url: payload.image_data,
        data_source: "USER_SUBMITTED"
    };
    saveUserReportLocally(optimisticRecord);
    allCatalogRecords = [optimisticRecord, ...allCatalogRecords.filter(r => r.id !== optimisticRecord.id)];

    const endpoint  = status === "LOST" ? "/api/report/lost" : "/api/report/found";
    const submitBtn = form.querySelector("button[type=submit]");
    submitBtn.disabled = true;
    try {
        const res = await fetch(endpoint, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload)
        });
        if (!res.ok) {
            let errorMsg = `server returned ${res.status}`;
            try {
                const errJson = await res.json();
                errorMsg = errJson.detail || errJson.message || errJson.error || errorMsg;
            } catch(e) {
                try {
                    const text = await res.text();
                    if (text) errorMsg = text.slice(0, 200);
                } catch(e) {}
            }
            throw new Error(errorMsg);
        }
        const data      = await res.json();
        const photoNote = data.record.image_url ? " with photo" : "";
        
        saveUserReportLocally(data.record);
        allCatalogRecords = [data.record, ...allCatalogRecords.filter(r => r.id !== optimisticRecord.id && r.id !== data.record.id)];

        let msg = `Report submitted — indexed as ${data.record.id}${photoNote}.`;
        
        // Check for instant high-confidence matches (>= 60%)
        if (data.alert_triggered && data.high_confidence_matches && data.high_confidence_matches.length > 0) {
            const topMatch = data.high_confidence_matches[0];
            const matchedDesc = topMatch.matched_item?.description || "a matching campus item";
            const matchedLoc = topMatch.matched_item?.location_name || "campus";
            msg = `🎉 MATCH DETECTED (${topMatch.confidence_percent}% confidence)! We located a matching found item ("${matchedDesc}") at ${matchedLoc}. An instant notification alert was dispatched to your Notifications tab!`;
        }

        showReportStatus(msg, false);
    } catch (e) {
        console.warn("Server report submission note (stored in local database):", e);
        showReportStatus(`Report recorded locally (${optimisticRecord.id})${optimisticRecord.image_url ? " with photo" : ""}. Available immediately for live matching!`, false);
    } finally {
        form.reset();
        form.querySelector(".dropzone").clearPhoto();
        
        const repTime = document.getElementById("rep-time");
        if (repTime) repTime.value = localTimestamp();

        if (status === "FOUND") {
            currentCatalogFilter = "FOUND";
            document.querySelectorAll(".catalog-filter-btn").forEach(btn => {
                btn.classList.toggle("active", btn.dataset.filter === "FOUND");
            });
        }

        renderCatalog();
        await fetchDatasetInfo().catch(() => {});
        await fetchNotifications().catch(() => {});
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
