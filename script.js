// script.js

// -------------------------
// Configuration
// -------------------------
const INCIDENTS_URL = 'incidents.json';

const SAFE_DISPLAY_CONFIG = {
  requireReviewed: true,
  reviewStatusField: 'status',
  reviewedValues: ['discovered'],
  minRelevancyScore: 0.40,
  excludedSources: ['arXiv'],
};

const UI_CONFIG = {
  pageSize: 6,
  mobileBreakpoint: 768,
};

// -------------------------
// State
// -------------------------
let allIncidents = [];
let safeIncidents = [];
let filteredIncidents = [];

let currentSort = 'date_desc';
let currentPage = 1;
let mobileVisibleCount = UI_CONFIG.pageSize;

let sourceOptions = [];
let vulnerableOptions = [];

const incidentGrid = document.getElementById('incident-grid');
const pageInfoEl = document.getElementById('page-info');
const prevPageBtn = document.getElementById('prev-page');
const nextPageBtn = document.getElementById('next-page');
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const fromDateInput = document.getElementById('filter-from');
const toDateInput = document.getElementById('filter-to');

let sortSelect = null;
let sourceFilterSelect = null;
let vulnerableFilterSelect = null;

// -------------------------
// Helpers
// -------------------------
function isMobile() {
  return window.innerWidth <= UI_CONFIG.mobileBreakpoint;
}

function safeLower(str) {
  return (str || '').toString().toLowerCase();
}

// Trim source to only the prefix before the first ":"
function cleanSource(str) {
  if (!str) return 'Unknown source';
  return str.split(':')[0].trim();
}

function parseDate(value) {
  if (!value) return null;
  const d = new Date(value);
  return isNaN(d.getTime()) ? null : d;
}

function getRelevancy(incident) {
  if (typeof incident.relevancy_score === 'number') return incident.relevancy_score;
  if (typeof incident.relevance_score === 'number') return incident.relevance_score;
  return null;
}

function passesSafeDisplay(incident) {
  if (
    SAFE_DISPLAY_CONFIG.excludedSources.length > 0 &&
    SAFE_DISPLAY_CONFIG.excludedSources.includes(incident.source)
  ) {
    return false;
  }

  const score = getRelevancy(incident);
  if (
    SAFE_DISPLAY_CONFIG.minRelevancyScore != null &&
    typeof score === 'number' &&
    score < SAFE_DISPLAY_CONFIG.minRelevancyScore
  ) {
    return false;
  }

  if (SAFE_DISPLAY_CONFIG.requireReviewed) {
    const fieldName = SAFE_DISPLAY_CONFIG.reviewStatusField;
    const statusValue = safeLower(incident[fieldName]);
    if (!SAFE_DISPLAY_CONFIG.reviewedValues.map(safeLower).includes(statusValue)) {
      return false;
    }
  }

  return true;
}

function normalizeIncident(raw, index) {
  const publication_date =
    raw.publication_date || raw.date || raw.published_at || raw.published || null;

  const relevancy_score = getRelevancy(raw);

  return {
    id: raw.id || raw.slug || raw.url || `incident-${index}`,
    title: raw.title || 'Untitled incident',
    url: raw.url || '#',
    source: cleanSource(raw.source || 'Unknown source'),
    summary: raw.summary || '',
    publication_date,
    vulnerable_populations:
      raw.vulnerable_populations || raw.vulnerable_population || '',
    status: raw.review_status || raw.status || '',
    relevancy_score,
  };
}

function showMessage(message, variant = 'info') {
  if (!incidentGrid) return;

  incidentGrid.innerHTML = '';
  const div = document.createElement('div');
  div.className = `incident-message incident-message-${variant}`;
  div.textContent = message;
  incidentGrid.appendChild(div);

  if (pageInfoEl) pageInfoEl.textContent = '';
  if (prevPageBtn) prevPageBtn.disabled = true;
  if (nextPageBtn) nextPageBtn.disabled = true;
}

function updateFooterYear() {
  const yearEl = document.getElementById('year');
  if (yearEl) {
    yearEl.textContent = new Date().getFullYear();
  }
}

// -------------------------
// Sort & Filter Logic
// -------------------------
function sortIncidents(list) {
  const sorted = [...list];

  sorted.sort((a, b) => {
    if (currentSort.startsWith('date')) {
      const da = parseDate(a.publication_date);
      const db = parseDate(b.publication_date);
      return currentSort === 'date_desc'
        ? (db ? db.getTime() : 0) - (da ? da.getTime() : 0)
        : (da ? da.getTime() : 0) - (db ? db.getTime() : 0);
    }

    if (currentSort.startsWith('relevance')) {
      const sa = getRelevancy(a) ?? -Infinity;
      const sb = getRelevancy(b) ?? -Infinity;
      return currentSort === 'relevance_desc' ? sb - sa : sa - sb;
    }

    return 0;
  });

  return sorted;
}

function applyFiltersAndSort() {
  if (!safeIncidents.length) {
    filteredIncidents = [];
    renderIncidents();
    return;
  }

  const query = safeLower(searchInput?.value || '');
  const fromDate = parseDate(fromDateInput?.value);
  const toDate = parseDate(toDateInput?.value);

  const selectedSource = sourceFilterSelect ? sourceFilterSelect.value : '';
  const selectedVuln = vulnerableFilterSelect ? vulnerableFilterSelect.value : '';

  let list = safeIncidents.filter((incident) => {
    if (query) {
      const text = [
        incident.title,
        incident.summary,
        incident.source,
        incident.vulnerable_populations,
      ]
        .map(safeLower)
        .join(' ');

      if (!text.includes(query)) return false;
    }

    const d = parseDate(incident.publication_date);
    if (fromDate && d && d < fromDate) return false;
    if (toDate && d && d > toDate) return false;

    if (selectedSource && incident.source !== selectedSource) return false;

    if (selectedVuln) {
      const vp = (incident.vulnerable_populations || '')
        .split(',')
        .map((s) => s.trim());
      if (!vp.includes(selectedVuln)) return false;
    }

    return true;
  });

  list = sortIncidents(list);
  filteredIncidents = list;

  currentPage = 1;
  mobileVisibleCount = UI_CONFIG.pageSize;

  renderIncidents();
}

// -------------------------
// Rendering
// -------------------------
function createIncidentCard(incident) {
  const card = document.createElement('article');
  card.className = 'incident-card';
  card.tabIndex = 0;

  const title = incident.title || 'Untitled incident';
  const source = incident.source || 'Unknown source';
  const pubDate = parseDate(incident.publication_date);
  const dateText = pubDate ? pubDate.toISOString().slice(0, 10) : 'Date not available';

  const summary = incident.summary || '';
  const vuln = incident.vulnerable_populations || '';

  const hideSummary =
    !summary ||
    summary.trim().length < 5 ||
    summary.toLowerCase().trim() === title.toLowerCase().trim();

  card.innerHTML = `
    <div class="incident-card-main">
      <h3 class="incident-title">${title}</h3>

      <div class="incident-meta">
        <span class="incident-source">${source}</span>
        <span class="incident-date">${dateText}</span>
      </div>

      ${
        hideSummary
          ? ''
          : `<p class="incident-summary">${summary}</p>`
      }

      ${
        vuln
          ? `<p class="incident-vulnerable"><strong>Keywords:</strong> ${vuln}</p>`
          : ''
      }
    </div>

    <div class="incident-actions" style="justify-content:flex-end;">
      <button type="button" class="button-ghost copy-link">Copy link</button>
    </div>
  `;

  // Clicking the card opens the article (except buttons)
  card.addEventListener('click', (e) => {
    if (e.target.closest('.copy-link')) return;
    if (incident.url && incident.url !== '#') {
      window.open(incident.url, '_blank', 'noopener,noreferrer');
    }
  });

  // Copy link
  const copyBtn = card.querySelector('.copy-link');
  if (copyBtn) {
    copyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (!incident.url) return;
      navigator.clipboard
        .writeText(incident.url)
        .then(() => {
          copyBtn.textContent = 'Copied';
          setTimeout(() => (copyBtn.textContent = 'Copy link'), 1500);
        });
    });
  }

  return card;
}

function renderIncidents() {
  if (!incidentGrid) return;

  incidentGrid.innerHTML = '';

  if (!filteredIncidents || filteredIncidents.length === 0) {
    showMessage('No sycophancy related incidents found', 'info');
    return;
  }

  if (isMobile()) {
    const count = Math.min(mobileVisibleCount, filteredIncidents.length);
    const items = filteredIncidents.slice(0, count);

    items.forEach((incident) => incidentGrid.appendChild(createIncidentCard(incident)));

    if (pageInfoEl) {
      pageInfoEl.textContent = `Showing ${count} of ${filteredIncidents.length}`;
    }

    if (prevPageBtn) prevPageBtn.style.display = 'none';
    if (nextPageBtn) {
      nextPageBtn.style.display = 'inline-flex';
      nextPageBtn.textContent =
        count < filteredIncidents.length ? 'Load more' : 'No more results';
      nextPageBtn.disabled = count >= filteredIncidents.length;
    }
  } else {
    const totalPages = Math.max(
      1,
      Math.ceil(filteredIncidents.length / UI_CONFIG.pageSize)
    );

    if (currentPage > totalPages) currentPage = totalPages;

    const startIdx = (currentPage - 1) * UI_CONFIG.pageSize;
    const endIdx = startIdx + UI_CONFIG.pageSize;
    const items = filteredIncidents.slice(startIdx, endIdx);

    items.forEach((incident) => incidentGrid.appendChild(createIncidentCard(incident)));

    if (pageInfoEl) {
      pageInfoEl.textContent = `Page ${currentPage} of ${totalPages}`;
    }

    if (prevPageBtn && nextPageBtn) {
      prevPageBtn.style.display = 'inline-flex';
      nextPageBtn.style.display = 'inline-flex';
      prevPageBtn.disabled = currentPage <= 1;
      nextPageBtn.disabled = currentPage >= totalPages;
      nextPageBtn.textContent = 'Next';
    }
  }
}

// -------------------------
// UI wiring
// -------------------------
function initSortControl() {
  const sectionHeader = document.querySelector('#incidents .section-header');
  if (!sectionHeader) return;

  const wrapper = document.createElement('div');
  wrapper.className = 'sort-wrapper';

  const label = document.createElement('label');
  label.setAttribute('for', 'sort-by');
  label.textContent = 'Sort by';

  const select = document.createElement('select');
  select.id = 'sort-by';
  select.className = 'input sort-select';
  select.innerHTML = `
    <option value="date_desc">Most recent</option>
    <option value="date_asc">Oldest first</option>
    <option value="relevance_desc">Highest relevance</option>
    <option value="relevance_asc">Lowest relevance</option>
  `;

  wrapper.appendChild(label);
  wrapper.appendChild(select);
  sectionHeader.appendChild(wrapper);

  sortSelect = select;
  sortSelect.addEventListener('change', () => {
    currentSort = sortSelect.value;
    applyFiltersAndSort();
  });
}

function initDynamicFilters() {
  const advancedGrid = document.querySelector('#search .advanced-grid');
  if (!advancedGrid) return;

  const sources = new Set();
  const vulns = new Set();

  safeIncidents.forEach((inc) => {
    if (inc.source) sources.add(inc.source);
    if (inc.vulnerable_populations) {
      inc.vulnerable_populations
        .split(',')
        .map((s) => s.trim())
        .forEach((v) => vulns.add(v));
    }
  });

  sourceOptions = Array.from(sources).sort();
  vulnerableOptions = Array.from(vulns).sort();

  if (sourceOptions.length > 0) {
    const field = document.createElement('div');
    field.className = 'field';

    const label = document.createElement('label');
    label.textContent = 'Source';

    const select = document.createElement('select');
    select.className = 'input';
    select.innerHTML =
      '<option value="">Any</option>' +
      sourceOptions.map((s) => `<option value="${s}">${s}</option>`).join('');

    field.appendChild(label);
    field.appendChild(select);
    advancedGrid.appendChild(field);

    sourceFilterSelect = select;
    sourceFilterSelect.addEventListener('change', applyFiltersAndSort);
  }

  if (vulnerableOptions.length > 0) {
    const field = document.createElement('div');
    field.className = 'field';

    const label = document.createElement('label');
    label.textContent = 'Keywords';

    const select = document.createElement('select');
    select.className = 'input';
    select.innerHTML =
      '<option value="">Any</option>' +
      vulnerableOptions.map((v) => `<option value="${v}">${v}</option>`).join('');

    field.appendChild(label);
    field.appendChild(select);
    advancedGrid.appendChild(field);

    vulnerableFilterSelect = select;
    vulnerableFilterSelect.addEventListener('change', applyFiltersAndSort);
  }
}

// -------------------------
// Initialization
// -------------------------
async function loadIncidents() {
  if (!incidentGrid) return;

  showMessage('Loading incidents…', 'info');

  try {
    const res = await fetch(INCIDENTS_URL, { cache: 'no-store' });
    if (!res.ok) {
      showMessage('No sycophancy related incidents found', 'info');
      return;
    }

    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) {
      showMessage('No sycophancy related incidents found', 'info');
      return;
    }

    allIncidents = data.map(normalizeIncident);
    safeIncidents = allIncidents.filter(passesSafeDisplay);

    if (safeIncidents.length === 0) {
      showMessage(
        'No incidents passed the current safety filters.',
        'info'
      );
      return;
    }

    initSortControl();
    initDynamicFilters();

    currentSort = 'date_desc';
    filteredIncidents = sortIncidents(safeIncidents);

    renderIncidents();
  } catch (err) {
    console.error(err);
    showMessage(
      'Unable to load incidents at this time. Please try again later.',
      'error'
    );
  }
}

document.addEventListener('DOMContentLoaded', () => {
  updateFooterYear();
  initEvents();
  loadIncidents();
});


