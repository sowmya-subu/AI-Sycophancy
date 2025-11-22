// script.js

// -------------------------
// Configuration
// -------------------------
const INCIDENTS_URL = 'incidents.json';

const SAFE_DISPLAY_CONFIG = {
  // Only show incidents that meet all these by default
  requireReviewed: true,
  reviewStatusField: 'status', // or 'status' if you change schema
  reviewedValues: ['discovered'], //change to ['approved', 'reviewed'] 
  minRelevancyScore: 0.40, // default threshold 40.0, but relevancy score is between 0 and 1
  excludedSources: ['arXiv','Hacker News'], // configurable, exact match on source field
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

let currentSort = 'date_desc'; // 'date_desc' | 'date_asc' | 'relevance_desc' | 'relevance_asc'
let currentPage = 1;
let mobileVisibleCount = UI_CONFIG.pageSize;

let sourceOptions = [];
let vulnerableOptions = [];

// Cached DOM refs
const incidentGrid = document.getElementById('incident-grid');
const pageInfoEl = document.getElementById('page-info');
const prevPageBtn = document.getElementById('prev-page');
const nextPageBtn = document.getElementById('next-page');
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const fromDateInput = document.getElementById('filter-from');
const toDateInput = document.getElementById('filter-to');

// Will be created dynamically
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

function parseDate(value) {
  if (!value) return null;
  const d = new Date(value);
  return isNaN(d.getTime()) ? null : d;
}

function getRelevancy(incident) {
  // Support both relevancy_score and relevance_score just in case
  if (typeof incident.relevancy_score === 'number') return incident.relevancy_score;
  if (typeof incident.relevance_score === 'number') return incident.relevance_score;
  return null;
}

// Clean "Google News: AI sycophancy 2023" -> "Google News"
function cleanSource(str) {
  if (!str) return 'Unknown source';
  return str.split(':')[0].trim();
}

// Extract "Psychology Today" from
// "When ... delusional thinking can result. - Psychology Today"
function splitTitleAndSource(title, fallbackSource) {
  let cleanTitle = title || 'Untitled incident';
  let source = fallbackSource || 'Unknown source';

  if (!title) {
    return { title: cleanTitle, source };
  }

  const separators = [' - ', ' – '];
  for (const sep of separators) {
    const idx = title.lastIndexOf(sep);
    if (idx > -1 && idx < title.length - sep.length) {
      const base = title.slice(0, idx).trim();
      const candidate = title.slice(idx + sep.length).trim();
      if (candidate) {
        cleanTitle = base || cleanTitle;
        source = candidate;
        break;
      }
    }
  }

  return { title: cleanTitle, source };
}

// Decide if summary is basically the same as title
function areTextsSimilar(title, summary) {
  const t = (title || '').toLowerCase().trim();
  const s = (summary || '').toLowerCase().trim();
  if (!t || !s) return false;
  if (t === s) return true;
  if (s.startsWith(t) || t.startsWith(s)) return true;
  if (Math.abs(t.length - s.length) < 25 && (t.includes(s) || s.includes(t))) {
    return true;
  }
  return false;
}

function passesSafeDisplay(incident) {
  // Exclude sources
  if (
    SAFE_DISPLAY_CONFIG.excludedSources.length > 0 &&
    SAFE_DISPLAY_CONFIG.excludedSources.includes(incident.source)
  ) {
    return false;
  }

  // Relevancy threshold
  const score = getRelevancy(incident);
  if (
    SAFE_DISPLAY_CONFIG.minRelevancyScore != null &&
    typeof score === 'number' &&
    score < SAFE_DISPLAY_CONFIG.minRelevancyScore
  ) {
    return false;
  }

  // Review status
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

  const baseTitle = raw.title || 'Untitled incident';
  const originalSourceRaw = raw.source || 'Unknown source';
  const cleanedOriginalSource = cleanSource(originalSourceRaw);

  let finalTitle = baseTitle;
  let finalSourceRaw = originalSourceRaw;

  if (cleanedOriginalSource.toLowerCase() === 'google news') {
    const split = splitTitleAndSource(baseTitle, originalSourceRaw);
    finalTitle = split.title;
    finalSourceRaw = split.source || originalSourceRaw;
  }

  return {
    id: raw.id || raw.slug || raw.url || `incident-${index}`,
    title: finalTitle,
    url: raw.url || '#',
    source: cleanSource(finalSourceRaw),
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
    if (currentSort === 'date_desc' || currentSort === 'date_asc') {
      const da = parseDate(a.publication_date);
      const db = parseDate(b.publication_date);

      const ta = da ? da.getTime() : 0;
      const tb = db ? db.getTime() : 0;

      return currentSort === 'date_desc' ? tb - ta : ta - tb;
    }

    if (currentSort === 'relevance_desc' || currentSort === 'relevance_asc') {
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
  const selectedVuln = vulnerableFilterSelect
    ? vulnerableFilterSelect.value
    : '';

  let list = safeIncidents.filter((incident) => {
    // Search text
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

    // Date range
    const d = parseDate(incident.publication_date);
    if (fromDate && d && d < fromDate) return false;
    if (toDate && d && d > toDate) return false;

    // Source filter
    if (selectedSource) {
      if (incident.source !== selectedSource) return false;
    }

    // Vulnerable population filter
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

  // Reset paging
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
  const dateText = pubDate
    ? pubDate.toISOString().slice(0, 10)
    : 'Date not available';
  const summary = incident.summary || '';
  const vuln = incident.vulnerable_populations || '';

  const hideSummary =
    !summary ||
    summary.trim().length < 5 ||
    areTextsSimilar(title, summary);

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
          : `<p class="incident-summary">
        ${summary}
      </p>`
      }
      ${
        vuln
          ? `<p class="incident-vulnerable"><strong>Vulnerable populations:</strong> ${vuln}</p>`
          : ''
      }
    </div>
    <div class="incident-actions">
      <a href="${incident.url}" target="_blank" rel="noopener noreferrer" class="button-link">
        View article
      </a>
      <button type="button" class="button-ghost copy-link">Copy link</button>
      <button type="button" class="button-ghost share-link">Share</button>
    </div>
  `;

  // Entire card click takes you to article
  card.addEventListener('click', (e) => {
    // Avoid double-activating when clicking buttons
    const target = e.target;
    if (
      target.closest('.copy-link') ||
      target.closest('.share-link') ||
      target.closest('.button-link')
    ) {
      return;
    }
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
        })
        .catch(() => {
          alert('Unable to copy link. Please copy manually.');
        });
    });
  }

  // Share
  const shareBtn = card.querySelector('.share-link');
  if (shareBtn) {
    shareBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!incident.url) return;

      const shareData = {
        title: incident.title || 'AI sycophancy incident',
        text: incident.summary || '',
        url: incident.url,
      };

      if (navigator.share) {
        try {
          await navigator.share(shareData);
        } catch (err) {
          // user cancelled, ignore
        }
      } else {
        // Fallback to copy
        navigator.clipboard
          .writeText(incident.url)
          .then(() => alert('Link copied to clipboard'))
          .catch(() => alert('Unable to share. Please copy link manually.'));
      }
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
    // Lazy-load vertical list
    const count = Math.min(mobileVisibleCount, filteredIncidents.length);
    const items = filteredIncidents.slice(0, count);

    items.forEach((incident) => {
      const card = createIncidentCard(incident);
      incidentGrid.appendChild(card);
    });

    if (pageInfoEl) {
      pageInfoEl.textContent = `Showing ${count} of ${filteredIncidents.length}`;
    }

    if (prevPageBtn) {
      prevPageBtn.style.display = 'none';
    }
    if (nextPageBtn) {
      nextPageBtn.style.display = 'inline-flex';
      nextPageBtn.textContent =
        count < filteredIncidents.length ? 'Load more' : 'No more results';
      nextPageBtn.disabled = count >= filteredIncidents.length;
    }
  } else {
    // Desktop pagination
    const totalPages = Math.max(
      1,
      Math.ceil(filteredIncidents.length / UI_CONFIG.pageSize)
    );
    if (currentPage > totalPages) currentPage = totalPages;

    const startIdx = (currentPage - 1) * UI_CONFIG.pageSize;
    const endIdx = startIdx + UI_CONFIG.pageSize;
    const items = filteredIncidents.slice(startIdx, endIdx);

    items.forEach((incident) => {
      const card = createIncidentCard(incident);
      incidentGrid.appendChild(card);
    });

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
// UI wiring (sort + filters)
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

  // Build options from safeIncidents
  const sources = new Set();
  const vulns = new Set();

  safeIncidents.forEach((inc) => {
    if (inc.source) sources.add(inc.source);
    if (inc.vulnerable_populations) {
      inc.vulnerable_populations
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .forEach((v) => vulns.add(v));
    }
  });

  sourceOptions = Array.from(sources).sort();
  vulnerableOptions = Array.from(vulns).sort();

  // Source filter
  if (sourceOptions.length > 0) {
    const field = document.createElement('div');
    field.className = 'field';

    const label = document.createElement('label');
    label.setAttribute('for', 'filter-source');
    label.textContent = 'Source';

    const select = document.createElement('select');
    select.id = 'filter-source';
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

  // Vulnerable population filter
  if (vulnerableOptions.length > 0) {
    const field = document.createElement('div');
    field.className = 'field';

    const label = document.createElement('label');
    label.setAttribute('for', 'filter-vulnerable');
    label.textContent = 'Vulnerable population';

    const select = document.createElement('select');
    select.id = 'filter-vulnerable';
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
// Event wiring
// -------------------------
function initEvents() {
  if (searchButton && searchInput) {
    searchButton.addEventListener('click', (e) => {
      e.preventDefault();
      applyFiltersAndSort();
    });

    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        applyFiltersAndSort();
      }
    });
  }

  if (fromDateInput) {
    fromDateInput.addEventListener('change', applyFiltersAndSort);
  }

  if (toDateInput) {
    toDateInput.addEventListener('change', applyFiltersAndSort);
  }

  if (prevPageBtn) {
    prevPageBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (isMobile()) return; // no prev on mobile
      if (currentPage > 1) {
        currentPage -= 1;
        renderIncidents();
      }
    });
  }

  if (nextPageBtn) {
    nextPageBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (isMobile()) {
        // Lazy-load more
        if (mobileVisibleCount < filteredIncidents.length) {
          mobileVisibleCount += UI_CONFIG.pageSize;
          renderIncidents();
        }
      } else {
        // Desktop pagination
        const totalPages = Math.max(
          1,
          Math.ceil(filteredIncidents.length / UI_CONFIG.pageSize)
        );
        if (currentPage < totalPages) {
          currentPage += 1;
          renderIncidents();
        }
      }
    });
  }

  window.addEventListener('resize', () => {
    // Reset paging mode when crossing breakpoint
    currentPage = 1;
    mobileVisibleCount = UI_CONFIG.pageSize;
    renderIncidents();
  });

  // Newsletter dummy handler to prevent page reload
  const newsletterForm = document.getElementById('newsletter-form');
  const newsletterMsg = document.getElementById('newsletter-message');
  if (newsletterForm && newsletterMsg) {
    newsletterForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const emailInput = document.getElementById('newsletter-email');
      const email = emailInput?.value || '';
      if (!email) return;
      newsletterMsg.textContent =
        'Thanks for your interest. Newsletter functionality is not yet implemented.';
      newsletterForm.reset();
    });
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
      if (res.status === 404) {
        showMessage('No sycophancy related incidents found', 'info');
        return;
      }
      throw new Error(`HTTP ${res.status}`);
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
        'No incidents passed the current safety filters. Adjust thresholds or review status.',
        'info'
      );
      return;
    }

    // Initialize sort & filters now that we have data
    initSortControl();
    initDynamicFilters();

    // Default sort by date descending (most recent)
    currentSort = 'date_desc';
    filteredIncidents = sortIncidents(safeIncidents);

    renderIncidents();
  } catch (err) {
    console.error('Error loading incidents:', err);
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
