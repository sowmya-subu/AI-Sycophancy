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
  excludedSources: ['arXiv', 'Hacker News'], // configurable, exact match on source field
};

const TRUSTED_SOURCE_PATTERNS = [
  'nature',
  'bmc psychiatry',
  'cnn',
  'pbs',
  'guardian',
  'washington post',
  'forbes',
  'new scientist',
  'cbc',
  'ctv news',
  'business insider',
  'ars technica',
  'engadget',
  'zdnet',
  'india today',
  'times of india',
  'psychology today',
  'fast company',
  'wbur',
  'yahoo',
  'the mirror',
];

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
let currentPage = 1;
let mobileVisibleCount = UI_CONFIG.pageSize;

// Cached DOM references
const incidentGrid = document.getElementById('incident-grid');
const pageInfoEl = document.getElementById('page-info');
const prevPageBtn = document.getElementById('prev-page');
const nextPageBtn = document.getElementById('next-page');
const showMoreBtn = document.getElementById('show-more-mobile');
const sortContainer = document.getElementById('sort-container');
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

function passesSafeDisplay(incident) {
  const sourceName = safeLower(incident.source);

  // Exclude sources
  const excluded = SAFE_DISPLAY_CONFIG.excludedSources.map(safeLower);
  if (excluded.length > 0 && excluded.includes(sourceName)) {
    return false;
  }

  // Trusted sources allowlist
  if (TRUSTED_SOURCE_PATTERNS.length > 0) {
    const isTrusted = TRUSTED_SOURCE_PATTERNS.some((pattern) =>
      sourceName.includes(pattern)
    );
    if (!isTrusted) return false;
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

function deriveSource(raw) {
  const rawSource = (raw.source || '').toString().trim();
  if (!rawSource) return 'Unknown source';

  // For Google News aggregator, the original source is at the end of the title after " - "
  if (rawSource.toLowerCase().includes('google news') && raw.title) {
    const title = raw.title.toString();
    const parts = title.split(' - ');
    if (parts.length > 1) {
      const candidate = parts[parts.length - 1].trim();
      if (candidate) return candidate;
    }
  }

  return rawSource;
}

function normalizeIncident(raw, index) {
  const publication_date =
    raw.publication_date || raw.date || raw.published_at || raw.published || null;

  const relevancy_score = getRelevancy(raw);
  const source = deriveSource(raw);

  return {
    id: raw.id || raw.slug || raw.url || `incident-${index}`,
    title: raw.title || 'Untitled incident',
    url: raw.url || '#',
    source,
    summary: raw.summary || '',
    publication_date,
    vulnerable_populations:
      raw.vulnerable_populations || raw.vulnerable_population || '',
    status: raw.review_status || raw.status || '',
    relevancy_score,  
    category: raw.category || raw.category_label || '',
    severity: raw.severity || raw.severity_label || raw.severity_level || '',
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
  if (!yearEl) return;
  yearEl.textContent = new Date().getFullYear();
}

// -------------------------
// Filtering & sorting
// -------------------------
function sortIncidents(list) {
  if (!sortSelect) return list;

  const value = sortSelect.value;
  const arr = [...list];

  switch (value) {
    case 'date-desc':
      arr.sort((a, b) => {
        const da = parseDate(a.publication_date);
        const db = parseDate(b.publication_date);
        if (!da && !db) return 0;
        if (!da) return 1;
        if (!db) return -1;
        return db - da;
      });
      break;
    case 'date-asc':
      arr.sort((a, b) => {
        const da = parseDate(a.publication_date);
        const db = parseDate(b.publication_date);
        if (!da && !db) return 0;
        if (!da) return 1;
        if (!db) return -1;
        return da - db;
      });
      break;
    case 'relevancy-desc':
      arr.sort((a, b) => {
        const ra = getRelevancy(a) ?? -Infinity;
        const rb = getRelevancy(b) ?? -Infinity;
        return rb - ra;
      });
      break;
    case 'relevancy-asc':
      arr.sort((a, b) => {
        const ra = getRelevancy(a) ?? Infinity;
        const rb = getRelevancy(b) ?? Infinity;
        return ra - rb;
      });
      break;
    default:
      break;
  }

  return arr;
}

function applyFiltersAndSort() {
  if (!safeIncidents || safeIncidents.length === 0) return;

  const query = safeLower(searchInput?.value || '').trim();
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
  const category = incident.category || '';
  const severity = incident.severity || '';
  const pubDate = parseDate(incident.publication_date);
  const dateText = pubDate
    ? pubDate.toLocaleDateString(undefined, {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
      })
    : ''; // no "Date not available"

  const summary = incident.summary || '';
  const vuln = incident.vulnerable_populations || '';

  const trimmedSummary = summary.trim();
  let finalSummary = '';

  const titleLc = safeLower(title).trim();
  const sumLc = safeLower(trimmedSummary).trim();

  if (titleLc && sumLc) {
    const minLen = Math.min(titleLc.length, sumLc.length);
    const maxLen = Math.max(titleLc.length, sumLc.length) || 1;
    const lengthRatio = minLen / maxLen;
    const contains = titleLc.includes(sumLc) || sumLc.includes(titleLc);

    // hide summary if it's the same / very similar as title
    if (sumLc === titleLc || (lengthRatio > 0.75 && contains)) {
      finalSummary = '';
    }
  }

  card.innerHTML = `
    <div class="incident-card-main">
      ${category ? `<span class="incident-tag">${category}</span>` : ''}
      <h3 class="incident-title">${title}</h3>
      <div class="incident-meta">
        ${dateText ? `<span class="incident-date">${dateText}</span>` : ''}
        ${
          severity
            ? `<span class="incident-severity">
                 <span class="incident-severity-dot"></span>${severity}
               </span>`
            : ''
        }
      </div>
      ${
        finalSummary
          ? `<p class="incident-summary">${finalSummary}</p>`
          : ''
      }
      ${
        vuln
          ? `<p class="incident-vulnerable"><strong>Vulnerable populations:</strong> ${vuln}</p>`
          : ''
      }
    </div>
    <div class="incident-actions">
      <a href="${incident.url}" target="_blank" rel="noopener noreferrer" class="button-link">
        View details &rarr;
      </a>
      <button type="button" class="button-ghost copy-link">Copy link</button>
    </div>
  `;

  card.addEventListener('click', (e) => {
    const target = e.target;
    if (target.closest('.copy-link') || target.closest('.button-link')) {
      return;
    }
    if (incident.url && incident.url !== '#') {
      window.open(incident.url, '_blank', 'noopener,noreferrer');
    }
  });

  card.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      card.click();
    }
  });

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

    if (showMoreBtn) {
      if (count >= filteredIncidents.length) {
        showMoreBtn.style.display = 'none';
      } else {
        showMoreBtn.style.display = 'inline-flex';
      }
    }

    if (prevPageBtn && nextPageBtn) {
      prevPageBtn.style.display = 'none';
      nextPageBtn.style.display = 'none';
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

    if (showMoreBtn) {
      showMoreBtn.style.display = 'none';
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
  wrapper.className = 'sort-control';

  const label = document.createElement('label');
  label.textContent = 'Sort by';

  const select = document.createElement('select');
  select.innerHTML = `
    <option value="date-desc">Most recent</option>
    <option value="date-asc">Oldest first</option>
    <option value="relevancy-desc">Most relevant</option>
    <option value="relevancy-asc">Least relevant</option>
  `;

  select.value = 'date-desc';
  select.addEventListener('change', applyFiltersAndSort);

  wrapper.appendChild(label);
  wrapper.appendChild(select);

  // insert into header right side
  sectionHeader.appendChild(wrapper);

  sortSelect = select;
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

  const sourceOptions = Array.from(sources).sort((a, b) =>
    a.localeCompare(b)
  );
  const vulnerableOptions = Array.from(vulns).sort((a, b) =>
    a.localeCompare(b)
  );

  // Only build filters if there's more than one option
  if (sourceOptions.length > 1) {
    const field = document.createElement('div');
    field.className = 'field';

    const label = document.createElement('label');
    label.textContent = 'Source';

    const select = document.createElement('select');
    select.innerHTML = `
      <option value="">All sources</option>
      ${sourceOptions.map((s) => `<option value="${s}">${s}</option>`).join('')}
    `;

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
    label.textContent = 'Vulnerable population';

    const select = document.createElement('select');
    select.innerHTML =
      '<option value="">All</option>' +
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
    searchButton.addEventListener('click', () => {
      applyFiltersAndSort();
    });

    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
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
    prevPageBtn.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage -= 1;
        renderIncidents();
      }
    });
  }

  if (nextPageBtn) {
    nextPageBtn.addEventListener('click', () => {
      if (!filteredIncidents || filteredIncidents.length === 0) return;
      const totalPages = Math.max(
        1,
        Math.ceil(filteredIncidents.length / UI_CONFIG.pageSize)
      );
      if (currentPage < totalPages) {
        currentPage += 1;
        renderIncidents();
      }
    });
  }

  if (showMoreBtn) {
    showMoreBtn.addEventListener('click', () => {
      if (!filteredIncidents || filteredIncidents.length === 0) return;

      if (isMobile()) {
        // Show more in vertical list
        mobileVisibleCount += UI_CONFIG.pageSize;
        if (mobileVisibleCount >= filteredIncidents.length) {
          mobileVisibleCount = filteredIncidents.length;
          showMoreBtn.style.display = 'none';
        }
        renderIncidents();
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

  // Re-render on resize to switch between mobile/desktop modes cleanly
  window.addEventListener('resize', () => {
    renderIncidents();
  });
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

    const rawData = await res.json();
    if (!Array.isArray(rawData)) {
      showMessage('No sycophancy related incidents found', 'info');
      return;
    }

    allIncidents = rawData.map(normalizeIncident);

    // Apply the safety gate (reviewed + relevancy + source rules)
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
    if (sortSelect) {
      sortSelect.value = 'date-desc';
    }

    applyFiltersAndSort();
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
