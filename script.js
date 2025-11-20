// Simple mock data for now.
// Later, we can swap this to fetch from a cloud DB / API.
const INCIDENTS = [
  {
    id: 1,
    title: 'Chatbot produced harmful medical advice',
    summary:
      'A consumer-facing chatbot provided incorrect medical instructions that conflicted with official health guidelines.',
    url: 'https://example.com/incident-1',
    publishedAt: '2025-01-10',
    category: 'safety',
    severity: 'high',
    keywords: ['health', 'chatbot', 'safety'],
  },
  {
    id: 2,
    title: 'Recruiting model showed gender bias in ranking',
    summary:
      'An AI-based screening tool ranked candidates in a way that systematically favored one gender over another.',
    url: 'https://example.com/incident-2',
    publishedAt: '2025-01-05',
    category: 'bias',
    severity: 'medium',
    keywords: ['hiring', 'bias', 'fairness'],
  },
  {
    id: 3,
    title: 'Image classifier mislabelled sensitive content',
    summary:
      'An image moderation model failed to correctly flag harmful imagery, leading to exposure on a public platform.',
    url: 'https://example.com/incident-3',
    publishedAt: '2024-12-18',
    category: 'safety',
    severity: 'critical',
    keywords: ['content moderation', 'safety'],
  },
  {
    id: 4,
    title: 'LLM leaked private training data in output',
    summary:
      'Users were able to coax a language model into revealing snippets of its proprietary training text.',
    url: 'https://example.com/incident-4',
    publishedAt: '2024-12-02',
    category: 'security',
    severity: 'high',
    keywords: ['security', 'privacy'],
  },
  {
    id: 5,
    title: 'Autonomous system behaved unpredictably in edge case',
    summary:
      'An automated control system failed during rare environmental conditions that were not represented in training.',
    url: 'https://example.com/incident-5',
    publishedAt: '2024-11-01',
    category: 'reliability',
    severity: 'medium',
    keywords: ['reliability', 'edge cases'],
  },
  {
    id: 6,
    title: 'Misleading AI-generated news went viral',
    summary:
      'AI-generated articles with fabricated details were widely shared before being corrected.',
    url: 'https://example.com/incident-6',
    publishedAt: '2024-10-20',
    category: 'safety',
    severity: 'low',
    keywords: ['misinformation', 'media'],
  },
  {
    id: 7,
    title: 'Credit scoring model disadvantaged specific region',
    summary:
      'A credit scoring model rated applicants from one region lower due to proxy features correlated with geography.',
    url: 'https://example.com/incident-7',
    publishedAt: '2024-09-15',
    category: 'bias',
    severity: 'high',
    keywords: ['finance', 'bias'],
  },
  {
    id: 8,
    title: 'Support chatbot revealed other users’ tickets',
    summary:
      'A support bot surfaced ticket snippets belonging to other customers in multi-turn conversations.',
    url: 'https://example.com/incident-8',
    publishedAt: '2024-08-01',
    category: 'security',
    severity: 'critical',
    keywords: ['support', 'security', 'privacy'],
  },
];

// Simple pagination + search state in memory
let currentPage = 1;
const PAGE_SIZE = 6;

function matchesSearch(incident, text, category, severity, fromDate, toDate) {
  const q = text.trim().toLowerCase();

  // Text match: title, summary, keywords
  const inText =
    !q ||
    incident.title.toLowerCase().includes(q) ||
    incident.summary.toLowerCase().includes(q) ||
    incident.keywords.some((k) => k.toLowerCase().includes(q));

  // Category
  const inCategory = !category || incident.category === category;

  // Severity
  const inSeverity = !severity || incident.severity === severity;

  // Date range
  let inDateRange = true;
  if (fromDate) {
    inDateRange = inDateRange && incident.publishedAt >= fromDate;
  }
  if (toDate) {
    inDateRange = inDateRange && incident.publishedAt <= toDate;
  }

  return inText && inCategory && inSeverity && inDateRange;
}

function getFilters() {
  const searchInput = document.getElementById('search-input');
  const categorySelect = document.getElementById('filter-category');
  const severitySelect = document.getElementById('filter-severity');
  const fromInput = document.getElementById('filter-from');
  const toInput = document.getElementById('filter-to');

  return {
    text: searchInput ? searchInput.value : '',
    category: categorySelect ? categorySelect.value : '',
    severity: severitySelect ? severitySelect.value : '',
    fromDate: fromInput ? fromInput.value : '',
    toDate: toInput ? toInput.value : '',
  };
}

function filterIncidents() {
  const { text, category, severity, fromDate, toDate } = getFilters();
  return INCIDENTS.filter((incident) =>
    matchesSearch(incident, text, category, severity, fromDate, toDate)
  ).sort((a, b) => (a.publishedAt < b.publishedAt ? 1 : -1));
}

function renderIncidents() {
  const container = document.getElementById('incident-grid');
  const pageInfo = document.getElementById('page-info');
  if (!container) return;

  const all = filterIncidents();
  const totalPages = Math.max(1, Math.ceil(all.length / PAGE_SIZE));
  if (currentPage > totalPages) currentPage = totalPages;

  const start = (currentPage - 1) * PAGE_SIZE;
  const pageItems = all.slice(start, start + PAGE_SIZE);

  container.innerHTML = '';

  if (pageItems.length === 0) {
    container.innerHTML =
      '<p class="section-note">No incidents match these filters yet.</p>';
  } else {
    for (const incident of pageItems) {
      const card = document.createElement('article');
      card.className = 'card incident-card';
      card.innerHTML = `
        <div>
          <h3 class="incident-title">${incident.title}</h3>
          <p class="incident-summary">${incident.summary}</p>
          <a href="${incident.url}" class="incident-link" target="_blank" rel="noreferrer">
            View original article
          </a>
          <div class="tags">
            ${incident.keywords
              .map(
                (tag) =>
                  `<span class="tag">${tag.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>`
              )
              .join('')}
          </div>
        </div>
        <div class="incident-footer">
          <span class="incident-date">Published: ${incident.publishedAt}</span>
          <div class="incident-actions">
            <button class="link-button" data-id="${incident.id}" data-action="share">Share</button>
            <button class="link-button" data-id="${incident.id}" data-action="copy">Copy link</button>
          </div>
        </div>
      `;
      container.appendChild(card);
    }
  }

  if (pageInfo) {
    pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
  }
}

function onSearch() {
  currentPage = 1;
  renderIncidents();
}

function onPageChange(delta) {
  const all = filterIncidents();
  const totalPages = Math.max(1, Math.ceil(all.length / PAGE_SIZE));
  const next = currentPage + delta;
  if (next >= 1 && next <= totalPages) {
    currentPage = next;
    renderIncidents();
  }
}

function onIncidentAction(e) {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return;
  const action = target.getAttribute('data-action');
  const id = target.getAttribute('data-id');
  if (!action || !id) return;

  const incident = INCIDENTS.find((x) => String(x.id) === id);
  if (!incident) return;

  const url = incident.url;

  if (action === 'copy') {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(
        () => {
          alert('Link copied to clipboard.');
        },
        () => {
          alert('Unable to copy link.');
        }
      );
    } else {
      alert('Clipboard not available in this browser.');
    }
  }

  if (action === 'share') {
    if (navigator.share) {
      navigator.share({
        title: incident.title,
        text: 'AI incident from AI Incident Hub',
        url,
      }).catch(() => {
        // user canceled; no-op
      });
    } else {
      alert('Share is not supported on this device. You can copy the link instead.');
    }
  }
}

function setupListeners() {
  const searchButton = document.getElementById('search-button');
  const searchInput = document.getElementById('search-input');
  const prevPage = document.getElementById('prev-page');
  const nextPage = document.getElementById('next-page');
  const incidentGrid = document.getElementById('incident-grid');
  const newsletterForm = document.getElementById('newsletter-form');
  const yearSpan = document.getElementById('year');

  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear().toString();
  }

  if (searchButton) searchButton.addEventListener('click', onSearch);
  if (searchInput) {
    searchInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        onSearch();
      }
    });
  }

  if (prevPage) prevPage.addEventListener('click', () => onPageChange(-1));
  if (nextPage) nextPage.addEventListener('click', () => onPageChange(1));

  if (incidentGrid) {
    incidentGrid.addEventListener('click', onIncidentAction);
  }

  if (newsletterForm) {
    newsletterForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const emailInput = document.getElementById('newsletter-email');
      const message = document.getElementById('newsletter-message');
      if (!emailInput || !message) return;

      const value = emailInput.value.trim();
      if (!value) {
        message.textContent = 'Please enter a valid email.';
        return;
      }
      // For now, just show a local confirmation.
      // Later, we’ll wire this to a real backend endpoint.
      message.textContent = 'Thanks for subscribing! (This is a local demo for now.)';
      emailInput.value = '';
    });
  }

  renderIncidents();
}

// Init on DOM ready
document.addEventListener('DOMContentLoaded', setupListeners);

