import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

# ==============================
# CONFIGURATION
# ==============================

# Airtable API settings
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY", "AIRTABLE_API_PAT")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.environ.get("AIRTABLE_TABLE_NAME", "AIRTABLE_TABLE_NAME")  # or your table name

# Optional: view to use (e.g. "Grid view")
AIRTABLE_VIEW = os.environ.get("AIRTABLE_VIEW", None)  # or set to "Published" etc.

# List of sources to exclude from the website (exact string match)
EXCLUDE_SOURCES = os.environ.get("EXCLUDE_SOURCES", "arXiv")
# EXCLUDE_SOURCES can be a comma-separated list in env, e.g. "Example News,BlogXYZ"
EXCLUDE_SOURCES_LIST = [s.strip() for s in EXCLUDE_SOURCES.split(",") if s.strip()]

# Local output path for the JSON file used by the website
OUTPUT_JSON_PATH = os.environ.get("OUTPUT_JSON_PATH", "incidents.json")

# ==============================
# AIRTABLE HELPER
# ==============================

AIRTABLE_API_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
}


def fetch_airtable_records() -> List[Dict[str, Any]]:
    """
    Fetch all records from the Airtable table, handling pagination.
    Returns a list of Airtable record objects (with 'id' and 'fields').
    """
    if AIRTABLE_API_KEY.startswith("YOUR_") or AIRTABLE_BASE_ID.startswith("YOUR_"):
        raise RuntimeError("Please configure AIRTABLE_API_KEY and AIRTABLE_BASE_ID before running.")

    records: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {}

    if AIRTABLE_VIEW:
        params["view"] = AIRTABLE_VIEW

    offset: Optional[str] = None

    while True:
        if offset:
            params["offset"] = offset

        resp = requests.get(AIRTABLE_API_URL, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()

        records.extend(data.get("records", []))
        offset = data.get("offset")

        if not offset:
            break

    return records


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one Airtable record into the JSON shape used by the website.
    Fields from Airtable:
      title, url, summary, source, publication_date, date_found,
      vulnerable_population, needs_review, severity, status,
      review_notes, relevance_score
    """
    fields = record.get("fields", {})
    record_id = record.get("id")

    def get_str(name: str) -> str:
        value = fields.get(name)
        return value if isinstance(value, str) else ("" if value is None else str(value))

    def get_bool(name: str) -> bool:
        value = fields.get(name)
        return bool(value)

    def get_float(name: str) -> Optional[float]:
        value = fields.get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Basic fields
    title = get_str("title")
    url = get_str("url")
    summary = get_str("summary")
    source = get_str("source")
    publication_date = get_str("publication_date")
    date_found = get_str("date_found")
    vulnerable_population = get_str("vulnerable_population")
    needs_review = get_bool("needs_review")
    severity = get_str("severity")
    status = get_str("status")
    review_notes = get_str("review_notes")
    relevance_score = get_float("relevance_score")

    # Prefer ISO strings, do not fail if dates are free-form.
    # If your publication_date is in ISO already, we just pass it through.
    def normalize_date(date_str: str) -> str:
        if not date_str:
            return ""
        # Try to parse common formats; if fail, return original.
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date_str

    publication_date_norm = normalize_date(publication_date)
    date_found_norm = normalize_date(date_found)

    return {
        "id": record_id,
        "title": title,
        "url": url,
        "summary": summary,
        "source": source,
        "publication_date": publication_date_norm,
        "date_found": date_found_norm,
        "vulnerable_population": vulnerable_population,
        "needs_review": needs_review,
        "severity": severity,
        "status": status,
        "review_notes": review_notes,
        "relevance_score": relevance_score,
    }


def should_exclude(record: Dict[str, Any]) -> bool:
    """
    Decide if a record should be excluded based on EXCLUDE_SOURCES_LIST.
    Optional place to add more filters (e.g., status != 'Published').
    """
    source = record.get("source", "")
    if EXCLUDE_SOURCES_LIST and source in EXCLUDE_SOURCES_LIST:
        return True

    # Example: only include records where status is empty or 'Published'
    # status = record.get("status", "")
    # if status and status.lower() != "published":
    #     return True

    return False


def sync_airtable_to_json() -> None:
    """
    Main sync function:
      - Fetch Airtable records
      - Normalize
      - Apply filters
      - Write incidents.json
    """
    print("Fetching records from Airtable...")
    raw_records = fetch_airtable_records()
    print(f"Fetched {len(raw_records)} raw records")

    normalized: List[Dict[str, Any]] = []
    for rec in raw_records:
        norm = normalize_record(rec)
        if should_exclude(norm):
            continue
        normalized.append(norm)

    print(f"{len(normalized)} records after filtering")

    # Optionally sort by publication_date descending
    def sort_key(item: Dict[str, Any]) -> str:
        return item.get("publication_date") or ""

    normalized_sorted = sorted(normalized, key=sort_key, reverse=True)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized_sorted, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(normalized_sorted)} incidents to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    # Run a single sync.
    # Frequency is controlled OUTSIDE this script by:
    #  - cron on your machine, OR
    #  - GitHub Actions schedule, OR
    #  - any other scheduler.
    sync_airtable_to_json()
