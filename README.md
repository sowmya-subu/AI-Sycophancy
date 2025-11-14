# AI-Sycophancy

Free AI Sycophancy Incident Scanner (2023-Present)

Scans RSS feeds, arXiv, and Hacker News to identify and catalog AI sycophancy incidents. Extracts full articles and creates extractive summaries using LSA (Latent Semantic Analysis).

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **NLTK Data Setup**
   
   The scanner will automatically download required NLTK data on first run. If you encounter issues, you can manually download:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

### Optional Dependencies

- **newspaper3k**: Better article extraction (recommended)
- **sumy**: LSA-based summarization (recommended)
- **sentence-transformers**: Semantic embeddings for relevancy scoring (optional, falls back to TF-IDF)

All are included in `requirements.txt` and will be installed automatically.

## Usage

### Basic Scan

Run a full historical scan (2023-present):
```bash
python scanner.py
```

This will:
- Search arXiv for research papers
- Search Google News historical archives
- Search Hacker News discussions
- Scan RSS feeds from major tech/news outlets
- Extract and summarize articles
- Generate `sycophancy_incidents_2023_present.csv`

### Command-Line Options

```bash
# Custom output filename
python scanner.py --output my_incidents.csv

# Save console output to log file
python scanner.py --log scan_output.txt

# Adjust RSS lookback period (default: 180 days)
python scanner.py --rss-days 90

# Use TF-IDF instead of embeddings (faster but less accurate)
python scanner.py --use-tfidf

# Skip historical sources (RSS feeds only)
python scanner.py --no-historical
```

### Output

The scanner generates a CSV file with the following columns:
- `title`: Article title
- `url`: Source URL
- `summary`: Extracted and summarized article content
- `source`: Source name (arXiv, Hacker News, RSS feed name, etc.)
- `publication_date`: Publication date
- `vulnerable_populations`: Mentioned vulnerable groups
- `relevancy_score`: Semantic relevancy score (0-1)
- `credibility_score`: Combined credibility score
- `source_tier`: Source tier (tier_1 through tier_4)
- `has_doi`: Whether research paper has DOI
- `doi`: DOI if available
- `needs_review`: Review flag
- `severity`: Severity rating (to be filled)
- `status`: Status (discovered, reviewed, etc.)
- `reviewer_notes`: Reviewer notes

## Features

- **Article Extraction**: Extracts full article text from URLs
- **True Summarization**: Uses LSA (Latent Semantic Analysis) for extractive summarization
- **Semantic Scoring**: Uses sentence transformers for relevancy scoring
- **Source Tiers**: Categorizes sources by credibility (research papers, news outlets, forums, etc.)
- **Historical Coverage**: Scans 2023-present data from multiple sources
- **HTML Cleaning**: Automatically cleans HTML from RSS feeds

## Notes

- Article extraction and summarization takes time.
- The scanner includes delays between requests to be respectful to sources
- First run will download NLTK data automatically
- Large scans may take 30+ minutes depending on number of articles
