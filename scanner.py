"""
Free AI Sycophancy Incident Scanner (2023-Present)
Uses RSS feeds, arXiv, and Hacker News for historical coverage

# Install dependencies
pip install requests pandas
# Full scan (2023-present) - includes historical data
python scanner.py

# Custom output filename
python scanner.py --output my_complete_scan.csv
# Adjust RSS lookback period
python scanner.py --rss-days 180

"""

import feedparser
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import time
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# RSS Feeds from major tech/news outlets
RSS_FEEDS = {
    # Tech News
    'TechCrunch': 'https://techcrunch.com/feed/',
    'The Verge': 'https://www.theverge.com/rss/index.xml',
    'Ars Technica': 'https://feeds.arstechnica.com/arstechnica/index',
    'VentureBeat': 'https://venturebeat.com/feed/',
    'Wired': 'https://www.wired.com/feed/rss',
    
    # AI Specific
    'MIT Tech Review AI': 'https://www.technologyreview.com/topic/artificial-intelligence/feed',
    
    # Mainstream
    'Reuters Tech': 'https://www.reuters.com/technology',
    'BBC Technology': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
    'NPR Technology': 'https://feeds.npr.org/1019/rss.xml',
    
    # Research/Safety
    'Partnership on AI': 'https://partnershiponai.org/feed/',
}

# Google News RSS (searches Google News via RSS)
GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"

# Search terms for Google News RSS
SEARCH_TERMS = [
    "AI sycophancy",
    "ChatGPT mental health harm",
    "AI chatbot dangerous advice",
    "AI therapy problem",
    "chatbot validation delusion",
    "AI flattery issue",
    "sycophantic AI",
    "AI agrees with everything",
    "ChatGPT stopped medication",
    "AI medical advice wrong",
    "Character.AI harm",
    "Meta AI chatbot incident",
    "OpenAI rollback sycophancy",
    "Claude sycophancy",
    "Gemini AI flattery",
]

# Date-specific searches for historical coverage (2023-2024)
HISTORICAL_SEARCHES = [
    # Major 2023 events
    "AI sycophancy research 2023",
    "ChatGPT problems 2023",
    "AI mental health 2023",
    
    # Major 2024 events  
    "AI sycophancy 2024",
    "ChatGPT mental health 2024",
    "OpenAI sycophancy 2024",
    
    # Major 2025 events
    "OpenAI April 2025 rollback",
    "GPT-4o sycophancy 2025",
    "AI sycophancy 2025",
    
    # Academic publications
    "Anthropic sycophancy study",
    "Google DeepMind sycophancy",
    "AI sycophancy benchmark",
    "ELEPHANT benchmark AI",
    "SycEval benchmark",
]

# Keywords to identify relevant incidents
SYCOPHANCY_KEYWORDS = [
    'sycophancy', 'sycophantic', 'flattery', 'flatter', 
    'validation', 'validate', 'agreeable', 'agrees with everything',
    'too positive', 'overly supportive', 'echo chamber',
    'confirmed delusion', 'validated belief', 'stop medication',
    'dangerous advice', 'harmful validation', 'too agreeable',
    'overly flattering', 'glazes too much', 'people pleasing',
    'reward hacking', 'human approval'
]

POPULATION_KEYWORDS = {
    'mental_health': ['mental health', 'depression', 'suicide', 'suicidal', 'therapy', 
                      'psychosis', 'delusion', 'psychiatric', 'crisis', 'counseling',
                      'anxiety', 'bipolar', 'schizophrenia', 'medication'],
    'medical': ['medical advice', 'diagnosis', 'doctor', 'patient', 
                'treatment', 'prescription', 'medicine', 'medical decision',
                'healthcare', 'clinical'],
    'minors': ['child', 'children', 'teenager', 'teen', 'student', 'minor', 
               'adolescent', 'youth', 'kid', 'school', 'young people'],
    'elderly': ['elderly', 'senior', 'older adult', 'aging', 'retirement', 
                'dementia', 'alzheimer'],
}

# ============================================================================
# HISTORICAL DATA SOURCES
# ============================================================================


def search_arxiv_papers() -> List[Dict]:
    """
    Search arXiv for sycophancy research papers (2023+)
    arXiv API is free and has no rate limits
    """
    incidents = []
    
    print("\n  Searching arXiv for research papers (2023-present)...")
    
    search_queries = [
        "sycophancy language models",
        "AI sycophancy",
        "LLM flattery",
        "chatbot validation bias",
        "language model alignment",
        "RLHF reward hacking",
        "AI people pleasing",
        "LLM harmful agreement"
    ]
    
    try:
        for query in search_queries:
            # Increase max_results to 200 to get more historical papers
            url = f"http://export.arxiv.org/api/query?search_query=all:{quote(query)}&start=0&max_results=200&sortBy=submittedDate&sortOrder=descending"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse XML response
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.content)
                
                # Namespace for arXiv
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                for entry in root.findall('atom:entry', ns):
                    # Get publication date
                    published = entry.find('atom:published', ns)
                    pub_date_str = published.text if published is not None else ''
                    
                    try:
                        pub_date = datetime.strptime(pub_date_str[:10], '%Y-%m-%d')
                        
                        # Only include 2023 or later
                        if pub_date >= datetime(2023, 1, 1):
                            title_elem = entry.find('atom:title', ns)
                            summary_elem = entry.find('atom:summary', ns)
                            id_elem = entry.find('atom:id', ns)
                            
                            title = title_elem.text if title_elem is not None else ''
                            summary = summary_elem.text if summary_elem is not None else ''
                            paper_url = id_elem.text if id_elem is not None else ''
                            
                            # Extract populations from abstract
                            full_text = f"{title} {summary}".lower()
                            populations = []
                            for pop_type, keywords in POPULATION_KEYWORDS.items():
                                if any(keyword in full_text for keyword in keywords):
                                    populations.append(pop_type)
                            
                            incidents.append({
                                'title': title.strip(),
                                'url': paper_url,
                                'summary': summary.strip()[:500],
                                'source': 'arXiv',
                                'publication_date': pub_date_str[:10],
                                'date_found': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'vulnerable_populations': ', '.join(populations),
                                'needs_review': True,
                                'severity': '',
                                'status': 'discovered',
                                'reviewer_notes': 'Research paper'
                            })
                    except:
                        pass
            
            time.sleep(3)  # Be respectful to arXiv
        
        print(f"    ✓ Found {len(incidents)} research papers from 2023+")
    
    except Exception as e:
        print(f"Error searching arXiv: {e}")
    
    return incidents


def get_google_news_historical() -> List[Dict]:
    """
    Search Google News RSS for historical articles (2023-2024)
    Uses date-specific searches to find older articles
    """
    incidents = []
    
    print("\n  Searching Google News historical archives (2023-2024)...")
    
    # Date-specific searches that Google News can find
    historical_queries = [
        # 2023 events
        "AI sycophancy 2023",
        "ChatGPT mental health 2023",
        "AI chatbot problem 2023",
        "ChatGPT therapy 2023",
        
        # 2024 events
        "AI sycophancy 2024",
        "ChatGPT mental health 2024",
        "AI chatbot dangerous 2024",
        "Character.AI 2024",
        "OpenAI alignment 2024",
        
        # Specific incidents
        "ChatGPT medication advice",
        "AI therapy harm",
        "chatbot emotional attachment",
        "AI validation delusion",
        "Replika AI emotional",
    ]
    
    for query in historical_queries:
        try:
            feed_url = GOOGLE_NEWS_RSS_BASE.format(quote(query))
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:20]:  # Top 20 results per query
                title = entry.get('title', '')
                summary = entry.get('summary', '') or entry.get('description', '')
                link = entry.get('link', '')
                
                # Parse publication date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                
                # Check if relevant
                if not is_relevant_content(title, summary, SYCOPHANCY_KEYWORDS):
                    continue
                
                # Extract populations
                populations = extract_populations(f"{title} {summary}")
                
                incidents.append({
                    'title': title,
                    'url': link,
                    'summary': summary[:500],
                    'source': 'Google News (Historical)',
                    'publication_date': pub_date.strftime('%Y-%m-%d %H:%M:%S') if pub_date else '',
                    'date_found': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'vulnerable_populations': ', '.join(populations),
                    'needs_review': True,
                    'severity': '',
                    'status': 'discovered',
                    'reviewer_notes': ''
                })
            
            time.sleep(2)  # Be respectful
            
        except Exception as e:
            print(f"    Error with query '{query}': {e}")
            continue
    
    print(f"    ✓ Found {len(incidents)} articles from Google News archives")
    return incidents


def get_hacker_news_historical() -> List[Dict]:
    """
    Search Hacker News for historical discussions (2023+)
    Uses Algolia's free API
    """
    incidents = []
    
    print("\n  Searching Hacker News (2023+)...")
    
    # Algolia HN API supports date filtering
    # Expanded search terms to catch more relevant discussions
    search_terms = [
        "AI sycophancy",
        "ChatGPT mental health",
        "AI dangerous advice",
        "sycophantic AI",
        "ChatGPT therapy",
        "AI chatbot harm",
        "LLM alignment problem",
        "ChatGPT agrees everything",
        "AI validation bias",
        "Character.AI",
        "Replika AI",
        "AI emotional attachment"
    ]
    
    try:
        # Unix timestamp for Jan 1, 2023
        start_timestamp = int(datetime(2023, 1, 1).timestamp())
        
        for term in search_terms:
            url = f"https://hn.algolia.com/api/v1/search?query={quote(term)}&tags=story&numericFilters=created_at_i>{start_timestamp}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for hit in data.get('hits', []):
                    created_at = datetime.fromtimestamp(hit.get('created_at_i', 0))
                    
                    # Extract populations
                    full_text = f"{hit.get('title', '')} {hit.get('story_text', '')}".lower()
                    populations = []
                    for pop_type, keywords in POPULATION_KEYWORDS.items():
                        if any(keyword in full_text for keyword in keywords):
                            populations.append(pop_type)
                    
                    incidents.append({
                        'title': hit.get('title', ''),
                        'url': hit.get('url', f"https://news.ycombinator.com/item?id={hit.get('objectID')}"),
                        'summary': hit.get('story_text', '')[:500],
                        'source': 'Hacker News',
                        'publication_date': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'date_found': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'vulnerable_populations': ', '.join(populations),
                        'hn_points': hit.get('points', 0),
                        'hn_comments': hit.get('num_comments', 0),
                        'needs_review': True,
                        'severity': '',
                        'status': 'discovered',
                        'reviewer_notes': ''
                    })
            
            time.sleep(1)
        
        print(f"    ✓ Found {len(incidents)} Hacker News stories from 2023+")
    
    except Exception as e:
        print(f"    Error searching Hacker News: {e}")
    
    return incidents


# ============================================================================
# RECENT DATA (RSS FEEDS)
# ============================================================================

def is_relevant_content(title: str, summary: str, keywords: List[str]) -> bool:
    """Check if content contains sycophancy-related keywords"""
    text = f"{title} {summary}".lower()
    return any(keyword.lower() in text for keyword in keywords)


def extract_populations(text: str) -> List[str]:
    """Extract mentioned vulnerable populations"""
    text_lower = text.lower()
    populations = []
    
    for pop_type, keywords in POPULATION_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            populations.append(pop_type)
    
    return populations


def parse_feed(feed_url: str, feed_name: str, days_back: int = 90) -> List[Dict]:
    """
    Parse RSS feed and extract relevant entries
    
    Args:
        feed_url: URL of RSS feed
        feed_name: Name of the feed source
        days_back: How many days back to include entries (default 90 for broader capture)
    
    Returns:
        List of incident dictionaries
    """
    incidents = []
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    try:
        feed = feedparser.parse(feed_url)
        
        for entry in feed.entries:
            # Get entry details
            title = entry.get('title', '')
            summary = entry.get('summary', '') or entry.get('description', '')
            link = entry.get('link', '')
            
            # Parse publication date
            pub_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                pub_date = datetime(*entry.updated_parsed[:6])
            
            # Skip if too old (but RSS usually only has recent anyway)
            if pub_date and pub_date < cutoff_date:
                continue
            
            # Check if relevant to sycophancy
            if not is_relevant_content(title, summary, SYCOPHANCY_KEYWORDS):
                continue
            
            # Extract populations
            populations = extract_populations(f"{title} {summary}")
            
            # Create incident record
            incident = {
                'title': title,
                'url': link,
                'summary': summary[:500],
                'source': feed_name,
                'publication_date': pub_date.strftime('%Y-%m-%d %H:%M:%S') if pub_date else '',
                'date_found': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'vulnerable_populations': ', '.join(populations),
                'needs_review': True,
                'severity': '',
                'status': 'discovered',
                'reviewer_notes': ''
            }
            
            incidents.append(incident)
    
    except Exception as e:
        print(f"    ⚠️  Error parsing {feed_name}: {e}")
    
    return incidents


def create_google_news_feeds(search_terms: List[str]) -> Dict[str, str]:
    """Create Google News RSS feed URLs for each search term"""
    feeds = {}
    for term in search_terms:
        feed_name = f"Google News: {term}"
        feed_url = GOOGLE_NEWS_RSS_BASE.format(quote(term))
        feeds[feed_name] = feed_url
    return feeds


def scan_rss_feeds(feeds: Dict[str, str], days_back: int = 90) -> List[Dict]:
    """Scan all RSS feeds for recent incidents"""
    all_incidents = []
    
    print(f"\n  Scanning {len(feeds)} RSS feeds (last {days_back} days)...")
    
    for feed_name, feed_url in feeds.items():
        incidents = parse_feed(feed_url, feed_name, days_back)
        
        if incidents:
            all_incidents.extend(incidents)
        
        time.sleep(0.5)  # Be polite
    
    print(f"    ✓ Found {len(all_incidents)} relevant articles from RSS")
    
    return all_incidents


# ============================================================================
# MAIN SCANNER
# ============================================================================

def remove_duplicates(incidents: List[Dict]) -> List[Dict]:
    """Remove duplicate incidents based on URL"""
    seen_urls = set()
    unique = []
    
    for incident in incidents:
        url = incident['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append(incident)
    
    return unique


def save_results(incidents: List[Dict], output_file: str = 'sycophancy_incidents_2023_present.csv'):
    """Save incidents to CSV with summary statistics"""
    if not incidents:
        print("\n No incidents found!")
        return
    
    df = pd.DataFrame(incidents)
    
    # Sort by publication date (oldest to newest for historical view)
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df = df.sort_values('publication_date', ascending=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved {len(incidents)} incidents to {output_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (2023-Present)")
    print("=" * 80)
    print(f"\nTotal unique incidents: {len(incidents)}")
    
    # Timeline breakdown
    print("\nIncidents by Year:")
    year_counts = df[df['publication_date'].notna()].groupby(df['publication_date'].dt.year).size()
    for year, count in year_counts.items():
        print(f"  - {int(year)}: {count}")
    
    # Monthly breakdown for 2025
    if 2025 in year_counts.index:
        print("\n2025 by Month:")
        df_2025 = df[df['publication_date'].dt.year == 2025]
        month_counts = df_2025.groupby(df_2025['publication_date'].dt.month).size()
        for month, count in month_counts.items():
            month_name = datetime(2025, int(month), 1).strftime('%B')
            print(f"  - {month_name}: {count}")
    
    # Vulnerable populations
    if df['vulnerable_populations'].any():
        print("\nVulnerable Populations Mentioned:")
        pop_counts = df['vulnerable_populations'].str.split(', ').explode().value_counts()
        for pop, count in pop_counts.head(10).items():
            if pop:
                print(f"  - {pop}: {count}")
    
    # Top sources
    print("\nTop Sources:")
    for source, count in df['source'].value_counts().head(15).items():
        print(f"  - {source}: {count}")
    
    # Date range
    valid_dates = df[df['publication_date'].notna()]['publication_date']
    if not valid_dates.empty:
        earliest = valid_dates.min()
        latest = valid_dates.max()
        print(f"\nDate Range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
    
    print("\n" + "=" * 80)
    print("\nNext Steps:")
    print("1. Review the CSV file and validate each incident")
    print("2. Fill in 'severity' field (low/medium/high/critical)")
    print("3. Update 'status' field as you review")
    print("4. Add notes in 'reviewer_notes' field")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Free RSS-based AI sycophancy scanner (2023-Present)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full historical scan (2023-present)
  python scanner.py
  
  # Custom output file
  python scanner.py --output my_incidents.csv
  
  # Skip historical sources (RSS only)
  python scanner.py --no-historical
        """
    )
    
    parser.add_argument(
        '--output',
        default='sycophancy_incidents_2023_present.csv',
        help='Output CSV file (default: sycophancy_incidents_2023_present.csv)'
    )
    
    parser.add_argument(
        '--no-historical',
        action='store_true',
        help='Skip historical sources (AIID, arXiv, HN) - only scan RSS feeds'
    )
    
    parser.add_argument(
        '--rss-days',
        type=int,
        default=90,
        help='Days back to scan for RSS feeds (default: 90)'
    )
    
    args = parser.parse_args()
    
    all_incidents = []
    
    print("=" * 80)
    print("AI SYCOPHANCY INCIDENT SCANNER (2023-PRESENT)")
    print("=" * 80)
    
    # Step 1: Get historical data from free sources
    if not args.no_historical:
        print("\n[1/5] COLLECTING HISTORICAL DATA (2023+)")
        print("-" * 80)
        
        # Google News historical archives (2023-2024)
        google_historical = get_google_news_historical()
        all_incidents.extend(google_historical)
        
        # arXiv research papers
        arxiv_incidents = search_arxiv_papers()
        all_incidents.extend(arxiv_incidents)
        
        # Hacker News historical
        hn_incidents = get_hacker_news_historical()
        all_incidents.extend(hn_incidents)
    
    # Step 2: Get recent data from RSS feeds
    print("\n[2/5] COLLECTING RECENT DATA (RSS Feeds)")
    print("-" * 80)
    
    # Combine all RSS feeds
    all_feeds = RSS_FEEDS.copy()
    
    # Add Google News searches
    google_feeds = create_google_news_feeds(SEARCH_TERMS + HISTORICAL_SEARCHES)
    all_feeds.update(google_feeds)
    
    print(f"  Total RSS feeds to scan: {len(all_feeds)}")
    
    rss_incidents = scan_rss_feeds(all_feeds, args.rss_days)
    all_incidents.extend(rss_incidents)
    
    # Step 3: Remove duplicates
    print("\n[3/5] DEDUPLICATING")
    print("-" * 80)
    print(f"  Total incidents before deduplication: {len(all_incidents)}")
    
    all_incidents = remove_duplicates(all_incidents)
    
    print(f"  Unique incidents: {len(all_incidents)}")
    
    # Step 4: Save results
    print("\n[4/5] SAVING RESULTS")
    print("-" * 80)
    
    save_results(all_incidents, args.output)


if __name__ == "__main__":
    main()
