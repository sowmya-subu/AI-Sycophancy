"""
Free AI Sycophancy Incident Scanner (2023-Present)
Uses RSS feeds, arXiv, and Hacker News for historical coverage

# Install dependencies
pip install requests pandas newspaper3k nltk sumy

# Run the scanner (NLTK data will download automatically on first run)
python scanner.py

# Custom output filename
python scanner.py --output my_complete_scan.csv
# Adjust RSS lookback period
python scanner.py --rss-days 180

"""

import feedparser
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
import json
import os
import re

# Article extraction libraries
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("⚠️  Warning: newspaper3k not installed. Install with: pip install newspaper3k")

# NLTK setup - MUST download data before importing sumy
try:
    import nltk
    NLTK_AVAILABLE = True
    
    print("Checking NLTK data...")
    
    # Download required NLTK data (punkt tokenizer)
    punkt_downloaded = False
    try:
        nltk.data.find('tokenizers/punkt')
        print("  ✓ punkt tokenizer found")
        punkt_downloaded = True
    except LookupError:
        print("  Downloading NLTK punkt tokenizer (one-time setup)...")
        try:
            nltk.download('punkt', quiet=False)
            punkt_downloaded = True
            print("  ✓ punkt downloaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to download punkt: {e}")
    
    # Also download punkt_tab for newer NLTK versions
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("  ✓ punkt_tab found")
    except LookupError:
        try:
            print("  Downloading punkt_tab for newer NLTK...")
            nltk.download('punkt_tab', quiet=False)
            print("  ✓ punkt_tab downloaded successfully")
        except Exception as e:
            # punkt_tab may not exist in older versions, that's okay
            print(f"  → punkt_tab not available (this is okay for older NLTK): {e}")
    
    # Download stopwords
    try:
        nltk.data.find('corpora/stopwords')
        print("  ✓ stopwords found")
    except LookupError:
        print("  Downloading stopwords...")
        try:
            nltk.download('stopwords', quiet=False)
            print("  ✓ stopwords downloaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to download stopwords: {e}")
    
    if not punkt_downloaded:
        print("\n⚠️  WARNING: punkt tokenizer failed to download!")
        print("   This will cause summarization to fail.")
        print("   Try manually: python -c \"import nltk; nltk.download('punkt')\"")
        
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  Warning: nltk not installed. Install with: pip install nltk")

# Summarization library - import AFTER NLTK setup
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False
    print("⚠️  Warning: sumy not installed. Install with: pip install sumy")
    print("   Using fallback summarization method.")

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
# ARTICLE EXTRACTION AND SUMMARIZATION
# ============================================================================

def clean_html(text: str) -> str:
    """
    Remove HTML tags and entities from text.
    This is critical for RSS feeds which often contain HTML.
    """
    if not isinstance(text, str) or not text:
        return text or ""
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    # Remove HTML entities
    clean_text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', clean_text)
    # Clean up extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text


def calculate_relevance_score(title: str, summary: str, source: str) -> float:
    """
    Calculate a relevance score (0-100) for how related an article is to AI sycophancy.
    
    Scoring criteria:
    - Primary keywords (sycophancy, flattery, etc.) = higher weight
    - Context keywords (AI, LLM, chatbot, etc.) = required for high scores
    - Harmful outcomes (dangerous advice, mental health, etc.) = bonus points
    - Generic AI research without specific sycophancy focus = lower scores
    
    Returns:
        float: Score from 0-100, where:
        - 80+: Highly relevant (directly about AI sycophancy)
        - 60-79: Relevant (discusses related issues)
        - 40-59: Somewhat relevant (tangentially related)
        - 0-39: Low relevance (false positive)
    """
    
    # Combine text for analysis
    full_text = f"{title} {summary}".lower()
    
    score = 0.0
    
    # PRIMARY KEYWORDS (Core sycophancy concepts) - 40 points max
    primary_keywords = {
        'sycophancy': 15,
        'sycophantic': 15,
        'flattery': 12,
        'flatter': 12,
        'flattering': 12,
        'people pleasing': 10,
        'people-pleasing': 10,
        'too agreeable': 10,
        'overly agreeable': 10,
        'agrees with everything': 12,
        'echo chamber': 8,
        'yes-man': 10,
        'yes man': 10,
        'bootlicking': 8,
    }
    
    for keyword, points in primary_keywords.items():
        if keyword in full_text:
            score += points
            break  # Only count highest-value primary keyword once
    
    # SECONDARY KEYWORDS (Validation/agreement issues) - 25 points max
    secondary_keywords = {
        'validation': 8,
        'validate': 8,
        'validates': 8,
        'validating': 8,
        'overly supportive': 10,
        'too supportive': 10,
        'confirmed delusion': 12,
        'validated belief': 10,
        'reinforced belief': 10,
        'reward hacking': 12,
        'human approval': 8,
        'user preference': 6,
    }
    
    secondary_score = 0
    for keyword, points in secondary_keywords.items():
        if keyword in full_text:
            secondary_score = max(secondary_score, points)
    score += secondary_score
    
    # AI/LLM CONTEXT (Must be present for relevance) - 15 points max
    ai_context = [
        'chatbot', 'chat bot', 'llm', 'large language model',
        'language model', 'gpt', 'chatgpt', 'claude', 'gemini',
        'ai assistant', 'conversational ai', 'character.ai',
        'anthropic', 'openai', 'deepseek'
    ]
    
    has_ai_context = any(term in full_text for term in ai_context)
    if has_ai_context:
        score += 15
    else:
        # Penalize heavily if no AI context - likely false positive
        score *= 0.3
    
    # HARMFUL OUTCOMES (Real-world impact) - 20 points bonus
    harmful_outcomes = {
        'dangerous advice': 15,
        'harmful advice': 15,
        'mental health': 10,
        'suicide': 18,
        'suicidal': 18,
        'stop medication': 15,
        'stopped medication': 15,
        'medical advice': 10,
        'therapy': 8,
        'delusion': 12,
        'psychosis': 12,
        'crisis': 10,
        'harm': 8,
        'dangerous': 10,
        'risk': 6,
    }
    
    harm_score = 0
    for keyword, points in harmful_outcomes.items():
        if keyword in full_text:
            harm_score += points
            if harm_score >= 20:  # Cap at 20 bonus points
                break
    score += min(harm_score, 20)
    
    # SPECIFIC INCIDENT MARKERS (Real cases) - 15 points bonus
    incident_markers = [
        'lawsuit', 'incident', 'case study', 'reported', 'rollback',
        'backlash', 'complaint', 'investigation', 'user pushback'
    ]
    
    if any(marker in full_text for marker in incident_markers):
        score += 15
    
    # NEGATIVE INDICATORS (Likely false positives) - penalties
    false_positive_indicators = {
        'code generation': -10,
        'software development': -10,
        'programming': -8,
        'syntax': -8,
        'compiler': -10,
        'algorithm': -5,
        'optimization': -5,
        'machine translation': -8,
        'image generation': -8,
        'computer vision': -8,
        'robotics': -8,
        'autonomous vehicle': -10,
        'quantum': -10,
        'blockchain': -10,
        'cybersecurity': -8,
    }
    
    for indicator, penalty in false_positive_indicators.items():
        if indicator in full_text and 'sycophancy' not in full_text:
            score += penalty
    
    
    return round(score, 2)


def filter_by_relevance(incidents: List[Dict], min_score: float = 10.0) -> List[Dict]:
    """
    Filter incidents by relevance score and add score to each incident.
    
    Args:
        incidents: List of incident dictionaries
        min_score: Minimum relevance score to keep (default: 40)
    
    Returns:
        List of filtered incidents with 'relevance_score' field added
    """
    
    scored_incidents = []
    
    for incident in incidents:
        score = calculate_relevance_score(
            incident.get('title', ''),
            incident.get('summary', ''),
            incident.get('source', '')
        )
        
        incident['relevance_score'] = score
        
        if score >= min_score:
            scored_incidents.append(incident)
    
    return scored_incidents


def summarize_text_lsa(text: str, num_sentences: int = 3) -> str:
    """
    Create a true extractive summary using LSA (Latent Semantic Analysis).
    This selects the most important sentences, not just the first ones.
    """
    if not text or len(text.strip()) < 100:
        return text
    
    try:
        if SUMY_AVAILABLE:
            # Use LSA summarization - ranks sentences by importance
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            
            # Get most important sentences
            summary_sentences = summarizer(parser.document, num_sentences)
            
            # Convert to string
            summary = ' '.join([str(sentence) for sentence in summary_sentences])
            
            if len(summary) > 500:
                summary = summary[:497] + '...'
            
            return summary.strip()
    except Exception as e:
        print(f"        LSA summarization failed: {str(e)[:50]}")
    
    # Fallback to basic extractive summarization
    return summarize_text_fallback(text, num_sentences)


def summarize_text_fallback(text: str, num_sentences: int = 3) -> str:
    """
    Fallback summarization using sentence scoring based on:
    - Sentence position (earlier sentences weighted higher)
    - Keyword frequency
    - Sentence length (not too short, not too long)
    """
    if not text or len(text.strip()) < 100:
        return text
    
    # Split into sentences
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Score each sentence
    scored_sentences = []
    
    for idx, sentence in enumerate(sentences):
        score = 0
        
        # Position score (earlier sentences are more important)
        position_score = 1.0 / (idx + 1)
        score += position_score * 2
        
        # Length score (prefer medium-length sentences)
        length = len(sentence.split())
        if 10 <= length <= 30:
            score += 1.5
        elif 5 <= length <= 40:
            score += 1.0
        
        # Keyword score (check for important keywords)
        sentence_lower = sentence.lower()
        important_keywords = ['research', 'study', 'found', 'shows', 'according', 
                            'report', 'expert', 'scientist', 'professor', 'revealed',
                            'ai', 'chatbot', 'model', 'system', 'users', 'people',
                            'sycophancy', 'flattery', 'validation', 'mental health',
                            'dangerous', 'harmful', 'problem', 'issue', 'concern']
        
        keyword_count = sum(1 for kw in important_keywords if kw in sentence_lower)
        score += keyword_count * 0.5
        
        scored_sentences.append((score, idx, sentence))
    
    # Sort by score (descending) and take top N
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    top_sentences = scored_sentences[:num_sentences]
    
    # Sort by original position to maintain flow
    top_sentences.sort(key=lambda x: x[1])
    
    # Join sentences
    summary = ' '.join([sent[2] for sent in top_sentences])
    
    if len(summary) > 500:
        summary = summary[:497] + '...'
    
    return summary.strip()


def extract_article_text(url: str) -> Optional[str]:
    """
    Extract full article text from URL.
    Returns None if extraction fails.
    """
    # Try newspaper3k first (best results)
    if NEWSPAPER_AVAILABLE:
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 200:
                return article.text
        except Exception as e:
            pass
    
    # Fallback to BeautifulSoup
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                script.decompose()
            
            # Try to find article content
            article_text = ''
            
            # Common article selectors
            article_selectors = [
                'article',
                '[role="article"]',
                '.article-content',
                '.article-body',
                '.post-content',
                '.entry-content',
                'main',
            ]
            
            for selector in article_selectors:
                article_elem = soup.select_one(selector)
                if article_elem:
                    # Get all paragraphs
                    paragraphs = article_elem.find_all('p')
                    if paragraphs:
                        article_text = ' '.join([p.get_text().strip() for p in paragraphs])
                        break
            
            # If no article found, try all paragraphs
            if not article_text:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    article_text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            if article_text and len(article_text) > 200:
                # Clean up whitespace
                article_text = re.sub(r'\s+', ' ', article_text).strip()
                return article_text
                
    except Exception as e:
        pass
    
    return None


def extract_article_summary(url: str, fallback_text: str = '') -> str:
    """
    Extract article and create a TRUE SUMMARY using extractive summarization.
    
    Args:
        url: Article URL to fetch
        fallback_text: RSS feed summary to use if extraction fails (may contain HTML)
    
    Returns:
        Article summary (500 chars max), HTML-cleaned
    """
    
    # Clean HTML from fallback text immediately
    clean_fallback = clean_html(fallback_text) if fallback_text else ""
    
    # Extract full article text
    article_text = extract_article_text(url)
    
    if article_text:
        # Create a REAL summary using LSA or fallback method
        summary = summarize_text_lsa(article_text, num_sentences=3)
        return clean_html(summary)  # Extra safety: clean any HTML in extracted text
    
    # If extraction failed, try to summarize fallback text
    if clean_fallback and len(clean_fallback) > 100:
        summary = summarize_text_lsa(clean_fallback, num_sentences=2)
        return clean_html(summary)
    
    # Last resort - return cleaned fallback or error message
    if clean_fallback:
        return clean_fallback[:500]
    
    return "Summary unavailable - article extraction failed"


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
                            abstract = summary_elem.text if summary_elem is not None else ''
                            paper_url = id_elem.text if id_elem is not None else ''
                            
                            # For arXiv, create a summary of the abstract
                            summary = summarize_text_lsa(abstract.strip(), num_sentences=3)
                            
                            # Extract populations from abstract
                            full_text = f"{title} {abstract}".lower()
                            populations = []
                            for pop_type, keywords in POPULATION_KEYWORDS.items():
                                if any(keyword in full_text for keyword in keywords):
                                    populations.append(pop_type)
                            
                            incidents.append({
                                'title': title.strip(),
                                'url': paper_url,
                                'summary': summary,
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
    NOW CREATES TRUE SUMMARIES
    """
    incidents = []
    
    print("\n  Searching Google News historical archives (2023-2024)...")
    print("     (Extracting and summarizing articles - this takes time)")
    
    # Date-specific searches that Google News can find
    historical_queries = [
        # 2023 events
        "AI sycophancy 2023",
        "ChatGPT mental health 2023",
        "AI chatbot problem 2023",
        "ChatGPT therapy 2023",
        
        # 2024 events
        "AI sycophancy 2024",
        "OpenAI sycophancy 2024",
        "ChatGPT agrees too much 2024",
        "AI validation problem 2024",
        
        # Specific incidents
        "Character.AI lawsuit",
        "ChatGPT mental health incident",
        "AI chatbot dangerous advice",
    ]
    
    for query in historical_queries:
        try:
            feed_url = GOOGLE_NEWS_RSS_BASE.format(quote(query))
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:200]:  # Limit to avoid too many requests
                try:
                    title = clean_html(entry.get('title', ''))  # Clean HTML from title
                    link = entry.get('link', '')
                    
                    # Check for sycophancy relevance
                    text_to_check = title.lower()
                    if not any(keyword in text_to_check for keyword in SYCOPHANCY_KEYWORDS):
                        continue
                    
                    # Get publication date
                    pub_date = None
                    if hasattr(entry, 'published_parsed'):
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # EXTRACT AND SUMMARIZE ARTICLE
                    rss_summary = entry.get('summary', '')
                    print(f"      Summarizing: {title[:60]}...")
                    article_summary = extract_article_summary(link, rss_summary)
                    
                    # Extract populations
                    full_text = f"{title} {article_summary}".lower()
                    populations = []
                    for pop_type, keywords in POPULATION_KEYWORDS.items():
                        if any(keyword in full_text for keyword in keywords):
                            populations.append(pop_type)
                    
                    incidents.append({
                        'title': title,
                        'url': link,
                        'summary': article_summary,
                        'source': f"Google News: {query}",
                        'publication_date': pub_date.strftime('%Y-%m-%d') if pub_date else '',
                        'date_found': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'vulnerable_populations': ', '.join(populations),
                        'needs_review': True,
                        'severity': '',
                        'status': 'discovered',
                        'reviewer_notes': ''
                    })
                    
                    time.sleep(2)  # Be respectful between article extractions
                    
                except Exception as e:
                    print(f"      Error processing entry: {e}")
                    continue
            
            time.sleep(1)
            
        except Exception as e:
            print(f"    Error with query '{query}': {e}")
    
    print(f"    ✓ Found {len(incidents)} historical articles with true summaries")
    return incidents


def get_hacker_news_historical() -> List[Dict]:
    """
    Search Hacker News archives (2023+) via Algolia API
    NOW CREATES TRUE SUMMARIES FROM LINKED ARTICLES
    """
    incidents = []
    
    print("\n  Searching Hacker News archives (2023-present)...")
    print("     (Summarizing linked articles)")
    
    search_terms = [
        "AI sycophancy",
        "ChatGPT mental health",
        "AI chatbot dangerous",
        "LLM flattery",
        "AI validation problem"
    ]
    
    base_url = "http://hn.algolia.com/api/v1/search"
    
    try:
        for term in search_terms:
            # Search last 3 years
            params = {
                'query': term,
                'tags': 'story',
                'numericFilters': f'created_at_i>{int(datetime(2023, 1, 1).timestamp())}'
            }
            
            response = requests.get(base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('hits', [])[:200]:  # Limit results
                    title = clean_html(item.get('title', ''))  # Clean HTML from title
                    url = item.get('url', '')
                    hn_url = f"https://news.ycombinator.com/item?id={item.get('objectID', '')}"
                    
                    # Skip if no external URL
                    if not url:
                        continue
                    
                    # Check relevance
                    text_check = title.lower()
                    if not any(keyword in text_check for keyword in SYCOPHANCY_KEYWORDS):
                        continue
                    
                    # Get timestamp
                    timestamp = item.get('created_at_i', 0)
                    pub_date = datetime.fromtimestamp(timestamp) if timestamp else None
                    
                    # EXTRACT AND SUMMARIZE ARTICLE
                    print(f"      Summarizing: {title[:60]}...")
                    article_summary = extract_article_summary(url, f"HN discussion: {title}")
                    
                    # Extract populations
                    full_text = f"{title} {article_summary}".lower()
                    populations = []
                    for pop_type, keywords in POPULATION_KEYWORDS.items():
                        if any(keyword in full_text for keyword in keywords):
                            populations.append(pop_type)
                    
                    incidents.append({
                        'title': title,
                        'url': hn_url,  # Link to HN discussion
                        'summary': article_summary,
                        'source': 'Hacker News',
                        'publication_date': pub_date.strftime('%Y-%m-%d') if pub_date else '',
                        'date_found': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'vulnerable_populations': ', '.join(populations),
                        'needs_review': True,
                        'severity': '',
                        'status': 'discovered',
                        'reviewer_notes': f"HN Score: {item.get('points', 0)}, Comments: {item.get('num_comments', 0)}"
                    })
                    
                    time.sleep(2)  # Be respectful
            
            time.sleep(1)
    
    except Exception as e:
        print(f"Error searching Hacker News: {e}")
    
    print(f"    ✓ Found {len(incidents)} HN items with true summaries")
    return incidents


def parse_feed(feed_url: str, feed_name: str, days_back: int = 180) -> List[Dict]:
    """
    Parse an RSS feed and extract incidents
    NOW CREATES TRUE SUMMARIES
    """
    incidents = []
    
    try:
        feed = feedparser.parse(feed_url)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for entry in feed.entries:
            # Get publication date
            pub_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                pub_date = datetime(*entry.updated_parsed[:6])
            
            # Skip if too old
            if pub_date and pub_date < cutoff_date:
                continue
            
            # Get basic info
            title = clean_html(entry.get('title', ''))  # Clean HTML from title
            link = entry.get('link', '')
            
            # Check for sycophancy keywords in title
            title_lower = title.lower()
            if not any(keyword in title_lower for keyword in SYCOPHANCY_KEYWORDS):
                continue
            
            # Get RSS summary (often blank or minimal)
            rss_summary = ''
            if hasattr(entry, 'summary'):
                rss_summary = entry.summary
            elif hasattr(entry, 'description'):
                rss_summary = entry.description
            elif hasattr(entry, 'content'):
                rss_summary = entry.content[0].value if entry.content else ''
            
            # EXTRACT AND SUMMARIZE ARTICLE
            print(f"      Summarizing: {title[:60]}...")
            article_summary = extract_article_summary(link, rss_summary)
            
            # Extract vulnerable populations
            full_text = f"{title} {article_summary}".lower()
            populations = []
            for pop_type, keywords in POPULATION_KEYWORDS.items():
                if any(keyword in full_text for keyword in keywords):
                    populations.append(pop_type)
            
            incident = {
                'title': title,
                'url': link,
                'summary': article_summary,
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
            time.sleep(2)  # Be respectful between extractions
    
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


def scan_rss_feeds(feeds: Dict[str, str], days_back: int = 180) -> List[Dict]:
    """Scan all RSS feeds for recent incidents"""
    all_incidents = []
    
    print(f"\n  Scanning {len(feeds)} RSS feeds (last {days_back} days)...")
    print("     (Creating true summaries - this takes time)")
    
    for feed_name, feed_url in feeds.items():
        incidents = parse_feed(feed_url, feed_name, days_back)
        
        if incidents:
            all_incidents.extend(incidents)
        
        time.sleep(1.0)  # Be polite
    
    print(f"    ✓ Found {len(all_incidents)} relevant articles with true summaries")
    
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
    
    # Sort by relevance score (highest first), then by publication date
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    
    if 'relevance_score' in df.columns:
        df = df.sort_values(['relevance_score', 'publication_date'], 
                           ascending=[False, True])
    else:
        df = df.sort_values('publication_date', ascending=True)
    
    # Save to CSV with proper quoting to handle commas and special characters
    df.to_csv(output_file, index=False, quoting=1, escapechar='\\')  # quoting=1 is csv.QUOTE_ALL
    
    print(f"\n✓ Saved {len(incidents)} incidents to {output_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (2023-Present)")
    print("=" * 80)
    print(f"\nTotal unique incidents: {len(incidents)}")
    
    # Relevance score distribution
    if 'relevance_score' in df.columns:
        print("\nRelevance Score Distribution:")
        high_relevance = (df['relevance_score'] >= 80).sum()
        good_relevance = ((df['relevance_score'] >= 60) & (df['relevance_score'] < 80)).sum()
        medium_relevance = ((df['relevance_score'] >= 40) & (df['relevance_score'] < 60)).sum()
        low_relevance = (df['relevance_score'] < 40).sum()
        
        print(f"  - Highly relevant (80-100): {high_relevance} ({high_relevance/len(df)*100:.1f}%)")
        print(f"  - Relevant (60-79): {good_relevance} ({good_relevance/len(df)*100:.1f}%)")
        print(f"  - Somewhat relevant (40-59): {medium_relevance} ({medium_relevance/len(df)*100:.1f}%)")
        if low_relevance > 0:
            print(f"  - Low relevance (<40): {low_relevance} ({low_relevance/len(df)*100:.1f}%)")
        
        avg_score = df['relevance_score'].mean()
        print(f"\n  Average relevance score: {avg_score:.1f}")
    
    # Check summary quality
    summaries_extracted = (df['summary'] != 'Summary unavailable - article extraction failed').sum()
    summary_quality = (summaries_extracted / len(df)) * 100
    print(f"\nArticles with extracted summaries: {summaries_extracted} ({summary_quality:.1f}%)")
    
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
  # Full historical scan (2023-present) with true summarization
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
        default=180,
        help='Days back to scan for RSS feeds (default: 90)'
    )
    
    parser.add_argument(
        '--min-relevance',
        type=float,
        default=10.0,
        help='Minimum relevance score (0-100) to include articles (default: 10)'
    )
    
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Skip relevance filtering (include all articles regardless of score)'
    )
    
    args = parser.parse_args()
    
    # Check for required libraries
    if not NEWSPAPER_AVAILABLE:
        print("\n⚠️  WARNING: newspaper3k not installed!")
        print("   Install it for better article extraction: pip install newspaper3k")
        print("   Continuing with basic extraction...\n")
    
    if not SUMY_AVAILABLE:
        print("\n⚠️  WARNING: sumy not installed!")
        print("   Install it for better summarization: pip install sumy")
        print("   Using fallback summarization method...\n")
    
    all_incidents = []
    
    print("=" * 80)
    print("AI SYCOPHANCY INCIDENT SCANNER (2023-PRESENT)")
    print("WITH TRUE EXTRACTIVE SUMMARIZATION")
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
    
    # Step 4: Filter by relevance score
    print("\n[4/5] FILTERING BY RELEVANCE")
    print("-" * 80)
    
    if args.no_filter:
        print("  ⚠️  Skipping relevance filtering (--no-filter enabled)")
        # Still add scores but don't filter
        all_incidents = filter_by_relevance(all_incidents, min_score=0.0)
    else:
        print(f"  Calculating relevance scores (min threshold: {args.min_relevance})...")
        filtered_incidents = filter_by_relevance(all_incidents, min_score=args.min_relevance)
        
        removed_count = len(all_incidents) - len(filtered_incidents)
        print(f"  Removed {removed_count} low-relevance articles")
        print(f"  Kept {len(filtered_incidents)} relevant incidents")
        
        all_incidents = filtered_incidents
    
    # Step 5: Save results
    print("\n[5/5] SAVING RESULTS")
    print("-" * 80)
    
    save_results(all_incidents, args.output)


if __name__ == "__main__":
    main()
