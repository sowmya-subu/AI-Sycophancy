"""
Free AI Sycophancy Incident Scanner (2023-Present)
Uses RSS feeds, arXiv, and Hacker News for historical coverage

# Install dependencies
pip install -r requirements.txt
# Full scan (2023-present) - includes historical data
python scanner.py

# Custom output filename
python scanner.py --output my_complete_scan.csv
# Save console output to log file
python scanner.py --log scan_output.txt
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
import sys
import re
import arxiv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
from pathlib import Path


# Try to import sentence transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

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
# TAXONOMY FOR KEYWORD CLASSIFICATION
# ============================================================================

# Taxonomy categories for classifying incidents
# Each category has a list of representative descriptions for semantic matching
TAXONOMY = {
    # Category 1: Sycophancy Behavior Type
    'behavior_type': {
        'validation': [
            'AI validates user beliefs without question',
            'chatbot confirms delusions or false beliefs',
            'AI reinforces harmful user opinions',
            'system validates incorrect information',
            'AI agrees with user regardless of truth'
        ],
        'flattery': [
            'AI provides excessive compliments',
            'chatbot overly praises user',
            'AI flatters user excessively',
            'system gives too much positive feedback',
            'AI is overly complimentary'
        ],
        'agreement': [
            'AI agrees with everything user says',
            'chatbot never disagrees with user',
            'AI is too agreeable',
            'system always says yes',
            'AI people-pleasing behavior'
        ],
        'reward_hacking': [
            'AI optimizes for human approval',
            'system rewards user satisfaction over accuracy',
            'AI prioritizes user preference over truth',
            'chatbot seeks human approval',
            'AI reward hacking behavior'
        ]
    },
    
    # Category 2: Impact Domain
    'impact_domain': {
        'mental_health': [
            'AI provides mental health advice',
            'chatbot gives therapy or counseling',
            'AI discusses depression or anxiety',
            'system addresses psychological issues',
            'AI mental health support'
        ],
        'medical': [
            'AI gives medical advice',
            'chatbot provides diagnosis or treatment',
            'AI discusses medication or prescriptions',
            'system offers healthcare guidance',
            'AI medical consultation'
        ],
        'therapy': [
            'AI acts as therapist',
            'chatbot provides therapeutic support',
            'AI offers counseling services',
            'system gives therapy advice',
            'AI therapeutic intervention'
        ],
        'general': [
            'AI provides non-health related advice',
            'chatbot gives advice on career, relationships, or education',
            'AI offers everyday conversation or general guidance',
            'system provides advice outside health or therapy contexts',
            'AI sycophancy in general conversation or non-medical topics',
            'chatbot agrees with user on non-health matters',
            'AI validation in everyday interactions'
        ]
    },
    
    # Category 3: Severity Level
    'severity': {
        'critical': [
            'AI causes suicide or self-harm',
            'chatbot leads to life-threatening situation',
            'AI results in serious physical harm',
            'system causes critical health crisis',
            'AI leads to death or severe injury'
        ],
        'high': [
            'AI causes significant harm',
            'chatbot leads to hospitalization',
            'AI results in serious mental health crisis',
            'system causes major health problems',
            'AI leads to dangerous outcomes'
        ],
        'medium': [
            'AI provides harmful advice',
            'chatbot gives dangerous guidance',
            'AI causes moderate harm',
            'system leads to concerning outcomes',
            'AI results in negative consequences'
        ],
        'low': [
            'AI exhibits sycophantic behavior',
            'chatbot is overly agreeable',
            'AI shows people-pleasing tendencies',
            'system demonstrates flattery',
            'AI validation without harm'
        ]
    },
    
    # Category 4: Affected Population
    'population': {
        'mental_health_users': [
            'people with mental health conditions',
            'users with depression or anxiety',
            'individuals with psychiatric disorders',
            'people seeking mental health support',
            'users with psychological conditions'
        ],
        'medical_patients': [
            'people seeking medical advice',
            'patients looking for diagnosis',
            'individuals with health concerns',
            'people needing medical guidance',
            'users with medical questions'
        ],
        'minors': [
            'children or teenagers',
            'young people under 18',
            'students or adolescents',
            'minors using AI systems',
            'youth interacting with chatbots'
        ],
        'elderly': [
            'elderly or senior citizens',
            'older adults using AI',
            'seniors seeking assistance',
            'aging population',
            'elderly users'
        ],
        'general_public': [
            'general users',
            'public at large',
            'average consumers',
            'general population',
            'typical users'
        ]
    }
}

# ============================================================================
# SOURCE TIER MAPPING
# ============================================================================

# Source tier mapping dictionary
SOURCE_TIERS = {
    # Tier 1: Peer-reviewed research
    'tier_1': {
        'sources': ['arXiv'],
        'weight': 1.0,
        'description': 'Academic research papers'
    },
    
    # Tier 2: Established news outlets
    'tier_2': {
        'sources': [
            'TechCrunch', 'The Verge', 'Ars Technica', 
            'VentureBeat', 'Wired', 'MIT Tech Review AI',
            'Reuters Tech', 'BBC Technology', 'NPR Technology',
            'Partnership on AI'
        ],
        'weight': 0.8,
        'description': 'Established news and tech publications'
    },
    
    # Tier 3: Community sources
    'tier_3': {
        'sources': ['Hacker News'],
        'weight': 0.6,
        'description': 'Community discussions and forums'
    },
    
    # Tier 4: Aggregators and other
    'tier_4': {
        'sources': ['Google News'],  # Any source starting with "Google News"
        'weight': 0.5,
        'description': 'News aggregators and other sources'
    }
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
# LAST SCAN TIMESTAMP TRACKING
# ============================================================================

LAST_SCAN_FILE = '.last_scan_time'

def get_last_scan_time() -> datetime:
    """
    Get the timestamp of the last successful scan.
    Returns datetime(2023, 1, 1) if no previous scan found (first run).
    """
    if os.path.exists(LAST_SCAN_FILE):
        try:
            with open(LAST_SCAN_FILE, 'r', encoding='utf-8') as f:
                timestamp_str = f.read().strip()
                if timestamp_str:
                    # Parse ISO format timestamp
                    return datetime.fromisoformat(timestamp_str)
        except Exception as e:
            print(f" Could not read last scan time: {e}")
    
    # Default to Jan 1, 2023 for first run
    return datetime(2023, 1, 1)


def save_last_scan_time() -> None:
    """Save the current timestamp as the last scan time."""
    try:
        current_time = datetime.now()
        with open(LAST_SCAN_FILE, 'w', encoding='utf-8') as f:
            f.write(current_time.isoformat())
    except Exception as e:
        print(f" Could not save last scan time: {e}")


def get_last_scan_time_from_incidents(incidents: List[Dict]) -> datetime:
    """
    Get the most recent date_found from existing incidents.
    Falls back to get_last_scan_time() if no incidents or no dates.
    """
    if not incidents:
        return get_last_scan_time()
    
    # Find the most recent date_found
    max_date = None
    for incident in incidents:
        date_found = incident.get('date_found', '')
        if date_found:
            try:
                # Try parsing different date formats
                if 'T' in date_found or len(date_found) > 10:
                    # ISO format or datetime
                    dt = datetime.fromisoformat(date_found.replace(' ', 'T'))
                else:
                    # Date only
                    dt = datetime.strptime(date_found, '%Y-%m-%d')
                
                if max_date is None or dt > max_date:
                    max_date = dt
            except Exception:
                continue
    
    return max_date if max_date else get_last_scan_time()


# ============================================================================
# HISTORICAL DATA SOURCES
# ============================================================================


def search_arxiv_papers(since_date: Optional[datetime] = None) -> List[Dict]:
    """
    Search arXiv for sycophancy research papers since the given date.
    Uses arxiv library for cleaner API access and direct DOI extraction.
    
    Args:
        since_date: Only include papers published on or after this date.
                   If None, uses get_last_scan_time()
    """
    if since_date is None:
        since_date = get_last_scan_time()
    
    incidents = []
    seen_entries = set()
    
    date_str = since_date.strftime('%Y-%m-%d')
    print(f"\n  Searching arXiv for research papers (since {date_str})...")
    
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
        client = arxiv.Client()
        
        for query in search_queries:
            search = arxiv.Search(
                query=query,
                max_results=200,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for result in client.results(search):
                # Only include papers published on or after since_date
                if result.published >= since_date:
                    # Create unique key to avoid duplicates
                    entry_key = (
                        result.title,
                        result.published,
                        result.entry_id
                    )
                    
                    if entry_key not in seen_entries:
                        seen_entries.add(entry_key)
                        
                        # Extract DOI (directly available from result)
                        doi = result.doi if result.doi else ''
                        has_doi = bool(doi)
                        
                        # Extract populations from abstract
                        full_text = f"{result.title} {result.summary}".lower()
                        populations = []
                        for pop_type, keywords in POPULATION_KEYWORDS.items():
                            if any(keyword in full_text for keyword in keywords):
                                populations.append(pop_type)
                        
                        # Create a summary of the abstract using LSA
                        abstract_summary = summarize_text_lsa(result.summary.strip(), num_sentences=3)
                        
                        incidents.append({
                            'title': result.title.strip(),
                            'url': result.entry_id,
                            'summary': abstract_summary,
                            'source': 'arXiv',
                            'publication_date': result.published.strftime('%Y-%m-%d'),
                            'date_found': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'vulnerable_populations': ', '.join(populations),
                            'has_doi': has_doi,
                            'doi': doi,
                            'needs_review': True,
                            'severity': '',
                            'status': 'discovered',
                            'reviewer_notes': 'Research paper'
                        })
            
            time.sleep(3)  # Be respectful to arXiv
        
        print(f"    ✓ Found {len(incidents)} research papers since {date_str}")
    
    except Exception as e:
        print(f"Error searching arXiv: {e}")
    
    return incidents

def get_google_news_historical(since_date: Optional[datetime] = None) -> List[Dict]:
    """
    Search Google News RSS for articles since the given date.
    Uses date-specific searches to find articles.
    NOW CREATES TRUE SUMMARIES
    
    Args:
        since_date: Only include articles published on or after this date.
                   If None, uses get_last_scan_time()
    """
    if since_date is None:
        since_date = get_last_scan_time()
    
    incidents = []
    
    date_str = since_date.strftime('%Y-%m-%d')
    print(f"\n  Searching Google News for articles (since {date_str})...")
    print("     (Extracting and summarizing articles - this takes time)")
    
    # General searches (not year-specific)
    search_queries = [
        "AI sycophancy",
        "ChatGPT mental health",
        "AI chatbot dangerous",
        "ChatGPT therapy",
        "AI validation problem",
        "ChatGPT medication advice",
        "AI therapy harm",
        "chatbot emotional attachment",
        "AI validation delusion",
        "Replika AI emotional",
        "Character.AI harm",
        "AI flattery issue",
    ]
    
    for query in search_queries:
        try:
            feed_url = GOOGLE_NEWS_RSS_BASE.format(quote(query))
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:20]:  # Top 20 results per query
                try:
                    title = clean_html(entry.get('title', ''))  # Clean HTML from title
                    link = entry.get('link', '')
                    
                    # Check for sycophancy relevance
                    text_to_check = title.lower()
                    if not any(keyword in text_to_check for keyword in SYCOPHANCY_KEYWORDS):
                        continue
                    
                    # Parse publication date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # Skip if article is older than since_date
                    if pub_date and pub_date < since_date:
                        continue
                    
                    # EXTRACT AND SUMMARIZE ARTICLE
                    rss_summary = entry.get('summary', '') or entry.get('description', '')
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
            continue
    
    print(f"    ✓ Found {len(incidents)} articles since {date_str} with true summaries")
    return incidents


def get_hacker_news_historical(since_date: Optional[datetime] = None) -> List[Dict]:
    """
    Search Hacker News for discussions since the given date.
    Uses Algolia's free API
    NOW CREATES TRUE SUMMARIES FROM LINKED ARTICLES
    
    Args:
        since_date: Only include posts created on or after this date.
                   If None, uses get_last_scan_time()
    """
    if since_date is None:
        since_date = get_last_scan_time()
    
    incidents = []
    
    date_str = since_date.strftime('%Y-%m-%d')
    print(f"\n  Searching Hacker News archives (since {date_str})...")
    print("     (Summarizing linked articles)")
    
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
        # Unix timestamp for since_date
        start_timestamp = int(since_date.timestamp())
        
        for term in search_terms:
            api_url = f"https://hn.algolia.com/api/v1/search?query={quote(term)}&tags=story&numericFilters=created_at_i>{start_timestamp}"
            
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for hit in data.get('hits', []):
                    title = clean_html(hit.get('title', ''))  # Clean HTML from title
                    article_url = hit.get('url', '')
                    hn_url = f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
                    
                    # Skip if no external URL
                    if not article_url:
                        continue
                    
                    # Check relevance
                    text_check = title.lower()
                    if not any(keyword in text_check for keyword in SYCOPHANCY_KEYWORDS):
                        continue
                    
                    # Get timestamp
                    created_at = datetime.fromtimestamp(hit.get('created_at_i', 0))
                    
                    # EXTRACT AND SUMMARIZE ARTICLE
                    print(f"      Summarizing: {title[:60]}...")
                    article_summary = extract_article_summary(article_url, f"HN discussion: {title}")
                    
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
                        'publication_date': created_at.strftime('%Y-%m-%d') if created_at else '',
                        'date_found': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'vulnerable_populations': ', '.join(populations),
                        'hn_points': hit.get('points', 0),
                        'hn_comments': hit.get('num_comments', 0),
                        'needs_review': True,
                        'severity': '',
                        'status': 'discovered',
                        'reviewer_notes': f"HN Score: {hit.get('points', 0)}, Comments: {hit.get('num_comments', 0)}"
                    })
                    
                    time.sleep(2)  # Be respectful
            
            time.sleep(1)
        
        date_str = since_date.strftime('%Y-%m-%d')
        print(f"    ✓ Found {len(incidents)} HN items since {date_str} with true summaries")
    
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


def parse_feed(feed_url: str, feed_name: str, since_date: Optional[datetime] = None, max_days_back: int = 180) -> List[Dict]:
    """
    Parse RSS feed and extract relevant entries
    NOW CREATES TRUE SUMMARIES
    
    Args:
        feed_url: URL of RSS feed
        feed_name: Name of the feed source
        since_date: Only include entries published on or after this date.
                   If None, defaults to max_days_back days ago
        max_days_back: Maximum days to look back (default 180). If since_date is older
                      than this, will use max_days_back instead.
    
    Returns:
        List of incident dictionaries with true summaries
    """
    incidents = []
    # Calculate cutoff date with 180-day maximum lookback
    max_cutoff_date = datetime.now() - timedelta(days=max_days_back)
    
    if since_date is None:
        cutoff_date = max_cutoff_date
    else:
        # Use the more recent of: since_date or max_days_back ago
        cutoff_date = max(since_date, max_cutoff_date)
    
    try:
        feed = feedparser.parse(feed_url)
        
        for entry in feed.entries:
            # Get entry details
            title = clean_html(entry.get('title', ''))  # Clean HTML from title
            link = entry.get('link', '')
            
            # Parse publication date
            pub_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                pub_date = datetime(*entry.updated_parsed[:6])
            
            # Skip if too old
            if pub_date and pub_date < cutoff_date:
                continue
            
            # Check if relevant to sycophancy (check title first for efficiency)
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
            
            # Extract populations
            full_text = f"{title} {article_summary}".lower()
            populations = []
            for pop_type, keywords in POPULATION_KEYWORDS.items():
                if any(keyword in full_text for keyword in keywords):
                    populations.append(pop_type)
            
            # Create incident record
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


def scan_rss_feeds(feeds: Dict[str, str], since_date: Optional[datetime] = None, max_days_back: int = 180) -> List[Dict]:
    """
    Scan all RSS feeds for incidents since the given date.
    Maximum lookback is capped at max_days_back (default 180 days).
    """
    all_incidents = []
    
    # Calculate actual cutoff (capped at max_days_back)
    max_cutoff_date = datetime.now() - timedelta(days=max_days_back)
    if since_date:
        actual_cutoff = max(since_date, max_cutoff_date)
        date_str = actual_cutoff.strftime('%Y-%m-%d')
        if actual_cutoff > since_date:
            print(f"\n  Scanning {len(feeds)} RSS feeds (last {max_days_back} days - capped from {since_date.strftime('%Y-%m-%d')})...")
        else:
            print(f"\n  Scanning {len(feeds)} RSS feeds (since {date_str})...")
    else:
        print(f"\n  Scanning {len(feeds)} RSS feeds (last {max_days_back} days - fallback)...")
    print("     (Creating true summaries - this takes time)")
    
    for feed_name, feed_url in feeds.items():
        incidents = parse_feed(feed_url, feed_name, since_date, max_days_back)
        
        if incidents:
            all_incidents.extend(incidents)
        
        time.sleep(1.0)  # Be polite
    
    print(f"    ✓ Found {len(all_incidents)} relevant articles with true summaries")
    
    return all_incidents


# ============================================================================
# LOGGING SETUP
# ============================================================================

class TeeOutput:
    """Class to write output to both console and log file"""
    def __init__(self, log_file: Optional[str] = None):
        self.terminal = sys.stdout
        self.log_file = None
        if log_file:
            self.log_file = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()


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


def compute_relevancy_scores(incidents: List[Dict], use_embeddings: bool = True) -> List[Dict]:
    """
    Compute relevancy scores for each incident using semantic embeddings or TF-IDF.
    Scores how relevant each article is to AI sycophancy topics.
    
    Args:
        incidents: List of incident dictionaries
        use_embeddings: If True, use semantic embeddings (better but slower). 
                       If False or unavailable, fall back to TF-IDF.
    
    Returns incidents with 'relevancy_score' field added (0.0 to 1.0).
    """
    if not incidents:
        return incidents
    
    # Try to use embeddings if requested and available
    if use_embeddings and EMBEDDINGS_AVAILABLE:
        return _compute_scores_with_embeddings(incidents)
    else:
        if use_embeddings and not EMBEDDINGS_AVAILABLE:
            print("\n[4/7] COMPUTING RELEVANCY SCORES")
            print("-" * 80)
            print("  ⚠️  sentence-transformers not available, using TF-IDF instead")
            print("  Install with: pip install sentence-transformers")
        else:
            print("\n[4/7] COMPUTING RELEVANCY SCORES (TF-IDF)")
            print("-" * 80)
        return _compute_scores_with_tfidf(incidents)


def _compute_scores_with_embeddings(incidents: List[Dict]) -> List[Dict]:
    """Compute relevancy scores using semantic embeddings"""
    print("\n[4/7] COMPUTING RELEVANCY SCORES (Semantic Embeddings)")
    print("-" * 80)
    print("  Loading embedding model... (this may take a moment on first run)")
    
    try:
        # Use a lightweight, fast model optimized for semantic similarity
        # all-MiniLM-L6-v2 is small (~80MB) and fast
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  ✓ Model loaded")
        
        # Create a comprehensive query that describes what we're looking for
        query_text = (
            "AI sycophancy incidents where chatbots or language models provide "
            "harmful validation, excessive flattery, dangerous advice, or overly agreeable responses. "
            "This includes cases where AI systems validate delusions, encourage stopping medication, "
            "provide harmful mental health advice, or exhibit people-pleasing behavior that causes harm. "
            "Sycophantic AI behavior, reward hacking, and validation bias in language models."
        )
        
        # Combine title and summary for each article
        article_texts = []
        for incident in incidents:
            title = incident.get('title', '')
            summary = incident.get('summary', '')
            article_text = f"{title}. {summary}".strip()
            article_texts.append(article_text)
        
        print(f"  Encoding {len(article_texts)} articles...")
        
        # Generate embeddings for query and all articles
        query_embedding = model.encode([query_text], convert_to_numpy=True)
        article_embeddings = model.encode(article_texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Compute cosine similarity between query and each article
        similarity_scores = cosine_similarity(query_embedding, article_embeddings)[0]
        
        # Add relevancy_score to each incident
        for i, incident in enumerate(incidents):
            # Round to 4 decimal places for readability
            incident['relevancy_score'] = round(float(similarity_scores[i]), 4)
        
        # Print statistics
        scores = [inc['relevancy_score'] for inc in incidents]
        print(f"  ✓ Computed relevancy scores for {len(incidents)} incidents")
        print(f"  Score range: {min(scores):.4f} to {max(scores):.4f}")
        print(f"  Average score: {np.mean(scores):.4f}")
        print(f"  High relevancy (≥0.5): {sum(1 for s in scores if s >= 0.5)} incidents")
        print(f"  Medium relevancy (0.3-0.5): {sum(1 for s in scores if 0.3 <= s < 0.5)} incidents")
        print(f"  Low relevancy (<0.3): {sum(1 for s in scores if s < 0.3)} incidents")
        
    except Exception as e:
        print(f"  Error computing embeddings: {e}")
        print("  Falling back to TF-IDF...")
        return _compute_scores_with_tfidf(incidents)
    
    return incidents


def _compute_scores_with_tfidf(incidents: List[Dict]) -> List[Dict]:
    """Compute relevancy scores using TF-IDF (fallback method)"""
    # Create query document from sycophancy keywords
    # Repeat keywords to give them more weight in the query
    query_text = ' '.join(SYCOPHANCY_KEYWORDS * 3)  # Repeat 3x for emphasis
    
    # Combine title and summary for each article
    article_texts = []
    for incident in incidents:
        title = incident.get('title', '')
        summary = incident.get('summary', '')
        article_text = f"{title} {summary}".strip()
        article_texts.append(article_text)
    
    # Add query as first document, then all articles
    all_texts = [query_text] + article_texts
    
    try:
        # Initialize TF-IDF vectorizer
        # Use ngram_range=(1,2) to capture phrases, max_features to limit vocabulary size
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english',
            lowercase=True,
            min_df=1,  # Word must appear in at least 1 document
            max_df=0.95  # Ignore words that appear in >95% of documents
        )
        
        # Fit and transform all texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Extract query vector (first row) and article vectors (rest)
        query_vector = tfidf_matrix[0:1]
        article_vectors = tfidf_matrix[1:]
        
        # Compute cosine similarity between query and each article
        similarity_scores = cosine_similarity(query_vector, article_vectors)[0]
        
        # Add relevancy_score to each incident
        for i, incident in enumerate(incidents):
            # Round to 4 decimal places for readability
            incident['relevancy_score'] = round(float(similarity_scores[i]), 4)
        
        # Print statistics
        scores = [inc['relevancy_score'] for inc in incidents]
        print(f"  ✓ Computed relevancy scores for {len(incidents)} incidents")
        print(f"  Score range: {min(scores):.4f} to {max(scores):.4f}")
        print(f"  Average score: {np.mean(scores):.4f}")
        print(f"  High relevancy (≥0.3): {sum(1 for s in scores if s >= 0.3)} incidents")
        print(f"  Medium relevancy (0.1-0.3): {sum(1 for s in scores if 0.1 <= s < 0.3)} incidents")
        print(f"  Low relevancy (<0.1): {sum(1 for s in scores if s < 0.1)} incidents")
        
    except Exception as e:
        print(f"  Error computing relevancy scores: {e}")
        print("  Continuing without relevancy scores...")
        # Add default score if computation fails
        for incident in incidents:
            incident['relevancy_score'] = 0.0
    
    return incidents

def assign_source_tier(source: str) -> tuple[str, float]:
    """
    Assign tier and weight to a source.
    
    Returns:
        (tier_name: str, weight: float)
    """
    # Check if it's arXiv
    if source == 'arXiv':
        return 'tier_1', 1.0
    
    # Check if it's Hacker News
    if source == 'Hacker News':
        return 'tier_3', 0.6
    
    # Check if it's Google News (any variant)
    if source.startswith('Google News'):
        return 'tier_4', 0.5
    
    # Check if it's in tier 2 (established news)
    tier_2_sources = [
        'TechCrunch', 'The Verge', 'Ars Technica', 
        'VentureBeat', 'Wired', 'MIT Tech Review AI',
        'Reuters Tech', 'BBC Technology', 'NPR Technology',
        'Partnership on AI'
    ]
    if source in tier_2_sources:
        return 'tier_2', 0.8
    
    # Default to tier 4
    return 'tier_4', 0.5

def calculate_credibility_score(
    relevancy_score: float,
    source_tier: str,
    tier_weight: float,
    has_doi: bool,
    is_research: bool
) -> float:
    """
    Calculate credibility score combining relevancy, source tier, and DOI.
    
    Formula:
        credibility = (relevancy × tier_weight) + doi_bonus
    
    Where:
        - doi_bonus = 0.1 if has_doi and is_research, else 0.0
        - tier_weight varies by source tier
    """
    # Base credibility from relevancy weighted by source tier
    base_credibility = relevancy_score * tier_weight
    
    # DOI bonus (only for research papers)
    doi_bonus = 0.1 if (has_doi and is_research) else 0.0
    
    # Final credibility score
    credibility_score = base_credibility + doi_bonus
    
    return min(credibility_score, 1.0)

def add_credibility_scores(incidents: List[Dict]) -> List[Dict]:
    """
    Add credibility scores, source tiers, and DOI detection to incidents.
    This runs after relevancy scores are computed.
    """
    for incident in incidents:
        source = incident.get('source', '')
        
        # Assign source tier
        tier_name, tier_weight = assign_source_tier(source)
        incident['source_tier'] = tier_name
        incident['tier_weight'] = tier_weight
        
        # For arXiv papers, DOI is already extracted in search_arxiv_papers()
        # For other sources, try to detect DOI
        if source == 'arXiv':
            # Use DOI already extracted from arxiv library
            has_doi = incident.get('has_doi', False)
            doi_value = incident.get('doi', '')
        else:
            # Non-arXiv sources don't have DOIs
            has_doi = False
            doi_value = ''
        
        incident['has_doi'] = has_doi
        incident['doi'] = doi_value if has_doi else ''
        
        # Determine if it's research
        is_research = (source == 'arXiv')
        
        # Calculate credibility score
        relevancy = incident.get('relevancy_score', 0.0)
        credibility = calculate_credibility_score(
            relevancy_score=relevancy,
            source_tier=tier_name,
            tier_weight=tier_weight,
            has_doi=has_doi,
            is_research=is_research
        )
        
        incident['credibility_score'] = round(credibility, 4)
    
    return incidents


def classify_article_taxonomy(title: str, summary: str, model: Optional[SentenceTransformer] = None) -> List[str]:
    """
    Classify an article into taxonomy categories using sentence transformers.
    Returns a list of 2-3 taxonomy keywords.
    
    Args:
        title: Article title
        summary: Article summary
        model: Pre-loaded SentenceTransformer model (optional, will load if None)
    
    Returns:
        List of taxonomy keywords (e.g., ['validation', 'mental_health', 'high'])
    """
    if not EMBEDDINGS_AVAILABLE:
        # Fallback to keyword-based classification if embeddings not available
        return _classify_with_keywords(title, summary)
    
    # Load model if not provided
    if model is None:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"  Warning: Could not load embedding model for taxonomy: {e}")
            return _classify_with_keywords(title, summary)
    
    # Combine title and summary
    article_text = f"{title}. {summary}".strip()
    
    if len(article_text) < 50:
        return _classify_with_keywords(title, summary)
    
    try:
        # Encode the article
        article_embedding = model.encode([article_text], convert_to_numpy=True)
        
        selected_keywords = []
        
        # For each taxonomy category, find the best match
        for category_name, category_items in TAXONOMY.items():
            if category_name == 'population':
                # Skip population - we already extract this separately
                continue
            
            best_match = None
            best_score = -1.0
            
            # Encode all descriptions for this category
            descriptions = []
            labels = []
            for label, desc_list in category_items.items():
                for desc in desc_list:
                    descriptions.append(desc)
                    labels.append(label)
            
            if descriptions:
                desc_embeddings = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
                
                # Find best matching description
                similarities = cosine_similarity(article_embedding, desc_embeddings)[0]
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                best_match = labels[best_idx]
                
                # Only add if similarity is above threshold (0.3 for semantic matching)
                if best_score >= 0.3:
                    selected_keywords.append(best_match)
        
        # Limit to 2-3 keywords
        if len(selected_keywords) > 3:
            # Keep top 3 by taking first 3 (they're already sorted by category importance)
            selected_keywords = selected_keywords[:3]
        elif len(selected_keywords) < 2:
            # If we have fewer than 2, add from keyword-based fallback
            keyword_fallback = _classify_with_keywords(title, summary)
            for kw in keyword_fallback:
                if kw not in selected_keywords:
                    selected_keywords.append(kw)
                    if len(selected_keywords) >= 2:
                        break
        
        return selected_keywords[:3]  # Ensure max 3 keywords
        
    except Exception as e:
        print(f"  Warning: Taxonomy classification failed: {e}")
        return _classify_with_keywords(title, summary)


def _classify_with_keywords(title: str, summary: str) -> List[str]:
    """
    Fallback keyword-based classification when embeddings are unavailable.
    """
    full_text = f"{title} {summary}".lower()
    keywords = []
    
    # Behavior type classification
    if any(kw in full_text for kw in ['validate', 'validates', 'validated', 'validation', 'confirmed delusion']):
        keywords.append('validation')
    elif any(kw in full_text for kw in ['flatter', 'flattery', 'flattering', 'compliment', 'praise']):
        keywords.append('flattery')
    elif any(kw in full_text for kw in ['agree', 'agrees', 'agreement', 'people pleasing', 'too agreeable']):
        keywords.append('agreement')
    elif any(kw in full_text for kw in ['reward hack', 'reward hacking', 'human approval']):
        keywords.append('reward_hacking')
    
    # Impact domain classification
    if any(kw in full_text for kw in ['mental health', 'depression', 'suicide', 'therapy', 'psychosis', 'anxiety']):
        keywords.append('mental_health')
    elif any(kw in full_text for kw in ['medical', 'diagnosis', 'medication', 'prescription', 'doctor', 'patient']):
        keywords.append('medical')
    elif any(kw in full_text for kw in ['therapy', 'therapist', 'counseling', 'counselor']):
        keywords.append('therapy')
    else:
        keywords.append('general')
    
    # Severity classification
    if any(kw in full_text for kw in ['suicide', 'death', 'died', 'killed', 'fatal']):
        keywords.append('critical')
    elif any(kw in full_text for kw in ['hospital', 'hospitalized', 'crisis', 'emergency', 'serious harm']):
        keywords.append('high')
    elif any(kw in full_text for kw in ['harmful', 'dangerous', 'risk', 'problem', 'issue']):
        keywords.append('medium')
    else:
        keywords.append('low')
    
    return keywords[:3]  # Return max 3


def add_taxonomy_keywords(incidents: List[Dict], use_embeddings: bool = True) -> List[Dict]:
    """
    Add taxonomy keywords to each incident using semantic classification.
    
    Args:
        incidents: List of incident dictionaries
        use_embeddings: If True, use sentence transformers (better accuracy)
    
    Returns:
        Incidents with 'keywords' field added (list of 2-3 taxonomy terms)
    """
    if not incidents:
        return incidents
    
    print("\n[6/7] CLASSIFYING ARTICLES WITH TAXONOMY")
    print("-" * 80)
    
    # Load model once if using embeddings
    model = None
    if use_embeddings and EMBEDDINGS_AVAILABLE:
        try:
            print("  Loading embedding model for taxonomy classification...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded")
        except Exception as e:
            print(f"Could not load embedding model: {e}")
            print("Using keyword-based classification instead")
            use_embeddings = False
    
    classified_count = 0
    for i, incident in enumerate(incidents):
        title = incident.get('title', '')
        summary = incident.get('summary', '')
        
        # Classify article
        keywords = classify_article_taxonomy(title, summary, model)
        
        # Keywords are computed but not stored in output
        # (Commented out - keywords not written to CSV/JSON)
        # incident['keywords'] = ', '.join(keywords)
        # incident['keywords_array'] = keywords
        
        classified_count += 1
        if (i + 1) % 50 == 0:
            print(f"Classified {i + 1}/{len(incidents)} articles...")
    
    print(f"Classified {classified_count} articles with taxonomy keywords")
    
    # Print keyword distribution
    all_keywords = []
    for inc in incidents:
        all_keywords.extend(inc.get('keywords_array', []))
    
    if all_keywords:
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        print(f"\n  Top taxonomy keywords:")
        for kw, count in keyword_counts.most_common(10):
            print(f"    - {kw}: {count}")
    
    return incidents


def load_existing_incidents(csv_file: str, json_file: str) -> List[Dict]:
    """
    Load existing incidents from CSV or JSON file (prefer CSV if both exist).
    Returns empty list if neither file exists.
    """
    existing_incidents = []
    
    # Try to load from CSV first (more reliable)
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            # Convert DataFrame to list of dicts
            existing_incidents = df.to_dict('records')
            print(f"  Loaded {len(existing_incidents)} existing incidents from {csv_file}")
        except Exception as e:
            print(f"  ⚠️  Could not load existing CSV: {e}")
    
    # Fallback to JSON if CSV doesn't exist or failed
    elif os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_incidents = json.load(f)
            print(f"  Loaded {len(existing_incidents)} existing incidents from {json_file}")
        except Exception as e:
            print(f"  ⚠️  Could not load existing JSON: {e}")
    
    return existing_incidents


def merge_and_deduplicate(new_incidents: List[Dict], existing_incidents: List[Dict]) -> List[Dict]:
    """
    Merge new and existing incidents, removing duplicates by URL.
    If a URL appears in both:
    - Keep the one with the higher credibility_score
    - If credibility scores are equal, keep the older one (based on date_found)
    
    Args:
        new_incidents: Newly discovered incidents
        existing_incidents: Previously saved incidents
    
    Returns:
        Merged and deduplicated list of incidents
    """
    # Create a dictionary keyed by URL for fast lookup
    merged_dict = {}
    
    # First, add all existing incidents
    for incident in existing_incidents:
        url = incident.get('url', '')
        if url:
            merged_dict[url] = incident
    
    # Then, add/update with new incidents
    new_count = 0
    updated_count = 0
    kept_existing_count = 0
    
    for incident in new_incidents:
        url = incident.get('url', '')
        if url:
            if url in merged_dict:
                # Compare credibility scores
                existing_cred = merged_dict[url].get('credibility_score', 0.0)
                new_cred = incident.get('credibility_score', 0.0)
                
                # Convert to float if they're strings or None
                try:
                    existing_cred = float(existing_cred) if existing_cred else 0.0
                except (ValueError, TypeError):
                    existing_cred = 0.0
                
                try:
                    new_cred = float(new_cred) if new_cred else 0.0
                except (ValueError, TypeError):
                    new_cred = 0.0
                
                # Keep the one with higher credibility score
                if new_cred > existing_cred:
                    merged_dict[url] = incident
                    updated_count += 1
                elif new_cred < existing_cred:
                    # Keep existing (it has higher credibility)
                    kept_existing_count += 1
                else:
                    # Credibility scores are equal, keep the older one (earlier date_found)
                    existing_date = merged_dict[url].get('date_found', '')
                    new_date = incident.get('date_found', '')
                    
                    # Compare dates (older = earlier date)
                    if existing_date and new_date:
                        if existing_date < new_date:
                            # Keep existing (it's older)
                            kept_existing_count += 1
                        elif new_date < existing_date:
                            # Keep new (it's older)
                            merged_dict[url] = incident
                            updated_count += 1
                        else:
                            # Dates are equal, keep existing
                            kept_existing_count += 1
                    else:
                        # If dates can't be compared, keep existing
                        kept_existing_count += 1
            else:
                # New incident (not a duplicate)
                merged_dict[url] = incident
                new_count += 1
    
    merged_list = list(merged_dict.values())
    
    if existing_incidents:
        print(f"  Merged: {new_count} new, {updated_count} updated (better credibility), {kept_existing_count} kept existing (better/equal), {len(merged_list)} total unique incidents")
    
    return merged_list


def save_all_outputs(
    incidents: List[Dict],
    csv_file: str = 'sycophancy_incidents_2023_present.csv',
    json_file: str = 'incidents.json',
    stats_file: str = 'sycophancy_incidents_2023_present_stats.txt',
    allow_empty: bool = False,
) -> None:
    """
    Save incidents in both CSV and JSON formats and write a single statistics file.
    
    APPENDS to existing files: Loads existing incidents, merges with new ones,
    deduplicates by URL, and saves the combined result.

    - Builds and sorts a DataFrame once
    - Uses that for both CSV + JSON
    - Uses the same DataFrame for statistics
    """
    if not incidents:
        if not allow_empty:
            print("\n No incidents found!")
            print("  (Use allow_empty=True to write empty files)")
            return
        else:
            print("\n No incidents found - saving empty files as requested")
    
    # Load existing incidents and merge
    print("\n  Checking for existing incidents...")
    existing_incidents = load_existing_incidents(csv_file, json_file)
    
    if existing_incidents:
        # Merge new with existing, deduplicating by URL
        print(f"  Found {len(existing_incidents)} existing incidents, merging with {len(incidents)} new incidents...")
        all_incidents = merge_and_deduplicate(incidents, existing_incidents)
        print(f"  Merged result: {len(all_incidents)} total incidents")
        
        # Update last scan time based on most recent date_found in merged incidents
        # This ensures we don't miss articles if the .last_scan_time file is missing
        actual_last_scan = get_last_scan_time_from_incidents(all_incidents)
        if actual_last_scan > get_last_scan_time():
            # Update the last scan time file if we found a more recent date
            try:
                with open(LAST_SCAN_FILE, 'w', encoding='utf-8') as f:
                    f.write(actual_last_scan.isoformat())
            except Exception:
                pass
    else:
        # No existing file, just use new incidents
        all_incidents = incidents
        print(f"  No existing file found, saving {len(incidents)} new incidents")

    # Build DataFrame once
    df = pd.DataFrame(all_incidents)
    
    # Remove keywords fields from output if they exist
    if 'keywords' in df.columns:
        df = df.drop(columns=['keywords'])
    if 'keywords_array' in df.columns:
        df = df.drop(columns=['keywords_array'])

    # Ensure publication_date is datetime for sorting/statistics
    if 'publication_date' in df.columns:
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')

    # Sort by credibility + relevancy + date (if present)
    sort_cols = []
    ascending = []

    if 'credibility_score' in df.columns:
        sort_cols.append('credibility_score')
        ascending.append(False)
    if 'relevancy_score' in df.columns:
        sort_cols.append('relevancy_score')
        ascending.append(False)
    if 'publication_date' in df.columns:
        sort_cols.append('publication_date')
        ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)

    # -----------------
    # 1) Write CSV
    # -----------------
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved {len(all_incidents)} total incidents to {csv_file} ({len(incidents)} new this run)")

    # -----------------
    # 2) Write JSON
    # -----------------
    json_df = df.copy()

    # Convert publication_date back to string for JSON
    if 'publication_date' in json_df.columns:
        json_df['publication_date'] = json_df['publication_date'].dt.strftime('%Y-%m-%d')
    
    # Handle keywords: use keywords_array if available, otherwise parse keywords string
    if 'keywords_array' in json_df.columns:
        # Convert keywords_array to proper list format for JSON
        json_records = json_df.to_dict('records')
        for record in json_records:
            if 'keywords_array' in record and record['keywords_array']:
                # Use the array if it exists
                if isinstance(record['keywords_array'], list):
                    record['keywords'] = record['keywords_array']
                elif isinstance(record['keywords_array'], str):
                    # Parse if it's stored as string representation
                    try:
                        import ast
                        record['keywords'] = ast.literal_eval(record['keywords_array'])
                    except:
                        record['keywords'] = [k.strip() for k in record['keywords_array'].split(',') if k.strip()]
                else:
                    record['keywords'] = []
                # Remove keywords_array to avoid duplication
                if 'keywords_array' in record:
                    del record['keywords_array']
            elif 'keywords' in record and isinstance(record['keywords'], str):
                # Fallback: parse keywords string
                record['keywords'] = [k.strip() for k in record['keywords'].split(',') if k.strip()]
        
        # Clean NaN values (convert to None/null for JSON)
        import math
        for record in json_records:
            for key, value in record.items():
                # Convert NaN, NaT, and other pandas null values to None
                if isinstance(value, float) and math.isnan(value):
                    record[key] = None
                elif pd.isna(value):
                    record[key] = None
        
        # Write JSON with proper array format
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_records, f, indent=2, ensure_ascii=False)
    else:
        # No keywords_array, use standard pandas JSON export
        # Replace NaN with None for proper JSON null values
        json_df = json_df.where(pd.notna(json_df), None)
        json_df.to_json(json_file, orient='records', indent=2, date_format='iso')
    
    print(f"✓ Saved {len(all_incidents)} total incidents to {json_file} ({len(incidents)} new this run)")

    # -----------------
    # 3) Statistics
    # -----------------
    stats_content = generate_statistics(df, all_incidents)

    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(stats_content)

    print(f"✓ Saved statistics to {stats_file}")
    print(stats_content)

def generate_statistics(df: pd.DataFrame, incidents: List[Dict]) -> str:
    """Generate statistics report as a string
    
    Args:
        df: DataFrame with incidents (publication_date should be datetime)
        incidents: Original list of incidents
        
    Returns:
        Formatted statistics report as string
    """
    lines = []
    
    # Only show statistics if we have incidents
    if not incidents:
        lines.append("No statistics to display (empty dataset)")
        return "\n".join(lines)
    
    # Header
    lines.append("=" * 80)
    lines.append("SUMMARY STATISTICS (2023-Present)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append(f"\nTotal unique incidents: {len(incidents)}")
    
    # Timeline breakdown
    lines.append("\nIncidents by Year:")
    year_counts = df[df['publication_date'].notna()].groupby(df['publication_date'].dt.year).size()
    for year, count in year_counts.items():
        lines.append(f"  - {int(year)}: {count}")
    
    # Monthly breakdown for 2025
    if 2025 in year_counts.index:
        lines.append("\n2025 by Month:")
        df_2025 = df[df['publication_date'].dt.year == 2025]
        month_counts = df_2025.groupby(df_2025['publication_date'].dt.month).size()
        for month, count in month_counts.items():
            month_name = datetime(2025, int(month), 1).strftime('%B')
            lines.append(f"  - {month_name}: {count}")
    
    # Vulnerable populations
    if df['vulnerable_populations'].any():
        lines.append("\nVulnerable Populations Mentioned:")
        pop_counts = df['vulnerable_populations'].str.split(', ').explode().value_counts()
        for pop, count in pop_counts.head(10).items():
            if pop:
                lines.append(f"  - {pop}: {count}")
    
    # Top sources
    lines.append("\nTop Sources:")
    for source, count in df['source'].value_counts().head(15).items():
        lines.append(f"  - {source}: {count}")
    
    # Date range
    valid_dates = df[df['publication_date'].notna()]['publication_date']
    if not valid_dates.empty:
        earliest = valid_dates.min()
        latest = valid_dates.max()
        lines.append(f"\nDate Range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
    
    # Relevancy score statistics
    if 'relevancy_score' in df.columns:
        lines.append("\nRelevancy Score Statistics:")
        lines.append(f"  Average: {df['relevancy_score'].mean():.4f}")
        lines.append(f"  Median: {df['relevancy_score'].median():.4f}")
        lines.append(f"  Top 10 scores: {', '.join([f'{s:.4f}' for s in df['relevancy_score'].nlargest(10).values])}")
        lines.append(f"  Articles sorted by relevancy score (highest first)")
    
    # Credibility score statistics
    if 'credibility_score' in df.columns:
        lines.append("\nCredibility Score Statistics:")
        lines.append(f"  Average: {df['credibility_score'].mean():.4f}")
        lines.append(f"  Median: {df['credibility_score'].median():.4f}")
        lines.append(f"  Top 10 scores: {', '.join([f'{s:.4f}' for s in df['credibility_score'].nlargest(10).values])}")
    
        # Breakdown by source tier
        lines.append("\nCredibility by Source Tier:")
        for tier in ['tier_1', 'tier_2', 'tier_3', 'tier_4']:
            tier_df = df[df['source_tier'] == tier]
            if not tier_df.empty:
                avg_cred = tier_df['credibility_score'].mean()
                count = len(tier_df)
                tier_desc = SOURCE_TIERS[tier]['description']
                lines.append(f"  {tier} ({tier_desc}): {count} articles, avg credibility: {avg_cred:.4f}")
        
        # DOI statistics
        if 'has_doi' in df.columns:
            doi_count = df['has_doi'].sum()
            lines.append(f"\nResearch papers with DOI: {doi_count}")
            if doi_count > 0:
                doi_avg_cred = df[df['has_doi']]['credibility_score'].mean()
                no_doi_avg_cred = df[~df['has_doi'] & (df['source'] == 'arXiv')]['credibility_score'].mean()
                lines.append(f"  Average credibility with DOI: {doi_avg_cred:.4f}")
                if not pd.isna(no_doi_avg_cred):
                    lines.append(f"  Average credibility without DOI: {no_doi_avg_cred:.4f}")
        
    lines.append(f"  Articles sorted by credibility score (highest first)")
    
    lines.append("\n" + "=" * 80)
    lines.append("\nNext Steps:")
    lines.append("1. Review the JSON file and validate each incident")
    lines.append("2. Fill in 'severity' field (low/medium/high/critical)")
    lines.append("3. Update 'status' field as you review")
    lines.append("4. Add notes in 'reviewer_notes' field")
    lines.append("=" * 80)
    
    return "\n".join(lines)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Free RSS-based AI sycophancy scanner (2023-Present)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Incremental scan (from last scan time - production mode)
  python scanner.py
  
  # Full historical scan from 2023 (testing mode)
  python scanner.py --full-scan
  
  # Custom CSV output file
  python scanner.py --output my_incidents.csv
  
  # Save console output to log file
  python scanner.py --log scan_output.txt
  
  # Use TF-IDF instead of embeddings (faster)
  python scanner.py --use-tfidf
  
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
        help='Days back to scan for RSS feeds (default: 180)'
    )
    
    parser.add_argument(
        '--log',
        type=str,
        default=None,
        help='Save console output to log file (e.g., --log scan_output.txt)'
    )
    
    parser.add_argument(
        '--use-tfidf',
        action='store_true',
        help='Use TF-IDF instead of semantic embeddings (faster but less accurate)'
    )
    
    parser.add_argument(
        '--full-scan',
        action='store_true',
        help='Perform full historical scan from 2023 (ignores last scan time, for testing)'
    )
    
    args = parser.parse_args()
    
    # Derive stats filename from the CSV output name
    output_path = Path(args.output)
    base_stem = output_path.stem  # e.g. "sycophancy_incidents_2023_present"
    
    # For GitHub Pages / static site:
    #   - JSON lives next to index.html as "incidents.json"
    #   - stats file lives in the repo root as "<base>_stats.txt"
    json_output = Path("incidents.json")
    stats_output = Path(f"{base_stem}_stats.txt")
    
    # Set up logging if requested
    tee = None
    if args.log:
        tee = TeeOutput(args.log)
        sys.stdout = tee
        print(f"Logging output to: {args.log}\n")
    
    all_incidents = []
    
    try:
        print("=" * 80)
        print("AI SYCOPHANCY INCIDENT SCANNER (2023-PRESENT)")
        print("=" * 80)
        
        # Get last scan time (for incremental scanning)
        if args.full_scan:
            # Testing mode: scan from 2023-01-01 regardless of last scan time
            last_scan_time = datetime(2023, 1, 1)
            print(f"\n  [TESTING MODE] Full scan from 2023-01-01 (ignoring last scan time)")
        else:
            # Production mode: use last scan time for incremental scanning
            last_scan_time = get_last_scan_time()
            last_scan_str = last_scan_time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n  Last scan time: {last_scan_str}")
            print(f"  Scanning for new incidents since then...")
        
        # Step 1: Get historical data from free sources
        if not args.no_historical:
            print("\n[1/7] COLLECTING DATA FROM HISTORICAL SOURCES")
            print("-" * 80)
            
            google_historical = get_google_news_historical(since_date=last_scan_time)
            all_incidents.extend(google_historical)
            
            arxiv_incidents = search_arxiv_papers(since_date=last_scan_time)
            all_incidents.extend(arxiv_incidents)
            
            hn_incidents = get_hacker_news_historical(since_date=last_scan_time)
            all_incidents.extend(hn_incidents)
        
        # Step 2: Get recent data from RSS feeds
        print("\n[2/7] COLLECTING RECENT DATA (RSS Feeds)")
        print("-" * 80)
        
        all_feeds = RSS_FEEDS.copy()
        google_feeds = create_google_news_feeds(SEARCH_TERMS + HISTORICAL_SEARCHES)
        all_feeds.update(google_feeds)
        
        print(f"  Total RSS feeds to scan: {len(all_feeds)}")
        
        # RSS feeds behavior:
        # - Full scan mode: scan 180 days (use None to trigger 180-day default)
        # - Normal mode: use incremental scanning (last scan time)
        if args.full_scan:
            # Full scan: scan last 180 days
            rss_since_date = None  # None means use max_days_back (180 days)
        else:
            # Normal mode: incremental scanning from last scan time
            rss_since_date = last_scan_time
        
        rss_incidents = scan_rss_feeds(all_feeds, since_date=rss_since_date)
        all_incidents.extend(rss_incidents)
        
        # Step 3: Remove duplicates
        print("\n[3/7] DEDUPLICATING")
        print("-" * 80)
        print(f"  Total incidents before deduplication: {len(all_incidents)}")
        
        all_incidents = remove_duplicates(all_incidents)
        
        print(f"  Unique incidents: {len(all_incidents)}")
        
        # Check if there are existing incidents and if we found any new ones
        # Only load URLs for efficient duplicate checking
        existing_urls = set()
        existing_count = 0
        
        # Try to load just URLs from CSV (most efficient)
        if os.path.exists(args.output):
            try:
                df = pd.read_csv(args.output, usecols=['url'], dtype={'url': str})
                existing_urls = set(df['url'].dropna().astype(str))
                existing_count = len(df)
            except Exception:
                # Fallback: try JSON
                if os.path.exists(str(json_output)):
                    try:
                        with open(str(json_output), 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            if isinstance(existing_data, list) and len(existing_data) > 0:
                                existing_urls = {inc.get('url', '') for inc in existing_data if inc.get('url')}
                                existing_count = len(existing_data)
                    except Exception:
                        pass
        
        if existing_count > 0:
            # Check if all new incidents are duplicates
            new_incident_urls = {inc.get('url', '') for inc in all_incidents if inc.get('url')}
            new_urls = new_incident_urls - existing_urls
            
            if len(new_urls) == 0:
                # No new incidents found (either all duplicates or none found)
                print("\n" + "=" * 80)
                print("NO NEW ARTICLES FOUND")
                print("=" * 80)
                if len(all_incidents) > 0:
                    print(f"\n  No new incidents found since last scan ({last_scan_str})")
                    print(f"  All {len(all_incidents)} articles found were already in the database.")
                else:
                    print(f"\n  No new incidents found since last scan ({last_scan_str})")
                print(f"  Total incidents in database: {existing_count}")
                print("\n  The scanner will exit without updating files.")
                print("=" * 80)
                return
        
        # Step 4: Compute relevancy scores
        print("\n[4/7] COMPUTING RELEVANCY SCORES")
        all_incidents = compute_relevancy_scores(
            all_incidents,
            use_embeddings=not args.use_tfidf
        )

        # Step 5: Add credibility scores
        print("\n[5/7] COMPUTING CREDIBILITY SCORES")
        print("-" * 80)
        all_incidents = add_credibility_scores(all_incidents)
        print(f"Added credibility scores to {len(all_incidents)} incidents")

        # Step 6: Add taxonomy keywords
        print("\n[6/7] ADDING TAXONOMY KEYWORDS")
        print("-" * 80)
        all_incidents = add_taxonomy_keywords(
            all_incidents,
            use_embeddings=not args.use_tfidf  # Use embeddings if available (unless TF-IDF was requested)
        )

        # Step 6: Save results
        print("\n[7/7] SAVING RESULTS")
        print("-" * 80)

        save_all_outputs(
            incidents=all_incidents,
            csv_file=args.output,           # e.g. sycophancy_incidents_2023_present.csv
            json_file=str(json_output),     # e.g. incidents.json (next to index.html)
            stats_file=str(stats_output),   # e.g. sycophancy_incidents_2023_present_stats.txt
            allow_empty=False,
        )
        
        # Save the current scan time for next run (unless in testing mode)
        if not args.full_scan:
            save_last_scan_time()
            print(f"\n Saved scan timestamp for next run")
        else:
            print(f"\n  [TESTING MODE] Skipped saving scan timestamp (full scan mode)")

    finally:
        if tee:
            sys.stdout = tee.terminal
            tee.close()
            if args.log:
                print(f"\n✓ Output saved to: {args.log}")


if __name__ == "__main__":
    main()
