"""
news_collector.py
-----------------
A module for fetching and deduplicating news articles from RSS feeds
based on user-specified interest categories.

Dependencies: feedparser (pip install feedparser)
"""

import html
import feedparser
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# 1. RSS Feed Registry
# ---------------------------------------------------------------------------

CATEGORY_FEEDS = {
    "Technology": [
        "https://feeds.feedburner.com/TechCrunch",
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
    ],
    "Politics": [
        "https://feeds.feedburner.com/ndtvnews-india-news",
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://indianexpress.com/feed/",
    ],
    "India": [
        "https://feeds.feedburner.com/ndtvnews-india-news",
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://indianexpress.com/feed/",
    ],
    "Business": [
        "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
        "https://www.livemint.com/rss/news",
    ],
    "Sports": [
        "https://feeds.feedburner.com/ndtvnews-sports",
        "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
    ],
    "Science": [
        "https://www.sciencedaily.com/rss/top/science.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    ],
    "Environment": [
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://www.theguardian.com/environment/rss",
    ],
    "Entertainment": [
        "https://feeds.feedburner.com/ndtvmovies-latest",
        "https://indianexpress.com/section/entertainment/feed/",
    ],
    "Education": [
        "https://indianexpress.com/section/education/feed/",
        "https://feeds.feedburner.com/ndtv-education",
    ],
}

MAX_TOTAL_ARTICLES = 100
MAX_PER_CATEGORY   = 20
DUPLICATE_THRESHOLD = 0.70
RECENCY_WINDOW_HOURS = 48


# ---------------------------------------------------------------------------
# 2. Date parser
# ---------------------------------------------------------------------------

def _parse_date(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            dt = datetime(*entry.published_parsed[:6])
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            pass
    return getattr(entry, "published", "Unknown")


# ---------------------------------------------------------------------------
# 3. Tokeniser for deduplication
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    import re
    words = re.findall(r"[a-z0-9]+", text.lower())
    return set(words)


# ---------------------------------------------------------------------------
# 4. Duplicate checker (Jaccard similarity)
# ---------------------------------------------------------------------------

def _are_duplicates(title_a: str, title_b: str) -> bool:
    words_a = _tokenize(title_a)
    words_b = _tokenize(title_b)
    if not words_a and not words_b:
        return True
    union        = words_a | words_b
    intersection = words_a & words_b
    return (len(intersection) / len(union)) > DUPLICATE_THRESHOLD


# ---------------------------------------------------------------------------
# 5. Deduplicator
# ---------------------------------------------------------------------------

def _deduplicate(articles: list) -> list:
    unique = []
    for candidate in articles:
        if not any(_are_duplicates(candidate["title"], a["title"]) for a in unique):
            unique.append(candidate)
    return unique


# ---------------------------------------------------------------------------
# 6. Recency filter
# ---------------------------------------------------------------------------

def is_recent(entry) -> bool:
    if not hasattr(entry, "published_parsed") or entry.published_parsed is None:
        return False
    try:
        published_dt = datetime(*entry.published_parsed[:6])
    except (TypeError, ValueError):
        return False
    return (datetime.utcnow() - published_dt) <= timedelta(hours=RECENCY_WINDOW_HOURS)


# ---------------------------------------------------------------------------
# 7. Feed fetcher
# ---------------------------------------------------------------------------

def _fetch_feed(feed_url: str, category: str) -> list:
    articles = []

    INDIAN_DOMAINS = {
        "inc42.com", "yourstory.com", "thehindu.com",
        "hindustantimes.com", "indianexpress.com", "ndtv.com",
        "economictimes.indiatimes.com", "livemint.com",
        "business-standard.com", "moneycontrol.com",
        "espncricinfo.com", "timesofindia.indiatimes.com",
        "thewire.in", "downtoearth.org.in", "indiaclimatedialogue.net",
    }

    # Extract source domain ONCE from the feed URL — used for priority
    # and for feedback boost calculation in personalizer.py
    source_domain = urlparse(feed_url).netloc.replace("www.", "")

    try:
        feed = feedparser.parse(feed_url)

        if feed.bozo:
            print(f"  [WARN] Malformed feed at {feed_url}: {feed.bozo_exception}")

        for entry in feed.entries:
            if not is_recent(entry):
                continue

            title = html.unescape(getattr(entry, "title", "")).strip()
            if not title:
                continue

            summary = html.unescape(
                getattr(entry, "summary", "") or getattr(entry, "description", "")
            ).strip()

            link      = getattr(entry, "link", "").strip()
            published = _parse_date(entry)
            priority  = "local" if any(d in feed_url for d in INDIAN_DOMAINS) else "international"

            articles.append({
                "title":         title,
                "summary":       summary,
                "link":          link,
                "published":     published,
                "category":      category,
                "priority":      priority,
                "source_domain": source_domain,   # ← properly defined here
            })

    except Exception as exc:
        print(f"  [ERROR] Failed to fetch {feed_url}: {exc}")

    return articles


# ---------------------------------------------------------------------------
# 8. Public API
# ---------------------------------------------------------------------------

def collect_news(interests: list) -> list:
    all_articles = []

    for category in interests:
        category = category.strip().title()

        if category not in CATEGORY_FEEDS:
            print(f"[SKIP] Unknown category '{category}'. "
                  f"Available: {list(CATEGORY_FEEDS.keys())}")
            continue

        print(f"[INFO] Fetching '{category}' articles...")
        feed_urls = CATEGORY_FEEDS[category]

        category_articles = []
        for url in feed_urls:
            print(f"  -> {url}")
            fetched = _fetch_feed(url, category)
            print(f"     {len(fetched)} articles retrieved")
            category_articles.extend(fetched)

        capped = category_articles[:MAX_PER_CATEGORY]
        print(f"     {len(capped)} kept after per-category cap")
        all_articles.extend(capped)

        time.sleep(0.5)

    print(f"\n[INFO] Total raw articles fetched: {len(all_articles)}")
    unique   = _deduplicate(all_articles)
    print(f"[INFO] After deduplication: {len(unique)} articles")
    final    = unique[:MAX_TOTAL_ARTICLES]
    print(f"[INFO] Returning {len(final)} articles (cap: {MAX_TOTAL_ARTICLES})\n")
    return final


if __name__ == "__main__":
    results = collect_news(["Technology", "Sports", "Politics"])
    for i, a in enumerate(results[:5], 1):
        print(f"[{i}] [{a['category']}] [{a['source_domain']}] {a['title']}")
