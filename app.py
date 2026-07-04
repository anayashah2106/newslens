"""
app.py
------
Flask entry point for the personalized news explainer app.

Pipeline per user:
  news_collector → personalizer → ai_processor + glossary → SQLite → feed.html

Routes:
  GET  /       — landing page (name + interest selection)
  POST /setup  — save user, redirect to /feed
  GET  /feed   — run full pipeline, render article cards
"""

from categorizer import classify
from newspaper import Article as NewsArticle
from dotenv import load_dotenv
load_dotenv()

import time
import json
import logging
import os
import sqlite3
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, redirect, render_template, request, session, url_for

from ai_processor import explain_article

from news_collector import collect_news
from personalizer import rank_articles

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# SECRET_KEY is required for session signing. In production, load this from
# an environment variable so it is not committed to source control.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-prod")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database bootstrap
#
# We use a single SQLite file for the whole app. Both tables are created here
# at startup so the rest of the code can assume they exist.
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("APP_DB_PATH", "news_app.db")


def get_db() -> sqlite3.Connection:
    """Open a SQLite connection with row_factory so rows behave like dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT    NOT NULL,
                interests TEXT    NOT NULL   -- comma-separated category names
            );

            CREATE TABLE IF NOT EXISTS articles (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER NOT NULL,
                title          TEXT    NOT NULL,
                link           TEXT,
                category       TEXT,
                source_domain TEXT,
                simple_summary TEXT,
                why_it_matters TEXT,
                background     TEXT,
                glossary_json  TEXT,           -- JSON object: {term: definition}
                processed_date TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
                           
            CREATE TABLE IF NOT EXISTS reactions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                article_id    INTEGER NOT NULL,
                category      TEXT,
                source_domain TEXT,
                reaction      INTEGER NOT NULL,   -- +1 thumbs up, -1 thumbs down
                reacted_at    TEXT,
                FOREIGN KEY (user_id)   REFERENCES users(id),
                FOREIGN KEY (article_id) REFERENCES articles(id),
                UNIQUE (user_id, article_id)      -- one reaction per article per user
            );
        """)


init_db()


# init_db()

# ── Startup check — confirms env vars are loaded before first request ──
# import google.generativeai as genai
# _startup_key = os.environ.get("GEMINI_API_KEY", "")
# if _startup_key:
#     print(f"[STARTUP] Gemini key loaded OK (ends in ...{_startup_key[-4:]})")
# else:
#     print("[STARTUP] WARNING: GEMINI_API_KEY is missing — summaries will be empty")

# ---------------------------------------------------------------------------
# Valid category choices (drives both the form and the collector)
# ---------------------------------------------------------------------------

ALL_CATEGORIES = [
    "Technology", "Politics", "Sports",
    "Science", "Business", "Environment","Entertainment", "Education",
]

# ---------------------------------------------------------------------------
# Core pipeline helper
#
# Extracted from the /feed route so the scheduler can call the exact same
# logic without duplicating code.
# ---------------------------------------------------------------------------

def run_pipeline(user_id: int, name: str, interests: list[str]) -> None:
    log.info("Pipeline start — user_id=%s interests=%s", user_id, interests)

    # Step 1 — fetch
    raw_articles = collect_news(interests)
    print(f"[DEBUG] collect_news returned {len(raw_articles)} articles")
    if raw_articles:
        print(f"[DEBUG] First article title: {raw_articles[0].get('title', '???')}")
        print(f"[DEBUG] First article summary length: {len(raw_articles[0].get('summary', ''))}")


   # Load this user's reaction history to apply feedback boosts
    with get_db() as conn:
        reaction_rows = conn.execute(
            "SELECT category, source_domain, reaction FROM reactions "
            "WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    feedback_rows = [dict(r) for r in reaction_rows]
    print(f"[DEBUG] Loaded {len(feedback_rows)} past reactions for user_id={user_id}")

    all_ranked = rank_articles(
        interests,
        raw_articles,
        top_n=len(raw_articles),
        feedback_rows=feedback_rows,
    )

    # Dynamic total based on how many categories the user selected —
    # more categories means more breadth is expected, so show more articles
    n_categories = max(len(interests), 1)
    if n_categories <= 2:
        TOTAL_ARTICLES = 15
    elif n_categories <= 4:
        TOTAL_ARTICLES = 20
    else:
        TOTAL_ARTICLES = 25

    print(f"[DEBUG] {n_categories} categories selected → targeting {TOTAL_ARTICLES} articles")

    slots_each = TOTAL_ARTICLES // n_categories
    remainder  = TOTAL_ARTICLES % n_categories

    # Group ranked articles by category, preserving relevance order within each
    from collections import defaultdict
    by_category = defaultdict(list)
    for art in all_ranked:
        by_category[art.get("category", "World News")].append(art)

    # Pull slots_each from each selected category in relevance order
    # Pull slots_each from each selected category in relevance order
    ranked = []
    shortfall = 0   # tracks unused slots from categories with too few articles
    for cat in interests:
        pool = by_category.get(cat, [])
        taken = pool[:slots_each]
        ranked.extend(taken)
        shortfall += (slots_each - len(taken))   # 0 if category had enough

    # Fill remainder slots PLUS any shortfall with the highest-scoring
    # articles not yet included — this redistributes unused slots from
    # empty/thin categories to whichever categories have surplus content
    included_links = {a.get("link") for a in ranked}
    extras = [a for a in all_ranked if a.get("link") not in included_links]
    ranked.extend(extras[:remainder + shortfall])

    # Final sort by relevance score so the feed reads best-first
    ranked.sort(key=lambda a: a.get("relevance_score", 0), reverse=True)
    ranked = ranked[:TOTAL_ARTICLES]

    print(f"[DEBUG] rank_articles returned {len(ranked)} articles (balanced)")
    for cat in interests:
        count = sum(1 for a in ranked if a.get("category") == cat)
        print(f"[DEBUG]   {cat}: {count} articles")
    today = datetime.now().strftime("%Y-%m-%d")

    with get_db() as conn:
        conn.execute(
            "DELETE FROM articles WHERE user_id = ? AND processed_date = ?",
            (user_id, today),
        )

        for i, article in enumerate(ranked):
            title    = article.get("title", "")
            link     = article.get("link", "")
            # Use the category assigned by news_collector (based on which
            # RSS feed the article came from) — more reliable than keyword matching
            # on short summaries. Map "India" → "Politics" for display consistency.
            _CATEGORY_MAP = {
                "India": "Politics",
            }
            source_category = _CATEGORY_MAP.get(
                article.get("category", "World News"),
                article.get("category", "World News"),
            )
            category = classify(
                title,
                article.get("summary", ""),
                source_category=source_category,
            )
            # Try to fetch full article text from the URL
            full_text = ""
            try:
                news_obj = NewsArticle(
                    article.get("link", ""),
                    browser_user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    request_timeout=10,
                )
                news_obj.download()
                news_obj.parse()
                full_text = news_obj.text.strip()
            except Exception:
                pass
            # Fall back to RSS summary if fetch failed
            article["full_text"] = full_text or article.get("summary", "") or title
            print(f"[DEBUG]   full_text after fetch: {len(article['full_text'])} chars")

            full_text_len = len(article.get("full_text", ""))
            print(f"\n[DEBUG] Article {i+1}/{len(ranked)}: '{title[:60]}'")
            print(f"[DEBUG]   category={category}, full_text length={full_text_len} chars")

            # AI explanation
            try:
                explanation = explain_article(article)
                print(f"[DEBUG]   simple_summary length : {len(explanation.get('simple_summary', ''))}")
                print(f"[DEBUG]   why_it_matters length : {len(explanation.get('why_it_matters', ''))}")
                print(f"[DEBUG]   background length     : {len(explanation.get('background', ''))}")
                if not explanation.get("simple_summary"):
                    print(f"[DEBUG]   ⚠ simple_summary is EMPTY — check Gemini key and full_text")
            except Exception as exc:
                print(f"[DEBUG]   ✗ explain_article RAISED: {exc}")
                explanation = {"simple_summary": "", "why_it_matters": "", "background": ""}

            # Glossary
            glossary= {}

            time.sleep(4)
            
            source_domain=article.get("source_domain","")
            conn.execute(
                """
                INSERT INTO articles
                    (user_id, title, link, category, source_domain,
                     simple_summary, why_it_matters, background,
                     glossary_json, processed_date)
                VALUES (?, ?, ?, ?,?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, title, link, category,source_domain,
                    explanation.get("simple_summary", ""),
                    explanation.get("why_it_matters", ""),
                    explanation.get("background", ""),
                    json.dumps(glossary),
                    today,
                ),
            )

    log.info("Pipeline complete — %d articles stored for user_id=%s", len(ranked), user_id)


# ---------------------------------------------------------------------------
# Route: GET /
# Landing page — collect name and interest preferences
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """
    Render the landing page.

    We pass ALL_CATEGORIES to the template so the checkbox list is driven
    by one source of truth rather than being hardcoded in HTML.
    """
    return render_template("index.html", categories=ALL_CATEGORIES)


# ---------------------------------------------------------------------------
# Route: POST /setup
# Save the user's profile and kick off the first pipeline run
# ---------------------------------------------------------------------------

@app.route("/setup", methods=["POST"])
def setup():
    """
    Validate form input, persist the user row, store user_id in session,
    then redirect to /feed which will immediately run the pipeline.

    We store interests as a comma-separated string in SQLite (simple and
    sufficient — no need for a junction table at this scale).
    """
    name = request.form.get("name", "").strip()
    # getlist returns all checked checkbox values for a given field name
    selected = request.form.getlist("interests")

    # Basic validation — bounce back to landing page on bad input
    if not name or not selected:
        return render_template(
            "index.html",
            categories=ALL_CATEGORIES,
            error="Please enter your name and select at least one interest.",
        )

    # Only accept categories from our known list (guards against form tampering)
    valid = [c for c in selected if c in ALL_CATEGORIES]
    if not valid:
        return render_template(
            "index.html",
            categories=ALL_CATEGORIES,
            error="Please select at least one valid interest category.",
        )

    interests_str = ",".join(valid)

    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO users (name, interests) VALUES (?, ?)",
            (name, interests_str),
        )
        user_id = cursor.lastrowid

    # Store only the user_id in the session — no sensitive data
    session["user_id"] = user_id
    session["user_name"] = name

    log.info("New user created — id=%s name=%s interests=%s", user_id, name, interests_str)

    return redirect(url_for("feed"))


# ---------------------------------------------------------------------------
# Route: GET /feed
# Run (or display) the personalised article feed for the logged-in user
# ---------------------------------------------------------------------------

@app.route("/feed", methods=["GET"])
def feed():
    """
    Full pipeline route.

    On each visit we check whether we already have articles for today.
    If yes — serve from cache (fast). If no — run the pipeline (slow, ~1 min
    for 15 articles with Gemini calls). This prevents redundant API calls
    when the user refreshes.

    The 'refresh' query param forces a re-run regardless of cache:
      /feed?refresh=1
    """
    user_id = session.get("user_id")
    if not user_id:
        # Not logged in — send them to the landing page
        return redirect(url_for("index"))

    # Load user row
    with get_db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()

    if not user:
        # User row missing (e.g. DB was wiped) — clear session and restart
        session.clear()
        return redirect(url_for("index"))

    interests = [i.strip() for i in user["interests"].split(",") if i.strip()]
    today = datetime.now().strftime("%Y-%m-%d")
    force_refresh = request.args.get("refresh") == "1"

    # Check for cached articles from today
    with get_db() as conn:
        cached = conn.execute(
            "SELECT * FROM articles WHERE user_id = ? AND processed_date = ? ORDER BY id",
            (user_id, today),
        ).fetchall()

    if not cached or force_refresh:
        log.info("No cache or refresh requested — running pipeline for user_id=%s", user_id)
        run_pipeline(user_id, user["name"], interests)

        with get_db() as conn:
            cached = conn.execute(
                "SELECT * FROM articles WHERE user_id = ? AND processed_date = ? ORDER BY id",
                (user_id, today),
            ).fetchall()

    # Convert Row objects to plain dicts and parse glossary_json
    articles = []
    for row in cached:
        art = dict(row)
        try:
            art["glossary"] = json.loads(art.get("glossary_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            art["glossary"] = {}
        articles.append(art)

    return render_template(
        "feed.html",
        articles=articles,
        user_name=session.get("user_name", ""),
        interests=interests,
        today=today,
    )



# ---------------------------------------------------------------------------
# Route: POST /react
# Store a thumbs up or thumbs down reaction for one article
# ---------------------------------------------------------------------------

@app.route("/react", methods=["POST"])
def react():
    """
    Receive a reaction (thumbs up/down) via JSON POST and store it.

    Expected JSON body: { "article_id": 42, "reaction": 1 }
    reaction: +1 = thumbs up, -1 = thumbs down

    Returns JSON: { "status": "ok" } or { "status": "error" }

    This route is called by the JavaScript fetch() in feed.html —
    the user never leaves the page when clicking thumbs up/down.
    """
    user_id = session.get("user_id")
    if not user_id:
        return {"status": "error", "message": "not logged in"}, 401

    data       = request.get_json()
    article_id = data.get("article_id")
    reaction   = data.get("reaction")   # +1 or -1

    if reaction not in (1, -1) or not article_id:
        return {"status": "error", "message": "invalid payload"}, 400

    # Fetch category and source_domain for boost calculation later
    with get_db() as conn:
        article = conn.execute(
            "SELECT category, link FROM articles WHERE id = ? AND user_id = ?",
            (article_id, user_id)
        ).fetchone()

        if not article:
            return {"status": "error", "message": "article not found"}, 404

        # Extract domain from link for source-level boost
        from urllib.parse import urlparse
        source_domain = urlparse(article["link"]).netloc.replace("www.", "")

        # INSERT OR REPLACE so changing your mind (up → down) updates the row
        conn.execute("""
            INSERT INTO reactions (user_id, article_id, category, source_domain,
                                   reaction, reacted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, article_id) DO UPDATE SET
                reaction   = excluded.reaction,
                reacted_at = excluded.reacted_at
        """, (
            user_id,
            article_id,
            article["category"],
            source_domain,
            reaction,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))

    log.info("Reaction stored — user_id=%s article_id=%s reaction=%s",
             user_id, article_id, reaction)
    return {"status": "ok"}



# ---------------------------------------------------------------------------
# APScheduler — daily refresh at 07:00
#
# Runs in a background thread alongside the Flask dev server.
# In production (Gunicorn/uWSGI), use a separate process scheduler
# (e.g. Celery Beat or a system cron) to avoid duplicate runs across workers.
# ---------------------------------------------------------------------------

def scheduled_refresh() -> None:
    """Re-run the pipeline for every user in the database."""
    log.info("Scheduled refresh starting...")
    with get_db() as conn:
        users = conn.execute("SELECT id, name, interests FROM users").fetchall()

    for user in users:
        interests = [i.strip() for i in user["interests"].split(",") if i.strip()]
        try:
            run_pipeline(user["id"], user["name"], interests)
        except Exception as exc:
            log.error("Scheduled refresh failed for user_id=%s: %s", user["id"], exc)

    log.info("Scheduled refresh complete for %d users", len(users))


scheduler = BackgroundScheduler()
scheduler.add_job(
    func=scheduled_refresh,
    trigger="cron",
    hour=7,
    minute=0,
    id="daily_refresh",
    replace_existing=True,
)
scheduler.start()

# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # debug=False because APScheduler + Flask reloader spawn duplicate
    # scheduler threads; disable the reloader if you need debug mode.
    app.run(host="0.0.0.0", port=5000, debug=False)
