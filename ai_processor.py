"""
ai_processor.py
---------------
Generates plain-English explanations for news articles using only
local, offline models — no external API calls whatsoever.

Pipeline:
  SIMPLE SUMMARY   → sumy (LexRank) extracts key sentences
                     → HuggingFace DistilBART turns them into fluent prose
  BACKGROUND       → spaCy finds the main topic noun from the title
                     → wikipedia library fetches the opening sentences
  WHY IT MATTERS   → keyword scan finds impact-related sentences

Dependencies:
    pip install sumy transformers torch wikipedia spacy
    python -m spacy download en_core_web_sm

⚠ LARGE DOWNLOAD WARNING:
    'sshleifer/distilbart-cnn-12-6' is ~1.2 GB and is downloaded once
    from HuggingFace the first time this file is imported. After that
    it is cached locally and loads in a few seconds.
"""
import os
from groq import Groq
import re
import spacy
import wikipedia

# sumy: a library for automatic text summarization
# LexRank is a graph-based algorithm — it finds sentences that are most
# "central" to the document, like finding the most-linked page on Wikipedia
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# transformers: HuggingFace library for running pretrained AI models locally
# pipeline() is the simplest way to use a model — it handles tokenization,
# inference, and decoding all in one call
# from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# Load heavy models ONCE at module level
#
# Loading a neural network model takes 3-10 seconds and uses ~1.5 GB RAM.
# If we loaded it inside the function, we'd pay that cost for every single
# article. Loading here means we pay it once when the app starts.
# ---------------------------------------------------------------------------

print("[ai_processor] Loading DistilBART summarization model...")
# sshleifer/distilbart-cnn-12-6 is a distilled (smaller, faster) version
# of Facebook's BART model, fine-tuned on CNN/DailyMail news articles —
# making it well-suited for summarizing exactly this kind of content.
# _bart_summarizer = hf_pipeline(
#     "summarization",
#     model="sshleifer/distilbart-cnn-12-6",
# )
# print("[ai_processor] Model loaded OK")

# spaCy: used here to find the main topic word in the article title
print("[ai_processor] Loading spaCy model...")
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )
print("[ai_processor] spaCy loaded OK")

# Words that signal a sentence is about impact or significance.
# We scan article sentences for these to find the "why it matters" content.
_IMPACT_WORDS = {
    "impact", "affect", "mean", "result", "lead", "cause", "change",
    "future", "major", "significant", "important", "critical", "historic",
    "first", "record", "risk", "threat", "concern", "opportunity",
    "boost", "cut", "rise", "fall", "ban", "allow", "plan", "demand",
    "supply", "rare", "never", "crisis", "emergency", "warning",
}


# Boilerplate phrases that indicate non-news content
_BOILERPLATE = [
    "subscribe", "newsletter", "sign up", "follow us", "editor at",
    "staff writer", "senior writer", "covered the", "years of experience",
    "posted in", "read more", "advertisement", "cookie", "privacy policy",
    "terms of service", "all rights reserved", "click here",
    "this article", "daily digest", "homepage feed", "email digest",
]

# Pronouns that signal a byline continuation when in the first 3 sentences
_BYLINE_PRONOUNS = {"he", "she", "they", "his", "her"}



def build_context_opener(title: str, category: str) -> str:
    """
    Build a clean, complete opening sentence from the article title.

    RSS feeds sometimes only give the tail end of a developing story
    ("...stepped up efforts to find the missing"). Prepending a title-based
    opener guarantees the summarizer always has full context to work with,
    even when the scraped article text starts mid-thought.
    """
    clean_title = re.sub(
        r'^(LIVE\s+updates?|Breaking|Explained|Watch|Read|Opinion)\s*:\s*',
        '', title, flags=re.IGNORECASE
    ).strip()

    return f"Here is what happened: {clean_title}."


def _validate_first_sentence(text: str, title: str, category: str) -> str:
    """
    Check whether the first sentence of generated text is coherent.
    Replaces it with the title-based opener if it fails any check —
    guarantees the summary never starts mid-thought.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        return text

    first = sentences[0]

    # Check 1: starts with a capital letter
    starts_capital = first[0].isupper() if first else False

    # Check 2: contains a subject (PROPN or NOUN) in first 4 tokens
    doc = _nlp(first)
    tokens = [t for t in doc if not t.is_space][:4]
    has_subject = any(t.pos_ in ("PROPN", "NOUN") for t in tokens)

    # Check 3: long enough to be a real sentence
    is_long_enough = len(first) > 40

    if starts_capital and has_subject and is_long_enough:
        return text   # first sentence is fine, leave as-is

    # Failed validation — replace first sentence with the context opener
    opener = build_context_opener(title, category)
    return opener + " " + " ".join(sentences[1:])

def _strip_leading_topic_colon(text: str) -> str:
    """
    Remove "Topic: " prefixes that bleed in from RSS titles, e.g.
    "Arunachal Floods: The district administration has..." becomes
    "The district administration has..."
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned = [
        re.sub(r'^[A-Za-z\s]+:\s*', '', sent) for sent in sentences
    ]
    return " ".join(cleaned).strip()


def clean_article_text(text: str, title: str = "", category: str = "General") -> str:
    import html as html_lib
    text = html_lib.unescape(text)

    doc = _nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    original_count = len(sentences)

    clean = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        if any(phrase in sent_lower for phrase in _BOILERPLATE):
            continue
        if len(sent) < 30:
            continue
        first_word = sent.split()[0].lower() if sent.split() else ""
        if i < 3 and first_word in _BYLINE_PRONOUNS:
            continue
        clean.append(sent)

    removed_fraction = 1 - (len(clean) / original_count) if original_count else 0
    if removed_fraction > 0.70:
        print(f"[WARN] clean_article_text: removed {removed_fraction:.0%} — falling back")
        result = text
    else:
        result = " ".join(clean)

    # Only prepend the opener if the result is genuinely empty
    # — not on every article, which causes it to bleed into summaries
    if title and not result.strip():
        opener = build_context_opener(title, category)
        return f"{opener} {result}"

    return result

# ---------------------------------------------------------------------------
# Helper: extract the N most important sentences using LexRank
# ---------------------------------------------------------------------------

def _extract_key_sentences(text: str, n: int = 5) -> str:
    """
    Use sumy's LexRank to pull out the most central sentences from text.

    LexRank works like Google's PageRank but for sentences:
    sentences that are similar to many other sentences in the document
    are considered more important and ranked higher.

    Returns a single string of the top N sentences joined together,
    which is then fed into the abstractive summarizer.
    """
    # PlaintextParser splits the text into sentences and words
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    # Get the N most important sentences (as sumy Sentence objects)
    key_sentences = summarizer(parser.document, sentences_count=n)

    # Join them into one block of text for the next stage
    return " ".join(str(s) for s in key_sentences)


# ---------------------------------------------------------------------------
# Helper: generate a fluent summary using DistilBART
# ---------------------------------------------------------------------------

def _abstractive_summary(text: str, max_length: int = 130,
                          min_length: int = 30) -> str:
    """
    Returns the extractive sentences from sumy directly.
    DistilBART was removed due to compatibility issues with Python 3.13.
    LexRank output is already clean enough for display.
    """
    if not text.strip():
        return ""

    # Truncate to a reasonable length for display
    words = text.split()
    if len(words) > max_length:
        text = " ".join(words[:max_length])

    return text.strip()

def clean_summary(text: str, fallback_sentences: str = "") -> str:
    """
    Clean DistilBART output by removing hashtags, symbols, duplicate
    sentences, and fragments before returning to the user.

    Falls back to the LexRank extractive sentences if the cleaned
    output is too short to be useful.
    """
    import re

    # Remove hashtag tokens (#, ##, ###...)
    text = re.sub(r'#\S*', '', text)

    # Remove any character that isn't a letter, digit, space,
    # or standard punctuation — catches random Unicode symbols
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)

    # Split into sentences, drop fragments under 20 chars
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= 20]

    # Remove duplicate sentences (model sometimes repeats itself)
    seen = set()
    unique = []
    for s in sentences:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    result = " ".join(unique).strip()

    # If fewer than 2 sentences survived, fall back to extractive output
    if len(unique) < 2 and fallback_sentences:
        return fallback_sentences.strip()

    return result

# ---------------------------------------------------------------------------
# Helper: find the main topic of the article from its title
# ---------------------------------------------------------------------------

# Pages that are useless as background context — direction names,
# disambiguation pages, name-meaning pages, etc.
_WIKI_BLACKLIST = [
    "cardinal direction", "compass", "given name",
    "surname", "disambiguation", "list of",
]


def _extract_main_topic(title: str) -> str:
    """
    Extract ALL meaningful proper nouns and nouns from the title
    and join them into a multi-word search query.

    e.g. "South Korea plans to train military as drone warriors"
         → "South Korea military drone warriors"

    A multi-word query gives wikipedia.search() enough context to
    return relevant results instead of matching the first word only.
    """
    doc = _nlp(title)
    tokens = []

    for token in doc:
        if token.pos_ in ("PROPN", "NOUN") and not token.is_stop:
            tokens.append(token.text)

    # Join all meaningful tokens — more context = better search results
    return " ".join(tokens) if tokens else title


def _get_wikipedia_background(topic: str) -> str:
    """
    Search Wikipedia with the full topic query and validate the result
    before using it, to avoid irrelevant pages like "South (direction)".

    Strategy:
      1. wikipedia.search() returns a ranked list of matching page titles
      2. Try up to the first 3 results
      3. For each: check that at least one title word appears in the
         Wikipedia page title (relevance check)
      4. Check that the page's first sentence isn't on the blacklist
      5. Return the first 2 sentences of the first valid page found
    """
    if not topic:
        return ""

    try:
        # search() returns a list of Wikipedia page titles ranked by relevance
        search_results = wikipedia.search(topic, results=5)
    except Exception:
        return ""

    # Words from the original article topic query — used for validation
    topic_words = set(topic.lower().split())

    for page_title in search_results[:3]:
        try:
            summary = wikipedia.summary(page_title, sentences=3,
                                        auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first disambiguation option
            try:
                summary = wikipedia.summary(e.options[0], sentences=3,
                                            auto_suggest=False)
                page_title = e.options[0]
            except Exception:
                continue
        except Exception:
            continue

        # --- Relevance check ---
        # Require at least 2 overlapping words (not just 1) — a single
        # word match like "swift" can coincidentally match a famous but
        # unrelated entity (e.g. Taylor Swift instead of the bird).
        page_title_words = set(page_title.lower().split())
        overlap = topic_words & page_title_words
        if len(overlap) < 2 and len(topic_words) > 1:
            continue

        # --- Blacklist check ---
        # Skip pages that are about directions, names, or disambiguation
        first_sentence = summary.split('.')[0].lower()
        if any(bad in first_sentence for bad in _WIKI_BLACKLIST):
            continue

        # Valid page found — return first 2 sentences only
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        return ". ".join(sentences[:2]) + "."

    return ""

# ---------------------------------------------------------------------------
# Helper: find "why it matters" sentences by scanning for impact words
# ---------------------------------------------------------------------------

def _extract_why_it_matters(text: str) -> str:
    """
    Scan the article sentence by sentence and return the first 2 sentences
    that contain any word from _IMPACT_WORDS.

    This works because journalists typically use words like "significant",
    "historic", or "affect" precisely when explaining why something matters.

    Falls back to the last 2 sentences if no impact words are found —
    news articles often end with a forward-looking statement of significance.
    """
    # Split text into sentences using a simple regex
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    impact_sentences = []
    for sentence in sentences:
        words = set(sentence.lower().split())
        if words & _IMPACT_WORDS:   # set intersection — any overlap?
            impact_sentences.append(sentence)
        if len(impact_sentences) == 2:
            break

    if impact_sentences:
        return " ".join(impact_sentences)

    # Fallback: last 2 sentences
    return " ".join(sentences[-2:]) if len(sentences) >= 2 else " ".join(sentences)


# ---------------------------------------------------------------------------
# Public API: explain_article()
# Signature matches the original exactly — app.py needs no changes.
# ---------------------------------------------------------------------------


# Groq client — initialised once at module level
# Groq's free tier: 14,400 requests/day, much more generous than Gemini
_groq_client = None

def _get_groq_client():
    """Lazy-initialise Groq client so missing key doesn't crash on import."""
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _why_it_matters_groq(title: str, summary: str, category: str) -> str:
    """
    Use Groq (llama3-8b-8192) to generate a genuine 2-sentence explanation
    of why this article matters to an everyday reader.

    Groq runs inference on Llama 3 at very high speed (~300 tokens/sec)
    and gives 14,400 free requests/day — plenty for a personal news app.
    Falls back to the extractive method if the API call fails.
    """
    client = _get_groq_client()
    if not client:
        return ""   # will trigger fallback in explain_article

    prompt = (
        f"Title: {title}\n"
        f"Article: {summary}\n"
        f"Category: {category}\n\n"
        f"Write exactly 2 sentences of context that help the reader understand this story better.\n\n"
        f"CRITICAL — your first sentence must be a direct, punchy statement of consequence, "
        f"like a journalist's hook — NOT an analyst's conclusion. "
        f"NEVER start with 'This is significant', 'This matters because', 'This is important', "
        f"'This highlights', or any rephrasing of those. Banned as an opener.\n\n"
        f"BAD opener: 'This is significant because Delhi recorded its warmest night in 4 years.'\n"
        f"GOOD opener: 'Delhi just had its warmest night in 4 years — and summer hasn't even peaked yet.'\n\n"
        f"Sentence 1: State the most striking fact, number, or consequence directly. No preamble.\n"
        f"Sentence 2: Add the broader context or what's changed because of it — "
        f"for people in general, or for Indians if the story directly involves India.\n\n"
        f"Vary your sentence structure — do not use the same opening pattern across different articles.\n"
        f"If the article is about entertainment, business performance, or pop culture, "
        f"it's okay to be casual or even slightly witty — match the tone to the topic.\n\n"
        f"Do NOT:\n"
        f"- Make predictions or say 'this could lead to'\n"
        f"- Give advice or tell the reader what to do\n"
        f"- Force an India connection if the story has nothing to do with India\n"
        f"- Use corporate or textbook language\n"
        f"- Repeat what already happened in the summary\n\n"
        f"Maximum 2 sentences. Be direct. Nothing more."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # fast, free, good quality
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.4,          # low temperature = more factual
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[WARN] Groq API failed for '{title}': {exc}")
        return ""



def explain_article(article: dict, context_snippet: str = "") -> dict:
    """
    Generate a plain-English explanation for a single news article.
    Handles short articles (< 5 sentences) and normal articles differently.
    """
    empty_result = {
        "simple_summary": "",
        "why_it_matters": "",
        "background":     "",
    }

    title     = article.get("title",     "").strip()
    full_text = article.get("full_text", "").strip()
    category  = article.get("category",  "General").strip()

    if not title or not full_text:
        print(f"[WARN] explain_article: missing title or full_text — skipping")
        return empty_result

    # ── Clean raw scraped text ───────────────────────────────────────────────
    cleaned_text = clean_article_text(full_text,title=title,category=category)
    if not cleaned_text:
        cleaned_text = full_text

    # Count sentences to decide which path to take
    sentences = [s.strip() for s in cleaned_text.split('.') if len(s.strip()) > 15]
    is_short  = len(sentences) < 5

    # ── SIMPLE SUMMARY ───────────────────────────────────────────────────────
    if is_short:
        # Skip DistilBART on short text — it produces poor output
        # Find the first sentence that starts with a subject (PROPN or NOUN)
        doc = _nlp(cleaned_text)
        good_sentences = []
        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_space]
            if not tokens:
                continue
            # Check first 3 tokens for a subject noun — avoids mid-thought starts
            has_subject = any(
                t.pos_ in ("PROPN", "NOUN") for t in tokens[:3]
            )
            if has_subject:
                good_sentences.append(sent.text.strip())

        simple_summary = " ".join(good_sentences) if good_sentences else cleaned_text

    else:
        # Check word count of cleaned text BEFORE deciding whether to
        # run DistilBART/LexRank — a model can't meaningfully summarize
        # text that's already too short to compress
        word_count = len(cleaned_text.split())
        print(f"[DEBUG] cleaned_text word count before summarizer: {word_count}")

        if word_count < 50:
            print(
                f"[DEBUG] cleaned_text under 50 words — skipping LexRank/DistilBART, "
                f"using first 2-3 sentences directly"
            )
            direct_sentences = [
                s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text.strip())
                if len(s.strip()) > 15
            ]
            simple_summary = " ".join(direct_sentences[:3])
        else:
            # Normal path: LexRank → DistilBART → clean
            key_sentences  = _extract_key_sentences(cleaned_text, n=5)

            print(f"[DEBUG] len(article_text) going into summarizer: {len(key_sentences or cleaned_text)}")

            raw_summary    = _abstractive_summary(key_sentences or cleaned_text)
            simple_summary = clean_summary(raw_summary, fallback_sentences=key_sentences)

    # Validate the opening sentence — replace if it starts mid-thought
    simple_summary = _validate_first_sentence(simple_summary, title, category)

    # Remove "Topic: " prefixes bleeding in from RSS titles
    simple_summary = _strip_leading_topic_colon(simple_summary)

    # Remove the context opener if it bled into the final summary visibly
    simple_summary = re.sub(
        r'^Here is what happened:\s*', '', simple_summary, flags=re.IGNORECASE
    ).strip()

    print(f"[DEBUG] simple_summary length after cleanup: {len(simple_summary)} chars")

    # ── WHY IT MATTERS ───────────────────────────────────────────────────────
    print(f"[DEBUG] len(article_text) going into Groq: {len(cleaned_text[:500])}")
    # Try Groq first — best quality, generous free tier
    why_it_matters = _why_it_matters_groq(title, cleaned_text[:500], category)

    # Fallback chain if Groq unavailable or fails:
    # 1. Signal word scan on article text
    # 2. Sentences 2-3 of the summary
    if not why_it_matters:
        why_it_matters = _extract_why_it_matters(cleaned_text)

    if not why_it_matters:
        summary_sents = [
            s.strip() for s in simple_summary.split('.')
            if len(s.strip()) > 15
        ]
        why_it_matters = ". ".join(summary_sents[1:3]).strip()
        if why_it_matters:
            why_it_matters += "."


    # ── DEDUPLICATION CHECK ──────────────────────────────────────────────────
    # Never let both sections say the exact same thing
    if simple_summary.strip() == why_it_matters.strip():
        summary_sents = [
            s.strip() for s in simple_summary.split('.')
            if len(s.strip()) > 15
        ]
        why_it_matters = ". ".join(summary_sents[1:3]).strip()
        if why_it_matters:
            why_it_matters += "."

    # ── FINAL SAFETY NET: never ship a one-sentence summary ──────────────────
    # If everything above still produced a tiny summary, pad it with one
    # more sentence from the original article that wasn't already included.
    if len(simple_summary) < 100:
        print(
            f"[WARN] simple_summary still under 100 chars "
            f"({len(simple_summary)}) — padding with an extra sentence"
        )
        original_sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text.strip())
            if len(s.strip()) > 20
        ]
        already_used = simple_summary.lower()
        for sent in original_sentences:
            if sent.lower() not in already_used:
                simple_summary = (simple_summary + " " + sent).strip()
                break

    # ── BACKGROUND CONTEXT ───────────────────────────────────────────────────
    main_topic = _extract_main_topic(title)
    background = _get_wikipedia_background(main_topic)

    return {
        "simple_summary": simple_summary,
        "why_it_matters": why_it_matters,
        "background":     background,
    } 

# ---------------------------------------------------------------------------
# Smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_article = {
        "title": "India wins the IPL Final in a thrilling last-over finish",
        "full_text": (
            "Mumbai Indians defeated Chennai Super Kings by 3 wickets in the IPL final "
            "held at Wankhede Stadium on Sunday night. Chasing 187, Mumbai needed 14 off "
            "the last over and Hardik Pandya hit two sixes to seal the victory. It was a "
            "historic win — Mumbai's sixth IPL title, making them the most successful "
            "franchise in the tournament's history. The result will significantly impact "
            "the future of T20 cricket in India. Millions of fans across India watched the "
            "match live, with social media erupting immediately after the winning shot."
        ),
        "category": "Sports",
    }

    result = explain_article(test_article)

    print("=" * 60)
    print("SIMPLE SUMMARY:")
    print(result["simple_summary"])
    print("\nWHY IT MATTERS:")
    print(result["why_it_matters"])
    print("\nBACKGROUND:")
    print(result["background"] or "(none found)")
