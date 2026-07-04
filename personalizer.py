"""
personalizer.py  (feedback-aware version)
-----------------------------------------
Ranks news articles by relevance to a user's interests using
TF-IDF + cosine similarity, boosted by the user's reaction history.

New concept added: feedback-weighted ranking
  final_score = tfidf_score × category_boost × source_boost

  category_boost: derived from how the user has reacted to articles
                  in each category historically. Liked many Tech articles?
                  Tech gets a boost > 1.0. Thumbed down Sports? Sports < 1.0.

  source_boost:   same idea but per RSS source domain. If articles from
                  livemint.com consistently get thumbs up, they score higher.

This is implicit feedback learning — the system infers preferences
from behaviour rather than asking the user to re-configure their settings.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def compute_feedback_boosts(feedback_rows: list) -> tuple[dict, dict]:
    """
    Compute category and source boost multipliers from the user's
    reaction history.

    Args:
        feedback_rows: List of dicts with keys:
                       category (str), source_domain (str), reaction (int)
                       reaction: +1 = thumbs up, -1 = thumbs down

    Returns:
        Tuple of (category_boosts, source_boosts):
          Both are dicts mapping name → float multiplier.
          1.0 = neutral, >1.0 = boosted, <1.0 = penalised.
          Minimum multiplier is 0.2 (never zero out a category entirely).
          Maximum multiplier is 2.0 (cap the boost so one category
          doesn't completely dominate).

    How the math works:
      For each category, sum all reactions (+1 or -1).
      Net score of +3 means 3 more likes than dislikes.
      We convert this to a multiplier using:
        boost = 1.0 + (net_score / max(total_reactions, 1)) × 0.5
      This produces a multiplier between 0.5 and 1.5 for moderate
      reaction counts, which we then clamp to [0.2, 2.0].

      Example: 8 likes, 2 dislikes in Technology
        net = 8 - 2 = 6
        total = 10
        boost = 1.0 + (6/10) × 0.5 = 1.0 + 0.30 = 1.30
      Technology articles score 30% higher than baseline.
    """
    if not feedback_rows:
        return {}, {}

    # Accumulate reactions per category and source
    cat_reactions   = defaultdict(list)
    src_reactions   = defaultdict(list)

    for row in feedback_rows:
        cat_reactions[row["category"]].append(row["reaction"])
        if row["source_domain"]:
            src_reactions[row["source_domain"]].append(row["reaction"])

    def reactions_to_boost(reactions: list) -> float:
        total   = len(reactions)
        net     = sum(reactions)   # positive = more likes than dislikes
        raw     = 1.0 + (net / total) * 0.5
        return max(0.2, min(2.0, raw))   # clamp to [0.2, 2.0]

    category_boosts = {cat: reactions_to_boost(rxns)
                       for cat, rxns in cat_reactions.items()}
    source_boosts   = {src: reactions_to_boost(rxns)
                       for src, rxns in src_reactions.items()}

    return category_boosts, source_boosts


def rank_articles(user_interests: list, articles: list,
                  top_n: int = 10,
                  feedback_rows: list = None) -> list:
    """
    Rank articles by relevance to the user's interests,
    optionally boosted by reaction history.

    Args:
        user_interests: List of interest keywords/phrases.
        articles:       List of dicts with at least title, summary, category keys.
        top_n:          Number of top articles to return.
        feedback_rows:  Optional list of past reaction dicts from SQLite.
                        If None or empty, behaves exactly like the original
                        version (pure TF-IDF, no feedback).

    Returns:
        List of up to top_n article dicts sorted by final_score descending,
        each with a new "relevance_score" key (float, 0.0–2.0+).
    """

    if not user_interests or not articles:
        return []

    # ── STEP 1: TF-IDF cosine similarity (unchanged from original) ────────

    article_texts = [
        f"{a.get('title','')} {a.get('title','')} {a.get('summary','')}".strip()
        for a in articles
    ]
    query_text = " ".join(user_interests)
    corpus     = [query_text] + article_texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix      = vectorizer.fit_transform(corpus)
    query_vector      = tfidf_matrix[0]
    article_matrix    = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(query_vector, article_matrix).flatten()

    # ── STEP 2: compute feedback boosts (new) ─────────────────────────────

    category_boosts, source_boosts = compute_feedback_boosts(
        feedback_rows or []
    )

    # ── STEP 3: apply boosts and build scored list ────────────────────────

    scored = []
    for article, tfidf_score in zip(articles, similarity_scores):
        category      = article.get("category", "")
        source_domain = article.get("source_domain", "")

        # Look up boosts — default to 1.0 (neutral) if no feedback yet
        cat_boost = category_boosts.get(category, 1.0)
        src_boost = source_boosts.get(source_domain, 1.0)

        # Final score: TF-IDF similarity × category preference × source preference
        final_score = tfidf_score * cat_boost * src_boost

        enriched = dict(article)
        enriched["relevance_score"] = round(float(final_score), 4)
        enriched["tfidf_score"]     = round(float(tfidf_score), 4)
        enriched["category_boost"]  = round(cat_boost, 3)
        enriched["source_boost"]    = round(src_boost, 3)
        scored.append(enriched)

    scored.sort(key=lambda a: a["relevance_score"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    articles = [
        {"title": "India wins IPL Final", "summary": "Mumbai won by 3 wickets.",
         "category": "Sports", "source_domain": "espncricinfo.com"},
        {"title": "OpenAI launches GPT-5", "summary": "New model with better reasoning.",
         "category": "Technology", "source_domain": "techcrunch.com"},
        {"title": "Sensex hits record high", "summary": "Markets up 500 points.",
         "category": "Business", "source_domain": "livemint.com"},
    ]

    feedback = [
        {"category": "Technology", "source_domain": "techcrunch.com", "reaction": 1},
        {"category": "Technology", "source_domain": "techcrunch.com", "reaction": 1},
        {"category": "Sports",     "source_domain": "espncricinfo.com", "reaction": -1},
    ]

    print("Without feedback:")
    for a in rank_articles(["technology", "AI"], articles, top_n=3):
        print(f"  {a['relevance_score']:.4f}  {a['title']}")

    print("\nWith feedback (Technology liked, Sports thumbed down):")
    for a in rank_articles(["technology", "AI"], articles, top_n=3,
                           feedback_rows=feedback):
        print(f"  {a['relevance_score']:.4f}  {a['title']}  "
              f"(cat_boost={a['category_boost']}, src_boost={a['source_boost']})")
