"""
categorizer.py
--------------
Classifies news articles into categories using:
  1. Hard rules — unambiguous title patterns, checked first (free, instant)
  2. Sentence embeddings + cosine similarity — semantic fallback

Dependencies: pip install sentence-transformers
"""

import re
from sentence_transformers import SentenceTransformer, util

print("[categorizer] Loading sentence-transformers model...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[categorizer] Model loaded OK")


# ---------------------------------------------------------------------------
# Category descriptions
# Richer natural language descriptions give the embedding model more
# semantic surface area to match against. Specificity matters —
# vague descriptions cause misclassification.
# ---------------------------------------------------------------------------

CATEGORY_DESCRIPTIONS = {
    "Technology": (
        "News about software, apps, artificial intelligence, gadgets, "
        "startups, the tech industry, smartphones, and computer chips."
    ),
    "Defence & World": (
        "News about military training, army operations, drone warfare, "
        "armed forces, war, weapons, missiles, nuclear programs, "
        "international conflict, geopolitical tension, and countries "
        "building or deploying their military capabilities."
    ),
    "Politics": (
        "News about Indian and global politics — elections, parliament, "
        "political parties like BJP, Congress, Trinamool, AAP, government "
        "decisions, ministers, chief ministers, policy announcements, and "
        "political leaders making statements or facing controversy."
    ),
    "Business": (
        "News about the stock market, companies, the economy, trade, "
        "inflation, corporate earnings, mergers, acquisitions, "
        "and financial markets."
    ),
    "Science": (
        "News about scientific research, space exploration, physics, "
        "biology, new discoveries, medical research, health studies, "
        "and academic findings published in journals."
    ),
    "Health": (
        "News about diseases, hospitals, medicine, vaccines, doctors, "
        "public health outbreaks, and medical treatment of patients."
    ),
    "Sports": (
        "News about cricket matches, football games, tennis tournaments, "
        "athletes competing, sports scores, and sporting events and leagues."
    ),
    "Environment": (
        "News about climate change, pollution, wildlife conservation, "
        "renewable energy, carbon emissions, endangered species, "
        "and ecosystems."
    ),
    "Disaster & Weather": (
        "News about floods, earthquakes, cyclones, extreme weather, "
        "natural disasters, evacuation, rescue operations, and "
        "emergency relief efforts."
    ),
    "Crime & Courts": (
        "News about specific criminal cases — murder, theft, fraud, "
        "arrests of named individuals, court verdicts, bail hearings, "
        "trials, and police investigations into crimes."
    ),
    "Entertainment": (
        "News about movies, Bollywood, celebrities, actors, music, "
        "TV shows, web series, box office collections, and pop culture."
    ),
    "Education": (
        "News about school and university board exams, exam results, "
        "college admissions, scholarships, and education policy changes."
    ),
}

print("[categorizer] Pre-computing category embeddings...")
_category_names      = list(CATEGORY_DESCRIPTIONS.keys())
_category_embeddings = _embedder.encode(
    list(CATEGORY_DESCRIPTIONS.values()),
    convert_to_tensor=True,
)
print("[categorizer] Category embeddings ready")


# ---------------------------------------------------------------------------
# Hard rules
#
# These fire BEFORE the embedding model is called.
# Only include triggers that are UNAMBIGUOUS — a word that appears in
# article titles only for one category, never as a false positive.
#
# Key decisions:
#   - "police" removed from Crime — too many political articles say
#     "police deployed" without being crime stories
#   - "investigation" removed from Crime — political/parliamentary
#     investigations are not crime stories
#   - "university" removed from Education — too many non-education articles
#     mention universities in passing
#   - Politics added with Indian-specific political proper nouns —
#     these are essentially impossible to misfire
# ---------------------------------------------------------------------------

_HARD_RULES = {
    "Disaster & Weather": [
        "weather", "rain lashes", "flood", "storm", "cyclone", "earthquake",
        "landslide", "heatwave", "drought", "tsunami", "cloudy", "forecast",
        "lightning", "thunder", "rescue operation", "missing persons",
        "evacuate", "evacuation", "orange alert", "red alert", "imd issues",
    ],
    "Crime & Courts": [
        "murder", "arrested", "in custody", "accused of", "verdict",
        "convicted", "fir filed", "sent to bail", "chargesheet",
        "life sentence", "death penalty", "acquitted", "rape case",
        "kidnapping", "theft case",
    ],
    "Sports": [
        "odi", "test match", "ipl", "fifa", "premier league", "grand slam",
        "medal", "wicket", "run chase", "batter", "innings",
        "championship", "world cup final", "hat-trick",
    ],
    "Business": [
        "sensex", "nifty", "ipo ", "stock market", "rbi rate",
        "gdp growth", "inflation rises", "trade deficit", "merger",
        "acquisition", "lakh crore", "crore deal", "funding round",
        "quarterly results", "revenue growth",
    ],
    "Politics": [
        "modi", "bjp", "rahul gandhi", "amit shah", "mamata",
        "kejriwal", "trinamool", "lok sabha", "rajya sabha",
        "parliament session", "chief minister", "home minister",
        "prime minister", "budget session", "no confidence",
        "election commission", "aap ", "congress party",
    ],
    "Entertainment": [
        "bollywood", "box office", "trailer launch", "ott release",
        "web series", "award show", "biopic", "film review",
        "music video", "celebrity", "actor arrested",
    ],
    "Education": [
        "board exam", "cbse", "icse", "neet result", "jee result",
        "result declared", "college admission", "syllabus change",
        "exam date sheet", "scholarship scheme",
    ],
    "Defence & World": [
        "military drill", "army deployed", "air strike", "missile launch",
        "nuclear test", "ceasefire", "drone strike", "troop deployment",
        "nato ", "border tension", "defence ministry", "armed forces",
        "fighter jet", "warship", "ballistic", "surgical strike",
    ],
    "Technology": [
        "apple ", "google", "microsoft", "meta ", "openai", "nvidia",
        "samsung", "amazon", "intel", "qualcomm", "tsmc", "chipmaker",
        "semiconductor", "iphone", "android", "chatgpt", "llm ",
        "artificial intelligence", "machine learning", "data center",
        "cybersecurity", "software update", "app store",
    ],
}

# "X vs Y" pattern — strong sports signal
_VS_PATTERN = re.compile(r'\b(india|australia|england|pakistan|newzealand|'
                          r'southafrica|srilanka|westindies|bangladesh|'
                          r'mumbai|chennai|delhi|kolkata|gujarat|rajasthan|'
                          r'hyderabad|punjab|sunrisers|rcb|csk|mi|kkr|dc|srh|lsg|pbks)'
                          r'\s+vs\.?\s+', re.IGNORECASE)

MIN_SIMILARITY = 0.32


def hard_classify(title: str) -> str | None:
    """
    Return a category if any hard rule fires, else None.
    Checks are ordered: more specific categories first to avoid
    broad triggers shadowing specific ones.
    """
    title_lower = title.lower()

    # Cricket/sports team vs team pattern
    if _VS_PATTERN.search(title_lower):
        return "Sports"

    for category, triggers in _HARD_RULES.items():
        for trigger in triggers:
            if trigger in title_lower:
                return category

    return None


def classify(title: str, summary: str,
             source_category: str = "World News") -> str:
    """
    Classify an article into a category.

    Step 0 — hard rules (instant, no model call)
    Step 1 — sentence embedding + cosine similarity
    Step 2 — fallback to source_category if similarity too low
    """
    # Step 0
    hard = hard_classify(title)
    if hard:
        return hard

    # Step 1
    article_text      = f"{title}. {title}. {summary}".strip()
    article_embedding = _embedder.encode(article_text, convert_to_tensor=True)
    
    similarities      = util.cos_sim(article_embedding, _category_embeddings)[0]

    best_idx      = similarities.argmax().item()
    best_score    = similarities[best_idx].item()
    best_category = _category_names[best_idx]

    # Tiebreaker: if Politics won but the title mentions a known tech company,
    # and Technology score is close, prefer Technology
    _TECH_COMPANIES = {
        "apple", "google", "microsoft", "meta", "openai", "nvidia",
        "samsung", "amazon", "intel", "qualcomm", "tsmc", "tesla"
    }
    if best_category == "Politics":
        title_words = set(title.lower().split())
        if title_words & _TECH_COMPANIES:
            tech_idx   = _category_names.index("Technology")
            tech_score = similarities[tech_idx].item()
            if best_score - tech_score < 0.05:
                best_category = "Technology"

    # ── Step 2: trust the match only if similarity is strong enough ──
    if best_score < MIN_SIMILARITY:
        return source_category

    return best_category


if __name__ == "__main__":
    tests = [
        ("'Kill Me To Stop Me': Defiant Mamata Banerjee Calls Trinamool Rebels Traitors",
         "Mamata Banerjee claimed rebels parted ways because of pressure."),
        ("South Korea plans to train entire military as drone warriors",
         "Half-million strong military will train on drones as universal combat tool."),
        ("Apple wants permission to buy memory from a blacklisted Chinese chip maker",
         "Apple is seeking a license to purchase semiconductors from CXMT."),
        ("India vs Australia: 3rd ODI preview",
         "India look to seal the series at home."),
        ("Weather LIVE Updates: Rain lashes Delhi NCR, orange alert issued",
         "IMD has issued an orange alert for the region."),
        ("Sensex jumps 500 points as IT stocks rally",
         "Strong Q1 earnings from major tech firms boosted investor sentiment."),
        ("CBSE Board Exam results declared for Class 10",
         "Over 20 lakh students appeared for the board exams this year."),
        ("Man arrested for murder of shopkeeper in Delhi",
         "Police said the accused was caught on CCTV footage near the scene."),
    ]

    print("Testing hard_classify only:")
    for title, summary in tests:
        hard = hard_classify(title)
        print(f"  hard={str(hard):25s} ← {title[:65]}")

    print("\nTesting full classify():")
    for title, summary in tests:
        result = classify(title, summary, source_category="Politics")
        print(f"  {result:25s} ← {title[:65]}")
