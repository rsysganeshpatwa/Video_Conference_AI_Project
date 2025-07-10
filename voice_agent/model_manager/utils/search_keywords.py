# search_keywords.py

# 🔍 Direct fallback for time-sensitive / web-needed keywords
# 🔍 Direct fallback for time-sensitive / web-needed keywords
SEARCH_TRIGGER_KEYWORDS = [
    "latest", "today", "now", "current", "tonight", "tomorrow",
    "weather", "temperature", "forecast",
    "news", "breaking", "headline",
    "who is", "who's", "ceo", "founder", "president", "minister",
    "what is", "when is", "where is",
    "update", "status", "recent", "event", "trending", "happening",
    "live", "score", "match", "result",
    "market", "stock", "price", "rate", "exchange rate",
    "holiday", "festival", "election", "vote", "poll",
    "announce", "announcement", "declared", "released",
    "launch", "release date", "version",
    "schedule", "program", "agenda",
    "list of", "top", "best", "rank", "ranking",
    "trailer", "movie", "show", "series",
    "crime", "accident", "disaster", "earthquake", "fire", "storm",
    "air quality", "pollution", "aqi","about", "info", "details", "summary", "highlights",

    # ✅ Additional conversational & fuzzy keywords
    "tell me about", "know about", "find info on", "get details about",
    "give me details", "learn about", "explain", "summary of", "highlights of",
    "what happened", "happened today", "happened yesterday", "right now", "just released",
    "new update", "hot topic", "just announced", "breaking news", "live update",
    
    # People / Org Roles
    "biography", "profile", "background", "management", "leadership team",
    "executive", "director", "chairman", "owner", "head of",

    # Business / Market
    "growth", "acquisition", "merger", "partnership", "earnings", "IPO", "valuation",
    "share price", "financials", "crypto", "token price",

    # Entertainment / Media
    "box office", "cast of", "plot of", "reviews", "imdb", "rotten tomatoes", "runtime",
    "streaming now", "available on", "platform", "episode", "season", "premiere",

    # Events / Schedules
    "event date", "event schedule", "start time", "venue", "calendar", "lineup",
    "guest list", "invite", "speaker", "hosted by", "organized by"
]


def needs_search(user_query: str) -> bool:
    query = user_query.lower()
    return any(keyword in query for keyword in SEARCH_TRIGGER_KEYWORDS)