import requests
from bs4 import BeautifulSoup

def search_snippets(query: str, max_results: int = 3):
    url = "https://html.duckduckgo.com/html"
    headers = {"User-Agent": "Mozilla/5.0"}
    data = {"q": query}
    resp = requests.post(url, data=data, headers=headers)

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.select(".result__body", limit=max_results):
        title = result.select_one(".result__title").get_text(strip=True)
        snippet = result.select_one(".result__snippet").get_text(strip=True)
        results.append(f"{title}:\n{snippet}")
    return results
