import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def duckduckgo_search(query, max_results=4):
    print(f"\n🔎 Scraping DuckDuckGo for: {query}")
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query}

    try:
        response = requests.post(url, data=data, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        results_html = soup.find_all("div", class_="result", limit=max_results)

        for result in results_html:
            title_tag = result.find("a", class_="result__a")
            snippet_tag = result.find("a", class_="result__snippet")
            link_tag = title_tag["href"] if title_tag and "href" in title_tag.attrs else None

            title = title_tag.get_text(strip=True) if title_tag else "No title"
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No snippet"

            content = fetch_page_text(link_tag) if link_tag else ""

            results.append(f"🔹 **{title}**\n📄 {snippet}\n🔗 {link_tag}\n📰 {content[:500]}...")

        return "\n\n---\n\n".join(results) if results else "No results found."

    except Exception as e:
        return f"[Error during scraping: {e}]"

def fetch_page_text(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.text.strip()) > 40)
        return text
    except Exception as e:
        return f"[Error fetching page: {e}]"

# 🔍 Try it
if __name__ == "__main__":
    query = "latest Bollywood movie released"
    full_article_text = duckduckgo_search(query)
    print("\n📝 Extracted Search Results:\n")
    print(full_article_text)
