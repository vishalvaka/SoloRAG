#!/usr/bin/env python
"""
Stripe Support full-site scraper (Sync Playwright + tqdm).

Usage (inside your venv):
    pip install playwright==1.44.0 beautifulsoup4 tqdm
    playwright install chromium        # once
    python -m playwright install-deps  # once (installs libnss3, fonts…)

    python scripts/stripe_scrape_sync.py
Outputs:
    data/raw/stripe_faqs_full.jsonl
"""

import collections, json, pathlib, re, time
from bs4 import BeautifulSoup
from tqdm import tqdm
from playwright.sync_api import sync_playwright

# ───── config ────────────────────────────────────────────────────────────
ROOT        = "https://support.stripe.com"
SEED        = ROOT + "/topics"
HEADERS     = {"User-Agent": "Mozilla/5.0"}
DELAY_SEC   = 0.35
MIN_WORDS   = 20
OUT_FILE    = pathlib.Path("data/raw/stripe_faqs_full.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TOPIC_RX = re.compile(r"^/topics/[a-z0-9\-]+$")
ART_RX   = re.compile(r"^/questions/[a-z0-9\-]+$")

# ───── main scraper ──────────────────────────────────────────────────────
def scrape():
    queue        = collections.deque([SEED])
    seen_pages   = set()
    paragraphs   = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page(extra_http_headers=HEADERS)

        bar  = tqdm(total=0, unit="page", desc="Crawled", colour="green")

        while queue:
            url = queue.popleft()
            if url in seen_pages:
                continue
            seen_pages.add(url)

            page.goto(url, timeout=90_000)
            soup = BeautifulSoup(page.content(), "html.parser")

            # -------- harvest article text ------------------------------
            if ART_RX.search(url.replace(ROOT, "")):
                # title = soup.select_one("h1").get_text(strip=True)
                h1 = soup.select_one("h1") or soup.select_one("h2")
                title = h1.get_text(strip=True) if h1 else "Untitled"
                for p_tag in soup.find_all("p"):
                    txt = p_tag.get_text(" ", strip=True)
                    if len(txt.split()) >= MIN_WORDS:
                        paragraphs.append({"url": url, "title": title, "text": txt})

            # -------- enqueue new links ---------------------------------
            for a in soup.find_all("a", href=True):
                href = a["href"].split("#")[0]
                if TOPIC_RX.match(href) or ART_RX.match(href):
                    full = href if href.startswith("http") else ROOT + href
                    if full not in seen_pages:
                        queue.append(full)

            # -------- progress update -----------------------------------
            bar.update(1)
            bar.set_postfix(pages=len(seen_pages), paras=len(paragraphs))

            time.sleep(DELAY_SEC)

        browser.close()
        bar.close()

    OUT_FILE.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in paragraphs),
        encoding="utf-8"
    )
    print(f"\n✅  Finished. Pages: {len(seen_pages)}, "
          f"paragraphs: {len(paragraphs)} → {OUT_FILE}")

# ───── entry-point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        scrape()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted — partial data (if any) kept.")
