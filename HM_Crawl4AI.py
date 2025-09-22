import asyncio
import json
import os
from pathlib import Path
import pandas as pd
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMExtractionStrategy,
    LLMConfig,
)

# -------- configuration --------
OPENAI_API_KEY = ""

CHUNK_SIZE = 50  # checkpoint frequency
logging.getLogger("crawl4ai").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

counter = 1  # will be initialized based on existing file

# -------- helpers --------
def extract_domain_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""

async def fetch_office_details(crawler: AsyncWebCrawler, office: dict) -> dict:
    """
    Crawl the office page and extract site and phone using crawl4ai's LLM schema extractor.
    """
    global counter

    run_config = CrawlerRunConfig(
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(provider="openai/gpt-4o", api_token=OPENAI_API_KEY),
            extraction_type="schema",
            schema={
                "site": {"selector": "a[href^='http']", "type": "href"},
                "phone": {"selector": "a[href^='tel:']", "type": "text"},
            },
            instruction="Extract the main website URL and phone number if present. Return empty string if not found.",
        ),
        cache_mode=CacheMode.BYPASS,
        page_timeout=15_000,
    )

    try:
        result = await crawler.arun(office["url"], config=run_config)
        data = getattr(result, "extracted_content", {})
    except Exception as e:
        print(f"❌ Fetch failed for {office['url']}: {e}")
        data = {}

    # Normalize output
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    elif isinstance(data, str):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list) and len(parsed) > 0:
                data = parsed[0]
            elif isinstance(parsed, dict):
                data = parsed
            else:
                data = {}
        except Exception:
            data = {}
    elif not isinstance(data, dict):
        data = {}

    site = data.get("site", "") if isinstance(data.get("site", ""), str) else ""
    if site and "housematch.be" in site:
        site = ""  # ignore internal links

    office["site"] = site
    office["phone"] = data.get("phone", "")

    domain = extract_domain_from_url(site)
    office["info_email"] = f"info@{domain}" if domain else ""
    office["hello_email"] = f"hello@{domain}" if domain else ""

    print(
        f"✅ {counter}: {office['name']} ({office['city']}) → {office['site']} | "
        f"{office['phone']} | {office['info_email']} | {office['hello_email']}"
    )
    counter += 1
    return office

# Blocking save routine executed in a thread
def _blocking_save_excel(filename: Path, data_snapshot: list, column_order: list):
    # Build DataFrame in the thread and write to a temporary file first
    df = pd.DataFrame(data_snapshot)
    for col in column_order:
        if col not in df.columns:
            df[col] = ""
    df = df.reindex(columns=column_order)

    # temp filename (eg offices_a.tmp.xlsx) then atomically replace
    tmp = filename.parent / (filename.stem + ".tmp.xlsx")
    df.to_excel(tmp, index=False, engine="openpyxl")
    os.replace(str(tmp), str(filename))  # atomic replace on most OSes

async def save_checkpoint(filename: Path, data: list, column_order: list):
    """
    Offload the blocking pandas to_excel call to a thread to avoid blocking the event loop.
    """
    snapshot = list(data)  # shallow copy to avoid mutation while writing
    try:
        print(f"💾 Saving checkpoint ({len(snapshot)} records) ...")
        # await the to-thread call so we know when it's finished, but the event loop isn't blocked
        await asyncio.to_thread(_blocking_save_excel, filename, snapshot, column_order)
        print(f"💾 Checkpoint saved: {len(snapshot)} records → {filename}")
    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")

# -------- main ----------
async def main():
    global counter

    # letter input
    while True:
        filter_letter = input("Which letter do you want to filter by? (A-Z): ").upper().strip()
        if len(filter_letter) == 1 and filter_letter.isalpha():
            break
        print("❌ Please enter a single letter (A-Z)")

    filename = Path(f"offices_{filter_letter.lower()}.xlsx")
    column_order = ["name", "city", "site", "phone", "info_email", "hello_email", "url"]

    # Load existing file (skip already scraped)
    scraped_urls = set()
    all_results = []
    if filename.exists():
        try:
            df_existing = pd.read_excel(filename, engine="openpyxl")
            if "url" in df_existing.columns:
                scraped_urls = set(df_existing["url"].dropna().astype(str).tolist())
            all_results = df_existing.to_dict(orient="records")
            print(f"📂 Loaded {len(all_results)} records from {filename} (unique urls: {len(scraped_urls)})")
        except Exception as e:
            print(f"⚠️ Failed to read existing file {filename}: {e}. Starting fresh.")
            scraped_urls = set()
            all_results = []

    counter = len(all_results) + 1

    # Crawl list and individual pages
    browser_cfg = BrowserConfig(headless=False, verbose=True)
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        run_conf = CrawlerRunConfig(
            wait_for="css:a.u-link.u-color-primary",
            cache_mode=CacheMode.BYPASS,
            page_timeout=60_000,
            verbose=False,
        )

        print("🌐 Fetching main locations page...")
        result = await crawler.arun("https://app.housematch.be/nl/locaties", config=run_conf)
        if not result.success:
            print("❌ Crawl failed:", result.error_message)
            return

        soup = BeautifulSoup(result.html, "html.parser")
        discovered_offices = []
        for a in soup.select("a.u-link.u-color-primary"):
            text = a.get_text(strip=True)
            href = a.get("href", "").strip()
            if "-" in text and (href.startswith("nl/") or href.startswith("/nl/")):
                name, city = [p.strip() for p in text.split("-", 1)]
                if name.upper().startswith(filter_letter):
                    url = f"https://app.housematch.be/{href.lstrip('/')}"
                    if url not in scraped_urls:
                        discovered_offices.append({"name": name, "city": city, "url": url})
                        print(f"📌 New office queued: {name} - {city} ({url})")
                    else:
                        print(f"⏭️ Skipping already scraped: {name} - {city} ({url})")

        if not discovered_offices:
            print(f"❌ No new offices found for '{filter_letter}'")
            return

        print(f"✅ Found {len(discovered_offices)} new offices to process")

        # Process and checkpoint during the run (offload save to a thread to avoid blocking)
        for office in discovered_offices:
            try:
                detailed_office = await fetch_office_details(crawler, office)
            except Exception as e:
                print(f"❌ Error while fetching details for {office.get('url')}: {e}")
                detailed_office = {
                    "name": office.get("name", ""),
                    "city": office.get("city", ""),
                    "url": office.get("url", ""),
                    "site": "",
                    "phone": "",
                    "info_email": "",
                    "hello_email": "",
                }

            all_results.append(detailed_office)
            scraped_urls.add(detailed_office.get("url", ""))

            # checkpoint when total results reaches a multiple of CHUNK_SIZE
            if len(all_results) % CHUNK_SIZE == 0:
                # this will not block the event loop (written in a separate thread)
                await save_checkpoint(filename, all_results, column_order)

    # After crawler closed, ensure final leftovers are saved
    if len(all_results) % CHUNK_SIZE != 0:
        await save_checkpoint(filename, all_results, column_order)
    else:
        # If last checkpoint was already saved inside the loop, ensure file exists (just in case)
        if not filename.exists() and all_results:
            await save_checkpoint(filename, all_results, column_order)

    print("🎉 Finished run.")

if __name__ == "__main__":
    asyncio.run(main())
