from typing import Any


import csv
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.planecrashinfo.com/"
DATABASE_URL = urljoin(BASE_URL, "database.htm")
OUTPUT_CSV = "planecrashinfo_accidents.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PlaneCrashInfoScraper/1.0)"}


def get_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_year_links():
    soup = get_soup(DATABASE_URL)
    year_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".htm") and "database" not in href.lower():
            absolute_url = urljoin(BASE_URL, href)
            year_links.append(absolute_url)

    year_links = sorted(set(year_links))
    return year_links


def get_accident_links_for_year(year_url):
    soup = get_soup(year_url)
    accident_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".htm") and "database" not in href.lower():
            absolute_url = urljoin(year_url, href)
            accident_links.append(absolute_url)

    accident_links = sorted(set(accident_links))
    return accident_links


def parse_accident_detail(accident_url):
    soup = get_soup(accident_url)
    record = {"detail_url": accident_url}

    table = soup.find("table")
    if not table:
        record["raw_text"] = soup.get_text(separator=" ", strip=True)
        return record

    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(" ", strip=True).rstrip(":")
        value = cells[1].get_text(" ", strip=True)
        norm_label = label.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("-", "_").strip("_")
        if norm_label:
            record[norm_label] = value

    summary = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(" ", strip=True)]
    if summary and "summary" not in record:
        record["summary"] = " ".join(summary)

    return record


def main():
    year_links = get_year_links()
    all_records, visited = [], set()


    for i, year_url in enumerate(year_links, start=1):
        print(f"[{i}/{len(year_links)}] Processing year page: {year_url}")
        try:
            accident_links = get_accident_links_for_year(year_url)
        except Exception:
            continue

        for acc_url in accident_links:
            if acc_url in visited:
                continue
            try:
                record = parse_accident_detail(acc_url)
                record["year_page_url"] = year_url
                all_records.append(record)
                visited.add(acc_url)
            except Exception:
                pass
            time.sleep(0.5)
        time.sleep(1.0)

    fieldnames = sorted({k for rec in all_records for k in rec.keys()})

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)


if __name__ == "__main__":
    main()
