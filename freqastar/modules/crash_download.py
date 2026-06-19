"""
Stage 5a -- NYC crash data downloader.

Downloads the NYC Motor Vehicle Collisions dataset (h9gi-nx95) from
NYC Open Data via the Socrata API, with server-side filtering to keep the
download small, then caches it locally as JSON.

Server-side filters (Socrata $where):
  * valid coordinates: latitude/longitude not null
  * time window:       crash_date >= cfg.crash_year_start  (e.g. 2020)
  * (severity is NOT filtered -- we count all collisions and let the
     beta-weighted 'killed' term encode severity, per the methodology)

Only the six columns we need are pulled ($select) to keep payloads small,
and results are paginated ($limit/$offset) because the API caps rows
per request.

Run locally (needs internet):
    python -m freqastar.modules.crash_download
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config

API_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"

# columns we actually use (Socrata API field names are lowercase/underscored)
SELECT_FIELDS = [
    "crash_date",
    "latitude",
    "longitude",
    "cross_street_name",
    "number_of_persons_injured",
    "number_of_persons_killed",
]


def _crash_cache(cfg: Config) -> Path:
    return cfg.data_dir / f"crashes_nyc_{cfg.crash_year_start}.json"


def _build_where(cfg: Config) -> str:
    """Server-side filter clause."""
    return (
        "latitude IS NOT NULL AND longitude IS NOT NULL "
        f"AND crash_date >= '{cfg.crash_year_start}-01-01T00:00:00'"
    )


def download_crashes(cfg: Config, force: bool = False, page_size: int = 50000):
    """Download (or load cached) filtered NYC crash records."""
    import requests

    cache = _crash_cache(cfg)
    if cache.exists() and not force:
        print(f"[crash_download] loading cached crashes: {cache}")
        with open(cache) as fh:
            records = json.load(fh)
        print(f"[crash_download] {len(records):,} records loaded")
        return records

    where = _build_where(cfg)
    select = ",".join(SELECT_FIELDS)
    print(f"[crash_download] querying NYC Open Data ...")
    print(f"[crash_download]   $where  = {where}")
    print(f"[crash_download]   $select = {select}")

    records = []
    offset = 0
    while True:
        params = {
            "$select": select,
            "$where": where,
            "$limit": page_size,
            "$offset": offset,
            "$order": "crash_date",     # stable paging order
        }
        resp = requests.get(API_URL, params=params, timeout=120)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        records.extend(page)
        print(f"[crash_download]   fetched {len(page):,} "
              f"(total {len(records):,}, offset {offset:,})")
        if len(page) < page_size:
            break                       # last page
        offset += page_size
        time.sleep(0.5)                 # be polite to the API

    with open(cache, "w") as fh:
        json.dump(records, fh)
    print(f"[crash_download] cached {len(records):,} records -> {cache}")
    return records


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    download_crashes(cfg)
