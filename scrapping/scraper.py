# ============================================================
# scraper.py  —  99acres Multi-City Property Scraper
# ============================================================
# FAST mode using ScraperAPI (recommended — 10-20× faster, no IP blocks)
# Get a free API key at https://scraperapi.com (1000 reqs/month free)
#
# Property types per city:
#   1. Apartments / Societies  → {city}_appartments.csv
#   2. Flats (individual)      → {city}_flats.csv
#   3. Independent Houses      → {city}_houses.csv
#   4. Residential Land        → {city}_residential_land.csv
#
# KEY IMPROVEMENTS over previous version:
#   - ScraperAPI integration: parallel-safe, no IP blocks, much faster
#   - Concurrent requests via ThreadPoolExecutor (10× speedup)
#   - Description parsing: extracts BHK, price range, area from
#     project-page descriptions when individual fields are missing
#   - Auto-deduplication by property_id
#   - Filters out empty rows (only saves listings with actual data)
#   - Skip-already-scraped: never re-fetches a property_id already in CSV
#
# INSTALL:
#   pip install requests beautifulsoup4 pandas
#   pip install selenium webdriver-manager   # only for --mode selenium
#
# USAGE:
#   python scraper.py --city bangalore --type flats --start 1 --end 50 \
#                     --api-key YOUR_SCRAPERAPI_KEY
#
#   python scraper.py --city gurgaon --type all --start 1 --end 50 \
#                     --api-key YOUR_SCRAPERAPI_KEY --workers 5
#
#   # Without ScraperAPI (slower, IP can get blocked):
#   python scraper.py --city bangalore --type flats --mode selenium
# ============================================================

import os
import re
import sys
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import pandas as pd


# ─────────────────────────────────────────────────────────────
# DEFAULT CONFIG
# ─────────────────────────────────────────────────────────────

DEFAULT_CITY     = 'gurgaon'
START_PAGE       = 1
END_PAGE         = 50
PAGES_PER_BATCH  = 5      # save CSV every N pages
MODE             = 'requests'   # 'requests' | 'selenium' | 'scraperapi'
SCRAPER_API_KEY  = None
WORKERS          = 5      # parallel detail-page fetches when using ScraperAPI
OUTPUT_DIR       = '.'

CITY_SLUGS = {
    'gurgaon':  'gurgaon',  'gurugram':  'gurgaon',
    'bangalore':'bangalore','bengaluru': 'bangalore',
    'mumbai':   'mumbai',   'delhi':     'delhi',
    'hyderabad':'hyderabad','chennai':   'chennai',
    'pune':     'pune',     'kolkata':   'kolkata',
    'noida':    'noida',
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
]


def _city_slug(city: str) -> str:
    return CITY_SLUGS.get(city.lower(), city.lower())


def _make_headers(referer: str) -> dict:
    return {
        'authority':                 'www.99acres.com',
        'accept':                    'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'accept-language':           'en-US,en;q=0.9',
        'cache-control':             'no-cache',
        'referer':                   referer,
        'upgrade-insecure-requests': '1',
        'user-agent':                random.choice(USER_AGENTS),
    }


# ─────────────────────────────────────────────────────────────
# FETCH LAYER
# ─────────────────────────────────────────────────────────────

def _is_blocked(resp) -> bool:
    if resp.status_code in (403, 429, 503):
        return True
    txt = resp.text.upper()
    bad = ('ACCESS DENIED', 'CAPTCHA', 'ROBOT',
           'CHROME-ERROR', 'NETERROR', 'ERR_NAME_NOT_RESOLVED',
           'ERR_CONNECTION', 'ERR_TIMED_OUT')
    return any(kw in txt for kw in bad)


def _fetch_scraperapi(url: str, key: str, retries: int = 2):
    """Fast, IP-rotating fetch via ScraperAPI."""
    proxy_url = (f'http://api.scraperapi.com/?api_key={key}'
                 f'&url={url}&country_code=in')
    for attempt in range(retries):
        try:
            resp = requests.get(proxy_url, timeout=70)
            if resp.status_code == 200 and not _is_blocked(resp):
                return resp
            print(f"  [SCRAPERAPI] status={resp.status_code}, retrying...")
            time.sleep(3)
        except Exception as e:
            print(f"  [SCRAPERAPI ERROR] {e} — retry {attempt+1}/{retries}")
            time.sleep(5)
    return None


def _fetch_requests(url: str, referer: str, retries: int = 3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_make_headers(referer), timeout=20)
            if _is_blocked(resp):
                wait = 60 * (attempt + 1)
                print(f"  [BLOCKED] retry {attempt+1}/{retries} — sleeping {wait}s")
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as e:
            print(f"  [ERROR] {e} — retrying in 10s")
            time.sleep(10)
    return None


def _fetch_selenium(url: str):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument(f'--user-agent={random.choice(USER_AGENTS)}')
    opts.add_experimental_option('excludeSwitches', ['enable-automation'])
    opts.add_experimental_option('useAutomationExtension', False)

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts,
        )
        driver.execute_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        driver.get(url)
        time.sleep(random.uniform(4, 7))
        html = driver.page_source
        driver.quit()
        return type('R', (), {
            'text': html, 'content': html.encode(), 'status_code': 200})()
    except Exception as e:
        print(f"  [SELENIUM ERROR] {e}")
        return None


def _fetch(url: str, referer: str = ''):
    """Routes to the right backend based on MODE / SCRAPER_API_KEY."""
    if SCRAPER_API_KEY:
        return _fetch_scraperapi(url, SCRAPER_API_KEY)
    if MODE == 'selenium':
        return _fetch_selenium(url)
    return _fetch_requests(url, referer or 'https://www.99acres.com/')


def _sleep(req_count: int):
    """Skip delays when using ScraperAPI — it handles rate limits."""
    if SCRAPER_API_KEY:
        return
    if req_count % 20 == 0:
        time.sleep(random.randint(50, 90))
    elif req_count % 4 == 0:
        time.sleep(random.randint(10, 18))
    else:
        time.sleep(random.uniform(1.5, 3.5))


# ─────────────────────────────────────────────────────────────
# PARSE HELPERS
# ─────────────────────────────────────────────────────────────

def _text(soup, selector: str, attr: str = None) -> str:
    try:
        el = soup.select_one(selector)
        if el is None:
            return ''
        return el[attr].strip() if attr else el.text.strip()
    except Exception:
        return ''


def _list(soup, container_sel: str, item_sel: str) -> list:
    try:
        container = soup.select_one(container_sel)
        if container is None:
            return []
        return [el.text.strip() for el in container.select(item_sel)]
    except Exception:
        return []


def _features(dsoup) -> list:
    furnish = _list(dsoup, '#FurnishDetails', 'li')
    idx = 1 if furnish else 0
    try:
        return [el.text.strip()
                for el in dsoup.select('#features')[idx].select('li')]
    except Exception:
        return []


def _ratings(dsoup) -> list:
    return [el.text for el in dsoup.select(
        'div.review__rightSide>div>ul>li>div div.ratingByFeature__circleWrap')]


# ─────────────────────────────────────────────────────────────
# DESCRIPTION PARSER  —  rescues data from project pages
# ─────────────────────────────────────────────────────────────
# Project pages don't have #bedRoomNum etc., but their descriptions
# follow a consistent pattern that can be parsed:
#   "Check out 2,3 BHK apartments in Whitefield ..."
#   "Prices of apartments in this project, vary between Rs. 94 L - 1.34 Cr"
#   "ranging between 1,245 - 1,865 sqft"

def parse_description(desc: str) -> dict:
    """Extract BHK, price range, area range from description text."""
    if not isinstance(desc, str) or not desc.strip():
        return {}
    out = {}

    # BHK → e.g. "2,3 BHK" or "1,2,3 BHK"
    m = re.search(r'(\d(?:\s*,\s*\d)*)\s*BHK', desc, re.IGNORECASE)
    if m:
        bhk_list  = [int(x.strip()) for x in m.group(1).split(',')]
        out['bedRoom_min'] = min(bhk_list)
        out['bedRoom_max'] = max(bhk_list)

    # Price range → "Rs. 94 L - 1.34 Cr" / "1.23 - 2.19 Cr" / "59.14 L - 1.77 Cr"
    m = re.search(
        r'Rs\.?\s*([\d.]+)\s*(L|Lac|Cr)?\s*[-–]\s*([\d.]+)\s*(L|Lac|Cr)',
        desc, re.IGNORECASE)
    if m:
        lo, lo_u, hi, hi_u = m.groups()
        lo_u = (lo_u or hi_u).upper()
        hi_u = hi_u.upper()
        out['price_min_cr'] = float(lo) / (100 if lo_u.startswith('L') else 1)
        out['price_max_cr'] = float(hi) / (100 if hi_u.startswith('L') else 1)

    # Area range → "1,245 - 1,865 sqft"
    m = re.search(
        r'([\d,]+)\s*[-–]\s*([\d,]+)\s*(sqft|sq\.\s*ft\.?)',
        desc, re.IGNORECASE)
    if m:
        out['area_min_sqft'] = int(m.group(1).replace(',', ''))
        out['area_max_sqft'] = int(m.group(2).replace(',', ''))

    return out


# ─────────────────────────────────────────────────────────────
# EMPTINESS CHECK  —  skip rows with no useful data
# ─────────────────────────────────────────────────────────────

def _is_empty(record: dict) -> bool:
    """A record is 'empty' if it has no price AND no bedRoom AND no area data."""
    has_price = (record.get('price') or '').strip()
    has_bed   = (record.get('bedRoom') or '').strip()
    has_area  = ((record.get('areaWithType') or '').strip() or
                 (record.get('area') or '').strip())
    has_desc_data = bool(parse_description(record.get('description', '')))
    return not (has_price or has_bed or has_area or has_desc_data)


# ─────────────────────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────────────────────

def _save(records: list, filename: str):
    """Append records to CSV, dropping duplicates and empty rows."""
    if not records:
        return
    df = pd.DataFrame(records)

    # Remove rows where everything important is empty
    df = df[~df.apply(lambda r: _is_empty(r.to_dict()), axis=1)]
    if df.empty:
        print(f"  [SKIP] All {len(records)} rows were empty.")
        return

    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.isfile(path):
        # Load existing IDs and skip duplicates
        existing = pd.read_csv(path, dtype=str)
        if 'property_id' in existing.columns:
            existing_ids = set(existing['property_id'].astype(str))
            df = df[~df['property_id'].astype(str).isin(existing_ids)]
        if df.empty:
            print(f"  [SKIP] All rows already in {path}")
            return
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, mode='w', header=True, index=False)
    print(f"  [SAVED] {len(df)} new rows → {path}")


def _load_seen_ids(filename: str) -> set:
    """Return set of property_ids already in the CSV (for resume)."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        return set()
    try:
        df = pd.read_csv(path, dtype=str, usecols=['property_id'])
        return set(df['property_id'].astype(str).dropna())
    except Exception:
        return set()


def _output_name(city: str, ptype: str) -> str:
    return f"{_city_slug(city)}_{ptype}.csv"


# ─────────────────────────────────────────────────────────────
# DETAIL PAGE PARSER  —  works for both individual + project pages
# ─────────────────────────────────────────────────────────────

def _parse_detail(dsoup, fallback: dict) -> dict:
    """
    Extract all fields from a detail page. Falls back to listing-card
    values (passed in 'fallback') for fields not present on this page type.
    """
    desc       = _text(dsoup, '#description') or fallback.get('description', '')
    price      = _text(dsoup, '#pdPrice2')   or fallback.get('price', '')
    bedRoom    = _text(dsoup, '#bedRoomNum')
    bathroom   = _text(dsoup, '#bathroomNum')
    balcony    = _text(dsoup, '#balconyNum')
    areaWithType = _text(dsoup, '#factArea') or fallback.get('areaWithType', '')

    # If individual fields missing, try parsing description
    parsed = parse_description(desc)
    if not bedRoom and 'bedRoom_min' in parsed:
        if parsed['bedRoom_min'] == parsed['bedRoom_max']:
            bedRoom = f"{parsed['bedRoom_min']} Bedrooms"
        else:
            bedRoom = f"{parsed['bedRoom_min']}-{parsed['bedRoom_max']} BHK"
    if not price and 'price_min_cr' in parsed:
        price = (f"₹ {parsed['price_min_cr']} - "
                 f"{parsed['price_max_cr']} Cr")
    if not areaWithType and 'area_min_sqft' in parsed:
        areaWithType = (f"{parsed['area_min_sqft']:,} - "
                        f"{parsed['area_max_sqft']:,} sqft")

    return {
        'price':           price,
        'area':            _text(dsoup, '#srp_tuple_price_per_unit_area') or
                           fallback.get('area', ''),
        'areaWithType':    areaWithType,
        'bedRoom':         bedRoom,
        'bathroom':        bathroom,
        'balcony':         balcony,
        'additionalRoom':  _text(dsoup, '#additionalRooms'),
        'address':         _text(dsoup, '#address'),
        'floorNum':        _text(dsoup, '#floorNumLabel'),
        'facing':          _text(dsoup, '#facingLabel'),
        'agePossession':   _text(dsoup, '#agePossessionLbl') or
                           fallback.get('agePossession', ''),
        'nearbyLocations': _list(dsoup,
                                 'div.NearByLocation__tagWrap',
                                 'span.NearByLocation__infoText'),
        'description':     desc,
        'furnishDetails':  _list(dsoup, '#FurnishDetails', 'li'),
        'features':        _features(dsoup),
        'rating':          _ratings(dsoup),
    }


# ─────────────────────────────────────────────────────────────
# LISTING CARD PARSER  —  works on the new tupleNew__ layout
# ─────────────────────────────────────────────────────────────

def _parse_listing_card(sec) -> dict:
    """
    Extract whatever's available from a single listing card (search-page).
    Handles both individual (FSL_TUPLE) and project (GROUPED_PROJECT_TUPLE).
    Returns dict with at minimum: link, property_id, property_name.
    """
    outer_id = sec.find('div', id=True)
    prop_id  = outer_id['id'].lstrip('DZ') if outer_id else ''

    # Individual listing
    link_el = sec.select_one('a.tupleNew__propertyHeading')
    if link_el:
        link      = link_el['href']
        prop_name = (_text(sec, 'h2.tupleNew__propType') or
                     link_el.get_text(strip=True))
        society   = _text(sec, 'div.tupleNew__locationName')
        psqft     = _text(sec, 'div.tupleNew__perSqftWrap')
        price_raw = _text(sec, 'div.tupleNew__priceValWrap span')
        area_els  = sec.select('span.tupleNew__area1Type')
        area_raw  = area_els[0].text.strip() if area_els else ''
        possess   = _text(sec, 'div.tupleNew__possessionBy')
    else:
        # Project/society page
        any_link = sec.find('a', href=lambda h: h and '99acres.com' in h)
        if not any_link:
            return None
        link      = any_link['href']
        society   = (_text(sec, 'div.tupleNew__locationName') or
                     _text(sec, '.PseudoTupleRevamp__heading a'))
        prop_name = society or link.split('/')[-1].replace('-', ' ').title()
        price_raw = _text(sec, 'div.tupleNew__priceValWrap span')
        area_els  = sec.select('span.tupleNew__area1Type')
        area_raw  = area_els[0].text.strip() if area_els else ''
        psqft     = _text(sec, 'div.tupleNew__perSqftWrap')
        possess   = _text(sec, 'div.tupleNew__possessionBy')

    return {
        'property_id':   prop_id,
        'property_name': prop_name,
        'link':          link,
        'society':       society,
        '_card_price':   price_raw,
        '_card_psqft':   psqft,
        '_card_area':    area_raw,
        '_card_possess': possess,
        '_card_desc':    _text(sec, 'p.tupleNew__descText'),
    }


# ─────────────────────────────────────────────────────────────
# CONCURRENT DETAIL PAGE FETCHER
# ─────────────────────────────────────────────────────────────

def _fetch_details_parallel(card_records: list, referer: str,
                            workers: int = WORKERS) -> list:
    """Fetch multiple detail pages in parallel (only with ScraperAPI)."""
    if not SCRAPER_API_KEY or workers <= 1:
        # Fall back to sequential for non-ScraperAPI modes
        results = []
        for i, card in enumerate(card_records, 1):
            res = _fetch_one_detail(card, referer)
            if res:
                results.append(res)
            print(f"    [{i}/{len(card_records)}] {res['property_name'][:50] if res else 'FAILED'}")
        return results

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one_detail, c, referer): c
                   for c in card_records}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result(timeout=120)
                if res:
                    results.append(res)
                print(f"    [{i}/{len(card_records)}] "
                      f"{res['property_name'][:50] if res else 'FAILED'}")
            except Exception as e:
                print(f"    [{i}/{len(card_records)}] ERROR: {e}")
    return results


def _fetch_one_detail(card: dict, referer: str) -> dict:
    """Fetch + parse a single detail page, merge with card data."""
    dresp = _fetch(card['link'], referer)
    if dresp is None:
        return None

    dsoup = BeautifulSoup(dresp.content, 'html.parser')
    fallback = {
        'price':         card.get('_card_price', ''),
        'area':          card.get('_card_psqft', ''),
        'areaWithType':  card.get('_card_area', ''),
        'agePossession': card.get('_card_possess', ''),
        'description':   card.get('_card_desc', ''),
    }
    detail = _parse_detail(dsoup, fallback)

    return {
        'property_name':  card['property_name'],
        'link':           card['link'],
        'society':        card['society'],
        **detail,
        'property_id':    _text(dsoup, '#Prop_Id') or card['property_id'],
    }


# ─────────────────────────────────────────────────────────────
# SCRAPER 2 — Flats  (most-used scraper, similar pattern for others)
# ─────────────────────────────────────────────────────────────

def scrape_flats(city=DEFAULT_CITY, start=START_PAGE, end=END_PAGE,
                 output=None, workers=WORKERS):
    city    = _city_slug(city)
    output  = output or _output_name(city, 'flats')
    referer = f'https://www.99acres.com/flats-in-{city}-ffid'

    print(f"\n{'='*60}")
    print(f"  FLATS | city={city} | pages {start}–{end-1}")
    print(f"  Output → {output}")
    backend = 'ScraperAPI' if SCRAPER_API_KEY else MODE
    print(f"  Backend: {backend}  Workers: {workers if SCRAPER_API_KEY else 1}")
    print(f"{'='*60}")

    seen_ids  = _load_seen_ids(output)
    if seen_ids:
        print(f"  Found {len(seen_ids)} property_ids already in CSV — will skip them.")

    records   = []
    req_count = 0

    try:
        for page_num in range(start, end):
            url = f'https://www.99acres.com/flats-in-{city}-ffid-page-{page_num}'
            print(f"\n[Page {page_num}]")

            resp = _fetch(url, referer)
            if resp is None:
                print(f"  Page fetch failed — skipping.")
                continue

            req_count += 1
            psoup      = BeautifulSoup(resp.content, 'html.parser')
            search_div = psoup.select_one('div[data-label="SEARCH"]')

            # Retry once on Chrome/network errors
            if search_div is None:
                if any(k in resp.text[:300].upper()
                       for k in ('CHROME-ERROR', 'NETERROR', 'ERR_')):
                    print(f"  Chrome failed — waiting 30s and retrying once...")
                    time.sleep(30)
                    resp = _fetch(url, referer)
                    if resp is not None:
                        psoup      = BeautifulSoup(resp.content, 'html.parser')
                        search_div = psoup.select_one('div[data-label="SEARCH"]')

            if search_div is None:
                print(f"  Search div not found (status={resp.status_code}).")
                print(f"  Snippet: {resp.text[:200]}")
                print(f"  Resume with --start {page_num}")
                break

            sections = search_div.select(
                'section[data-hydration-on-demand="true"]')

            # Parse listing cards
            cards = []
            for sec in sections:
                card = _parse_listing_card(sec)
                if card is None:
                    continue
                if card['property_id'] in seen_ids:
                    continue   # already scraped
                seen_ids.add(card['property_id'])
                cards.append(card)

            print(f"  Page {page_num}: {len(cards)} new cards "
                  f"(of {len(sections)} on page)")

            if not cards:
                continue

            # Fetch detail pages (parallel if ScraperAPI)
            page_records = _fetch_details_parallel(cards, referer, workers)
            records.extend(page_records)
            req_count += len(cards)

            print(f"  Page {page_num}: {len(page_records)} flats detailed | "
                  f"Total: {len(records)}")

            # Batch save
            if (page_num - start + 1) % PAGES_PER_BATCH == 0:
                _save(page_records, output)
                page_records.clear()

            _sleep(req_count)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving collected data...")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
    finally:
        _save(records, output)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# SCRAPER 3 — Houses
# ─────────────────────────────────────────────────────────────

def scrape_houses(city=DEFAULT_CITY, start=START_PAGE, end=END_PAGE,
                  output=None, workers=WORKERS):
    """Same logic as scrape_flats but with 'rate' / 'noOfFloor' columns."""
    city    = _city_slug(city)
    output  = output or _output_name(city, 'houses')
    referer = f'https://www.99acres.com/independent-house-in-{city}-ffid'

    print(f"\n{'='*60}")
    print(f"  HOUSES | city={city} | pages {start}–{end-1}")
    print(f"  Output → {output}")
    print(f"{'='*60}")

    seen_ids = _load_seen_ids(output)
    records  = []
    req_count = 0

    try:
        for page_num in range(start, end):
            url = (f'https://www.99acres.com/'
                   f'independent-house-in-{city}-ffid-page-{page_num}')
            print(f"\n[Page {page_num}]")

            resp = _fetch(url, referer)
            if resp is None:
                continue

            req_count += 1
            psoup      = BeautifulSoup(resp.content, 'html.parser')
            search_div = psoup.select_one('div[data-label="SEARCH"]')
            if search_div is None:
                print(f"  Search div not found. Resume with --start {page_num}")
                break

            sections = search_div.select(
                'section[data-hydration-on-demand="true"]')
            cards = []
            for sec in sections:
                card = _parse_listing_card(sec)
                if card is None or card['property_id'] in seen_ids:
                    continue
                seen_ids.add(card['property_id'])
                cards.append(card)

            print(f"  Page {page_num}: {len(cards)} new cards")
            if not cards:
                continue

            page_records = _fetch_details_parallel(cards, referer, workers)
            # Add house-specific columns
            for r in page_records:
                r['rate']      = r.pop('area', '')
                r['noOfFloor'] = r.pop('floorNum', '')
            records.extend(page_records)
            req_count += len(cards)

            if (page_num - start + 1) % PAGES_PER_BATCH == 0:
                _save(page_records, output)
                page_records.clear()
            _sleep(req_count)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving...")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
    finally:
        _save(records, output)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# SCRAPER 4 — Land
# ─────────────────────────────────────────────────────────────

def scrape_land(city=DEFAULT_CITY, start=START_PAGE, end=END_PAGE,
                output=None, workers=WORKERS):
    city    = _city_slug(city)
    output  = output or _output_name(city, 'residential_land')
    referer = f'https://www.99acres.com/residential-land-in-{city}-ffid'

    print(f"\n{'='*60}")
    print(f"  LAND | city={city} | pages {start}–{end-1}")
    print(f"  Output → {output}")
    print(f"{'='*60}")

    seen_ids = _load_seen_ids(output)
    records  = []
    req_count = 0

    try:
        for page_num in range(start, end):
            url = (f'https://www.99acres.com/'
                   f'residential-land-in-{city}-ffid-page-{page_num}')
            print(f"\n[Page {page_num}]")

            resp = _fetch(url, referer)
            if resp is None:
                continue

            req_count += 1
            psoup      = BeautifulSoup(resp.content, 'html.parser')
            search_div = psoup.select_one('div[data-label="SEARCH"]')
            if search_div is None:
                print(f"  Search div not found. Resume with --start {page_num}")
                break

            sections = search_div.select(
                'section[data-hydration-on-demand="true"]')
            cards = []
            for sec in sections:
                card = _parse_listing_card(sec)
                if card is None or card['property_id'] in seen_ids:
                    continue
                seen_ids.add(card['property_id'])
                cards.append(card)

            print(f"  Page {page_num}: {len(cards)} new cards")
            if not cards:
                continue

            page_records = _fetch_details_parallel(cards, referer, workers)
            # Land-specific columns
            land_records = []
            for r in page_records:
                land_records.append({
                    'property_name':   r['property_name'],
                    'link':            r['link'],
                    'society':         r['society'],
                    'price':           r.get('price', ''),
                    'areaWithType':    r.get('areaWithType', ''),
                    'address':         r.get('address', ''),
                    'floorNumAllowed': r.get('floorNum', ''),
                    'noOfOpenSides':   '',
                    'possession':      r.get('agePossession', ''),
                    'nearbyLocations': r.get('nearbyLocations', []),
                    'description':     r.get('description', ''),
                    'features':        r.get('features', []),
                    'rating':          r.get('rating', []),
                    'property_id':     r.get('property_id', ''),
                })
            records.extend(land_records)
            req_count += len(cards)

            if (page_num - start + 1) % PAGES_PER_BATCH == 0:
                _save(land_records, output)
                land_records.clear()
            _sleep(req_count)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving...")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
    finally:
        _save(records, output)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# SCRAPER 1 — Apartments / Society listings
# ─────────────────────────────────────────────────────────────

def scrape_apartments(city=DEFAULT_CITY, start=START_PAGE, end=END_PAGE,
                      output=None, workers=WORKERS):
    """Society-level scraper (each row = a project, not a unit)."""
    city    = _city_slug(city)
    output  = output or _output_name(city, 'appartments')
    referer = f'https://www.99acres.com/property-in-{city}-ffid'

    print(f"\n{'='*60}")
    print(f"  APARTMENTS | city={city} | pages {start}–{end-1}")
    print(f"  Output → {output}")
    print(f"{'='*60}")

    seen_ids = _load_seen_ids(output)
    records  = []
    req_count = 0

    try:
        for page_num in range(start, end):
            url = f'https://www.99acres.com/property-in-{city}-ffid-page-{page_num}'
            print(f"\n[Page {page_num}]")

            resp = _fetch(url, referer)
            if resp is None:
                continue

            req_count += 1
            psoup      = BeautifulSoup(resp.content, 'html.parser')
            search_div = psoup.select_one('div[data-label="SEARCH"]')
            if search_div is None:
                print(f"  Search div not found. Resume with --start {page_num}")
                break

            sections = search_div.select(
                'section[data-hydration-on-demand="true"]')
            cards = []
            for sec in sections:
                card = _parse_listing_card(sec)
                if card is None or card['property_id'] in seen_ids:
                    continue
                seen_ids.add(card['property_id'])
                cards.append(card)

            print(f"  Page {page_num}: {len(cards)} new cards")
            if not cards:
                continue

            page_records = _fetch_details_parallel(cards, referer, workers)
            records.extend(page_records)
            req_count += len(cards)

            if (page_num - start + 1) % PAGES_PER_BATCH == 0:
                _save(page_records, output)
                page_records.clear()
            _sleep(req_count)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving...")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
    finally:
        _save(records, output)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# CLEANUP TOOL  —  fix existing scraped CSV
# ─────────────────────────────────────────────────────────────

def clean_existing_csv(input_path: str, output_path: str = None):
    """
    Fix a scraped CSV that has many empty rows:
      1. Parse descriptions to fill missing bedRoom/price/areaWithType
      2. Drop fully-empty rows
      3. Deduplicate by property_id
    """
    if output_path is None:
        output_path = input_path.replace('.csv', '_cleaned.csv')

    print(f"\n[CLEAN] {input_path} → {output_path}")
    df = pd.read_csv(input_path, dtype=str).fillna('')
    print(f"  Loaded {len(df)} rows")

    # Parse descriptions to fill gaps
    filled_bed   = filled_price = filled_area = 0
    for idx, row in df.iterrows():
        if (not row.get('bedRoom', '').strip() or
            not row.get('price', '').strip() or
            not row.get('areaWithType', '').strip()):
            parsed = parse_description(row.get('description', ''))
            if not row['bedRoom'].strip() and 'bedRoom_min' in parsed:
                if parsed['bedRoom_min'] == parsed['bedRoom_max']:
                    df.at[idx, 'bedRoom'] = f"{parsed['bedRoom_min']} Bedrooms"
                else:
                    df.at[idx, 'bedRoom'] = (
                        f"{parsed['bedRoom_min']}-{parsed['bedRoom_max']} BHK")
                filled_bed += 1
            if not row['price'].strip() and 'price_min_cr' in parsed:
                df.at[idx, 'price'] = (
                    f"₹ {parsed['price_min_cr']} - {parsed['price_max_cr']} Cr")
                filled_price += 1
            if not row['areaWithType'].strip() and 'area_min_sqft' in parsed:
                df.at[idx, 'areaWithType'] = (
                    f"{parsed['area_min_sqft']:,} - "
                    f"{parsed['area_max_sqft']:,} sqft")
                filled_area += 1

    print(f"  Filled from descriptions: "
          f"{filled_bed} bedRoom, {filled_price} price, {filled_area} area")

    # Drop fully empty rows
    before = len(df)
    df = df[~df.apply(lambda r: _is_empty(r.to_dict()), axis=1)]
    print(f"  Dropped {before - len(df)} fully-empty rows")

    # Dedup
    before = len(df)
    df = df.drop_duplicates(subset=['property_id'])
    print(f"  Dropped {before - len(df)} duplicate property_ids")

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df)} rows → {output_path}")
    return df


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city',   default=DEFAULT_CITY)
    parser.add_argument('--type',
                        choices=['apartments','flats','houses','land','all','clean'],
                        default='flats')
    parser.add_argument('--start',  type=int, default=START_PAGE)
    parser.add_argument('--end',    type=int, default=END_PAGE)
    parser.add_argument('--mode',   choices=['requests','selenium'],
                        default=MODE)
    parser.add_argument('--api-key', default=None,
                        help='ScraperAPI key (recommended)')
    parser.add_argument('--workers', type=int, default=WORKERS,
                        help='Parallel detail-page fetches (ScraperAPI only)')
    parser.add_argument('--input',  default=None,
                        help='For --type clean: input CSV path')
    args = parser.parse_args()

    MODE            = args.mode
    SCRAPER_API_KEY = args.api_key or SCRAPER_API_KEY
    WORKERS         = args.workers

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"  City   : {args.city}")
    print(f"  Type   : {args.type}")
    if args.type != 'clean':
        print(f"  Pages  : {args.start} → {args.end}")
        print(f"  Mode   : {'ScraperAPI' if SCRAPER_API_KEY else MODE}")
        print(f"  Workers: {WORKERS if SCRAPER_API_KEY else 1}")

    if args.type == 'clean':
        if not args.input:
            args.input = _output_name(args.city, 'flats')
        clean_existing_csv(args.input)
    elif args.type in ('apartments', 'all'):
        scrape_apartments(args.city, args.start, args.end, workers=WORKERS)
    if args.type in ('flats', 'all'):
        scrape_flats(args.city, args.start, args.end, workers=WORKERS)
    if args.type in ('houses', 'all'):
        scrape_houses(args.city, args.start, args.end, workers=WORKERS)
    if args.type in ('land', 'all'):
        scrape_land(args.city, args.start, args.end, workers=WORKERS)

    print("\n✓ Done.")
