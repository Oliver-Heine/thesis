import asyncio
import aiohttp
import csv
import tldextract
import math
import ssl
import socket
import whois
import time
import pickle
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import os
import logging

# =========================
# CONFIG (use env vars for Docker)
# =========================
NUM_TABS = int(os.getenv("NUM_TABS", 4))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 15000))
BROWSER_RESTART_INTERVAL = int(os.getenv("BROWSER_RESTART_INTERVAL", 100))
BENIGN_OUTPUT = os.getenv("BENIGN_OUTPUT", "benign_domains.csv")
MALICIOUS_OUTPUT = os.getenv("MALICIOUS_OUTPUT", "dataset_malicious.csv")
SEEN_DOMAINS_FILE = os.getenv("SEEN_DOMAINS_FILE", "seen_domains.pkl")
OPENPHISH_FEED = os.getenv("OPENPHISH_FEED", "https://openphish.com/feed.txt")
PHISHTANK_FEED = os.getenv("PHISHTANK_FEED", "https://data.phishtank.com/data/online-valid.csv")
SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", 300))  # seconds

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DISCORD_UPDATE_INTERVAL = int(os.getenv("DISCORD_UPDATE_INTERVAL", 3600))

# =========================
# GLOBAL STATE
# =========================
stats = {
    "benign":0,
    "malicious":0,
    "errors":0,
    "browser_fail":0,
    "start_time":time.time()
}
seen_domains = set()
pending_domains = set()
write_queue = asyncio.Queue()
malicious_queue = asyncio.Queue()
benign_queue = asyncio.Queue()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# =========================
# Persist seen_domains
# =========================
def save_seen_domains():
    with open(SEEN_DOMAINS_FILE, "wb") as f:
        pickle.dump(seen_domains, f)


def load_seen_domains():
    global seen_domains
    if os.path.exists(SEEN_DOMAINS_FILE):
        with open(SEEN_DOMAINS_FILE, "rb") as f:
            seen_domains = pickle.load(f)
    else:
        seen_domains = set()

# =========================
# Periodic autosave task
# =========================
async def autosave_seen_domains():
    while True:
        await asyncio.sleep(SAVE_INTERVAL)
        save_seen_domains()
        logger.info(f"Total seen domains {len(seen_domains)}")
        print(f"[{datetime.utcnow()}] Autosaved {len(seen_domains)} domains to {SEEN_DOMAINS_FILE}")


async def send_discord(msg: str):
    """Send a message to Discord webhook."""
    if not DISCORD_WEBHOOK:
        return
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                DISCORD_WEBHOOK,
                json={"content": msg}
            )
    except Exception as e:
        print("Discord webhook error:", e)


async def daily_report():
    """Send a hourly report every 24h."""
    while True:
        await asyncio.sleep(DISCORD_UPDATE_INTERVAL)
        msg = f"""
**Crawler Report**
Benign: {stats['benign']}
Malicious: {stats['malicious']}
Errors: {stats['errors']}
Browser Failures: {stats['browser_fail']}
Start Time: {time.ctime(stats['start_time'])}
"""
        await send_discord(msg)


# =========================
# Domain features
# =========================
def tld_entropy(domain):
    tld = tldextract.extract(domain).suffix
    if not tld:
        return 0
    probs = [tld.count(c)/len(tld) for c in set(tld)]
    return -sum(p*math.log2(p) for p in probs)


async def domain_age(domain):
    def _whois():
        try:
            w = whois.whois(domain)
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if not creation:
                return -1
            return (datetime.utcnow() - creation).days
        except:
            return -1
    return await asyncio.to_thread(_whois)


async def certificate_valid(domain):
    def _cert():
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                s.settimeout(3)
                s.connect((domain, 443))
                s.getpeercert()
            return 1
        except:
            return 0
    return await asyncio.to_thread(_cert)


# =========================
# Playwright feature extraction
# =========================
async def playwright_features(context, url):
    page = await context.new_page()
    parsed = urlparse(url)
    page_domain = parsed.netloc

    redirect_chain = []
    third_party = set()
    uses_eval = 0
    popup_window = 0
    location_change = 0
    canvas_fp = 0

    async def request_listener(request):
        host = urlparse(request.url).netloc
        if host and host != page_domain:
            third_party.add(host)

    page.on("request", request_listener)

    page.add_init_script("""
        window.__eval_used = false;
        const originalEval = window.eval;
        window.eval = function() {
            window.__eval_used = true;
            return originalEval.apply(this, arguments);
        };
        window.__location_changed = false;
        const originalAssign = window.location.assign;
        window.location.assign = function(val) {
            window.__location_changed = true;
            return originalAssign.apply(this, [val]);
        };
        window.__canvas_fp = false;
        const originalCanvas = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function() {
            window.__canvas_fp = true;
            return originalCanvas.apply(this, arguments);
        };
    """)

    try:
        resp = await page.goto(url, timeout=REQUEST_TIMEOUT, wait_until="domcontentloaded")
        if resp is None or resp.status >= 400:
            raise Exception(f"http_status_{resp.status if resp else 'none'}")
        redirect_chain.append(url)
        redirect_chain.append(resp.url)
    except Exception as e:
        #logger.error("Exception occured", exc_info=True)
        page.remove_listener("request", request_listener)
        await page.close()
        raise e

    html = await page.content()
    uses_eval = 1 if await page.evaluate("window.__eval_used") else 0
    location_change = 1 if await page.evaluate("window.__location_changed") else 0
    canvas_fp = 1 if await page.evaluate("window.__canvas_fp") else 0

    soup = BeautifulSoup(html, "html.parser")
    features = {
        "redirect_count": len(redirect_chain),
        "third_party_domains": len(third_party),
        "login_form": 1 if soup.find("form") else 0,
        "password_input": 1 if soup.find("input", {"type": "password"}) else 0,
        "num_inputs": len(soup.find_all("input")),
        "num_iframes": len(soup.find_all("iframe")),
        "external_scripts": sum(1 for s in soup.find_all("script") if s.get("src") and page_domain not in s.get("src")),
        "uses_eval": uses_eval,
        "popup_window": popup_window,
        "document_location_change": location_change,
        "canvas_fingerprint": canvas_fp,
        "page_size": len(html)/1024
    }

    page.remove_listener("request", request_listener)
    await page.close()
    return features


# =========================
# CSV writer with headers
# =========================
async def csv_writer():
    # Define feature labels
    feature_labels = [
        "redirect_count",
        "third_party_domains",
        "login_form",
        "password_input",
        "num_inputs",
        "num_iframes",
        "external_scripts",
        "uses_eval",
        "popup_window",
        "document_location_change",
        "canvas_fingerprint",
        "page_size",
        "tld_entropy",
        "cert_valid",
        "domain_age"
    ]

    # Prepare header row
    header = ["domain"] + feature_labels + ["label"]

    # Open files
    with open(BENIGN_OUTPUT, "a", newline="") as f_b, open(MALICIOUS_OUTPUT, "a", newline="") as f_m:
        writer_b = csv.writer(f_b)
        writer_m = csv.writer(f_m)

        # Write header if file is empty
        if os.stat(BENIGN_OUTPUT).st_size == 0:
            writer_b.writerow(header)
        if os.stat(MALICIOUS_OUTPUT).st_size == 0:
            writer_m.writerow(header)

        # Main loop
        while True:
            domain, features, label = await write_queue.get()
            row = [domain] + [features.get(f, "") for f in feature_labels] + [label]
            (writer_b if label == 0 else writer_m).writerow(row)
            write_queue.task_done()


# =========================
# Worker
# =========================
async def worker(worker_id):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        processed = 0

        while True:
            batch = []
            for _ in range(NUM_TABS):

                if not malicious_queue.empty():
                    batch.append(await malicious_queue.get())

                elif not benign_queue.empty():
                    batch.append(await benign_queue.get())

                else:
                    break

            if not batch:
                await asyncio.sleep(0.1)
                continue

            results = await asyncio.gather(*[playwright_features(context, domain) for domain, label in batch],
                                           return_exceptions=True)

            for (domain, label), res in zip(batch, results):
                if isinstance(res, Exception):
                    logging.error("Failed to extract features for domain %s", domain)
                    logging.error(res)
                    stats["browser_fail"] += 1
                    stats["errors"] += 1
                else:
                    res["tld_entropy"] = tld_entropy(domain)
                    res["cert_valid"] = await certificate_valid(domain)
                    res["domain_age"] = await domain_age(domain)
                    await write_queue.put((domain, res, label))
                    seen_domains.add(domain)
                    pending_domains.remove(domain)
                    stats["benign" if label == 0 else "malicious"] += 1
                if label == 1:
                    malicious_queue.task_done()
                else:
                    benign_queue.task_done()
                processed += 1

            if processed >= BROWSER_RESTART_INTERVAL:
                await context.close()
                await browser.close()
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                processed = 0


# =========================
# Feed ingestion
# =========================
async def fetch_feed(session, url, label):
    try:
        HEADERS = {
            "User-Agent": "Mozilla/5.0 (QRPhishCrawler Research Bot)"
        }

        async with session.get(url, headers=HEADERS) as r:
            text = await r.text()
            lines = text.splitlines()
            if label == 1 and url == PHISHTANK_FEED:
                for row in csv.reader(lines[1:]):
                    if len(row) > 1:
                        domain = row[1].strip()
                        if domain and domain not in seen_domains and domain not in pending_domains:
                            pending_domains.add(domain)
                            await malicious_queue.put((domain, label))
            else:
                for line in lines:
                    domain = line.strip()
                    if domain and domain not in seen_domains and domain not in pending_domains:
                        pending_domains.add(domain)
                        await malicious_queue.put((domain, label))
    except Exception as e:
        #logger.error("Unable to fetch feed for {}".format(url))
        print("Feed fetch error:", e)


async def load_benign(file, label=0):
    with open(file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                domain = row[1].strip()
                if domain and domain not in seen_domains and domain not in pending_domains:
                    pending_domains.add(domain)
                    await benign_queue.put((domain, label))


# =========================
# Periodic feed refresher
# =========================
async def feed_loop():
    headers = {
        "User-Agent": "Mozilla/5.0 (QRPhishCrawler Research Bot)"
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            try:
                logger.info("Refreshing phishing feeds")

                await asyncio.gather(
                    fetch_feed(session, OPENPHISH_FEED, 1),
                    fetch_feed(session, PHISHTANK_FEED, 1),
                )

                logger.info(
                    "Feed refresh complete | queue size: %s | seen urls: %s",
                    malicious_queue.qsize(),
                    len(seen_domains)
                )

            except Exception as e:
                print("Feed refresh error:", e)
                # logger.error("Feed refresh failed: %s", e)

            # Wait 30 minutes
            await asyncio.sleep(1800)

# =========================
# Main
# =========================
async def main():
    logger.info("Starting crawling domains")
    load_seen_domains()

    asyncio.create_task(feed_loop())
    asyncio.create_task(daily_report())
    asyncio.create_task(csv_writer())
    asyncio.create_task(autosave_seen_domains())

    async with aiohttp.ClientSession() as session:
        logger.info("downloading data")
        await asyncio.gather(
            load_benign("/app/benign_domains.csv", 0)
        )

    workers = [asyncio.create_task(worker(i)) for i in range(NUM_TABS)]

    await asyncio.Event().wait()

    save_seen_domains()


if __name__ == "__main__":
    asyncio.run(main())