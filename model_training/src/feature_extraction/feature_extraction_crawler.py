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
import zipfile
import io

# =========================
# CONFIG (use env vars for Docker)
# =========================
NUM_TABS = int(os.getenv("NUM_TABS", 4))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 15000))
BROWSER_RESTART_INTERVAL = int(os.getenv("BROWSER_RESTART_INTERVAL", 100))
BENIGN_OUTPUT = os.getenv("BENIGN_OUTPUT", "benign_domains.csv")
MALICIOUS_OUTPUT = os.getenv("MALICIOUS_OUTPUT", "dataset_malicious.csv")
FAILED_OUTPUT = os.getenv("FAILED_OUTPUT", "failed_domains.csv")
SEEN_URLS_FILE = os.getenv("SEEN_URLS_FILE", "seen_urls.pkl")
OPENPHISH_FEED = os.getenv("OPENPHISH_FEED")
PHISHTANK_FEED = os.getenv("PHISHTANK_FEED")
PHISHTANK_USERNAME = os.getenv("PHISHTANK_USERNAME")
URLHAUS_FEED = os.getenv("URLHAUS_FEED")
SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", 300))  # seconds
FEED_REFRESH_TIME = int(os.getenv("FEED_REFRESH_TIME", 300))  # seconds

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DISCORD_UPDATE_INTERVAL = int(os.getenv("DISCORD_UPDATE_INTERVAL", 3600))

# =========================
# GLOBAL STATE
# =========================
stats = {
    "benign":0,
    "malicious":0,
    "errors":0,
    "dns_fail": 0,
    "timeout": 0,
    "http_error": 0,
    "connection_fail": 0,
    "other_error": 0,
    "start_time":time.time()
}
seen_urls = set()
pending_domains = set()
write_queue = asyncio.Queue()
malicious_queue = asyncio.Queue()
benign_queue = asyncio.Queue()
failed_queue = asyncio.Queue()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# =========================
# Persist seen_urls
# =========================
def save_seen_urls():
    with open(SEEN_URLS_FILE, "wb") as f:
        pickle.dump(seen_urls, f)


def load_seen_urls():
    global seen_urls
    if os.path.exists(SEEN_URLS_FILE):
        with open(SEEN_URLS_FILE, "rb") as f:
            seen_urls = pickle.load(f)
    else:
        seen_urls = set()

# =========================
# Periodic autosave task
# =========================
async def autosave_seen_urls():
    while True:
        await asyncio.sleep(SAVE_INTERVAL)
        save_seen_urls()
        logger.info(f"Total seen domains {len(seen_urls)}")
        print(f"[{datetime.utcnow()}] Autosaved {len(seen_urls)} domains to {SEEN_URLS_FILE}")


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
    """Send hourly report."""
    while True:
        await asyncio.sleep(DISCORD_UPDATE_INTERVAL)

        total_processed = stats["benign"] + stats["malicious"]
        elapsed = time.time() - stats["start_time"]
        rate = total_processed / elapsed if elapsed > 0 else 0

        remaining = malicious_queue.qsize() + benign_queue.qsize()
        eta = remaining / rate if rate > 0 else -1

        msg = f"""
**Crawler Report**
Processed: {total_processed}
Elapsed time: {format_eta(elapsed)}
Processing rate: {rate:.2f} domains/sec

Remaining: {remaining}
ETA: {format_eta(eta)}

Benign: {stats['benign']}
Benign Queue size: {benign_queue.qsize()}

Malicious: {stats['malicious']}
Malicious Queue size: {malicious_queue.qsize()}

Errors: {stats['errors']}

Timeouts: {stats['timeout']}
DNS Fail: {stats['dns_fail']}
Connection Fail: {stats['connection_fail']}
HTTP Errors: {stats['http_error']}
Other: {stats['other_error']}
Start Time: {time.ctime(stats['start_time'])}
"""
        await send_discord(msg)

def format_eta(seconds):
    if seconds < 0:
        return "N/A"
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m {secs}s"

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

    await page.add_init_script("""
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
        "page_size": float(len(html) / 1024)
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
        writer_b = csv.writer(f_b, quoting=csv.QUOTE_ALL)
        writer_m = csv.writer(f_m, quoting=csv.QUOTE_ALL)

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

async def failed_writer():
    header = ["timestamp", "domain", "label", "error_type", "error_message"]

    with open(FAILED_OUTPUT, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if empty
        if os.stat(FAILED_OUTPUT).st_size == 0:
            writer.writerow(header)
            f.flush()

        while True:
            domain, label, error_type, error_msg = await failed_queue.get()

            writer.writerow([
                datetime.utcnow().isoformat(),
                domain,
                label,
                error_type,
                error_msg[:200]  # truncate to avoid huge logs
            ])
            f.flush()

            failed_queue.task_done()

# =========================
# Worker
# =========================
async def worker(worker_id):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        processed = 0

        last_restart = time.time()

        while True:
            batch = []
            for _ in range(NUM_TABS):

                try:
                    batch.append(malicious_queue.get_nowait())
                except asyncio.QueueEmpty:
                    try:
                        batch.append(benign_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            if not batch:
                await asyncio.sleep(0.1)
                continue

            results = await asyncio.gather(
                *[safe_playwright(context, domain) for domain, label in batch],
                return_exceptions=True
            )

            for (domain, label), res in zip(batch, results):
                try:
                    if isinstance(res, Exception):
                        err_str = str(res).lower()

                        logger.error("Domain failed: %s | Error: %s", domain, err_str)

                        stats["errors"] += 1

                        if "timeout" in err_str:
                            error_type = "timeout"
                            stats["timeout"] += 1

                        elif "net::err_name_not_resolved" in err_str or "dns" in err_str:
                            error_type = "dns_fail"
                            stats["dns_fail"] += 1

                        elif "connection refused" in err_str or "net::err_connection" in err_str:
                            error_type = "connection_fail"
                            stats["connection_fail"] += 1

                        elif "http_status_" in err_str:
                            error_type = "http_error"
                            stats["http_error"] += 1

                        else:
                            error_type = "other"
                            stats["other_error"] += 1

                        await failed_queue.put((domain, label, error_type, str(res)))

                    else:
                        res["tld_entropy"] = tld_entropy(urlparse(domain).netloc)
                        res["cert_valid"] = await certificate_valid(urlparse(domain).netloc)
                        res["domain_age"] = await domain_age(urlparse(domain).netloc)

                        await write_queue.put((domain, res, label))

                        stats["benign" if label == 0 else "malicious"] += 1

                finally:
                    pending_domains.discard(domain)
                    seen_urls.add(domain)

                    if label == 1:
                        malicious_queue.task_done()
                    else:
                        benign_queue.task_done()

                    processed += 1

            if processed >= BROWSER_RESTART_INTERVAL or time.time() - last_restart > 300:
                await context.close()
                await browser.close()
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                processed = 0
                last_restart = time.time()

async def safe_playwright(context, url):
    try:
        return await asyncio.wait_for(
            playwright_features(context, url),
            timeout=20  # hard cap (seconds)
        )
    except asyncio.TimeoutError:
        raise Exception("timeout")

# =========================
# Feed ingestion
# =========================
async def fetch_feed_Phishtank(session, url, label):
    try:
        HEADERS = {
            "User-Agent": f"phishtank/{PHISHTANK_USERNAME or 'research'}"
        }

        async with session.get(url, headers=HEADERS) as r:
            if r.status != 200:
                logger.error(f"Feed fetch for {url} error:", r)
                return
            text = await r.text()
            lines = text.splitlines()

            for row in csv.reader(lines[1:]):
                if len(row) > 1:
                    url_entry = row[1].strip()
                    await enqueue_url(url_entry, label)
    except Exception as e:
        print("Feed fetch error:", e)

async def fetch_feed_Urlhause(session, url, label):
    try:
        async with session.get(url) as r:
            if r.status != 200:
                logger.error(f"Feed fetch for {url} failed with status {r.status}")
                return

            text = await r.text()

            reader = csv.reader(
                line for line in io.StringIO(text)
                if not line.startswith("#")
            )

            for row in reader:
                if len(row) < 3:
                    continue

                url_entry = row[2].strip()
                await enqueue_url(url_entry, label)

    except Exception as e:
        logger.error(f"Feed fetch error for {url}: {e}")

async def fetch_feed_Openphish(session, url, label):
    try:
        HEADERS = {
            "User-Agent": "Mozilla/5.0 (QRPhishCrawler Research Bot)"
        }

        async with session.get(url, headers=HEADERS) as r:
            if r.status != 200:
                logger.error(f"Feed fetch for {url} error:", r)
                return
            text = await r.text()
            lines = text.splitlines()
            for line in lines:
                url_entry = line.strip()
                await enqueue_url(url_entry, label)
    except Exception as e:
        print("Feed fetch error:", e)

async def enqueue_url(url_entry, label):
    if url_entry and url_entry not in seen_urls and url_entry not in pending_domains:
        pending_domains.add(url_entry)
        await malicious_queue.put((url_entry, label))

async def load_benign(file, label=0):
    with open(file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                url_entry = row[1].strip()
                if url_entry and url_entry not in seen_urls and url_entry not in pending_domains:
                    pending_domains.add(url_entry)
                    await benign_queue.put((url_entry, label))


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
                    fetch_feed_Openphish(session, OPENPHISH_FEED, 1),
                    fetch_feed_Phishtank(session, PHISHTANK_FEED, 1),
                    fetch_feed_Urlhause(session, URLHAUS_FEED, 1)
                )

                logger.info(
                    "Feed refresh complete | queue size: %s | seen urls: %s",
                    malicious_queue.qsize(),
                    len(seen_urls)
                )

            except Exception as e:
                print("Feed refresh error:", e)
                # logger.error("Feed refresh failed: %s", e)

            # Wait 30 minutes
            await asyncio.sleep(FEED_REFRESH_TIME)

# =========================
# Main
# =========================
async def main():
    logger.info("Starting crawling domains")
    load_seen_urls()

    asyncio.create_task(feed_loop())
    asyncio.create_task(daily_report())
    asyncio.create_task(csv_writer())
    asyncio.create_task(failed_writer())
    asyncio.create_task(autosave_seen_urls())

    async with aiohttp.ClientSession() as session:
        logger.info("downloading data")
        await asyncio.gather(
            load_benign("/app/benign_domains.csv", 0)
        )

    workers = [asyncio.create_task(worker(i)) for i in range(NUM_TABS)]

    await asyncio.Event().wait()

    save_seen_urls()


if __name__ == "__main__":
    asyncio.run(main())