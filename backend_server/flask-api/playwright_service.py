from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import ssl
import socket
from utils import logger

REQUEST_TIMEOUT = 15000


class PlaywrightService:
    def __init__(self):
        self.p = None
        self.browser = None
        self.context = None

    def start(self):
        logger.info("Starting Playwright")
        self.p = sync_playwright().start()
        self.browser = self.p.chromium.launch(
            headless=True,
            args=["--no-sandbox"]
        )
        self.context = self.browser.new_context()

    def stop(self):
        logger.info("Stopping Playwright")
        self.context.close()
        self.browser.close()
        self.p.stop()

    def extract(self, url: str):
        logger.info(f"Extracting URL: {url}")

        page = self.context.new_page()

        parsed = urlparse(url)
        page_domain = parsed.netloc
        domain = parsed.hostname

        redirect_chain = 0
        server_redirect_count = 0
        third_party = set()

        def request_listener(request):
            nonlocal redirect_chain
            host = urlparse(request.url).netloc
            if host and host != page_domain:
                third_party.add(host)
            if request.is_navigation_request():
                redirect_chain += 1

        def response_listener(response):
            nonlocal server_redirect_count
            if response.status in [300, 301, 302, 303, 307, 308]:
                server_redirect_count += 1

        page.on("request", request_listener)
        page.on("response", response_listener)

        logger.info("Injecting scripts")

        page.add_init_script("""
            window.__eval_used = false;
            const originalEval = window.eval;
            window.eval = function() {
                window.__eval_used = true;
                return originalEval.apply(this, arguments);
            };

            window.__canvas_fp = false;
            const originalCanvas = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function() {
                window.__canvas_fp = true;
                return originalCanvas.apply(this, arguments);
            };
        """)

        logger.info("Navigating")

        resp = page.goto(url, timeout=REQUEST_TIMEOUT, wait_until="domcontentloaded")

        if resp is None or resp.status >= 400:
            page.close()
            raise Exception("http_error")

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        logger.info("Building features")

        features = {
            "redirect_count": max(redirect_chain - 1, 0),
            "server_redirect_count": server_redirect_count,
            "third_party_domains": len(third_party),
            "login_form": 1 if soup.find("form") else 0,
            "password_input": 1 if soup.find("input", {"type": "password"}) else 0,
            "num_inputs": len(soup.find_all("input")),
            "num_iframes": page.evaluate("document.querySelectorAll('iframe').length"),
            "external_scripts": sum(
                1 for s in soup.find_all("script")
                if s.get("src") and page_domain not in s.get("src")
            ),
            "uses_eval": 1 if page.evaluate("window.__eval_used") else 0,
            "canvas_fingerprint": 1 if page.evaluate("window.__canvas_fp") else 0,
            "page_size": float(len(html) / 1024),
            "cert_valid": certificate_valid(domain)
        }

        page.close()
        return features


def certificate_valid(domain):
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(3)
            s.connect((domain, 443))
            s.getpeercert()
        return 1
    except:
        return 0