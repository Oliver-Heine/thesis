from flask import Flask, request, jsonify
from playwright_service import PlaywrightService
from normalize import bucketize, build_output
from utils import logger

app = Flask(__name__)

playwright_service = PlaywrightService()
playwright_service.start()   # no asyncio anymore


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "Missing URL"}), 400

    try:
        logger.info(f"Analyzing {url}")

        features = playwright_service.extract(url)

        logger.info("Formatting output")
        formatted = build_output(url, features)

        return jsonify({
            "url": url,
            "formatted_result": formatted
        })

    except Exception as e:
        logger.exception("Error during analysis")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, threaded=False)