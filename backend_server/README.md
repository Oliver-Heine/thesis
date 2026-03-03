# Backend Server — QR Quishing Detector

Spring Boot REST API that provides deep URL analysis for cases where the
on-device TFLite model is uncertain.

## Requirements

| Tool | Version |
|------|---------|
| JDK | 21+ |
| Maven | 3.9+ |
| Docker + Compose | 24+ (optional) |

---

## Local Development

### 1. Configure API keys

```bash
export OPENAI_API_KEY="sk-..."
export VT_API_KEY="<your-virustotal-key>"
```

Neither key is required for the server to start; missing keys cause the
corresponding check to be skipped (verdict defaults to `"uncertain"`).

### 2. Build and run

```bash
cd backend_server
mvn spring-boot:run
```

The server starts on **http://localhost:8080**.

### 3. Run tests

```bash
mvn test
```

---

## Docker

```bash
# Copy and fill in API keys:
cp .env.example .env

# Build and start:
docker compose up --build

# Stop:
docker compose down
```

The container runs as UID/GID 1000 with:
- `no-new-privileges`
- All Linux capabilities dropped
- Read-only root filesystem (writable `/tmp` via tmpfs)

---

## API Reference

### `POST /validate`

Validate a URL extracted from a QR code.

**Request**

```json
{
  "url": "https://example.com/path?param=value"
}
```

| Field | Type   | Constraints |
|-------|--------|-------------|
| url   | string | Required. Must start with `http://` or `https://`. Max 2048 chars. |

**Response (200 OK)**

```json
{
  "verdict": "benign",
  "confidence": 0.97,
  "features": {
    "redirect_count": 0,
    "final_url": "https://example.com/path?param=value",
    "has_login_form": false,
    "uses_eval": false,
    "final_url_https": true,
    "vt_flag": false,
    "page_title": "Example Domain"
  }
}
```

| Field      | Type   | Values |
|------------|--------|--------|
| verdict    | string | `"malicious"` \| `"benign"` \| `"uncertain"` |
| confidence | float  | 0.0 – 1.0 |
| features   | object | Extracted URL features |

**Error responses**

| Status | Reason |
|--------|--------|
| 400    | Invalid request body or SSRF-blocked URL |
| 429    | Rate limit exceeded (10 req / 60 s per IP) |
| 502    | Feature extraction failed (site unreachable) |
| 503    | LLM service unavailable |
| 500    | Unexpected internal error |

---

## Security Notes

- **SSRF prevention**: private/loopback IP addresses are blocked before any
  browser navigation.
- **Prompt injection**: URL features are serialised to JSON before being sent
  to the LLM — raw URL strings are never interpolated into the prompt.
- **Rate limiting**: simple in-memory fixed-window limiter (10 req / 60 s / IP).
- **API keys**: loaded from environment variables only; never logged.
