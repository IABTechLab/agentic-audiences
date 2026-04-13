# Contextual Signal Encoder

Reference implementation for generating **contextual embedding signals** compatible with the [Agentic Audiences](../../specs/v1.0/) embedding exchange protocol.

The [scoring service](../user-embedding-to-campaign-scoring/) consumes pre-computed embeddings, but the spec does not include an implementation for **producing** them. This service fills that gap for the **contextual** signal type: content in, ORTB-compatible embedding out.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8081

# Encode text
curl -X POST http://localhost:8081/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "NFL playoff predictions: Chiefs vs Bills AFC Championship"}'
```

## API

### POST /encode

```bash
curl -X POST http://localhost:8081/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Latest electric vehicle reviews and Tesla Model Y comparison"}'
```

**Response:**
```json
{
  "data": {
    "id": "contextual-signal-encoder",
    "name": "contextual-embeddings",
    "segment": [{
      "id": "ctx-a1b2c3d4",
      "name": "contextual-text",
      "ext": {
        "ver": "1.0",
        "vector": [0.042, -0.118, 0.203, "..."],
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "type": "context"
      }
    }]
  },
  "source": "text",
  "content_length": 64
}
```

The `data` object goes directly into an OpenRTB `user.data[]` for scoring:

```json
{"id": "bid-001", "user": {"id": "page", "data": [<response>.data]}, "top_k": 5}
```

| Field | Type | Description |
|---|---|---|
| `text` | string | Raw text to encode |
| `url` | string | URL to fetch and extract text from |

### GET /health

```json
{"status": "ok", "provider": "sentence_transformers"}
```

## Architecture

```
Content ──► Extract ──► Embed ──► ORTB Segment ──► Scoring Service
                         │                              │
              sentence-transformers            POST /score (existing)
              (pluggable via EmbeddingProvider)
```

The provider interface (`app/providers/base.py`) is designed for extension — implement `EmbeddingProvider` to add custom embedding backends.

## Development

```bash
pip install -r requirements.txt
pytest tests/ -v
```

```
├── app/
│   ├── main.py                # FastAPI app
│   ├── models/encode.py       # ORTB-compatible models
│   ├── engine/
│   │   ├── encoder.py         # Encoding orchestrator
│   │   └── extractor.py       # URL text extraction
│   ├── providers/
│   │   ├── base.py            # EmbeddingProvider interface
│   │   └── sentence_transformers.py
│   └── routes/encode.py       # POST /encode
└── tests/test_encode.py
```
