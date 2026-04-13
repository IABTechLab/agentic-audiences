# Contextual Signal Encoder

A reference implementation for generating **contextual embedding signals** compatible with the [Agentic Audiences](../../specs/v1.0/) embedding exchange protocol.

The Agentic Audiences spec defines a protocol for exchanging and scoring embeddings across six signal types (identity, contextual, reinforcement, creative, inventory, intent). The [scoring service](../user-embedding-to-campaign-scoring/) consumes pre-computed embeddings — but the spec does not include a reference implementation for **producing** them. This service fills that gap for the **contextual** signal type.

## What It Does

Takes raw content (text, URLs, images, video) and produces contextual embedding vectors in the exact wire format the scoring service expects:

```
Content → Extract → Embed → ORTB Segment
```

The output is an OpenRTB-compatible `data` object that can be placed directly into a `user.data[]` array in a scoring request.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8081

# Health check
curl http://localhost:8081/health

# Encode text
curl -X POST http://localhost:8081/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "NFL playoff predictions: Chiefs vs Bills AFC Championship"}'
```

### Docker

```bash
docker build -t aa-contextual-encoder .
docker run -p 8081:8081 aa-contextual-encoder
```

## API

### POST /encode

Generate a contextual embedding from content.

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
    "segment": [
      {
        "id": "ctx-a1b2c3d4",
        "name": "contextual-text",
        "ext": {
          "ver": "1.0",
          "vector": [0.042, -0.118, 0.203, "..."],
          "model": "all-MiniLM-L6-v2",
          "dimension": 384,
          "type": "context"
        }
      }
    ]
  },
  "metadata": {
    "source": "text",
    "provider": "sentence_transformers",
    "content_length": 64,
    "iab_categories": null
  }
}
```

The `data` object is designed to be placed directly into an OpenRTB `user.data[]` array:

```json
{
  "id": "bid-req-001",
  "user": {
    "id": "page-context",
    "data": [<encoder response>.data]
  },
  "top_k": 5
}
```

#### Input Fields

| Field | Type | Description |
|---|---|---|
| `text` | string | Raw text content to encode |
| `url` | string | URL to fetch and extract text from |
| `image_url` | string | Image URL (requires Mixpeek provider) |
| `video_url` | string | Video URL (requires Mixpeek provider) |
| `provider` | string | Override the configured provider for this request |

At least one content field is required. Priority: `video_url` > `image_url` > `url` > `text`.

### POST /encode/batch

Encode multiple content items in a single request.

```bash
curl -X POST http://localhost:8081/encode/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "NFL playoff highlights"},
    {"text": "Electric vehicle comparison guide"},
    {"url": "https://example.com/recipe"}
  ]'
```

### GET /health

```json
{"status": "ok", "provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"}
```

## Providers

The encoder supports pluggable embedding backends:

### sentence_transformers (default)

Open-source, runs locally, text only. Uses [sentence-transformers](https://sbert.net/) models.

- Default model: `all-MiniLM-L6-v2` (384 dimensions)
- Any sentence-transformers model can be swapped in via config
- No API keys required
- Text content only (image/video requests return 400)

### mixpeek

Production multimodal encoding via [Mixpeek](https://mixpeek.com). Supports text, images, and video with optional IAB taxonomy enrichment.

```bash
export MIXPEEK_API_KEY=your_key
export MIXPEEK_NAMESPACE=your_namespace
export ENCODER_PROVIDER=mixpeek
```

When configured with an IAB taxonomy retriever, the Mixpeek provider returns IAB content category classifications alongside the embedding:

```json
{
  "metadata": {
    "iab_categories": [
      {"name": "American Football", "tier1": "Sports", "path": "Sports > American Football", "score": 0.92},
      {"name": "NFL", "tier1": "Sports", "path": "Sports > American Football > NFL", "score": 0.87}
    ]
  }
}
```

Features:
- Multimodal: text, image, and video content
- IAB v3.0 taxonomy classification
- Auto-discovers IAB retrievers in your namespace
- Production-grade latency and throughput

### Adding a Custom Provider

Implement the `EmbeddingProvider` interface:

```python
from app.providers.base import EmbeddingProvider, EmbeddingResult

class MyProvider(EmbeddingProvider):
    async def encode_text(self, text: str) -> EmbeddingResult:
        vector = my_model.encode(text)
        return EmbeddingResult(
            vector=vector.tolist(),
            model="my-model-v1",
            dimension=len(vector),
        )

    async def close(self) -> None:
        pass
```

## Multi-Model Encoding (Named Vectors)

The encoder supports running content through **multiple models simultaneously**, producing named vectors with full model metadata. This addresses the named vector / model permutation problem: instead of separate ORTB segments with string-based `model:type` partition keys, content is encoded once and stored as a single point with multiple named vectors, each independently queryable.

### POST /encode/multi

```bash
curl -X POST http://localhost:8081/encode/multi \
  -H "Content-Type: application/json" \
  -d '{
    "text": "NFL playoff bracket set after wild-card weekend",
    "image_url": "https://example.com/nfl-playoffs.jpg",
    "models": [
      {"name": "text-minilm", "provider": "sentence_transformers", "model": "all-MiniLM-L6-v2", "modality": "text"},
      {"name": "text-mpnet", "provider": "sentence_transformers", "model": "all-mpnet-base-v2", "modality": "text"},
      {"name": "image-multimodal", "provider": "mixpeek", "modality": "image"}
    ]
  }'
```

**Response includes:**

- **`named_vectors`** — one per model, each carrying metadata (architecture, pooling, normalization, training domain, version):

```json
{
  "named_vectors": [
    {
      "name": "text-minilm",
      "vector": "[384d]",
      "model": "all-MiniLM-L6-v2",
      "dimension": 384,
      "modality": "text",
      "metadata": {
        "embedding_type": "context",
        "architecture": "transformer",
        "pooling": "mean",
        "normalization": "l2_unit",
        "training_domain": ["general", "semantic-similarity"],
        "version": "2.0"
      }
    },
    {
      "name": "image-multimodal",
      "vector": "[1152d]",
      "model": "mixpeek-multimodal",
      "dimension": 1152,
      "modality": "image",
      "metadata": {
        "architecture": "multimodal-fusion",
        "training_domain": ["general", "multimodal", "adtech"]
      }
    }
  ]
}
```

- **`storage_example`** — shows how named vectors map to a single storage point with indexed payload metadata:

```json
{
  "point_id": "content-a1b2c3d4",
  "named_vectors": {
    "text-minilm": {"values": "[384d]", "dimension": 384},
    "text-mpnet": {"values": "[768d]", "dimension": 768},
    "image-multimodal": {"values": "[1152d]", "dimension": 1152}
  },
  "payload": {
    "models": {
      "text-minilm": {"model": "all-MiniLM-L6-v2", "architecture": "transformer", "version": "2.0"},
      "image-multimodal": {"model": "mixpeek-multimodal", "architecture": "multimodal-fusion", "version": "1.0"}
    }
  }
}
```

- **`ortb_data`** — backwards-compatible ORTB segments for the scoring service (each model maps to a separate segment, routed by `model:type` partition).

### Why named vectors matter

The current spec puts model routing into the scoring service via `model:type` partition keys. This breaks down when:

- **Multiple models per entity**: A page has text, image, and video embeddings — they should be one point, not three separate segments.
- **Model versioning**: Upgrading a model requires changing the partition key string, breaking existing campaign heads.
- **Metadata filtering**: You want to query "all text embeddings from transformer models with mean pooling" — not possible with string-based partitioning.
- **Cross-model search**: Find content where the text embedding matches AND the image embedding matches — requires named vectors on the same point.

Named vectors with payload metadata push model routing to the infrastructure layer, where it can be indexed, filtered, and versioned independently.

See `examples/named_vectors_multi_model.json` for a complete walkthrough.

## End-to-End Pipeline

The encoder is designed to work with the [scoring service](../user-embedding-to-campaign-scoring/):

```
┌─────────────────────┐     ┌─────────────────────┐
│  Contextual Signal  │     │   Campaign Scoring   │
│      Encoder        │     │      Service         │
│   (this service)    │     │  (scoring service)   │
│                     │     │                      │
│  Content ──► ORTB   │────►│  ORTB ──► Scores     │
│  segment with       │     │  against campaign    │
│  contextual vector  │     │  head vectors        │
└─────────────────────┘     └─────────────────────┘
        :8081                       :8080
```

1. **Register campaign heads** with the scoring service (these represent what campaigns want to target)
2. **Encode page content** with this encoder at bid time
3. **Score** the contextual embedding against campaign heads
4. **Bid/no-bid** decision by the DSP based on scores

See `examples/end_to_end_pipeline.json` for a complete walkthrough.

## Configuration

Via `config.yaml` (override path with `CONFIG_PATH` env var):

```yaml
encoder:
  provider: sentence_transformers    # or "mixpeek"
  model_name: all-MiniLM-L6-v2
  dimension: 384
  embedding_type: context            # wire type → resolved to "contextual" by scoring service
  max_tokens: 512

mixpeek:
  # api_key: set via MIXPEEK_API_KEY env var
  # namespace: set via MIXPEEK_NAMESPACE env var
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `CONFIG_PATH` | Path to YAML config | `config.yaml` |
| `ENCODER_PROVIDER` | Provider override | `sentence_transformers` |
| `ENCODER_MODEL` | Model name override | `all-MiniLM-L6-v2` |
| `MIXPEEK_API_KEY` | Mixpeek API key | — |
| `MIXPEEK_BASE_URL` | Mixpeek API base URL | `https://api.mixpeek.com` |
| `MIXPEEK_NAMESPACE` | Mixpeek namespace | — |
| `MIXPEEK_RETRIEVER_ID` | Explicit retriever ID (skips auto-discovery) | — |

## Development

```bash
pip install -r requirements.txt

# Run tests (mocked — no model download)
pytest tests/ -v

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload
```

### Project Structure

```
├── Dockerfile
├── requirements.txt
├── config.yaml
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── config.py            # Configuration loader
│   ├── models/
│   │   └── encode.py        # Request/response + ORTB models
│   ├── engine/
│   │   ├── encoder.py       # Single-model encoding orchestrator
│   │   ├── multi_encoder.py # Multi-model named vector encoding
│   │   └── extractor.py     # URL text extraction
│   ├── providers/
│   │   ├── base.py          # EmbeddingProvider interface
│   │   ├── sentence_transformers.py  # Open-source default
│   │   └── mixpeek.py       # Production multimodal
│   └── routes/
│       └── encode.py        # POST /encode, /encode/batch, /encode/multi
├── tests/
│   ├── test_encode.py              # Single-model unit tests
│   ├── test_multi_encode.py        # Multi-model / named vector tests
│   └── test_scoring_integration.py # Scoring service compatibility
└── examples/
    ├── sample_encode_request.json
    ├── sample_encode_response.json
    ├── end_to_end_pipeline.json
    └── named_vectors_multi_model.json
```
