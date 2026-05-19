# Contextual Signal Encoder

Reference implementation for generating contextual embedding signals conforming to the [Agentic Audiences](../../specs/v1.0/) protocol and [embedding format schema](../../specs/v1.0/embedding_format.schema.json).

The [scoring service](../user-embedding-to-campaign-scoring/) consumes embeddings but the spec has no reference for **producing** them. This service fills that gap: content in, ORTB-compatible embedding out — with the spec-defined model metadata fields (`version`, `embedding_space_id`, `metric`) so the receiving party knows how to process the vector.

## Public lane / private lane

Ships with sentence-transformers as the "public lane" — a standardized open-source model any party can use. The `EmbeddingProvider` interface supports private-lane providers. The spec metadata fields travel with every embedding regardless of which lane produced it.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8081

curl -X POST http://localhost:8081/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "NFL playoff predictions: Chiefs vs Bills"}'
```

**Response:**
```json
{
  "data": {
    "id": "contextual-signal-encoder",
    "name": "contextual-embeddings",
    "segment": [{
      "id": "ctx-a1b2c3d4",
      "ext": {
        "ver": "1.0",
        "vector": [0.042, -0.118, "..."],
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "type": "context",
        "version": "2.0.0",
        "embedding_space_id": "aa://spaces/contextual/sentence-transformers/minilm-l6-v2",
        "metric": "cosine"
      }
    }]
  },
  "source": "text",
  "content_length": 42
}
```

The `version`, `embedding_space_id`, and `metric` fields come from `embedding_format.schema.json`. The `data` object plugs directly into the scoring service as a `user.data[]` entry.

## Adding a private-lane provider

```python
from app.providers.base import EmbeddingProvider, EmbeddingResult

class AcmeProvider(EmbeddingProvider):
    async def encode_text(self, text: str) -> EmbeddingResult:
        vector = my_model.encode(text)
        return EmbeddingResult(
            vector=vector.tolist(),
            model="acme-contextual-v3",
            version="3.1.0",
            dimension=len(vector),
            embedding_space_id="aa://spaces/private/acme-corp/v3",
            metric="dot",
        )
    async def close(self) -> None: pass
```

## Development

```bash
pytest tests/ -v
```

```
├── app/
│   ├── main.py                      # FastAPI app (public lane default)
│   ├── models/encode.py             # ORTB models using spec-defined fields
│   ├── engine/
│   │   ├── encoder.py               # Encoding orchestrator
│   │   └── extractor.py             # URL text extraction
│   ├── providers/
│   │   ├── base.py                  # EmbeddingProvider interface
│   │   └── sentence_transformers.py # Public lane
│   └── routes/encode.py
└── tests/test_encode.py
```
