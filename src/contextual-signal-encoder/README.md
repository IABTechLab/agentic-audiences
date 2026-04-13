# Contextual Signal Encoder

Reference implementation for generating contextual embedding signals with the [Agentic Audiences](../../specs/v1.0/) protocol.

The [scoring service](../user-embedding-to-campaign-scoring/) consumes embeddings but the spec has no reference for **producing** them. This service fills that gap: content in, ORTB-compatible embedding out — with a **model descriptor envelope** so the receiving party knows the model name, version, embedding space, and metric needed to process the vector.

## Public lane / private lane

The encoder implements the "public lane" concept: a standardized open-source model ([sentence-transformers](https://sbert.net/)) that any party can use to produce and read embeddings in a shared embedding space. Private-lane providers can be added by implementing the `EmbeddingProvider` interface — the model descriptor travels with the embedding so the receiving party always knows what model was used.

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
        "model_descriptor": {
          "name": "all-MiniLM-L6-v2",
          "version": "2.0.0",
          "embedding_space_id": "aa://spaces/contextual/sentence-transformers/minilm-l6-v2",
          "metric": "cosine"
        }
      }
    }]
  },
  "source": "text",
  "content_length": 42
}
```

The `model_descriptor` is the interoperability primitive — it tells the receiving party:
- **name** + **version**: which model to load for reading/matching
- **embedding_space_id**: two vectors are only comparable within the same space
- **metric**: how to compute similarity (cosine, dot, l2)

The `data` object plugs directly into the scoring service:
```json
{"id": "bid-001", "user": {"id": "page", "data": [<response>.data]}, "top_k": 5}
```

## Adding a private-lane provider

Implement `EmbeddingProvider` with your proprietary model. The model descriptor ensures the receiving party knows how to process your embeddings:

```python
from app.providers.base import EmbeddingProvider, EmbeddingResult

class AcmeProvider(EmbeddingProvider):
    async def encode_text(self, text: str) -> EmbeddingResult:
        vector = my_proprietary_model.encode(text)
        return EmbeddingResult(
            vector=vector.tolist(),
            model="acme-contextual-v3",
            version="3.1.0",
            dimension=len(vector),
            embedding_space_id="aa://spaces/private/acme-corp/v3",
            metric="dot",
        )

    async def close(self) -> None:
        pass
```

## Development

```bash
pytest tests/ -v
```

```
├── app/
│   ├── main.py                      # FastAPI app (public lane default)
│   ├── models/encode.py             # ORTB models + ModelDescriptor envelope
│   ├── engine/
│   │   ├── encoder.py               # Encoding orchestrator
│   │   └── extractor.py             # URL text extraction
│   ├── providers/
│   │   ├── base.py                  # EmbeddingProvider interface
│   │   └── sentence_transformers.py # Public lane (open-source)
│   └── routes/encode.py
└── tests/test_encode.py
```
