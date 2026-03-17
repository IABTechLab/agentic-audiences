# Agentic Audiences Scoring Service

A containerized embedding-similarity scoring service for RTB environments. Replaces traditional segment-ID lookups with vector similarity scoring of user embeddings against pre-trained campaign "heads."

The service is **not a bidder** — it returns scores only. Bid/no-bid decisions remain with the DSP.

## Quick Start

```bash
# Build
docker build -t aa-scoring .

# Run
docker run -p 8080:8080 aa-scoring

# Health check
curl http://localhost:8080/health
```

## API

### POST /campaigns/heads

Register campaign head weight vectors. Heads are partitioned by `model:embedding_type` and only scored against matching embeddings.

Model configuration (metric, normalization, compatibility) travels with the head registration — no pre-declaration required.

```bash
curl -X POST http://localhost:8080/campaigns/heads \
  -H "Content-Type: application/json" \
  -d '{
    "heads": [
      {
        "campaign_id": "camp-001",
        "campaign_head_id": "head-001a",
        "weights": [0.1, -0.2, 0.3, ...],
        "model": "minilm-l6-v2",
        "dimension": 384,
        "type": "contextual",
        "metric": "cosine",
        "apply_l2_norm": true,
        "embedding_space_id": "aa://spaces/contextual/en-v1",
        "compatible_with": ["sbert-mini-ctx-001"]
      }
    ]
  }'
```

**Response:**
```json
{
  "registered": 1,
  "ids": ["head-001a"]
}
```

#### Campaign Head Fields

| Field | Required | Default | Description |
|---|---|---|---|
| `campaign_id` | yes | — | Campaign identifier |
| `campaign_head_id` | yes | — | Unique head identifier |
| `weights` | yes | — | Weight vector. Length must equal `dimension`. |
| `model` | yes | — | Model name (e.g. `minilm-l6-v2`) |
| `dimension` | yes | — | Expected vector length. Must be consistent across all heads for the same model. |
| `type` | yes | — | Embedding type (canonical or alias) |
| `metric` | no | `cosine` | Similarity metric: `cosine`, `dot`, or `l2` |
| `apply_l2_norm` | no | `false` | L2-normalize vectors before scoring |
| `embedding_space_id` | no | `""` | Identifier for the embedding space (informational) |
| `compatible_with` | no | `[]` | Other model names whose heads can be scored against this model's embeddings. Bidirectional. |

Validation rules:
- `len(weights)` must equal `dimension`
- `dimension` must be consistent for all heads registered under the same `model` name
- `type` can be a canonical type or a wire-format alias (see [Type Aliases](#type-aliases))

### POST /score

Score user embeddings from an OpenRTB-shaped bid request against registered campaign heads.

```bash
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{
    "id": "req-12345",
    "user": {
      "id": "user-abc",
      "data": [
        {
          "id": "embedding-provider-1",
          "segment": [
            {
              "id": "seg-ctx-001",
              "ext": {
                "ver": "1.0",
                "vector": [0.15, -0.22, 0.31, ...],
                "model": "minilm-l6-v2",
                "dimension": 384,
                "type": "context"
              }
            }
          ]
        }
      ]
    },
    "top_k": 5
  }'
```

**Response:**
```json
[
  {
    "request_id": "req-12345",
    "scores": [
      {
        "campaign_id": "camp-001",
        "campaign_head_id": "head-001a",
        "score": 0.87432
      }
    ],
    "model": "minilm-l6-v2",
    "embedding_type": "contextual",
    "metric": "cosine"
  }
]
```

The response is a list — one entry per embedding found in the request. Each entry contains the top-K scored campaign heads sorted descending by score.

Embeddings are extracted from `user.data[].segment[].ext`. Segments without an `ext` field are ignored.

If no heads are registered for the given model+type, empty scores are returned (not an error).

### GET /analytics

Retrieve scoring analytics with PCA-reduced centroids bucketed by score percentile.

```bash
# All campaigns
curl http://localhost:8080/analytics

# Filter by campaign
curl http://localhost:8080/analytics?campaign_id=camp-001

# Filter by specific head
curl http://localhost:8080/analytics?campaign_head_id=head-001a
```

**Response:**
```json
{
  "campaigns": [
    {
      "campaign_id": "camp-001",
      "campaign_head_id": "head-001a",
      "total_scored": 500,
      "score_buckets": [
        {
          "bucket_label": "p0-p25",
          "count": 125,
          "reduced_centroid": [0.12, -0.45, 0.78]
        },
        {
          "bucket_label": "p25-p50",
          "count": 125,
          "reduced_centroid": [0.34, 0.11, -0.22]
        }
      ]
    }
  ],
  "reduction_method": "pca",
  "reduced_dimensions": 3
}
```

Each bucket's `reduced_centroid` is the PCA-reduced mean of all embeddings that scored in that percentile range. This answers: "embeddings that look like *this* score in the Nth percentile for this campaign head."

### GET /health

Returns `{"status": "ok"}`. Use for container orchestration liveness probes.

## Configuration

The service is configured via `config.yaml` (override path with `CONFIG_PATH` env var). Config is optional — the service starts with sensible defaults if no file is present.

```yaml
embedding_types:
  - identity
  - contextual
  - reinforcement
  - capi
  - intent

type_aliases:
  context: contextual
  creative: capi
  user_intent: intent
  query: intent
  inventory: contextual

analytics:
  pca_dimensions: 3
  score_buckets: [0, 25, 50, 75, 100]
  max_embeddings_stored: 10000
```

Model configuration (dimension, metric, normalization, compatibility) is no longer declared in config — it travels with campaign head registration. The first head registered for a model name establishes its configuration; subsequent heads for the same model must have a matching dimension.

### Type Aliases

Wire-format embedding types from OpenRTB extensions may differ from the canonical types used internally. The `type_aliases` map resolves them:

| Wire Type | Canonical Type |
|---|---|
| `context` | `contextual` |
| `creative` | `capi` |
| `user_intent` | `intent` |
| `query` | `intent` |
| `inventory` | `contextual` |

Campaign heads registered with type `contextual` will match score requests with type `context` (and vice versa). Heads and embeddings with different canonical types are never scored against each other.

### Analytics

| Field | Description |
|---|---|
| `pca_dimensions` | Number of PCA components for centroid reduction (default: 3). |
| `score_buckets` | Percentile boundaries for bucketing scores (default: 0/25/50/75/100). |
| `max_embeddings_stored` | Max scoring records kept per campaign head. FIFO eviction when exceeded. |

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `CONFIG_PATH` | Path to the YAML config file | `config.yaml` (relative to app root) |

## Scoring Metrics

| Metric | Formula | Notes |
|---|---|---|
| `cosine` | cos(embedding, head) | Range [-1, 1]. Ignores magnitude. |
| `dot` | embedding . head | Unbounded. Sensitive to magnitude. |
| `l2` | -\|\|embedding - head\|\| | Negated so higher = more similar. |

When `apply_l2_norm` is `true`, both vectors are unit-normalized before scoring. For cosine similarity, normalization is always applied regardless of this setting.

## Architecture

```
┌──────────────────────────────────────────────┐
│              FastAPI Application             │
├──────────┬──────────────┬────────────────────┤
│ POST     │ POST         │ GET                │
│ /score   │ /campaigns/  │ /analytics         │
│          │   heads      │                    │
├──────────┴──────────────┴────────────────────┤
│                  Engine                       │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐ │
│  │ Scorer  │  │  Store   │  │  Analytics   │ │
│  │ (NumPy) │  │ (dict)   │  │  (PCA/FIFO)  │ │
│  └─────────┘  └─────────┘  └──────────────┘ │
└──────────────────────────────────────────────┘
```

- **Scorer**: Batch NumPy similarity computation. Stacks heads into a matrix for vectorized scoring.
- **Store**: In-memory dict partitioned by `model:type`. Thread-safe via asyncio.Lock. Model config (metric, l2_norm, compatibility) is captured from head registration data.
- **Analytics**: Records scoring events in a bounded FIFO buffer. On query, fits PCA and computes per-bucket centroids.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run locally (without Docker)
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Project Structure

```
├── Dockerfile
├── requirements.txt
├── config.yaml              # optional — type aliases + analytics config
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── config.py             # Pydantic config loader
│   ├── models/
│   │   ├── ortb.py           # OpenRTB extension models
│   │   ├── campaigns.py      # Campaign head models
│   │   ├── scoring.py        # Score response models
│   │   └── analytics.py      # Analytics response models
│   ├── engine/
│   │   ├── scorer.py         # Similarity scoring (cosine/dot/L2)
│   │   ├── store.py          # Campaign head store + model config
│   │   └── analytics.py      # PCA + score bucketing
│   └── routes/
│       ├── score.py          # POST /score
│       ├── campaigns.py      # POST /campaigns/heads
│       └── analytics.py      # GET /analytics
├── tests/
│   ├── conftest.py
│   ├── test_scorer.py
│   ├── test_campaigns.py
│   ├── test_score.py
│   └── test_analytics.py
└── examples/
    ├── sample_score_request.json
    └── sample_campaign_heads.json
```

## Design Decisions

- **In-memory store** — acceptable for a reference implementation. Production use would swap in Redis or a vector database.
- **NumPy over FAISS** — brute-force similarity is sub-millisecond for hundreds of campaign heads; FAISS adds complexity without benefit at this scale.
- **PCA over UMAP** — deterministic, fast, no hyperparameters to tune.
- **float16 storage, float32 math** — halves memory usage. NumPy auto-upcasts during computation for numerical stability.
- **Config travels with registration** — model configuration (dimension, metric, normalization, compatibility) is declared per-campaign-head at registration time, not pre-declared in config. This allows different campaigns to use different models without redeploying.
