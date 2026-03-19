from __future__ import annotations


def _register_heads(client, heads=None):
    if heads is None:
        heads = [
            {
                "campaign_id": "camp-1",
                "campaign_head_id": "head-1",
                "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "model": "test-model",
                "dimension": 8,
                "type": "contextual",
                "metric": "cosine",
                "apply_l2_norm": True,
                "compatible_with": ["compat-model"],
            },
            {
                "campaign_id": "camp-2",
                "campaign_head_id": "head-2",
                "weights": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "model": "test-model",
                "dimension": 8,
                "type": "contextual",
                "metric": "cosine",
                "apply_l2_norm": True,
                "compatible_with": ["compat-model"],
            },
        ]
    resp = client.post("/campaigns/heads", json={"heads": heads})
    assert resp.status_code == 200
    return resp


def _score_request(vector, model="test-model", dim=8, emb_type="contextual", top_k=5):
    return {
        "id": "test-req",
        "user": {
            "id": "user-1",
            "data": [
                {
                    "id": "provider-1",
                    "segment": [
                        {
                            "id": "seg-1",
                            "ext": {
                                "ver": "1.0",
                                "vector": vector,
                                "model": model,
                                "dimension": dim,
                                "type": emb_type,
                            },
                        }
                    ],
                }
            ],
        },
        "top_k": top_k,
    }


def test_score_end_to_end(client):
    _register_heads(client)
    vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    resp = client.post("/score", json=_score_request(vector))
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    scores = data[0]["scores"]
    assert len(scores) == 2
    assert scores[0]["campaign_head_id"] == "head-1"
    assert scores[0]["score"] > scores[1]["score"]


def test_score_type_alias(client):
    """Wire type 'context' should resolve to 'contextual'."""
    _register_heads(client)
    vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    resp = client.post("/score", json=_score_request(vector, emb_type="context"))
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["embedding_type"] == "contextual"
    assert len(data[0]["scores"]) == 2


def test_score_type_filtering(client):
    """Intent heads should not match contextual embeddings."""
    heads = [
        {
            "campaign_id": "camp-1",
            "campaign_head_id": "head-intent",
            "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "model": "test-model",
            "dimension": 8,
            "type": "intent",
            "metric": "cosine",
            "apply_l2_norm": True,
        },
    ]
    _register_heads(client, heads)
    vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    resp = client.post("/score", json=_score_request(vector, emb_type="contextual"))
    assert resp.status_code == 200
    assert resp.json()[0]["scores"] == []


def test_score_unregistered_model(client):
    """Scoring against an unregistered model returns empty scores, not an error."""
    vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    resp = client.post("/score", json=_score_request(vector, model="unknown-model"))
    assert resp.status_code == 200
    assert resp.json()[0]["scores"] == []


def test_score_no_embeddings(client):
    req = {"user": {"data": [{"segment": [{"id": "seg-1"}]}]}}
    resp = client.post("/score", json=req)
    assert resp.status_code == 400


def test_score_compatible_models(client):
    """Heads registered with compat-model should match test-model embeddings."""
    # First register test-model heads (establishes compatible_with)
    _register_heads(client)
    # Then register compat-model heads
    compat_heads = [
        {
            "campaign_id": "camp-compat",
            "campaign_head_id": "head-compat",
            "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "model": "compat-model",
            "dimension": 8,
            "type": "contextual",
            "metric": "cosine",
            "apply_l2_norm": True,
        },
    ]
    _register_heads(client, compat_heads)
    vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    resp = client.post("/score", json=_score_request(vector, model="test-model"))
    assert resp.status_code == 200
    scores = resp.json()[0]["scores"]
    assert any(s["campaign_head_id"] == "head-compat" for s in scores)


def test_dimension_consistency(client):
    """Registering heads with different dimensions for the same model should fail."""
    head1 = {
        "campaign_id": "camp-1",
        "campaign_head_id": "head-1",
        "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "model": "dim-model",
        "dimension": 8,
        "type": "contextual",
    }
    resp = client.post("/campaigns/heads", json={"heads": [head1]})
    assert resp.status_code == 200

    head2 = {
        "campaign_id": "camp-2",
        "campaign_head_id": "head-2",
        "weights": [1.0, 0.0, 0.0, 0.0],
        "model": "dim-model",
        "dimension": 4,
        "type": "contextual",
    }
    resp = client.post("/campaigns/heads", json={"heads": [head2]})
    assert resp.status_code == 400
    assert "Dimension mismatch" in resp.json()["detail"]
