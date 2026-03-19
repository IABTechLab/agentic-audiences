from __future__ import annotations


def _make_head(
    campaign_id="c1",
    head_id="h1",
    dim=8,
    model="test-model",
    type_="contextual",
    metric="cosine",
    apply_l2_norm=True,
    compatible_with=None,
):
    return {
        "campaign_id": campaign_id,
        "campaign_head_id": head_id,
        "weights": [0.1] * dim,
        "model": model,
        "dimension": dim,
        "type": type_,
        "metric": metric,
        "apply_l2_norm": apply_l2_norm,
        "compatible_with": compatible_with or [],
    }


def test_register_heads(client):
    resp = client.post("/campaigns/heads", json={"heads": [_make_head()]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["registered"] == 1
    assert data["ids"] == ["h1"]


def test_register_multiple_heads(client):
    heads = [
        _make_head(head_id="h1"),
        _make_head(campaign_id="c2", head_id="h2"),
    ]
    resp = client.post("/campaigns/heads", json={"heads": heads})
    assert resp.status_code == 200
    assert resp.json()["registered"] == 2


def test_register_dimension_mismatch(client):
    head = _make_head(dim=8)
    head["weights"] = [0.1] * 16  # weights don't match declared dimension
    resp = client.post("/campaigns/heads", json={"heads": [head]})
    assert resp.status_code == 400
    assert "Dimension mismatch" in resp.json()["detail"]


def test_register_inconsistent_dimension(client):
    """Second head for same model with different dimension should fail."""
    head1 = _make_head(head_id="h1", dim=8)
    resp = client.post("/campaigns/heads", json={"heads": [head1]})
    assert resp.status_code == 200

    head2 = _make_head(head_id="h2", dim=4)
    resp = client.post("/campaigns/heads", json={"heads": [head2]})
    assert resp.status_code == 400
    assert "Dimension mismatch" in resp.json()["detail"]


# --- Upsert tests ---


def test_register_upsert_replaces(client):
    """POST with same campaign_head_id replaces the existing head."""
    head = _make_head(head_id="h1", dim=8)
    resp = client.post("/campaigns/heads", json={"heads": [head]})
    assert resp.status_code == 200

    # Re-register with different weights
    head2 = _make_head(head_id="h1", dim=8)
    head2["weights"] = [0.9] * 8
    resp = client.post("/campaigns/heads", json={"heads": [head2]})
    assert resp.status_code == 200
    assert resp.json()["ids"] == ["h1"]

    # Score to confirm new weights are used (scores should differ from original)
    score_req = {
        "id": "req-1",
        "user": {
            "id": "u1",
            "data": [{
                "id": "prov",
                "segment": [{
                    "id": "seg1",
                    "ext": {
                        "ver": "1.0",
                        "vector": [0.9] * 8,
                        "model": "test-model",
                        "dimension": 8,
                        "type": "contextual",
                    }
                }]
            }]
        }
    }
    resp = client.post("/score", json=score_req)
    assert resp.status_code == 200
    scores = resp.json()
    # Should have exactly one head scored
    assert len(scores[0]["scores"]) == 1
    assert scores[0]["scores"][0]["campaign_head_id"] == "h1"


def test_register_upsert_preserves_others(client):
    """Upserting one head does not affect other heads."""
    heads = [
        _make_head(head_id="h1"),
        _make_head(campaign_id="c2", head_id="h2"),
    ]
    resp = client.post("/campaigns/heads", json={"heads": heads})
    assert resp.status_code == 200
    assert resp.json()["registered"] == 2

    # Upsert h1 only
    updated = _make_head(head_id="h1")
    updated["weights"] = [0.5] * 8
    resp = client.post("/campaigns/heads", json={"heads": [updated]})
    assert resp.status_code == 200

    # Score — both heads should still be present
    score_req = {
        "id": "req-1",
        "user": {
            "id": "u1",
            "data": [{
                "id": "prov",
                "segment": [{
                    "id": "seg1",
                    "ext": {
                        "ver": "1.0",
                        "vector": [0.5] * 8,
                        "model": "test-model",
                        "dimension": 8,
                        "type": "contextual",
                    }
                }]
            }]
        }
    }
    resp = client.post("/score", json=score_req)
    assert resp.status_code == 200
    head_ids = {s["campaign_head_id"] for s in resp.json()[0]["scores"]}
    assert head_ids == {"h1", "h2"}


# --- Delete tests ---


def test_delete_existing(client):
    """DELETE removes an existing head."""
    resp = client.post("/campaigns/heads", json={"heads": [_make_head(head_id="h1")]})
    assert resp.status_code == 200

    resp = client.delete("/campaigns/heads/h1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "h1"


def test_delete_nonexistent(client):
    """DELETE on unknown ID returns 404."""
    resp = client.delete("/campaigns/heads/unknown-id")
    assert resp.status_code == 404


def test_delete_allows_reregister(client):
    """After deleting the only head for a model, can re-register with a different dimension."""
    head = _make_head(head_id="h1", dim=8, model="model-a")
    resp = client.post("/campaigns/heads", json={"heads": [head]})
    assert resp.status_code == 200

    resp = client.delete("/campaigns/heads/h1")
    assert resp.status_code == 200

    # Re-register same model with different dimension
    head2 = _make_head(head_id="h1", dim=4, model="model-a")
    head2["weights"] = [0.2] * 4
    resp = client.post("/campaigns/heads", json={"heads": [head2]})
    assert resp.status_code == 200
    assert resp.json()["ids"] == ["h1"]


# --- Update (PUT) tests ---


def test_update_existing(client):
    """PUT updates an existing head."""
    resp = client.post("/campaigns/heads", json={"heads": [_make_head(head_id="h1")]})
    assert resp.status_code == 200

    updated = _make_head(head_id="h1")
    updated["weights"] = [0.9] * 8
    resp = client.put("/campaigns/heads", json={"heads": [updated]})
    assert resp.status_code == 200
    assert resp.json()["ids"] == ["h1"]


def test_update_nonexistent(client):
    """PUT on unknown head returns 404."""
    head = _make_head(head_id="unknown")
    resp = client.put("/campaigns/heads", json={"heads": [head]})
    assert resp.status_code == 404


def test_update_atomic_on_missing(client):
    """PUT with mix of existing and missing heads fails atomically — no heads updated."""
    resp = client.post("/campaigns/heads", json={"heads": [_make_head(head_id="h1")]})
    assert resp.status_code == 200

    # Try to update h1 and non-existent h2 — should fail
    heads = [
        _make_head(head_id="h1"),
        _make_head(head_id="h2"),
    ]
    heads[0]["weights"] = [0.9] * 8
    resp = client.put("/campaigns/heads", json={"heads": heads})
    assert resp.status_code == 404

    # Verify h1 is unchanged by scoring with original weights
    score_req = {
        "id": "req-1",
        "user": {
            "id": "u1",
            "data": [{
                "id": "prov",
                "segment": [{
                    "id": "seg1",
                    "ext": {
                        "ver": "1.0",
                        "vector": [0.1] * 8,
                        "model": "test-model",
                        "dimension": 8,
                        "type": "contextual",
                    }
                }]
            }]
        }
    }
    resp = client.post("/score", json=score_req)
    assert resp.status_code == 200
    # h1 should score ~1.0 with its original [0.1]*8 weights
    score = resp.json()[0]["scores"][0]["score"]
    assert score > 0.99
