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
