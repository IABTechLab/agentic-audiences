from __future__ import annotations


def _register_and_score(client, n_scores=20):
    heads = [
        {
            "campaign_id": "camp-1",
            "campaign_head_id": "head-1",
            "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "model": "test-model",
            "dimension": 8,
            "type": "contextual",
        },
    ]
    client.post("/campaigns/heads", json={"heads": heads})

    for i in range(n_scores):
        val = (i + 1) / n_scores
        vector = [val, 1 - val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        req = {
            "user": {
                "data": [
                    {
                        "segment": [
                            {
                                "ext": {
                                    "ver": "1.0",
                                    "vector": vector,
                                    "model": "test-model",
                                    "dimension": 8,
                                    "type": "contextual",
                                }
                            }
                        ]
                    }
                ]
            }
        }
        client.post("/score", json=req)


def test_analytics_empty(client):
    resp = client.get("/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["campaigns"] == []
    assert data["reduction_method"] == "pca"


def test_analytics_with_scores(client):
    _register_and_score(client, n_scores=20)
    resp = client.get("/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["campaigns"]) == 1
    campaign = data["campaigns"][0]
    assert campaign["campaign_id"] == "camp-1"
    assert campaign["total_scored"] == 20
    assert len(campaign["score_buckets"]) == 4


def test_analytics_filter_by_campaign(client):
    _register_and_score(client, n_scores=10)
    resp = client.get("/analytics", params={"campaign_id": "camp-1"})
    assert resp.status_code == 200
    assert len(resp.json()["campaigns"]) == 1

    resp = client.get("/analytics", params={"campaign_id": "nonexistent"})
    assert resp.status_code == 200
    assert len(resp.json()["campaigns"]) == 0


def test_analytics_pca_centroids(client):
    _register_and_score(client, n_scores=20)
    resp = client.get("/analytics")
    data = resp.json()
    for bucket in data["campaigns"][0]["score_buckets"]:
        if bucket["count"] > 0:
            assert bucket["reduced_centroid"] is not None
            assert len(bucket["reduced_centroid"]) == 3


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "device" in data
