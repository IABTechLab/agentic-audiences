# Agentic Audiences Contextual Embedding Exchange Specification (Draft v0.1)

Status: Draft  
Scope: Defines a vendor-neutral wire format for exchanging contextual embeddings between agents in the Agentic Audiences ecosystem.  
Primary transport: HTTPS JSON (optionally NDJSON for streaming). Binary variants MAY use CBOR with identical field names.

---

## 1. Design Goals

- Interoperability: any agent that understands the same embedding space can consume vectors safely.
- Interpretable metadata: receivers can determine dimensionality, metric, normalization, tokenizer, pooling, projection, quantization, training domain, and license.
- Privacy and consent: explicit fields for consent, TTL, purposes, and permissible uses.
- Security: integrity, authentication, and optional attestation.
- Versioning: clear envelope and schema version semantics.

---

## 2. Content Type and Versioning

- HTTP Content-Type: `application/vnd.ucp.embedding+json; v=1`
- Top-level `spec_version`: semantic version string of this spec (e.g., `"1.0.0"`).
- Backwards-compatible additions use new optional fields. Breaking changes bump major version.

---

## 3. Published Schemas

- schema/embedding_format.schema.json

---

## 4. Embedding Object

Required fields:
- id string, unique per embedding within message.
- type one of context, creative, user_intent, inventory, query.
- Either vector (array of numbers) or quantized_b64 (base64).
- dimension integer equals model.dimension.
- dtype one of float32, float16, int8, uint8.

Optional fields:
- scale number used for dequantization.
- compressed boolean.
- hash content hash of raw vector bytes after normalization.
- origin to explain how it was produced.
- usage_hints metric, thresholds, target agents.

---

## 5. Model Descriptor

Required:
- id, version, dimension, metric (cosine|dot|l2), type (encoder|llm|slm), embedding_space_id.

Recommended:
- tokenizer name/version and canonical vocab id.
- pooling (mean|max|cls|weighted).
- normalization (none|l2_unit).
- projection info if PCA, whitening, or learned projection applied.
- quantization scheme if any.
- training_domain tags describing primary data regimes.
- licensing license id and URL.
- compatibility compatible_spaces and guidance.

Rationale: receivers must know how to compare vectors and whether mixing spaces will degrade quality.

---

## 6. Context Descriptor
- url, page_title, keywords (ordered, deduped), language BCP-47.
- content_hash hash of the processed text or DOM excerpt used to create the embedding.
- placement and device are optional but useful for reproducibility and analysis.
- geography is coarse-level only and MUST honor consent and policy.

---

## 7. Consent, Purpose, and TTL
- consent.framework and consent_string reference the governing framework (e.g., IAB TCF, US state signals).
- permissible_uses enumerates allowed downstream uses. If absent, default is "activation_scoring" only.
- ttl_seconds defines the retention and re-use horizon for the embedding and associated metadata.

Receivers MUST enforce TTL and purposes before storing or training.

---

## 8. Security and Attestation
- Transport: HTTPS with mTLS is RECOMMENDED for inter-agent links.
- Integrity: sign the envelope; include key_id resolvable via JWKS or equivalent.
- Attestation: optional policy hash or enclave report so consumers can trust how vectors were produced, especially for on-device SLM execution.

---

## 9. Error Handling
- 400: schema or consent invalid.
- 409: incompatible embedding_space_id or dimension.
- 422: metric or normalization mismatch.
- 429: throttling.
- 5xx: transient server errors.

Responses SHOULD include message_id, status, and errors[] with code, field, detail.

---

## 10. Streaming (NDJSON)

When sending high-volume events:

Content-Type: application/x-ndjson; charset=utf-8

Each line is a complete envelope. Receivers MAY accept batched arrays over HTTP/2 as an alternative.

---

## 11. Security and Operational Guidance
- Authenticate producers and consumers via mTLS or OAuth 2.0 with private JWKS per tenant.
- Sign envelopes; reject messages with unknown key_id or invalid signature.
- Enforce consent.purposes and ttl_seconds at ingestion and storage layers.
- Validate embedding_space_id, dimension, and metric before indexing.
- Log message_id, context.context_id, and hash for auditability.

---

## 12. Interoperability Rules
- Two vectors are comparable if and only if:
- embedding_space_id matches, or a known converter maps between spaces, and
- dimension, metric, and normalization are compatible, and
- any projection or quantization differences are accounted for.
- Receivers SHOULD normalize to the model’s declared normalization before scoring.

---

## 13. IANA-like Registry Stubs (to be formalized)
- Embedding Space IDs: ucp://spaces/{domain}/{lang}-v{n}
- Metrics: cosine, dot, l2
- Pooling: mean, max, cls, weighted
- Quantization: none, int8, uint8, pq, sq

