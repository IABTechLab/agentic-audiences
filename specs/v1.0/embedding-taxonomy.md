# Agentic Audiences Embedding Taxonomy

**Version:** 0.1 (Draft)
**Status:** Proposal

---

## Overview

This document defines a taxonomy for classifying embeddings exchanged via Agentic Audiences. Embeddings encode different types of signals—identity, contextual, and reinforcement—and understanding their semantic purpose is critical for proper interpretation, combination, and usage by agents.

The taxonomy categorizes embeddings along three dimensions:
1. **Signal Type** - What kind of information the embedding encodes
2. **Temporal Scope** - Time horizon the embedding represents
3. **Composition** - Whether it encodes a single signal type or multiple types

---

## 1. Signal Type Classification

### 1.1 Identity Embeddings

**Purpose:** Represent a user's persistent identity, enabling cross-session recognition and historical behavior understanding.

**Subcategories:**

#### 1.1.1 PII-Derived Identity Embeddings
- **Description:** Learned representations of offline identity derived from personally identifiable information (PII)
- **Source Data:** Email addresses, phone numbers, postal addresses, device IDs (hashed/encrypted)
- **Model Type:** Transformer models trained on tokenized, anonymized PII
- **Use Cases:** Cross-device identity resolution, household linkage, deterministic matching
- **Privacy:** Must preserve k-anonymity; embeddings should not be reversible to raw PII
- **Example:** A text-based PII identifier transformer that encodes hashed email → 512-dim vector

#### 1.1.2 Behavioral Identity Embeddings
- **Description:** Representations of a user based on long-term behavioral patterns
- **Source Data:** Purchase history, browsing patterns, app usage, content consumption over weeks/months
- **Model Type:** Recurrent networks (LSTM/GRU), transformers with temporal attention
- **Use Cases:** Lookalike modeling, lifetime value prediction, segment discovery
- **Temporal Scope:** Weeks to months of historical data
- **Example:** User journey embedding capturing 90-day behavioral fingerprint

#### 1.1.3 Demographic Identity Embeddings
- **Description:** Representations of inferred or declared demographic attributes
- **Source Data:** Age, gender, income bracket, education, location, interests
- **Model Type:** Categorical embeddings, entity embeddings from tabular data
- **Use Cases:** Demographic targeting, audience extension, census-level aggregation
- **Example:** Combined demographic vector encoding {age_bucket, region, interest_category}

#### 1.1.4 Graph-Based Identity Embeddings
- **Description:** Representations derived from identity graphs (device graphs, household graphs, social graphs)
- **Source Data:** Device co-occurrence, shared network connections, household relationships
- **Model Type:** Graph neural networks (GNN), node2vec, DeepWalk
- **Use Cases:** Probabilistic identity linking, fraud detection, household targeting
- **Example:** Node embedding in a device graph representing probabilistic linkage to 5 devices

### 1.2 Contextual Embeddings

**Purpose:** Represent the current situational context in which a user is operating, enabling real-time intent understanding.

**Subcategories:**

#### 1.2.1 Content Contextual Embeddings
- **Description:** Semantic representations of page/app content where ads may appear
- **Source Data:** Page text, article body, video metadata, app category
- **Model Type:** Sentence transformers (SBERT), BERT variants, multimodal models
- **Use Cases:** Contextual targeting, brand safety, semantic matching
- **Temporal Scope:** Instantaneous (current page view)
- **Example:** Page embedding from article about "electric vehicle financing" → 768-dim BERT embedding

#### 1.2.2 Temporal Contextual Embeddings
- **Description:** Representations of time-based context (time of day, day of week, seasonality)
- **Source Data:** Timestamp, timezone, calendar features (weekday/weekend, holiday)
- **Model Type:** Sinusoidal encodings, learned temporal embeddings
- **Use Cases:** Dayparting optimization, seasonal campaign tuning
- **Example:** Time embedding encoding {Monday, 9am PST, non-holiday} → 64-dim vector

#### 1.2.3 Geospatial Contextual Embeddings
- **Description:** Representations of location-based context
- **Source Data:** Country, region, DMA, postal code, lat/long (coarse), POI proximity
- **Model Type:** Geohashing embeddings, hierarchical location embeddings
- **Use Cases:** Local targeting, geo-fencing, regional campaign optimization
- **Privacy:** Must use coarse granularity (ZIP/postal code level, not GPS coordinates)
- **Example:** Location embedding for "San Francisco Bay Area, CA" → 128-dim vector

#### 1.2.4 Device/Environment Contextual Embeddings
- **Description:** Representations of device and browsing environment
- **Source Data:** Device type, OS, browser, screen size, connection type, app vs web
- **Model Type:** Categorical embeddings, device fingerprint encoders
- **Use Cases:** Creative optimization (mobile vs desktop), format selection
- **Example:** Device embedding for {iOS, Safari, mobile, 5G} → 32-dim vector

#### 1.2.5 Session Contextual Embeddings
- **Description:** Representations of current browsing session behavior
- **Source Data:** Pages visited in session, dwell time, scroll depth, engagement signals
- **Model Type:** Session RNNs, attention-based sequence models
- **Use Cases:** In-session intent prediction, urgency detection
- **Temporal Scope:** Minutes to hours (current session)
- **Example:** Session embedding capturing "researching → comparing → near-purchase" journey state

### 1.3 Reinforcement Embeddings

**Purpose:** Represent feedback signals from user interactions with advertising, enabling model updates and campaign optimization.

**Subcategories:**

#### 1.3.1 Engagement Reinforcement Embeddings
- **Description:** Representations of ad engagement behaviors short of conversion
- **Source Data:** Impressions, viewability, clicks, video completions, hover time, scroll-through
- **Model Type:** Event sequence encoders, survival models, interaction transformers
- **Use Cases:** Click prediction, viewability optimization, engagement modeling
- **Temporal Scope:** Seconds to hours post-exposure
- **Example:** Engagement embedding encoding {5 impressions, 2 clicks, 30s avg dwell} → 256-dim vector

#### 1.3.2 Conversion Reinforcement Embeddings
- **Description:** Representations of conversion events and their context
- **Source Data:** Purchase, sign-up, download, form submission, attributed conversions
- **Model Type:** Conversion path encoders, attribution models
- **Use Cases:** Conversion rate prediction, incrementality measurement, ROAS optimization
- **Temporal Scope:** Hours to days post-exposure
- **Example:** Conversion embedding capturing {purchase, $150 AOV, 48hr lag, 3-touch path} → 512-dim vector

#### 1.3.3 Attribution Reinforcement Embeddings
- **Description:** Representations of multi-touch attribution weights across touchpoints
- **Source Data:** Full conversion path, touchpoint timestamps, channel mix, credited value
- **Model Type:** Markov chain models, Shapley value calculators, path transformers
- **Use Cases:** Budget allocation, channel optimization, incrementality testing
- **Example:** Attribution embedding encoding contribution weights across {display, social, search} path

#### 1.3.4 Feedback Reinforcement Embeddings
- **Description:** Representations of negative signals or policy violations
- **Source Data:** Ad fatigue indicators, frequency cap violations, user complaints, brand safety violations
- **Model Type:** Anomaly detection models, policy classifiers
- **Use Cases:** Frequency optimization, ad quality improvement, brand safety enforcement
- **Example:** Feedback embedding flagging {over-frequency, user opted out, creative underperforming}

### 1.4 Creative Embeddings

**Purpose:** Represent advertising creative assets, enabling semantic matching and creative optimization.

**Subcategories:**

#### 1.4.1 Visual Creative Embeddings
- **Description:** Representations of image/video creative elements
- **Source Data:** Image pixels, video frames, visual elements (objects, colors, composition)
- **Model Type:** CNN encoders (ResNet, EfficientNet), vision transformers (ViT), CLIP
- **Use Cases:** Visual similarity matching, A/B testing, dynamic creative optimization
- **Example:** Image embedding from display ad creative → 2048-dim ResNet embedding

#### 1.4.2 Textual Creative Embeddings
- **Description:** Representations of ad copy, headlines, CTAs
- **Source Data:** Ad text, headlines, descriptions, call-to-action phrases
- **Model Type:** Sentence transformers, advertising-specific language models
- **Use Cases:** Copy testing, message-market fit, semantic creative matching
- **Example:** Text embedding from headline "Save 30% on Electric Vehicles" → 384-dim SBERT vector

#### 1.4.3 Multimodal Creative Embeddings
- **Description:** Joint representations of visual + textual + audio creative elements
- **Source Data:** Combined image, text, audio from video ads or rich media
- **Model Type:** CLIP-style models, multimodal transformers, unified embedding spaces
- **Use Cases:** Holistic creative understanding, cross-modal retrieval, dynamic assembly
- **Example:** Multimodal embedding from 30s video ad with voiceover → 1024-dim joint vector

#### 1.4.4 Creative Performance Embeddings
- **Description:** Representations combining creative features with performance history
- **Source Data:** Creative attributes + historical CTR/CVR/engagement metrics
- **Model Type:** Performance-aware encoders, metric-conditioned embeddings
- **Use Cases:** Creative ranking, performance prediction, winner prediction
- **Example:** Creative+performance embedding: {image_vector, historical_CTR=2.3%} → 768-dim vector

### 1.5 Inventory Embeddings

**Purpose:** Represent available advertising inventory, enabling supply-demand matching.

**Subcategories:**

#### 1.5.1 Publisher Inventory Embeddings
- **Description:** Representations of publisher properties and their characteristics
- **Source Data:** Domain, content category, audience reach, brand safety score, viewability rates
- **Model Type:** Publisher encoders, domain embedding models
- **Use Cases:** Publisher selection, PMPs, inventory quality scoring
- **Example:** Publisher embedding for "premium news site, politics category" → 256-dim vector

#### 1.5.2 Placement Inventory Embeddings
- **Description:** Representations of specific ad placements/units
- **Source Data:** Format (banner/video/native), size, position (above/below fold), context
- **Model Type:** Placement feature encoders
- **Use Cases:** Format selection, placement optimization, yield management
- **Example:** Placement embedding for "300×250 banner, above-fold, homepage" → 128-dim vector

#### 1.5.3 Audience Inventory Embeddings
- **Description:** Representations of targetable audience segments available in inventory
- **Source Data:** Segment definitions, reach, overlap, refresh rates, data source
- **Model Type:** Segment taxonomy embeddings, audience characteristic encoders
- **Use Cases:** Audience discovery, segment recommendation, overlap analysis
- **Example:** Segment embedding for "in-market auto shoppers, 2M reach" → 512-dim vector

### 1.6 Query/Intent Embeddings

**Purpose:** Represent user intent signals or agent queries for matching against inventory or audiences.

**Subcategories:**

#### 1.6.1 Search Query Embeddings
- **Description:** Representations of search queries indicating commercial intent
- **Source Data:** Search terms, query refinements, search session context
- **Model Type:** Query encoders, BERT-based search models
- **Use Cases:** Search retargeting, intent capture, keyword expansion
- **Example:** Query embedding for "best electric SUV 2025" → 768-dim vector

#### 1.6.2 Buyer Intent Embeddings
- **Description:** Representations of what a buyer agent is seeking
- **Source Data:** Campaign goals, target audience description, creative requirements, budget constraints
- **Model Type:** Intent specification encoders, goal-aware transformers
- **Use Cases:** Inventory matching, seller discovery, programmatic negotiation
- **Example:** Buyer intent: "reach tech-savvy millennials interested in sustainable products" → 512-dim vector

#### 1.6.3 Seller Offer Embeddings
- **Description:** Representations of what a seller agent is offering
- **Source Data:** Available inventory characteristics, pricing, audience profiles, context
- **Model Type:** Offer specification encoders
- **Use Cases:** Buyer-seller matching, marketplace efficiency, price discovery
- **Example:** Seller offer: "CTV inventory, sports content, 18-34 males, $15 CPM" → 512-dim vector

---

## 2. Temporal Scope Classification

Embeddings can be classified by the time horizon they represent:

### 2.1 Persistent Embeddings
- **Time Horizon:** Weeks to months
- **Update Frequency:** Weekly to monthly
- **Examples:** PII-derived identity, behavioral identity, LTV predictions
- **Characteristics:** Stable, long-term representations

### 2.2 Session Embeddings
- **Time Horizon:** Minutes to hours
- **Update Frequency:** Per session or hourly
- **Examples:** Session context, current intent, in-session behavior
- **Characteristics:** Medium-term, updated within browsing sessions

### 2.3 Real-Time Embeddings
- **Time Horizon:** Seconds to minutes
- **Update Frequency:** Per event or continuously
- **Examples:** Current page context, immediate device context, ad request context
- **Characteristics:** Instantaneous, reflects current moment

### 2.4 Retrospective Embeddings
- **Time Horizon:** Historical (post-event analysis)
- **Update Frequency:** Batch updates after campaigns complete
- **Examples:** Attribution embeddings, incrementality measurements, campaign performance
- **Characteristics:** Backward-looking, enable learning for future campaigns

---

## 3. Composition Classification

Embeddings can combine multiple signal types:

### 3.1 Atomic Embeddings
- **Definition:** Encode a single signal type from a single source
- **Examples:**
  - Pure content embedding (just the page text)
  - Pure PII embedding (just the hashed email)
  - Pure device embedding (just device characteristics)
- **Use Cases:** Building blocks for more complex representations, interpretability, debugging

### 3.2 Composite Embeddings
- **Definition:** Combine multiple related signals of the same type
- **Examples:**
  - User identity = PII embedding + behavioral embedding + demographic embedding
  - Full context = content + temporal + device + geo embeddings
- **Method:** Concatenation, weighted averaging, learned fusion
- **Use Cases:** Richer representations, holistic understanding within a signal type

### 3.3 Graph Embeddings
- **Definition:** Encode relational structures between entities
- **Examples:**
  - Device graph embeddings (device-device relationships)
  - User journey graph (page-to-page navigation)
  - Conversion path graph (touchpoint sequences)
  - Creative similarity graph (creative-creative relationships)
- **Model Type:** Graph Neural Networks (GNN), node2vec, GraphSAGE
- **Use Cases:** Relationship discovery, transitive inference, network effects

### 3.4 Cross-Signal Fusion Embeddings
- **Definition:** Combine multiple signal types (identity + context + reinforcement)
- **Examples:**
  - User-in-context: identity embedding + current contextual embedding
  - Predictive fusion: identity + context → predicted engagement
  - Feedback-informed identity: baseline identity + historical reinforcement signals
- **Method:** Multimodal fusion, cross-attention, gating mechanisms
- **Use Cases:** Comprehensive user understanding, real-time scoring, personalized predictions

### 3.5 Hierarchical Embeddings
- **Definition:** Multi-level representations with coarse-to-fine granularity
- **Examples:**
  - Geographic hierarchy: country → state → DMA → postal code
  - Taxonomic hierarchy: IAB category L1 → L2 → L3
  - Temporal hierarchy: year → month → week → day → hour
- **Model Type:** Hierarchical encoders, tree-structured embeddings
- **Use Cases:** Multi-resolution targeting, privacy-aware aggregation, drill-down analysis

---

## 4. Embedding Metadata Schema

To properly interpret and use embeddings, agents must exchange metadata. The following fields should accompany embeddings:

### Required Metadata
```json
{
  "embedding_id": "unique-id",
  "taxonomy_class": {
    "signal_type": "identity|contextual|reinforcement|creative|inventory|query",
    "subtype": "pii_derived|behavioral|content|engagement|...",
    "temporal_scope": "persistent|session|realtime|retrospective",
    "composition": "atomic|composite|graph|fusion|hierarchical"
  },
  "dimension": 512,
  "model": {
    "id": "model-identifier",
    "version": "1.0.0",
    "architecture": "transformer|cnn|gnn|...",
    "embedding_space_id": "ucp://spaces/identity/pii-v1"
  },
  "vector": [0.01, 0.02, ...],
  "normalization": "l2_unit|none",
  "metric": "cosine|dot|l2"
}
```

### Optional Metadata
```json
{
  "source_signals": ["hashed_email", "behavioral_history"],
  "temporal_window": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-24T00:00:00Z",
    "scope": "90_days"
  },
  "privacy": {
    "k_anonymity": 100,
    "differential_privacy": false,
    "reversibility_risk": "low"
  },
  "quality_metrics": {
    "confidence": 0.95,
    "coverage": 0.87,
    "staleness_hours": 2
  },
  "interpretability": {
    "top_features": ["feature1", "feature2"],
    "attribution_method": "integrated_gradients"
  }
}
```

---

## 5. Usage Guidelines

### 5.1 Embedding Selection

**For Buyer Agents:**
- Use **identity embeddings** (PII + behavioral) to understand who to target
- Use **contextual embeddings** (content + temporal + geo) to find when/where to show ads
- Use **creative embeddings** to select appropriate messaging
- Use **query/intent embeddings** to express what you're looking for

**For Seller Agents:**
- Use **inventory embeddings** to represent what you're offering
- Use **contextual embeddings** to describe placement environment
- Use **audience embeddings** to communicate available segments

**For Measurement Agents:**
- Use **reinforcement embeddings** (engagement + conversion) to provide feedback
- Use **attribution embeddings** to credit touchpoints
- Use **feedback embeddings** to flag quality issues

### 5.2 Embedding Combination

When combining embeddings from different classes:

1. **Ensure compatible embedding spaces** - Check `embedding_space_id` and model compatibility
2. **Normalize before fusion** - Use consistent normalization (typically L2)
3. **Weight appropriately** - Identity may deserve higher weight than device context
4. **Consider temporal freshness** - Don't mix stale persistent embeddings with real-time context
5. **Preserve privacy** - Fusion should not reduce k-anonymity below thresholds

### 5.3 Interoperability

For cross-agent embedding exchange:

- **Shared embedding spaces** - Agents using the same `embedding_space_id` can directly compare embeddings
- **Transfer learning** - Embeddings from compatible spaces can be projected into common space
- **Metadata transparency** - Always include taxonomy classification in metadata
- **Version compatibility** - Specify model version; newer versions should maintain backward compatibility when possible

---

## 6. Future Extensions

This taxonomy is a living document. Anticipated future additions:

### 6.1 Additional Signal Types
- **Attention embeddings** - Representations of visual attention patterns (eye-tracking derived)
- **Emotional embeddings** - Affective responses to creative (sentiment, emotional arousal)
- **Trust embeddings** - Brand safety, verification, fraud risk signals
- **Privacy embeddings** - Consent state, privacy preferences, regulatory compliance signals

### 6.2 Advanced Compositions
- **Causal embeddings** - Encode causal relationships (not just correlations)
- **Counterfactual embeddings** - "What would have happened without the ad?"
- **Ensemble embeddings** - Weighted combinations of multiple models' embeddings
- **Meta-embeddings** - Embeddings of embeddings (second-order representations)

### 6.3 Dynamic Embeddings
- **Streaming embeddings** - Continuously updated via online learning
- **Adaptive embeddings** - Self-adjusting based on prediction accuracy
- **Context-conditional embeddings** - Same user/content but different embeddings based on query context

---

## 7. References

- Agentic Audiences Embedding Format Specification (`embedding_format.schema.json`)
- AI/ML Models in Agentic Digital Advertising Era (whitepaper)

---

## 8. Change Log

- **v0.1 (2025-10-24)**: Initial draft taxonomy proposal
  - Defined 6 primary signal types with subcategories
  - Established temporal scope and composition classifications
  - Added metadata schema and usage guidelines

---

**Maintainers:** LiveRamp Agentic Audiences Working Group
**Feedback:** Submit issues or PRs to the Agentic Audiences repository
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
