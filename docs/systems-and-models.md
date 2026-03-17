# Agentic Audiences Systems and Models

---

## Overview

This document describes the systems and models architecture for Agentic Audiences, presented as a "10K Foot View."

Today's programmatic signals are sparse -- segment IDs, key-value pairs, and coarse demographic buckets. These representations are a poor fit for deep learning models that thrive on dense, high-dimensional inputs. Agentic Audiences addresses this by embedding richer information -- contextual, behavioral, and identity signals -- into the bid stream as vectors. The result is a shift from "is this user in segment X?" to "how appropriate is this campaign for this context right now?", without adding latency to the RTB critical path.

This is an open, collaborative initiative developed with IAB involvement toward global standards. It is not a walled-garden approach. The protocol, embedding taxonomy, and a reference scoring implementation are provided openly so that any participant in the ecosystem can build toward the same specification.

The diagram below depicts a header bidding flow for vectorized payloads, building on top of ATS and Prebid. This is not meant to be the only transport mechanism, but rather to demonstrate one such embodiment.

---

## System Architecture

The architecture is organized into the following layers:

### Audience Model

The audience model must produce quality embeddings from various types of inputs, including but not limited to; contextual data, identity data, behavioral data. Inputs are generated directly from a tag on browser or from the publisher's server or CDN. These embeddings may be produced from one or more models, often relying on specialized model providers to produce high quality embeddings. Quality embeddings from contextual data can be produced from open source models such as `MiniLM` or `bge-small`. This model, or set of models, should evolve towards a domain-specific semantic understanding of audience data, i.e. it will have a deep understanding of sequence of events leading to clicks/conversions. Embedding generation from inputs should be asynchronous (i.e. not in line with an RTB request).

LiveRamp does not aim to build all models internally. The architecture supports a competitive marketplace where external model builders produce embeddings compatible with the protocol and compete on quality -- analogous to how Ramp IDs create a network of identity providers. An embedding taxonomy classifies models by the class of data they were trained on (contextual, behavioral, identity, etc.), which determines compatibility and whether alignment between models is feasible.

### Browser / Edge

The browser interacts with a publisher-side tag (such as LiveRamp's ATS.js) that manages embedding storage in a first-party cookie. The specification is open -- ATS.js is LiveRamp's implementation, but other parties can build their own implementations toward the same protocol.

The tag does not crawl or scrape pages. Publishers provide title, keywords, and other page-level signals through standard locations (meta tags, data layers) or server-side. This keeps integration lightweight for publishers.

Embeddings from the audience model are ideally stored in the first-party cookie (increasing privacy as well as decreasing latencies during embedding retrieval), but could reside server-side as well. Prebid.js will construct a BidRequest object, placing the embedding in an ORTB2 Segment ext object. IAB is adding an extension to this segment object to carry the embedding vector and its metadata (model, dimension, type) in a standardized format.

[Link to ORTB2 Segment ext schema]

Storing embeddings in first-party cookies provides a privacy advantage over server-side approaches. Reduced representations of embeddings can be transmitted back as feedback signals, providing an adjustable dial to meet any regulatory surface.

### Campaign Scoring Service

The responsibility of the campaign scoring service is to perform vector distance calculations between a user's embedding and a list of vectors representing the campaign. This container replaces the question of "is this user being targeted in any active campaigns?" with "how appropriate is it to show this campaign in this context right now?". Note that returning a valid BidResponse, controlling bidding logic such as budget pacing or arbitrage is outside the scope of this container.

A reference implementation suitable for deployment in execution platforms is included in this repository at [`src/user-embedding-to-campaign-scoring/`](../src/user-embedding-to-campaign-scoring/). See its [README](../src/user-embedding-to-campaign-scoring/README.md) for API details and deployment instructions. The service is provided as a Docker image designed to run as a sidecar on DSP infrastructure. Its API surface covers three operations: registering campaign head weights, scoring user embeddings against those heads (cosine, dot product, or L2 similarity), and retrieving scoring analytics.

The method of scoring, tagging and validation of compatible models, and normalization functions are registered along with the weights representing the campaign. Model configuration travels with the head registration, allowing different campaigns to use different embedding models without redeployment.

Note on latencies -- a single GPU can support many thousands of campaigns with sub-millisecond latency.

Downstream of the bid, standard RTB flows work as usual, making this compatible with any ad server that supports OpenRTB. Downstream use of user embeddings is not part of the current scope of this project, but it is not a large leap to see how they could be applied in bidding logic or creative rendering.

### Campaign Training

The responsibility of this component is to produce a vector of weights (or vector set of weights) representing the campaign. This is accomplished by training with data that the advertiser has access to, such as CRM and CAPI, to target embeddings that active in the bid stream data. Targeting the embedding space on the supply-side requires two things; an audience model to generate embeddings over the advertisers known universe and reinforcement signals being reported back from actual events (e.g. impression or click beacons, campaign scoring responses).

The audience model on the supply side and the demand side is ideally the same model, however this training step can additionally serve as an alignment step for cross-model compatibility, using techniques facilitated by infrastructure providers such as LiveRamp. This allows the advertisers model to learn the geometry and optimal set of weights to target for a given campaign goal.

The information transmitted back to a signal aggregator can be a **reduced representation** of the embedding that itself can be used as a feature in campaign training. The genericity of using embeddings allows us to reduce either contextual, event series, or identity signals with the same mechanism, giving us an adjustable dial to meet any regulatory surface.

**Training loop:** (Marketplace, CRM, CAPI) -> Campaign Model Training -> Campaign Head -> Campaign Scoring -> Signal Aggregator -> Loop

---

## Component Ownership

A common question is "who runs what?" The table below clarifies operational responsibility for each component.

| Component | Operated By | Notes |
|---|---|---|
| Tag (e.g. ATS.js) | Publisher (LiveRamp provides ATS.js; others can build to spec) | Manages 1P cookie and embedding storage |
| Inference Server | LiveRamp / Publisher | Generates embeddings from page content and brand data |
| Campaign Scoring Function | DSP | Provided as a Docker image; runs as a sidecar on DSP infrastructure |
| Campaign Training | Advertiser / Clean Room | Produces campaign head weights from CRM, CAPI, marketplace data |
| Signal Aggregator | LiveRamp | Collects reduced representations and event signals for the training loop |
| Prebid.js + OpenRTB Transport | Publisher / SSP | Standard header bidding flow; embedding travels in ORTB2 segment ext |

---

## Model Interoperability

Different models produce incomparable vector spaces. Two contextual models both producing 384-dimensional vectors are not inherently comparable -- the geometry of the learned space differs between models. This is a first-class architectural concern.

The protocol addresses this in two ways. First, model metadata and an embedding taxonomy travel with every vector. The scoring service partitions heads by `model:embedding_type` and only scores against matching embeddings. The `embedding_space_id` and `compatible_with` fields in the campaign head schema (see the [scoring service README](../src/user-embedding-to-campaign-scoring/README.md)) make compatibility explicit at the protocol level.

Second, alignment techniques can map one embedding space onto another when the underlying models were trained on the same class of data. Rotation is one such technique, but alignment can employ a range of methods -- some proprietary -- facilitated by infrastructure providers such as LiveRamp. The key insight is that models trained on the same class of data learn similar structure, making alignment feasible. The embedding taxonomy (contextual, identity, behavioral, reinforcement, CAPI, intent) is what determines whether alignment is possible and meaningful.

---

## FAQ

### Isn't this just contextual advertising?

It is not, for two reasons. There is a deeper semantic understanding that is being taken advantage of provided by the underlying LLM or GNN. There is an inherent first party identity being used as well through the updates to an embedding already living in the 1P cookie. So the embedding doesn't just represent the current context, but actually the user's journey across this publisher's space.

### What about third-party Identity?

This prototype will limit itself to campaign level reporting and optimization, but Identity itself can also be included in the embedding through an experimental Identity model built by LiveRamp. This gives the system flexibility into the level of precision on which to optimize; from global campaign level all the way down to person level precision, all with the turn of a dial.

### Embeddings aren't interoperable between models, so how does this work?

Embedding providers are analogous to Identity providers in Agentic Audiences. There exist a handful of Identity providers in the industry, and IDs are not interoperable without setting up complicated ID syncs replete with conflicts. Embeddings are generally not interoperable out of the box (although in some cases they can be aligned with minimal effort). Embeddings have the advantage of being learnable by proprietary models, allowing data controllers with privileged data access to take full and unique advantage of their assets.

### Why is the scoring function open source?

To lower adoption barriers for DSPs. The scoring function must run at DSP scale and latency -- DSPs need to audit, customize, and trust the code running on their infrastructure. Open-sourcing the reference implementation aligns with the IAB standards work and signals that this is a collaborative, ecosystem-wide effort rather than a proprietary lock-in.

### What data does a publisher need to provide?

At minimum, title and keywords. Publishers provide this through standard locations (meta tags, data layers) or server-side. The tag does not crawl or scrape pages -- the integration is lightweight by design. Additional signals such as content categories or behavioral events improve embedding quality but are not required.
