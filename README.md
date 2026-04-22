# Agentic Audiences

An Open Protocol for Intelligent Interoperability Across Advertising Agents

> **Note:** Agentic Audiences was formerly known as the User Context Protocol (UCP).

> **Note:** This specification represents LiveRamp's initial proposal. We have open-sourced this repository to enable the community to collaboratively define and reach collective agreement on a standard for embedding exchange in agentic advertising.

---

## Overview

Agentic Audiences is an open standard proposed by LiveRamp to enable intelligent agents in advertising and marketing to interoperate through the exchange of **signals**—identity, contextual, and reinforcement information—that represent a consumer's true real-time intent and response to advertising.

As the industry transitions into the agentic web, where autonomous buyer, seller, and measurement agents powered by AI/ML models act on behalf of users and organizations, advertising decisions increasingly rely on these models to process billions of signals per second. Agentic Audiences defines a protocol for agents to exchange **embeddings**—compact, learned vector representations that efficiently encode identity signals (who the user is), contextual signals (what they're doing right now), and reinforcement signals (how they respond to ads) in a privacy-preserving, interoperable format.

This repository contains:
- **Technical specifications** for embedding exchange formats and schemas
- **AI/ML model architecture guidance** ([`/docs/AI_ML Models in Agentic Digital Advertising Era.pdf`](docs/AI_ML%20Models%20in%20Agentic%20Digital%20Advertising%20Era.pdf)) explaining how 15+ model categories across the advertising lifecycle consume and produce embeddings
- **Systems and models architecture** ([`/docs/systems-and-models.md`](docs/systems-and-models.md)) — a "10K foot view" of the end-to-end system: audience model, browser/edge layer, campaign scoring service, campaign training loop, component ownership, and model interoperability
- **Reference schemas and examples** demonstrating real-world protocol usage (in-progress)

---

## Motivation

### The Challenge: Agents, Models, and Signals

Next-gen advertising will operate through **agentic AI systems** that make millions of autonomous decisions per second. These agents will rely on **AI/ML models**—from click prediction to conversion modeling to multi-touch attribution—that process vast arrays of **signals** to understand user intent and optimize outcomes.

**Signals** come in three critical forms:
- **Identity signals**: Who the user is (hashed identifiers, segments, behavioral history)
- **Contextual signals**: What the user is doing right now (page content, time of day, device, engagement patterns)
- **Reinforcement signals**: How users respond to advertising (impressions, clicks, conversions, engagement metrics)

Today's advertising systems struggle to efficiently exchange these signals:
- **Text-based prompts** are too verbose and slow for real-time bidding (<100ms response time)
- **Raw feature vectors** lack semantic meaning and don't transfer across systems
- **Proprietary formats** prevent interoperability between buyer, seller, and measurement agents

### The Solution: Embeddings as Signal Carriers

**Embeddings** solve this problem by encoding identity, contextual, and reinforcement signals into dense, learned vector representations that:
- **Compress information**: 256-1024 dimensions vs. thousands of raw features across all signal types
- **Capture semantics**: Similar intents and behaviors have similar embeddings (vector similarity)
- **Enable transfer learning**: Models trained by one agent can be understood by others
- **Preserve privacy**: Embeddings can represent intent and response patterns without exposing raw user data
- **Support real-time inference**: Fast vector operations enable sub-100ms decisions
- **Unify signal types**: A single embedding can simultaneously encode who the user is, what they're doing, and how they've responded to past interactions

Agentic Audiences defines how agents exchange these embeddings, transforming advertising from prompt-driven coordination to embedding-based interoperability that spans the entire decision-feedback loop.

1. **Phase 1 – Agent Interoperability Layer**  
   Enable existing LLM agents to exchange structured marketing context using standardized inputs and outputs.
   Focus on context engineering, schema alignment, and real-time messaging between agents such as, but not limited to, buyer, seller, and measurement agents.

2. **Phase 2 – Context Learning Layer**  
   Train deep learning models on the contextual and behavioral data exchanged through the protocol.
   These models learn to represent user journeys, ad impressions, conversions, and marketplace signals as dynamic embeddings.

3. **Phase 3 – Embedding Intelligence Layer**
   Agents evolve from exchanging textual context to exchanging embeddings that encode understanding of user intent, campaign state, and performance.
   These embeddings act as transferable memory between agents that share a compatible vector space, enabling near real-time optimization without large prompt contexts.

> **📄 Deep Dive: AI/ML Models in Agentic Advertising**
> The [`/docs/AI_ML Models in Agentic Digital Advertising Era.pdf`](docs/AI_ML%20Models%20in%20Agentic%20Digital%20Advertising%20Era.pdf) whitepaper provides comprehensive coverage of the 15+ model categories—from Audience Discovery and Lifetime Value Prediction to Multi-Touch Attribution and Incrementality Measurement—that power agentic advertising systems. These models both **consume** embeddings (using them as input features) and **produce** embeddings (generating vector representations of users, contexts, and creatives) that are exchanged via Agentic Audiences. Understanding this model ecosystem is essential for implementing Agentic Audiences-compatible agents.

---

## Core Principles

- **Interoperability:** Define clear input and output contracts for all agent types.
- **Context Engineering:** Maintain relevant and bounded context to keep agents aligned on goals.
- **Incremental Evolution:** Support LLM agents and prompt orchestration today while enabling learned models tomorrow.
- **Identity and Privacy:** Preserve user trust with privacy-safe handling of identity and behavioral signals.
- **Composability:** Allow independent agents to cooperate through standardized schemas and embeddings.

---

## Agent Ecosystem

Agentic Audiences sits alongside IAB Tech Lab's [**buyer agent**](https://github.com/IABTechLab/buyer-agent) and [**seller agent**](https://github.com/IABTechLab/seller-agent)—reference implementations for buyer- and seller-side automation (discovery, negotiation, booking, activation)—complementing that stack with a dedicated data plane for embeddings.

**How Agentic Audiences fits in:**

- **[Buyer agent](https://github.com/IABTechLab/buyer-agent)** / **[seller agent](https://github.com/IABTechLab/seller-agent)** (IAB Tech Lab) — programmatic workflows between buyers and sellers (media kits, deals, OpenDirect-style fulfillment, platform activation)
- **Agentic Audiences** — the data plane: how agents exchange embeddings that encode identity, contextual, and reinforcement signals

Together, these layers enable a complete agentic advertising ecosystem:

| Layer | Focus | Purpose |
|-------|-------|---------|
| **Control** | IAB Tech Lab [buyer agent](https://github.com/IABTechLab/buyer-agent) / [seller agent](https://github.com/IABTechLab/seller-agent) | Connect buyers and sellers; activate audiences, execute buys, manage inventory |
| **Data** | Agentic Audiences | Agent-to-agent embedding exchange (share learned representations of users, contexts, and outcomes) |

**Example flow:**
1. A buyer agent (for example, the Tech Lab [buyer agent](https://github.com/IABTechLab/buyer-agent)) discovers audience signals—for example, "Find premium sports enthusiasts interested in running shoes"—via seller-facing workflows
2. The seller stack (for example, the Tech Lab [seller agent](https://github.com/IABTechLab/seller-agent)) returns candidate audiences or features
3. The buyer agent uses **Agentic Audiences** to exchange contextual and identity embeddings with a seller agent
4. The seller agent uses embeddings to match inventory in real-time via vector similarity
5. Reinforcement signals (impressions, conversions) flow back through **Agentic Audiences** to update models
6. The measurement agent reports results through the buyer/seller automation path and uses **Agentic Audiences** to share learned embeddings

Together with the IAB Tech Lab buyer and seller agent ecosystem and broader platform integrations, Agentic Audiences supports the transition from prompt-based advertising automation to embedding-based intelligence to drive efficiencies by eliminating the need for massive copies of user-level datasets across the ecosystem.

---

## Technical Vision

Agentic Audiences defines:

1. **Protocol Interfaces** - APIs and schemas for exchanging context, signals, and results.
2. **Context Management** - Strategies for maintaining scoped, composable context windows in LLM-driven agents.
3. **Embedding Interoperability** - Standards for shared embedding structures, dimensional alignment, and vector-space identity.
4. **Agent Coordination Flows** - Request and response patterns for cross-agent actions.
5. **Privacy and Consent Controls** - Mechanisms for secure signal sharing, security and authentication, permissible uses, and time-to-live (TTL) of consented data.
6. **Agentic Attestation** - Ensures confidentiality and integrity of code and information accessed or executed through agents, including provenance and controlled execution environments.
7. **Token Exchange and Settlement** - Enables agents to exchange tokens or perform value transfers for advertising events, supporting integration with emerging payment and attribution protocols such as AP2 and X402.

By evolving from structured text exchanges to compact vector exchanges, Agentic Audiences will enable major gains in speed, scale, and cost efficiency for campaign optimization.

> **🏗 System Architecture Deep Dive**
> [`/docs/systems-and-models.md`](docs/systems-and-models.md) provides a 10K foot view of the end-to-end Agentic Audiences architecture — covering the audience model, browser/edge embedding storage (ATS.js + Prebid), campaign scoring service (Docker sidecar), campaign training loop, and component ownership. It also addresses model interoperability and answers common architectural questions.

---

## Example Evolution Path

1. **Today:**  
   - A buyer agent prompts a seller agent:  
     "Provide available CTV inventory for users interested in electric vehicles in San Francisco this week."  
   - The seller agent responds using the Agentic Audiences schema, returning JSON data on available segments.
   - A measurement agent records conversions and feeds updates.

2. **Future:**  
   - The buyer agent receives a user embedding representing current context.
   - It queries seller embeddings directly in vector space to find optimal matches.
   - Feedback embeddings from the measurement agent continuously refine the shared context model.

---

## Roadmap

1. **Specification Draft** - Schema and interfaces for prompt-driven interoperability.
2. **Reference SDKs** - Python and JavaScript libraries for MCP-compatible agents.
3. **Context Engine Framework** - Tools for managing context window updates and relevance.
4. **Embedding Schema Standard** - Common representation for learned user and campaign embeddings.
5. **Industry Working Group** - Partnership with open-source and adtech leaders to align adoption.

---

## Contributing

This repository hosts the evolving Agentic Audiences specification and reference implementations.
We welcome contributions from engineers, researchers, and organizations shaping the next generation of agentic advertising.

To get involved:
- Read [`/docs/systems-and-models.md`](docs/systems-and-models.md) for a 10K foot view of the system architecture
- Read [`/docs/AI_ML Models in Agentic Digital Advertising Era.pdf`](docs/AI_ML%20Models%20in%20Agentic%20Digital%20Advertising%20Era.pdf) to understand the model ecosystem that Agentic Audiences enables
- Fork the repo and explore the `/specs` directory for technical specifications
- Propose changes via pull request
- Join or start a working group under `/community`

---

## License

- Specification and Documentation: Creative Commons Attribution 4.0 International (CC BY 4.0)  
- Reference Implementations: Apache License 2.0  

---