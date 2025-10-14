# User Context Protocol™ (UCP)

An Open Protocol for Intelligent Interoperability Across Advertising Agents

---

## Overview

The User Context Protocol (UCP) is an open standard proposed by LiveRamp to enable intelligent agents in advertising and marketing to interoperate with precision and efficiency.

As the industry transitions into the agentic web, where autonomous buyer, seller, and measurement agents act on behalf of users and organizations, UCP provides a common language for exchanging context, intent, and results.
It defines how agents communicate, how context is shared and updated, and how intelligence can evolve from prompt-based reasoning to learned representations.

---

## Motivation

Today, the rise of large language models (LLMs) and agent frameworks has made it possible for general-purpose agents to plan, reason, and execute tasks using text-based prompts.
Frameworks such as the Model Context Protocol (MCP) define how these agents interact with external tools and APIs.
However, prompt-driven coordination is inefficient for high-frequency, data-intensive workflows like advertising optimization.

UCP bridges this gap by defining a path from prompt-based interaction to embedding-based interoperability.

1. **Phase 1 – Agent Interoperability Layer**  
   Enable existing LLM agents to exchange structured marketing context using standardized inputs and outputs.
   Focus on context engineering, schema alignment, and real-time messaging between agents such as, but not limited to, buyer, seller, and measurement agents.

2. **Phase 2 – Context Learning Layer**  
   Train deep learning models on the contextual and behavioral data exchanged through the protocol.
   These models learn to represent user journeys, ad impressions, conversions, and marketplace signals as dynamic embeddings.

3. **Phase 3 – Embedding Intelligence Layer**  
   Agents evolve from exchanging textual context to exchanging embeddings that encode understanding of user intent, campaign state, and performance.
   These embeddings act as transferable memory between agents that share a compatible vector space, enabling near real-time optimization without large prompt contexts.

---

## Core Principles

- **Interoperability:** Define clear input and output contracts for all agent types.
- **Context Engineering:** Maintain relevant and bounded context to keep agents aligned on goals.
- **Incremental Evolution:** Support LLM agents and prompt orchestration today while enabling learned models tomorrow.
- **Identity and Privacy:** Preserve user trust with privacy-safe handling of identity and behavioral signals.
- **Composability:** Allow independent agents to cooperate through standardized schemas and embeddings.

---

## Agent Ecosystem

| Agent Type | Role | Early-Stage Interface | Future Interface |
|-------------|------|----------------------|------------------|
| Buyer Agent | Plans and executes ad placements | Prompt plus JSON I/O | Embedding-based optimization |
| Seller Agent | Publishes and prices inventory | Context-aware API | Vector similarity negotiation |
| Measurement Agent | Tracks outcomes and updates models | Event feed | Feedback embedding update |

---

## Technical Vision

UCP defines:

1. **Protocol Interfaces** - APIs and schemas for exchanging context, signals, and results.
2. **Context Management** - Strategies for maintaining scoped, composable context windows in LLM-driven agents.
3. **Embedding Interoperability** - Standards for shared embedding structures, dimensional alignment, and vector-space identity.
4. **Agent Coordination Flows** - Request and response patterns for cross-agent actions.
5. **Privacy and Consent Controls** - Mechanisms for secure signal sharing, security and authentication, permissible uses, and time-to-live (TTL) of consented data.
6. **Agentic Attestation** - Ensures confidentiality and integrity of code and information accessed or executed through agents, including provenance and controlled execution environments.
7. **Token Exchange and Settlement** - Enables agents to exchange tokens or perform value transfers for advertising events, supporting integration with emerging payment and attribution protocols such as AP2 and X402.

By evolving from structured text exchanges to compact vector exchanges, UCP will enable major gains in speed, scale, and cost efficiency for campaign optimization.

---

## Example Evolution Path

1. **Today:**  
   - A buyer agent prompts a seller agent:  
     "Provide available CTV inventory for users interested in electric vehicles in San Francisco this week."  
   - The seller agent responds using the UCP schema, returning JSON data on available segments.
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

This repository hosts the evolving UCP specification and reference implementations.
We welcome contributions from engineers, researchers, and organizations shaping the next generation of agentic advertising.

To get involved:
- Fork the repo and explore the `/specs` and `/reference` directories.
- Propose changes via pull request.
- Join or start a working group under `/community`.

---

## License

- Specification and Documentation: Creative Commons Attribution 4.0 International (CC BY 4.0)  
- Reference Implementations: Apache License 2.0  

"User Context Protocol" is a trademark of LiveRamp Holdings, Inc.  
Use of these marks is subject to the [TRADEMARK.md](./TRADEMARK.md) policy.

---