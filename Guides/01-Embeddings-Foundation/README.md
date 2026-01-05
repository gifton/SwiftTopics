# Chapter 1: Embeddings Foundation

> **Understanding how text becomes geometry—the semantic foundation for topic discovery.**

Before we can cluster documents into topics, we need to represent them as **numbers**. This chapter explains how neural embeddings transform text into vectors where semantic similarity becomes geometric proximity.

---

## Chapter Contents

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [1.1 Why Embeddings for Topics](./01-Why-Embeddings-For-Topics.md) | Text as geometry | Semantic similarity = vector proximity |
| [1.2 Embedding Spaces](./02-Embedding-Spaces.md) | High-dimensional vectors | Dimensions, curse of dimensionality preview |
| [1.3 Distance Metrics](./03-Distance-Metrics.md) | Euclidean vs Cosine | When to use which, normalization |

---

## What You'll Learn

By the end of this chapter, you'll understand:

- Why text must become numbers for machine learning
- How embedding models encode semantic meaning into vectors
- What it means for embeddings to be "768-dimensional"
- Why similar texts have nearby vectors (and vice versa)
- How to measure similarity between embeddings
- Why SwiftTopics is embedding-agnostic

---

## The Core Insight

Topic modeling is fundamentally a **clustering problem**. But you can't cluster text directly—"Had a great run" and "Went jogging this morning" look completely different as strings.

Embeddings solve this by mapping text to a **geometric space** where:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING SPACE                                  │
│                                                                         │
│                    ★ "neural networks"                                  │
│                       "deep learning" ★                                 │
│            ★ "backpropagation"                                          │
│                                                                         │
│                                                                         │
│                                                                         │
│   ★ "morning run"                                                       │
│      "went jogging" ★                                                   │
│         ★ "5K personal best"                                            │
│                                                                         │
│                                                                         │
│                             ★ "coffee brewing"                          │
│                                "espresso machine" ★                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Similar concepts cluster together in vector space.
This geometric structure enables clustering algorithms.
```

---

## SwiftTopics and Embeddings

SwiftTopics follows the **BYOE pattern** (Bring Your Own Embeddings). This design choice provides flexibility:

| Embedding Source | Pros | Cons |
|------------------|------|------|
| **Apple NL** (via SwiftTopicsApple) | On-device, fast, private | 512D, English-focused |
| **CoreML models** | On-device, customizable | Requires model deployment |
| **Remote APIs** (OpenAI, etc.) | High quality, multilingual | Network dependency, cost |
| **Pre-computed** | Offline capability | Storage overhead |

```swift
// SwiftTopics accepts embeddings from any source
let model = TopicModel(configuration: .default)

// Option 1: Pre-computed embeddings
let result = try await model.fit(
    documents: documents,
    embeddings: precomputedEmbeddings
)

// Option 2: EmbeddingProvider protocol
let result = try await model.fit(
    documents: documents,
    embeddingProvider: myProvider
)
```

---

## Prerequisites Check

Before starting this chapter, ensure you understand:

- [ ] What a vector is (array of numbers)
- [ ] Basic geometry (distance between points)
- [ ] Why "similar" is subjective without a formal definition

---

## Start Here

**[→ 1.1 Why Embeddings for Topics](./01-Why-Embeddings-For-Topics.md)**

---

*Chapter 1 of 6 • SwiftTopics Learning Guide*
