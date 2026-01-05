# Chapter 4: Topic Representation

> **From clusters to keywords—making topics human-readable.**

---

## The Challenge

After clustering, you have groups of similar documents. But what does each group *mean*?

```
HDBSCAN output:                    What we want:

Cluster 0: [doc_0, doc_4, doc_7]   Topic 0: "fitness, running, exercise"
Cluster 1: [doc_1, doc_5, doc_8]   Topic 1: "work, meeting, deadline"
Cluster 2: [doc_2, doc_3, doc_6]   Topic 2: "anxiety, stress, feeling"

Raw indices tell us nothing.       Keywords tell the whole story.
```

**Topic representation** bridges this gap—extracting keywords that characterize each cluster.

---

## The c-TF-IDF Approach

SwiftTopics uses **class-based TF-IDF (c-TF-IDF)**, an innovation from BERTopic that adapts traditional TF-IDF for topic modeling.

```
Traditional TF-IDF:              c-TF-IDF:
─────────────────────            ──────────────────────────────────
Compares terms across            Compares terms across CLUSTERS
DOCUMENTS                        (treating each cluster as one
                                 mega-document)

"What terms distinguish          "What terms distinguish THIS TOPIC
this document from               from OTHER TOPICS?"
other documents?"
```

The key insight: **merge all documents in a cluster into one virtual document**, then ask which terms make that virtual document distinctive.

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Topic Representation Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Documents          Clustered           Tokenized &        Keywords    │
│  + Clusters    →    Documents      →    Aggregated    →    per Topic   │
│                     by Topic            by Topic                        │
│                                                                         │
│  ┌─────────┐       ┌─────────┐        ┌─────────────┐    ┌───────────┐ │
│  │ Doc 0   │       │ Topic 0 │        │ Topic 0:    │    │ fitness   │ │
│  │ (→ T0)  │       │ [0,4,7] │        │ "great run  │    │ running   │ │
│  ├─────────┤       ├─────────┤        │  morning    │    │ exercise  │ │
│  │ Doc 1   │       │ Topic 1 │        │  5k shoes"  │    │ health    │ │
│  │ (→ T1)  │  →    │ [1,5,8] │   →    ├─────────────┤ →  ├───────────┤ │
│  ├─────────┤       ├─────────┤        │ Topic 1:    │    │ work      │ │
│  │ Doc 2   │       │ Topic 2 │        │ "meeting    │    │ deadline  │ │
│  │ (→ T2)  │       │ [2,3,6] │        │  project    │    │ project   │ │
│  └─────────┘       └─────────┘        │  deadline"  │    │ meeting   │ │
│                                       └─────────────┘    └───────────┘ │
│                                                                         │
│     Input           Group              Merge &              c-TF-IDF   │
│                                        Tokenize             Scoring    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What You'll Learn

| Guide | Topic | Key Concepts |
|-------|-------|--------------|
| [4.1 From Clusters to Topics](./01-From-Clusters-To-Topics.md) | The representation problem | Why clusters need keywords |
| [4.2 TF-IDF Refresher](./02-TF-IDF-Refresher.md) | Classic TF-IDF | Term frequency, inverse document frequency |
| [4.3 Class-Based TF-IDF](./03-Class-Based-TF-IDF.md) | The c-TF-IDF innovation | Cluster-as-document, class frequency |
| [4.4 Keyword Diversification](./04-Keyword-Diversification.md) | MMR diversification | Avoiding redundant keywords |

---

## The SwiftTopics Implementation

### Core Components

```
Sources/SwiftTopics/Representation/
├── cTFIDF.swift              ← Core c-TF-IDF computation
├── CTFIDFRepresenter.swift   ← TopicRepresenter implementation
├── Tokenizer.swift           ← Text tokenization
└── Vocabulary.swift          ← Term→index mapping

Sources/SwiftTopics/Core/
└── Topic.swift               ← Topic struct with keywords

Sources/SwiftTopics/Protocols/
└── TopicRepresenter.swift    ← Protocol + configuration
```

### Quick Start

```swift
// 📍 See: Sources/SwiftTopics/Representation/CTFIDFRepresenter.swift

let representer = CTFIDFRepresenter(configuration: .default)

let topics = try await representer.represent(
    documents: documents,
    assignment: clusterAssignment
)

// Each topic has ranked keywords
for topic in topics {
    print("Topic \(topic.id): \(topic.keywordSummary())")
    // Output: "Topic 0: fitness, running, exercise, health, morning"
}
```

---

## Key Formulas

### c-TF-IDF Formula

```
c-TF-IDF(t, c) = tf(t, c) × log(1 + A / tf(t, corpus))

Where:
  tf(t, c)      = Frequency of term t in cluster c
  tf(t, corpus) = Frequency of term t across all clusters
  A             = Average number of tokens per cluster
```

### Why It Works

```
High c-TF-IDF score:
  - Term appears frequently in THIS cluster (high tf(t,c))
  - Term appears infrequently in OTHER clusters (low tf(t,corpus))
  - This term is DISTINCTIVE to this topic

Low c-TF-IDF score:
  - Term is rare in this cluster, OR
  - Term is common across many clusters
  - This term doesn't characterize the topic
```

---

## Configuration Options

```swift
// 📍 See: Sources/SwiftTopics/Protocols/TopicRepresenter.swift

public struct CTFIDFConfiguration {
    /// Number of keywords to extract per topic
    public let keywordsPerTopic: Int       // Default: 10

    /// Minimum document frequency for term inclusion
    public let minDocumentFrequency: Int   // Default: 1

    /// Maximum document frequency ratio (filters common words)
    public let maxDocumentFrequencyRatio: Float  // Default: 0.95

    /// Whether to include bigrams
    public let useBigrams: Bool            // Default: false

    /// Whether to apply MMR diversification
    public let diversify: Bool             // Default: false

    /// Diversity weight for MMR
    public let diversityWeight: Float      // Default: 0.3
}
```

---

## Example Output

```swift
// Real output from a journal topic model:

Topic 0 (Size: 127):
  Keywords: running, exercise, workout, fitness, morning
  Score:    1.00    0.87      0.81      0.76     0.68

Topic 1 (Size: 89):
  Keywords: meeting, project, deadline, client, review
  Score:    1.00    0.92      0.84      0.79     0.71

Topic 2 (Size: 54):
  Keywords: anxiety, feeling, stress, sleep, worried
  Score:    1.00    0.89      0.83      0.75     0.69

Outliers (Size: 23):
  Documents that don't fit any topic cleanly
  (No keywords—not a real topic)
```

---

## Prerequisites

Before starting this chapter, you should understand:

- ✅ Document embeddings (Chapter 1)
- ✅ How HDBSCAN produces cluster assignments (Chapter 3)
- ✅ Basic statistics (frequency counts)

---

## 💡 Key Insight

Clusters are **geometric**—they group embeddings that are close in vector space. Keywords are **linguistic**—they describe what humans understand.

c-TF-IDF bridges geometry and linguistics by asking: *"What words appear disproportionately often in the documents that ended up in this cluster?"*

```
Embedding Space                   Keyword Space
───────────────                   ─────────────
    ●●●●                          "fitness"
   ●●●●●●●  ──────────────────►  "running"
    ●●●●                          "exercise"

The dots are vectors.             The words are meaning.
HDBSCAN sees density.             c-TF-IDF extracts labels.
```

---

## Let's Begin

Ready to understand how clusters become human-readable topics?

**[→ 4.1 From Clusters to Topics](./01-From-Clusters-To-Topics.md)**

---

*Chapter 4 of 6 • SwiftTopics Learning Guide*
