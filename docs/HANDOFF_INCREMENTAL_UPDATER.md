# Handoff: IncrementalUpdater Research & Planning Spike

## Overview

**Task**: Research and design an incremental update system for TopicModel
**Location**: `/Users/goftin/dev/real/GournalV2/SwiftTopics`
**Type**: Research spike - NO IMPLEMENTATION CODE
**Deliverable**: `DESIGN_INCREMENTAL_UPDATER.md`

---

## ⚠️ IMPORTANT: THIS IS A RESEARCH SPIKE

**DO NOT** write any Swift implementation code.
**DO NOT** modify existing source files.
**DO NOT** create placeholder Swift files.

**DO** create a comprehensive design document that will guide future implementation.

---

## Objective

Design an incremental update system for `TopicModel` that avoids full retraining when new documents are added. This is complex because HDBSCAN and UMAP are fundamentally batch algorithms designed for static datasets.

---

## Read First

1. `SPEC.md` (Section 2.6, 4.2) - Original design intent for incremental updates
2. `Sources/SwiftTopics/Model/TopicModel.swift` - Current fit/transform flow
3. `Sources/SwiftTopics/Model/TopicModelState.swift` - Persisted state structure
4. `Sources/SwiftTopics/Clustering/HDBSCAN/HDBSCAN.swift` - Clustering algorithm
5. `Sources/SwiftTopics/Reduction/UMAP/UMAP.swift` - Dimensionality reduction
6. `Sources/SwiftTopics/Reduction/UMAP/NearestNeighborGraph.swift` - k-NN for UMAP

---

## Research Questions to Answer

### 1. Literature Review

Search for and summarize:

- **BERTopic incremental updates**: How does BERTopic's `update_topics()` work? What are its limitations?
- **HDBSCAN approximate_predict()**: What does scikit-learn's implementation do internally?
- **Online/streaming HDBSCAN**: Are there papers on incremental density-based clustering?
- **UMAP transform()**: How does umap-learn project new points through a fitted model?
- **Streaming dimensionality reduction**: Any research on online UMAP or similar?

Key sources to check:
- BERTopic GitHub issues/discussions on incremental learning
- HDBSCAN documentation on `approximate_predict()`
- Original UMAP paper and supplementary materials
- Recent ML papers on streaming topic modeling

### 2. Architecture Options

Document **at least 3** distinct approaches:

#### Option A: Transform-Only (Simplest)
- New documents use fitted reducer + nearest-centroid assignment
- No retraining of UMAP or HDBSCAN
- Topics can only grow, never split/merge from new data
- Quality degrades as data distribution shifts

#### Option B: Periodic Batch Refresh
- Buffer new documents until threshold reached
- Trigger full retrain when buffer size exceeds N
- Handle topic ID stability across retrains

#### Option C: Hybrid Incremental
- Transform for small updates (< threshold)
- Partial component retraining based on drift detection
- Most complex, best quality preservation

#### For Each Option, Analyze:

| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Computational complexity | ? | ? | ? |
| Memory requirements | ? | ? | ? |
| Topic ID stability | ? | ? | ? |
| Quality over time | ? | ? | ? |
| Implementation complexity | ? | ? | ? |
| Use case fit | ? | ? | ? |

### 3. Component-Level Analysis

For each pipeline stage, research and document:

#### Embedding Layer
- Embeddings are deterministic - no incremental concerns
- Just embed new documents with same provider

#### UMAP Reduction
- **Question**: Can we project new points through a fitted UMAP?
- **Research**: How does `umap.transform()` work in Python?
- **State needed**: What must be persisted? k-NN graph? Fuzzy simplicial set? Low-dim coordinates?
- **Quality**: How much does transform quality degrade vs. including point in fit?

#### HDBSCAN Clustering
- **Research**: What does `approximate_predict()` do internally?
- **Limitation**: Can only assign to EXISTING clusters - cannot discover new ones
- **Question**: Can we incrementally update the MST when adding points?
- **Question**: How do we detect when a new cluster should form?

#### c-TF-IDF Representation
- **Easiest to update**: Just recompute term frequencies
- **Consideration**: Vocabulary expansion for new terms
- **Consideration**: IDF values change as corpus grows

### 4. State Persistence Analysis

What additional state must `TopicModelState` store for incremental updates?

Current state (verify by reading TopicModelState.swift):
- Topics with keywords
- Document-topic assignments
- Configuration

Potentially needed:
- [ ] Original document embeddings (for similarity search, retraining)
- [ ] Reduced embeddings (for cluster assignment)
- [ ] Cluster centroids (for nearest-centroid assignment)
- [ ] UMAP k-NN graph (for transform)
- [ ] UMAP low-dimensional coordinates (for optimization context)
- [ ] Vocabulary with term frequencies (for c-TF-IDF updates)

**Estimate storage size** for different corpus sizes:

| Documents | Embeddings (512d) | Reduced (50d) | k-NN Graph (k=15) | Total |
|-----------|-------------------|---------------|-------------------|-------|
| 10,000 | ? MB | ? MB | ? MB | ? MB |
| 100,000 | ? MB | ? MB | ? MB | ? MB |
| 1,000,000 | ? MB | ? MB | ? MB | ? MB |

### 5. API Design Options

Propose and evaluate different API designs:

#### Option 1: Explicit Mode Selection
```swift
public enum UpdateMode {
    case transformOnly      // Fast, no retraining
    case partialRetrain     // Retrain affected components
    case fullRetrain        // Complete retraining
}

public func update(documents: [Document], mode: UpdateMode) async throws -> TopicModelResult
```

#### Option 2: Automatic Strategy Selection
```swift
// Automatically selects strategy based on:
// - Number of new documents vs existing
// - Detected distribution drift
// - Time since last full retrain
public func update(documents: [Document]) async throws -> TopicModelResult
```

#### Option 3: Separate Methods
```swift
public func addDocuments(_ documents: [Document]) async throws  // Transform only
public func retrain() async throws                              // Full retrain
public func retrainIfNeeded() async throws                      // Auto-detect
```

**Evaluate each**:
- Clarity of intent
- Flexibility
- Footgun potential (user picks wrong mode)
- Consistency with existing API

### 6. Drift Detection

How do we know when transform-only is degrading quality?

Research and propose metrics:
- Average distance to assigned cluster centroid (increasing = drift)
- Coherence score trend over time
- Percentage of new documents marked as outliers
- Embedding space coverage changes

**Questions to answer**:
- What threshold triggers retraining?
- Should this be configurable?
- How do we avoid false positives (legitimate new topics vs. drift)?

---

## Deliverable: DESIGN_INCREMENTAL_UPDATER.md

Create a comprehensive design document with these sections:

### Required Sections

1. **Executive Summary** (1 paragraph)
   - Problem statement
   - Recommended approach in one sentence
   - Key tradeoff acknowledged

2. **Literature Review Findings**
   - Summary of how BERTopic/HDBSCAN/UMAP handle this
   - Relevant papers or implementations
   - Key insights that informed the design

3. **Architecture Options Comparison**
   - Table comparing all options
   - Pros/cons for each
   - Use case recommendations

4. **Recommended Approach**
   - Which option and why
   - Tradeoffs accepted
   - Justification based on SwiftTopics use cases

5. **Detailed Design**
   - Data flow diagram (ASCII art is fine)
   - State requirements with storage estimates
   - API specification (Swift signatures)
   - Error handling strategy
   - Thread safety considerations

6. **Implementation Phases**
   - Ordered by dependency
   - Each phase should be independently testable
   - Estimated LOC per phase

7. **Open Questions & Risks**
   - Unknowns that need prototyping
   - Performance concerns
   - Edge cases to handle

8. **Effort Estimate**
   - Total estimated LOC
   - Estimated time (days/weeks)
   - Suggested milestones

---

## Research Resources

### Web Search Queries
- "BERTopic incremental learning update_topics"
- "HDBSCAN approximate_predict implementation"
- "UMAP transform new points"
- "streaming topic modeling"
- "online density-based clustering"
- "incremental dimensionality reduction"

### GitHub Repositories
- https://github.com/MaartenGr/BERTopic (check issues for "incremental")
- https://github.com/scikit-learn-contrib/hdbscan
- https://github.com/lmcinnes/umap

### Documentation
- BERTopic: https://maartengr.github.io/BERTopic/
- HDBSCAN: https://hdbscan.readthedocs.io/
- UMAP: https://umap-learn.readthedocs.io/

---

## Exit Criteria

- [ ] `DESIGN_INCREMENTAL_UPDATER.md` created in project root
- [ ] All 8 required sections completed
- [ ] At least 3 architecture options documented with comparison
- [ ] Clear recommendation with justification
- [ ] Realistic effort estimates based on research
- [ ] No Swift code written or modified

---

## Verification

```bash
# Check deliverable exists
ls -la DESIGN_INCREMENTAL_UPDATER.md

# Verify no source changes
git status Sources/

# Word count (should be substantial - 1500+ words)
wc -w DESIGN_INCREMENTAL_UPDATER.md
```

---

*Created: January 2025*
*Priority: Research spike before implementation*
