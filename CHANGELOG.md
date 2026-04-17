# Changelog

All notable changes to SwiftTopics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-beta.1] - 2026-01-04

### Added

#### Core Pipeline
- **Embedding Layer**: `EmbeddingProvider` protocol with `PrecomputedEmbeddingProvider` implementation
- **Dimension Reduction**: Full UMAP implementation with spectral initialization, PCA fallback
- **Clustering**: Complete HDBSCAN with Ball Tree spatial indexing, mutual reachability, MST, EOM extraction
- **Topic Representation**: c-TF-IDF keyword extraction with configurable tokenization
- **Coherence Evaluation**: NPMI scoring, diversity metrics, topic quality assessment

#### TopicModel Orchestrator
- `TopicModel` actor for thread-safe pipeline coordination
- Configuration presets: `.default`, `.fast`, `.quality`
- Full `fit()`, `transform()`, `fitTransform()` API
- Topic merging and reduction operations
- Semantic search capabilities

#### Incremental Updates System
- **Phase 1 - Storage**: `TopicModelStorage` protocol, `FileBasedTopicModelStorage`, `BufferedEntry`, `IncrementalVocabulary`
- **Phase 2 - Interruptible Training**: `TrainingCheckpoint`, `InterruptibleTrainingRunner` with checkpoint/resume
- **Phase 3 - Topic Matching**: `HungarianMatcher` (O(n³) optimal bipartite), `TopicMatcher`, `TopicIDGenerator`, `ModelMerger`
- **Phase 4 - Incremental Updater**: `IncrementalTopicUpdater` actor, `IncrementalUpdateConfiguration`, `IncrementalTopicAssignment`

#### Platform Support
- `SwiftTopicsApple` target with `AppleNLProvider` and `EmbedKitAdapter`
- GPU acceleration via VectorAccelerate (Metal 4)
- Full Swift 6 concurrency support with strict Sendable conformance

### Technical Details

- **Test Coverage**: 228 tests covering all components
- **Platforms**: iOS 26+, macOS 26+, visionOS 26+
- **Dependencies**: VectorAccelerate 0.3.1+, VectorCore 0.1.6+

### Known Limitations

- KDTree spatial index not implemented (Ball Tree preferred for high-dimensional data)
- GPU kernels for HDBSCAN operations not yet contributed to VectorAccelerate
- Hierarchical/dynamic topics deferred to future release

---

## [Unreleased]

## [0.1.0-beta.2] - 2026-04-14

### Changed
- **Architectural Decoupling**: Refactored `TopicModel` to depend on `DimensionReducer`, `ClusteringEngine`, and `TopicRepresenter` protocols, guided by the open/closed principle.
- **Memory Efficiency**: Prevented `FittedTopicModelState` from retaining documents and embeddings indefinitely, massively lowering memory consumption on large corpora.
- **Probability Normalization**: Swapped softmax for UMAP's student-t distribution during topic assignment to preserve mathematical soundness in high-dimensional spaces.
- **MST Performance**: Eliminated an O(N² log N) bottleneck inside Prim's algorithm using array-backed dense processing. 
- **Storage Bottlenecks**: Remedied high allocation and memory fragmentation rates when serializing large `Embedding` buffers to binary via pointer-oriented appends.
- **Interruption/Task Cancelation**: Replaced the custom unsafe `shouldContinue` closures with native Swift `Task.checkCancellation()` patterns.
- **Dependencies**: Bumped `VectorAccelerate` to 0.4.4 and `EmbedKit` to 0.3.4.

### Planned
- Hierarchical topic support
- Dynamic/temporal topic modeling
- Guided/semi-supervised topics
- SwiftTopicsMetal target for large corpora (>50k docs)
