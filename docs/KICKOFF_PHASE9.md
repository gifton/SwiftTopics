# Phase 9: Apple Integration Target - Kickoff (Revised)

## Overview

**Duration**: 1 day (reduced from 2-3 days)
**LOC**: ~80 (reduced from ~200)
**Milestone**: M6 preparation (Apple-native embedding providers)

## Key Discovery: Leverage EmbedKit

**EmbedKit** (`/Users/goftin/dev/gsuite/VSK/EmbedKit`) already provides:

| Component | What It Does |
|-----------|-------------|
| `AppleNLContextualModel` | Wraps Apple's `NLContextualEmbedding` (512-dim) |
| `EmbeddingModel` protocol | Actor-based embedding interface with batch support |
| `EmbeddingGenerator` | High-level VectorProducer wrapper with progress |

Instead of reimplementing Apple's NL embedding support, we create a **thin adapter** that bridges:
- `EmbedKit.EmbeddingModel` → `SwiftTopics.EmbeddingProvider`

---

## Files to Create

### 1. Sources/SwiftTopicsApple/EmbedKitAdapter.swift (~50 LOC)

Adapter that wraps any `EmbedKit.EmbeddingModel` to conform to `SwiftTopics.EmbeddingProvider`.

```swift
import SwiftTopics
import EmbedKit

/// Adapts an EmbedKit `EmbeddingModel` to SwiftTopics' `EmbeddingProvider` protocol.
///
/// This enables using any EmbedKit model (CoreML, Apple NL, ONNX) with SwiftTopics.
///
/// ## Usage
/// ```swift
/// import SwiftTopics
/// import SwiftTopicsApple
/// import EmbedKit
///
/// // Use Apple's NLContextualEmbedding
/// let nlModel = try AppleNLContextualModel(language: "en")
/// let provider = EmbedKitAdapter(model: nlModel)
///
/// let model = TopicModel(configuration: .default)
/// let result = try await model.fit(documents: docs, embeddingProvider: provider)
/// ```
public struct EmbedKitAdapter<Model: EmbeddingModel>: EmbeddingProvider {

    private let model: Model

    public var dimension: Int {
        model.dimensions
    }

    /// Creates an adapter wrapping an EmbedKit embedding model.
    ///
    /// - Parameter model: The EmbedKit model to wrap.
    public init(model: Model) {
        self.model = model
    }

    public func embed(_ text: String) async throws -> SwiftTopics.Embedding {
        do {
            let embedding = try await model.embed(text)
            return SwiftTopics.Embedding(vector: embedding.vector)
        } catch {
            throw EmbeddingError.modelError(underlying: error)
        }
    }

    public func embedBatch(_ texts: [String]) async throws -> [SwiftTopics.Embedding] {
        do {
            let embeddings = try await model.embedBatch(texts, options: .default)
            return embeddings.map { SwiftTopics.Embedding(vector: $0.vector) }
        } catch {
            throw EmbeddingError.modelError(underlying: error)
        }
    }
}
```

---

### 2. Sources/SwiftTopicsApple/AppleNLProvider.swift (~30 LOC)

Convenience factory for Apple NL embeddings.

```swift
import SwiftTopics
import EmbedKit
import NaturalLanguage

/// Convenience factory for Apple NaturalLanguage embedding providers.
public enum AppleNLProvider {

    /// Creates an embedding provider using Apple's NLContextualEmbedding.
    ///
    /// Uses the newer contextual embedding API (iOS 17+/macOS 14+) which provides
    /// 512-dimensional embeddings with better semantic understanding than the
    /// older `NLEmbedding.sentenceEmbedding`.
    ///
    /// - Parameters:
    ///   - language: The language code (default: "en" for English).
    ///   - normalize: Whether to L2-normalize embeddings (default: true).
    /// - Returns: An embedding provider for SwiftTopics.
    /// - Throws: If the language is not supported.
    ///
    /// ## Supported Languages
    /// English (en), German (de), French (fr), Spanish (es), Italian (it),
    /// Portuguese (pt), Japanese (ja), Chinese (zh), Korean (ko), Russian (ru)
    ///
    /// ## Example
    /// ```swift
    /// let provider = try AppleNLProvider.contextual(language: "en")
    /// let model = TopicModel(configuration: .default)
    /// let result = try await model.fit(documents: docs, embeddingProvider: provider)
    /// ```
    public static func contextual(
        language: String = NLLanguage.english.rawValue,
        normalize: Bool = true
    ) throws -> some EmbeddingProvider {
        var config = EmbeddingConfiguration()
        config.normalizeOutput = normalize

        let nlModel = try AppleNLContextualModel(
            language: language,
            configuration: config
        )

        return EmbedKitAdapter(model: nlModel)
    }

    /// Checks if contextual embeddings are available for a language.
    ///
    /// - Parameter language: The language code to check.
    /// - Returns: True if embeddings are available.
    public static func isAvailable(language: String) -> Bool {
        let lang = NLLanguage(rawValue: language)
        return NLContextualEmbedding(language: lang) != nil
    }
}
```

---

### 3. Update Package.swift

Add EmbedKit dependency to SwiftTopicsApple target:

```swift
.target(
    name: "SwiftTopicsApple",
    dependencies: [
        "SwiftTopics",
        .product(name: "EmbedKit", package: "EmbedKit"),
    ]
),
```

And add the package dependency:

```swift
dependencies: [
    .package(url: "https://github.com/gifton/VectorAccelerate.git", from: "0.4.4"),
    .package(path: "../VSK/EmbedKit"),  // Local path or git URL
],
```

---

## What EmbedKit Provides (No Need to Reimplement)

| Feature | EmbedKit Implementation |
|---------|------------------------|
| NL Contextual Embeddings | `AppleNLContextualModel` |
| Asset downloading | `embedding.requestAssets()` |
| Mean pooling over tokens | Built-in |
| L2 normalization | `configuration.normalizeOutput` |
| Error handling | `EmbedKitError` types |
| Metrics/profiling | `metricsData`, `profiler` |
| Batch processing | `embedBatch(_:options:)` |
| Language support matrix | `NLContextualDimensions` |

---

## Comparison: Original vs Revised Approach

| Aspect | Original Plan | Revised (EmbedKit) |
|--------|--------------|-------------------|
| Lines of code | ~200 | ~80 |
| Duration | 2-3 days | 1 day |
| Asset management | Manual | Built-in |
| Error handling | Custom | Reuse EmbedKit |
| Batch optimization | Basic | Advanced (micro-batching) |
| Profiling | None | Built-in |
| Maintenance | Duplicate code | Single source of truth |

---

## Exit Criteria

- [ ] `EmbedKitAdapter` wraps any `EmbeddingModel` to `EmbeddingProvider`
- [ ] `AppleNLProvider.contextual()` creates working provider
- [ ] `swift build` succeeds for SwiftTopicsApple target
- [ ] Integration test: embed text → non-zero 512-dim vector

---

## Usage Example (Post-Implementation)

```swift
import SwiftTopics
import SwiftTopicsApple

// Create provider using Apple's built-in embeddings
let provider = try AppleNLProvider.contextual(language: "en")

// Load documents
let documents = [
    Document(id: "1", content: "Machine learning transforms data"),
    Document(id: "2", content: "Neural networks learn patterns"),
    Document(id: "3", content: "Stock market analysis trends"),
]

// Run topic modeling
let model = TopicModel(configuration: .default)
let result = try await model.fit(documents: documents, embeddingProvider: provider)

print("Found \(result.topics.count) topics")
```

---

## Dependencies

- **Requires**: Phase 0 (Core types, protocols)
- **EmbedKit**: Already exists at `/Users/goftin/dev/gsuite/VSK/EmbedKit`
- **SwiftTopicsApple target**: Already configured in Package.swift

---

★ Insight ─────────────────────────────────────
**Why adapter pattern over reimplementation:**

1. **DRY principle**: EmbedKit already solves NL embedding with production-quality code

2. **Future-proof**: When EmbedKit adds new models (ONNX, CoreML transformers), they automatically work with SwiftTopics through the same adapter

3. **VSK ecosystem alignment**: Both EmbedKit and SwiftTopics are part of VSK - natural integration point

4. **`NLContextualEmbedding` vs `NLEmbedding`**: EmbedKit uses the newer, better API (iOS 17+) that provides contextual token embeddings with mean pooling, rather than the older sentence embedding API
─────────────────────────────────────────────────

---

*Phase 9 Kickoff (Revised) - Created: January 2025*
