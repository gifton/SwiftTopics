// EmbeddingProvider.swift
// SwiftTopics
//
// Protocol for obtaining document embeddings from various sources

import Foundation

// MARK: - Embedding Provider Protocol

/// A source of document embeddings.
///
/// SwiftTopics is embedding-agnostic: it works with any source of dense vectors.
/// Implement this protocol to connect your embedding model or pre-computed vectors.
///
/// ## Common Implementations
/// - `PrecomputedEmbeddingProvider`: Use pre-computed vectors from your database
/// - `AppleNLEmbeddingProvider`: Use Apple's NaturalLanguage framework (in SwiftTopicsApple)
/// - Custom: Wrap your own embedding service
///
/// ## Requirements
/// - **Determinism**: Same text must always produce the same embedding
/// - **Consistent Dimension**: All embeddings must have the same dimension
/// - **Thread Safety**: Implementation must be safe to call from any thread
///
/// ## Example Implementation
/// ```swift
/// struct MyEmbeddingProvider: EmbeddingProvider {
///     let dimension = 384
///
///     func embed(_ text: String) async throws -> Embedding {
///         // Call your embedding model
///         let vector = try await myModel.encode(text)
///         return Embedding(vector: vector)
///     }
///
///     func embedBatch(_ texts: [String]) async throws -> [Embedding] {
///         // Batch encode for efficiency
///         let vectors = try await myModel.encodeBatch(texts)
///         return vectors.map { Embedding(vector: $0) }
///     }
/// }
/// ```
public protocol EmbeddingProvider: Sendable {

    /// The dimension of embeddings produced by this provider.
    ///
    /// All embeddings from this provider must have exactly this many dimensions.
    var dimension: Int { get }

    /// Embeds a single text string.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: The embedding vector.
    /// - Throws: `EmbeddingError` if embedding fails.
    func embed(_ text: String) async throws -> Embedding

    /// Embeds multiple texts in a batch.
    ///
    /// Default implementation calls `embed(_:)` for each text sequentially.
    /// Override for more efficient batch processing.
    ///
    /// - Parameter texts: The texts to embed.
    /// - Returns: Embeddings in the same order as input texts.
    /// - Throws: `EmbeddingError` if embedding fails.
    func embedBatch(_ texts: [String]) async throws -> [Embedding]

    /// Embeds documents, returning document embeddings.
    ///
    /// Default implementation uses `embedBatch(_:)` on document contents.
    ///
    /// - Parameter documents: The documents to embed.
    /// - Returns: Document embeddings in the same order.
    /// - Throws: `EmbeddingError` if embedding fails.
    func embed(documents: [Document]) async throws -> [DocumentEmbedding]
}

// MARK: - Default Implementations

extension EmbeddingProvider {

    /// Default batch implementation: embeds texts sequentially.
    public func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        var embeddings: [Embedding] = []
        embeddings.reserveCapacity(texts.count)
        for text in texts {
            let embedding = try await embed(text)
            embeddings.append(embedding)
        }
        return embeddings
    }

    /// Default document embedding implementation.
    public func embed(documents: [Document]) async throws -> [DocumentEmbedding] {
        let texts = documents.map(\.content)
        let embeddings = try await embedBatch(texts)

        return zip(documents, embeddings).map { document, embedding in
            DocumentEmbedding(documentID: document.id, embedding: embedding)
        }
    }
}

// MARK: - Embedding Error

/// Errors that can occur during embedding generation.
public enum EmbeddingError: Error, Sendable {

    /// The text is empty and cannot be embedded.
    case emptyText

    /// The text exceeds the maximum length for this provider.
    case textTooLong(maxLength: Int)

    /// The embedding dimension doesn't match expected.
    case dimensionMismatch(expected: Int, actual: Int)

    /// The provider is not available (e.g., model not loaded).
    case providerUnavailable(reason: String)

    /// An underlying model error occurred.
    case modelError(underlying: Error)

    /// A batch operation partially failed.
    case partialFailure(succeeded: Int, failed: Int, firstError: Error)

    /// Unknown error.
    case unknown(String)
}

extension EmbeddingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyText:
            return "Cannot embed empty text"
        case .textTooLong(let maxLength):
            return "Text exceeds maximum length of \(maxLength) characters"
        case .dimensionMismatch(let expected, let actual):
            return "Embedding dimension mismatch: expected \(expected), got \(actual)"
        case .providerUnavailable(let reason):
            return "Embedding provider unavailable: \(reason)"
        case .modelError(let underlying):
            return "Embedding model error: \(underlying.localizedDescription)"
        case .partialFailure(let succeeded, let failed, let firstError):
            return "Batch embedding partially failed: \(succeeded) succeeded, \(failed) failed. First error: \(firstError.localizedDescription)"
        case .unknown(let message):
            return "Embedding error: \(message)"
        }
    }
}

// MARK: - Precomputed Embedding Provider

/// An embedding provider that uses pre-computed vectors.
///
/// Use this when you have embeddings stored in a database or file,
/// and don't need to compute them on-the-fly.
///
/// ## Example Usage
/// ```swift
/// let provider = PrecomputedEmbeddingProvider(
///     embeddings: [
///         "document1": Embedding(vector: [...]),
///         "document2": Embedding(vector: [...]),
///     ]
/// )
/// ```
public struct PrecomputedEmbeddingProvider: EmbeddingProvider {

    private let embeddings: [String: Embedding]

    public let dimension: Int

    /// Creates a precomputed embedding provider.
    ///
    /// - Parameter embeddings: Dictionary mapping text or IDs to embeddings.
    /// - Precondition: All embeddings must have the same dimension.
    public init(embeddings: [String: Embedding]) {
        precondition(!embeddings.isEmpty, "Must provide at least one embedding")

        self.embeddings = embeddings
        self.dimension = embeddings.values.first!.dimension

        // Validate all dimensions match
        for embedding in embeddings.values {
            precondition(
                embedding.dimension == dimension,
                "All embeddings must have the same dimension"
            )
        }
    }

    /// Creates a precomputed embedding provider from raw vectors.
    ///
    /// - Parameter vectors: Dictionary mapping text or IDs to float arrays.
    public init(vectors: [String: [Float]]) {
        let embeddings = vectors.mapValues { Embedding(vector: $0) }
        self.init(embeddings: embeddings)
    }

    public func embed(_ text: String) async throws -> Embedding {
        guard let embedding = embeddings[text] else {
            throw EmbeddingError.unknown("No precomputed embedding for text: \(text.prefix(50))...")
        }
        return embedding
    }

    /// Checks if an embedding exists for the given text.
    public func hasEmbedding(for text: String) -> Bool {
        embeddings[text] != nil
    }

    /// All available keys.
    public var availableKeys: [String] {
        Array(embeddings.keys)
    }
}

// MARK: - Passthrough Embedding Provider

/// An embedding provider that passes through pre-computed embeddings by index.
///
/// Useful when you have embeddings computed externally and want to use them
/// with the topic model without a lookup step.
public struct PassthroughEmbeddingProvider: EmbeddingProvider {

    private let embeddingList: [Embedding]
    private var currentIndex: Int = 0

    public let dimension: Int

    /// Creates a passthrough provider.
    ///
    /// - Parameter embeddings: List of embeddings to return in order.
    public init(embeddings: [Embedding]) {
        precondition(!embeddings.isEmpty, "Must provide at least one embedding")

        self.embeddingList = embeddings
        self.dimension = embeddings[0].dimension
    }

    public func embed(_ text: String) async throws -> Embedding {
        // This is stateless - can't track index across async calls
        // Use embedBatch instead for proper ordering
        throw EmbeddingError.unknown("Use embedBatch for passthrough provider")
    }

    public func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        guard texts.count <= embeddingList.count else {
            throw EmbeddingError.unknown(
                "Not enough precomputed embeddings: have \(embeddingList.count), need \(texts.count)"
            )
        }
        return Array(embeddingList.prefix(texts.count))
    }
}

// MARK: - Caching Embedding Provider

/// An embedding provider that caches results from an underlying provider.
///
/// Useful for avoiding redundant embedding computations when the same
/// texts appear multiple times.
public actor CachingEmbeddingProvider<Base: EmbeddingProvider>: EmbeddingProvider {

    private let base: Base
    private var cache: [String: Embedding] = [:]

    public nonisolated var dimension: Int {
        base.dimension
    }

    /// Creates a caching wrapper around another provider.
    ///
    /// - Parameter base: The underlying embedding provider.
    public init(base: Base) {
        self.base = base
    }

    public func embed(_ text: String) async throws -> Embedding {
        if let cached = cache[text] {
            return cached
        }
        let embedding = try await base.embed(text)
        cache[text] = embedding
        return embedding
    }

    public func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        var results: [Embedding] = []
        results.reserveCapacity(texts.count)

        var uncachedTexts: [(index: Int, text: String)] = []

        // Check cache first
        for (index, text) in texts.enumerated() {
            if let cached = cache[text] {
                results.append(cached)
            } else {
                uncachedTexts.append((index, text))
                results.append(Embedding.zeros(dimension: dimension)) // Placeholder
            }
        }

        // Batch embed uncached texts
        if !uncachedTexts.isEmpty {
            let newEmbeddings = try await base.embedBatch(uncachedTexts.map(\.text))

            for (i, (index, text)) in uncachedTexts.enumerated() {
                let embedding = newEmbeddings[i]
                cache[text] = embedding
                results[index] = embedding
            }
        }

        return results
    }

    /// Clears the cache.
    public func clearCache() {
        cache.removeAll()
    }

    /// The number of cached embeddings.
    public var cacheSize: Int {
        cache.count
    }
}
