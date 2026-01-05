// EmbedKitAdapter.swift
// SwiftTopicsApple
//
// Adapter bridging EmbedKit.EmbeddingModel to SwiftTopics.EmbeddingProvider

// Import specific types to avoid naming conflicts
// SwiftTopics module has an enum also named `SwiftTopics`, causing shadowing
import struct SwiftTopics.Embedding
import protocol SwiftTopics.EmbeddingProvider
import enum SwiftTopics.EmbeddingError
import struct SwiftTopics.Document
import struct SwiftTopics.DocumentEmbedding

import EmbedKit

// MARK: - EmbedKit Adapter

/// Adapts an EmbedKit `EmbeddingModel` to SwiftTopics' `EmbeddingProvider` protocol.
///
/// This enables using any EmbedKit model (CoreML, Apple NL, ONNX) with SwiftTopics
/// topic modeling pipelines.
///
/// ## Architecture
/// EmbedKit models are actors for thread safety. This adapter wraps the actor
/// and forwards calls through its async interface, converting between the two
/// embedding types.
///
/// ## Type Conversion
/// - `EmbedKit.Embedding` → `SwiftTopics.Embedding`: Extracts vector, discards metadata
/// - This is intentional: SwiftTopics doesn't need EmbedKit's detailed metrics
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

    // MARK: - Properties

    /// The wrapped EmbedKit model.
    private let model: Model

    /// Batch options for embedBatch calls.
    private let batchOptions: BatchOptions

    // MARK: - EmbeddingProvider Protocol

    /// The dimension of embeddings produced by this provider.
    ///
    /// Forwards to `EmbedKit.EmbeddingModel.dimensions`.
    public var dimension: Int {
        model.dimensions
    }

    // MARK: - Initialization

    /// Creates an adapter wrapping an EmbedKit embedding model.
    ///
    /// - Parameters:
    ///   - model: The EmbedKit model to wrap.
    ///   - batchOptions: Options for batch embedding operations (default: `.default`).
    public init(model: Model, batchOptions: BatchOptions = .default) {
        self.model = model
        self.batchOptions = batchOptions
    }

    // MARK: - Embedding Methods

    /// Embeds a single text string.
    ///
    /// Forwards to the wrapped EmbedKit model and converts the result
    /// to SwiftTopics' embedding format.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: The embedding vector.
    /// - Throws: `EmbeddingError.modelError` if embedding fails.
    public func embed(_ text: String) async throws -> Embedding {
        do {
            let ekEmbedding = try await model.embed(text)
            return Embedding(vector: ekEmbedding.vector)
        } catch {
            throw EmbeddingError.modelError(underlying: error)
        }
    }

    /// Embeds multiple texts in a batch.
    ///
    /// Uses EmbedKit's optimized batch processing with the configured
    /// batch options.
    ///
    /// - Parameter texts: The texts to embed.
    /// - Returns: Embeddings in the same order as input texts.
    /// - Throws: `EmbeddingError.modelError` if embedding fails.
    public func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        guard !texts.isEmpty else { return [] }

        do {
            let ekEmbeddings = try await model.embedBatch(texts, options: batchOptions)
            return ekEmbeddings.map { Embedding(vector: $0.vector) }
        } catch {
            throw EmbeddingError.modelError(underlying: error)
        }
    }
}

// MARK: - Convenience Extensions

extension EmbedKitAdapter {

    /// Creates an adapter with high-throughput batch options.
    ///
    /// Uses larger batch sizes and dynamic batching for maximum throughput.
    /// Ideal for processing large document corpora.
    ///
    /// - Parameter model: The EmbedKit model to wrap.
    /// - Returns: An adapter configured for high throughput.
    public static func highThroughput(model: Model) -> EmbedKitAdapter<Model> {
        EmbedKitAdapter(model: model, batchOptions: .highThroughput)
    }

    /// Creates an adapter with low-latency batch options.
    ///
    /// Uses smaller batch sizes for faster response times.
    /// Ideal for real-time applications.
    ///
    /// - Parameter model: The EmbedKit model to wrap.
    /// - Returns: An adapter configured for low latency.
    public static func lowLatency(model: Model) -> EmbedKitAdapter<Model> {
        EmbedKitAdapter(model: model, batchOptions: .lowLatency)
    }
}
