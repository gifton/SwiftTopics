// Embedding.swift
// SwiftTopics
//
// Vector wrapper with dimension validation for embeddings

import Foundation
import VectorCore

// MARK: - Embedding

/// A dense vector representation of a document.
///
/// Embeddings are the numerical representation of documents used for similarity
/// computation and clustering. This type wraps a float vector with dimension
/// validation and provides efficient access patterns.
///
/// ## Dimensionality
/// All embeddings in a corpus must have the same dimensionality. Common dimensions:
/// - 384: Sentence transformers (e.g., all-MiniLM-L6-v2)
/// - 512: Apple NL sentence embeddings
/// - 768: BERT-base, RoBERTa-base
/// - 1024: BERT-large, RoBERTa-large
/// - 1536: OpenAI ada-002
///
/// ## Thread Safety
/// `Embedding` is `Sendable` and can be safely shared across concurrency domains.
///
/// ## Performance
/// The underlying storage uses a contiguous array for cache-friendly access
/// and compatibility with SIMD/GPU operations.
public struct Embedding: Sendable, Codable, Hashable {

    /// The vector components.
    public let vector: [Float]

    /// The dimensionality of this embedding.
    @inlinable
    public var dimension: Int {
        vector.count
    }

    /// Creates an embedding from a float array.
    ///
    /// - Parameter vector: The embedding components.
    /// - Precondition: Vector must not be empty.
    public init(vector: [Float]) {
        precondition(!vector.isEmpty, "Embedding vector cannot be empty")
        self.vector = vector
    }

    /// Creates an embedding from a `DynamicVector`.
    ///
    /// - Parameter dynamicVector: A VectorCore DynamicVector.
    public init(dynamicVector: DynamicVector) {
        self.vector = dynamicVector.toArray()
    }

    /// Creates a zero embedding of the specified dimension.
    ///
    /// - Parameter dimension: The number of dimensions.
    /// - Returns: An embedding with all zero components.
    public static func zeros(dimension: Int) -> Embedding {
        Embedding(vector: [Float](repeating: 0, count: dimension))
    }

    /// Creates a random embedding of the specified dimension.
    ///
    /// Useful for testing and initialization. Values are uniformly distributed
    /// in the range [-1, 1].
    ///
    /// - Parameters:
    ///   - dimension: The number of dimensions.
    ///   - seed: Optional random seed for reproducibility.
    /// - Returns: A random embedding.
    public static func random(dimension: Int, seed: UInt64? = nil) -> Embedding {
        var generator: RandomNumberGenerator
        if let seed = seed {
            generator = SeededRandomGenerator(seed: seed)
        } else {
            generator = SystemRandomNumberGenerator()
        }
        let values = (0..<dimension).map { _ in
            Float.random(in: -1...1, using: &generator)
        }
        return Embedding(vector: values)
    }

    /// Converts this embedding to a `DynamicVector`.
    ///
    /// - Returns: A VectorCore DynamicVector for use with VectorAccelerate kernels.
    public func toDynamicVector() -> DynamicVector {
        DynamicVector(vector)
    }
}

// MARK: - Embedding Collection Operations

extension Embedding {

    /// Computes the L2 (Euclidean) norm of this embedding.
    ///
    /// ||v||₂ = √(Σ vᵢ²)
    ///
    /// - Returns: The L2 norm.
    @inlinable
    public var l2Norm: Float {
        var sumSquares: Float = 0
        for value in vector {
            sumSquares += value * value
        }
        return sumSquares.squareRoot()
    }

    /// Returns a normalized copy of this embedding.
    ///
    /// v_normalized = v / ||v||₂
    ///
    /// - Returns: The normalized embedding, or the original if norm is zero.
    public func normalized() -> Embedding {
        let norm = l2Norm
        guard norm > Float.ulpOfOne else { return self }
        let normalizedVector = vector.map { $0 / norm }
        return Embedding(vector: normalizedVector)
    }

    /// Computes the dot product with another embedding.
    ///
    /// v · w = Σ vᵢ × wᵢ
    ///
    /// - Parameter other: The other embedding.
    /// - Returns: The dot product.
    /// - Precondition: Both embeddings must have the same dimension.
    @inlinable
    public func dot(_ other: Embedding) -> Float {
        precondition(dimension == other.dimension, "Embeddings must have same dimension")
        var result: Float = 0
        for i in 0..<dimension {
            result += vector[i] * other.vector[i]
        }
        return result
    }

    /// Computes the cosine similarity with another embedding.
    ///
    /// cos(θ) = (v · w) / (||v||₂ × ||w||₂)
    ///
    /// - Parameter other: The other embedding.
    /// - Returns: The cosine similarity in range [-1, 1].
    /// - Precondition: Both embeddings must have the same dimension.
    public func cosineSimilarity(_ other: Embedding) -> Float {
        let dotProduct = dot(other)
        let normProduct = l2Norm * other.l2Norm
        guard normProduct > Float.ulpOfOne else { return 0 }
        return dotProduct / normProduct
    }

    /// Computes the Euclidean distance to another embedding.
    ///
    /// d(v, w) = ||v - w||₂ = √(Σ (vᵢ - wᵢ)²)
    ///
    /// - Parameter other: The other embedding.
    /// - Returns: The Euclidean distance.
    /// - Precondition: Both embeddings must have the same dimension.
    public func euclideanDistance(_ other: Embedding) -> Float {
        precondition(dimension == other.dimension, "Embeddings must have same dimension")
        var sumSquares: Float = 0
        for i in 0..<dimension {
            let diff = vector[i] - other.vector[i]
            sumSquares += diff * diff
        }
        return sumSquares.squareRoot()
    }
}

// MARK: - Document Embedding

/// An embedding associated with a specific document.
///
/// Links a document ID to its vector representation, enabling downstream
/// operations to track which embedding belongs to which document.
public struct DocumentEmbedding: Sendable, Codable, Hashable, Identifiable {

    /// The ID of the associated document.
    public let documentID: DocumentID

    /// The embedding vector.
    public let embedding: Embedding

    public var id: DocumentID { documentID }

    /// Creates a document embedding.
    ///
    /// - Parameters:
    ///   - documentID: The ID of the associated document.
    ///   - embedding: The embedding vector.
    public init(documentID: DocumentID, embedding: Embedding) {
        self.documentID = documentID
        self.embedding = embedding
    }

    /// Creates a document embedding from raw values.
    ///
    /// - Parameters:
    ///   - documentID: The ID of the associated document.
    ///   - vector: The embedding vector components.
    public init(documentID: DocumentID, vector: [Float]) {
        self.documentID = documentID
        self.embedding = Embedding(vector: vector)
    }
}

// MARK: - Embedding Matrix

/// A collection of embeddings stored as a matrix for efficient batch operations.
///
/// The matrix is stored in row-major order where each row is one embedding.
/// This format is compatible with VectorAccelerate kernels for GPU-accelerated
/// distance computations.
public struct EmbeddingMatrix: Sendable {

    /// The underlying float storage in row-major order.
    public let storage: [Float]

    /// The number of embeddings (rows).
    public let count: Int

    /// The dimension of each embedding (columns).
    public let dimension: Int

    /// Creates an embedding matrix from a collection of embeddings.
    ///
    /// - Parameter embeddings: The embeddings to store.
    /// - Precondition: All embeddings must have the same dimension.
    public init(embeddings: [Embedding]) {
        precondition(!embeddings.isEmpty, "Cannot create empty embedding matrix")

        let dimension = embeddings[0].dimension
        precondition(
            embeddings.allSatisfy { $0.dimension == dimension },
            "All embeddings must have the same dimension"
        )

        self.count = embeddings.count
        self.dimension = dimension

        // Flatten to row-major storage
        var storage = [Float]()
        storage.reserveCapacity(count * dimension)
        for embedding in embeddings {
            storage.append(contentsOf: embedding.vector)
        }
        self.storage = storage
    }

    /// Creates an embedding matrix from raw storage.
    ///
    /// - Parameters:
    ///   - storage: Row-major float storage.
    ///   - count: Number of embeddings (rows).
    ///   - dimension: Dimension of each embedding (columns).
    public init(storage: [Float], count: Int, dimension: Int) {
        precondition(storage.count == count * dimension, "Storage size mismatch")
        self.storage = storage
        self.count = count
        self.dimension = dimension
    }

    /// Accesses the embedding at the given index.
    ///
    /// - Parameter index: The embedding index.
    /// - Returns: The embedding at that index.
    public subscript(index: Int) -> Embedding {
        precondition(index >= 0 && index < count, "Index out of bounds")
        let start = index * dimension
        let end = start + dimension
        return Embedding(vector: Array(storage[start..<end]))
    }

    /// Returns all embeddings as an array.
    public var embeddings: [Embedding] {
        (0..<count).map { self[$0] }
    }
}

// MARK: - Seeded Random Generator

/// A simple seeded random number generator for reproducible random embeddings.
private struct SeededRandomGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // xorshift64* algorithm
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return state &* 0x2545F4914F6CDD1D
    }
}
