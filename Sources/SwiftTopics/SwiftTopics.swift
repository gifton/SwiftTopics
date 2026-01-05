// SwiftTopics
// High-fidelity, on-device topic extraction with GPU acceleration
//
// Powered by VectorAccelerate for Metal 4 GPU-accelerated operations

import Foundation
import VectorAccelerate
import VectorCore

// MARK: - Library Version

/// SwiftTopics library version
public enum SwiftTopics {
    /// Current library version (semantic versioning)
    public static let version = "0.1.0-beta.1"

    /// Whether this is a beta/pre-release version
    public static let isBeta = true

    /// Minimum platform requirements
    public static let platformRequirements = """
        iOS 26.0+
        macOS 26.0+
        visionOS 26.0+
        """
}

// MARK: - Re-exports

// Re-export VectorAccelerate types that consumers may need
@_exported import struct VectorCore.DynamicVector

// MARK: - Module Documentation

/// # SwiftTopics
///
/// A pure-Swift library for high-fidelity, on-device topic extraction.
///
/// ## Overview
///
/// SwiftTopics implements a production-grade topic modeling pipeline inspired by BERTopic,
/// optimized for Apple platforms with GPU acceleration via VectorAccelerate.
///
/// ## Architecture
///
/// The pipeline consists of four main stages:
/// 1. **Embedding** - Convert documents to dense vectors (consumer-provided)
/// 2. **Reduction** - Compress high-dim vectors to low-dim (PCA/UMAP)
/// 3. **Clustering** - Group similar vectors into topics (HDBSCAN)
/// 4. **Representation** - Extract keywords for each topic (c-TF-IDF)
///
/// ## GPU Acceleration
///
/// Heavy operations use VectorAccelerate's Metal 4 kernels:
/// - `L2DistanceKernel` - Pairwise distances
/// - `FusedL2TopKKernel` - k-NN queries
/// - `MatrixMultiplyKernel` - Covariance, PCA projection
/// - `StatisticsKernel` - Mean centering
///
/// ## Example Usage
///
/// ```swift
/// // Create topic model with default configuration
/// let model = TopicModel(configuration: .default)
///
/// // Provide embeddings (from your embedding service)
/// let embeddings: [[Float]] = await embeddingService.embed(documents)
///
/// // Fit and transform
/// let result = try await model.fitTransform(
///     documents: documents,
///     embeddings: embeddings
/// )
///
/// // Access topics
/// for topic in result.topics {
///     print("Topic \(topic.id): \(topic.keywords.map(\.term).joined(separator: ", "))")
/// }
/// ```
///
/// ## See Also
///
/// - ``TopicModel`` - Main orchestrator
/// - ``HDBSCANConfiguration`` - Clustering parameters
/// - ``TopicModelResult`` - Pipeline output
