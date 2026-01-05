// AppleNLProvider.swift
// SwiftTopicsApple
//
// Convenience factory for Apple NL embedding providers

// Import specific types to avoid naming conflicts with the SwiftTopics enum
import protocol SwiftTopics.EmbeddingProvider

import EmbedKit
import NaturalLanguage

// MARK: - Apple NL Provider

/// Convenience factory for Apple NaturalLanguage embedding providers.
///
/// This provides a simple API for creating embedding providers using
/// Apple's built-in NLContextualEmbedding, which is available on
/// iOS 17+/macOS 14+ without any additional model downloads.
///
/// ## Features
/// - 512-dimensional contextual embeddings
/// - Mean pooling over token vectors
/// - Optional L2 normalization
/// - Automatic asset download when needed
///
/// ## Supported Languages
/// English (en), German (de), French (fr), Spanish (es), Italian (it),
/// Portuguese (pt), Japanese (ja), Chinese (zh), Korean (ko), Russian (ru)
///
/// ## Usage
/// ```swift
/// import SwiftTopics
/// import SwiftTopicsApple
///
/// let provider = try AppleNLProvider.contextual(language: "en")
///
/// let model = TopicModel(configuration: .default)
/// let result = try await model.fit(documents: docs, embeddingProvider: provider)
/// ```
public enum AppleNLProvider {

    // MARK: - Factory Methods

    /// Creates an embedding provider using Apple's NLContextualEmbedding.
    ///
    /// Uses the newer contextual embedding API (iOS 17+/macOS 14+) which provides
    /// 512-dimensional embeddings with better semantic understanding than the
    /// older `NLEmbedding.sentenceEmbedding`.
    ///
    /// - Parameters:
    ///   - language: The language code (default: "en" for English).
    ///   - normalize: Whether to L2-normalize embeddings (default: true).
    ///   - batchOptions: Options for batch processing (default: `.default`).
    /// - Returns: An embedding provider for SwiftTopics.
    /// - Throws: If the language is not supported.
    ///
    /// ## Example
    /// ```swift
    /// let provider = try AppleNLProvider.contextual(language: "en")
    /// let model = TopicModel(configuration: .default)
    /// let result = try await model.fit(documents: docs, embeddingProvider: provider)
    /// ```
    public static func contextual(
        language: String = NLLanguage.english.rawValue,
        normalize: Bool = true,
        batchOptions: BatchOptions = .default
    ) throws -> some EmbeddingProvider {
        var config = EmbeddingConfiguration()
        config = EmbeddingConfiguration(
            maxTokens: config.maxTokens,
            truncationStrategy: config.truncationStrategy,
            paddingStrategy: config.paddingStrategy,
            includeSpecialTokens: config.includeSpecialTokens,
            poolingStrategy: config.poolingStrategy,
            normalizeOutput: normalize,
            inferenceDevice: config.inferenceDevice,
            minElementsForGPU: config.minElementsForGPU
        )

        let nlModel = try AppleNLContextualModel(
            language: language,
            configuration: config
        )

        return EmbedKitAdapter(model: nlModel, batchOptions: batchOptions)
    }

    /// Creates a high-throughput embedding provider for batch processing.
    ///
    /// Optimized for processing large document corpora with larger batch
    /// sizes and dynamic batching enabled.
    ///
    /// - Parameters:
    ///   - language: The language code (default: "en").
    ///   - normalize: Whether to L2-normalize embeddings (default: true).
    /// - Returns: An embedding provider optimized for throughput.
    /// - Throws: If the language is not supported.
    public static func highThroughput(
        language: String = NLLanguage.english.rawValue,
        normalize: Bool = true
    ) throws -> some EmbeddingProvider {
        try contextual(
            language: language,
            normalize: normalize,
            batchOptions: .highThroughput
        )
    }

    // MARK: - Availability Checks

    /// Checks if contextual embeddings are available for a language.
    ///
    /// - Parameter language: The language code to check.
    /// - Returns: True if embeddings are available.
    public static func isAvailable(language: String) -> Bool {
        let lang = NLLanguage(rawValue: language)
        return NLContextualEmbedding(language: lang) != nil
    }

    /// Returns all supported language codes.
    ///
    /// Based on Apple's documented supported languages for NLContextualEmbedding.
    public static var supportedLanguages: [String] {
        [
            NLLanguage.english.rawValue,
            NLLanguage.german.rawValue,
            NLLanguage.french.rawValue,
            NLLanguage.spanish.rawValue,
            NLLanguage.italian.rawValue,
            NLLanguage.portuguese.rawValue,
            NLLanguage.japanese.rawValue,
            NLLanguage.simplifiedChinese.rawValue,
            NLLanguage.korean.rawValue,
            NLLanguage.russian.rawValue
        ]
    }

    /// The embedding dimension for NLContextualEmbedding (512 for all languages).
    public static let embeddingDimension: Int = 512
}
