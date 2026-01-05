// Vocabulary.swift
// SwiftTopics
//
// Vocabulary building with term and document frequencies

import Foundation

// MARK: - Vocabulary Configuration

/// Configuration for vocabulary building.
public struct VocabularyConfiguration: Sendable, Codable, Hashable {

    /// Minimum document frequency for a term to be included.
    ///
    /// Terms appearing in fewer than this many documents are excluded.
    public let minDocumentFrequency: Int

    /// Maximum document frequency ratio for a term (0-1).
    ///
    /// Terms appearing in more than this fraction of documents are excluded.
    /// Helps filter overly common words that weren't in the stop list.
    public let maxDocumentFrequencyRatio: Float

    /// Maximum vocabulary size (nil = no limit).
    ///
    /// If set, only the most frequent terms are kept.
    public let maxVocabularySize: Int?

    /// Creates a vocabulary configuration.
    ///
    /// - Parameters:
    ///   - minDocumentFrequency: Minimum document frequency (default: 1).
    ///   - maxDocumentFrequencyRatio: Maximum document frequency ratio (default: 0.95).
    ///   - maxVocabularySize: Maximum vocabulary size (default: nil).
    public init(
        minDocumentFrequency: Int = 1,
        maxDocumentFrequencyRatio: Float = 0.95,
        maxVocabularySize: Int? = nil
    ) {
        precondition(minDocumentFrequency >= 1, "minDocumentFrequency must be at least 1")
        precondition(maxDocumentFrequencyRatio > 0 && maxDocumentFrequencyRatio <= 1,
                     "maxDocumentFrequencyRatio must be in (0, 1]")

        self.minDocumentFrequency = minDocumentFrequency
        self.maxDocumentFrequencyRatio = maxDocumentFrequencyRatio
        self.maxVocabularySize = maxVocabularySize
    }

    /// Default configuration.
    public static let `default` = VocabularyConfiguration()

    /// Configuration for small corpora.
    public static let small = VocabularyConfiguration(
        minDocumentFrequency: 1,
        maxDocumentFrequencyRatio: 0.9
    )

    /// Configuration for large corpora.
    public static let large = VocabularyConfiguration(
        minDocumentFrequency: 5,
        maxDocumentFrequencyRatio: 0.95,
        maxVocabularySize: 50000
    )
}

// MARK: - Term Statistics

/// Statistics for a term in the vocabulary.
public struct TermStatistics: Sendable, Hashable {

    /// The term string.
    public let term: String

    /// Index in the vocabulary.
    public let index: Int

    /// Number of documents containing this term.
    public let documentFrequency: Int

    /// Total occurrences across all documents.
    public let totalFrequency: Int

    /// Document frequency ratio (df / total documents).
    public let documentFrequencyRatio: Float

    /// Creates term statistics.
    public init(
        term: String,
        index: Int,
        documentFrequency: Int,
        totalFrequency: Int,
        documentFrequencyRatio: Float
    ) {
        self.term = term
        self.index = index
        self.documentFrequency = documentFrequency
        self.totalFrequency = totalFrequency
        self.documentFrequencyRatio = documentFrequencyRatio
    }
}

// MARK: - Vocabulary

/// A vocabulary mapping terms to indices with frequency statistics.
///
/// ## Building
/// Use `VocabularyBuilder` to construct a vocabulary from tokenized documents:
///
/// ```swift
/// let builder = VocabularyBuilder(configuration: .default)
/// let vocab = builder.build(from: tokenizedDocs)
/// ```
///
/// ## Thread Safety
/// `Vocabulary` is immutable and safe to use from any thread.
public struct Vocabulary: Sendable {

    /// Configuration used to build this vocabulary.
    public let configuration: VocabularyConfiguration

    /// Number of documents used to build the vocabulary.
    public let documentCount: Int

    /// Term to index mapping.
    private let termToIndex: [String: Int]

    /// Index to term mapping.
    private let indexToTerm: [String]

    /// Document frequency per term (indexed by term index).
    private let documentFrequencies: [Int]

    /// Total frequency per term (indexed by term index).
    private let totalFrequencies: [Int]

    /// Creates a vocabulary.
    ///
    /// - Parameters:
    ///   - configuration: Configuration used.
    ///   - documentCount: Number of documents.
    ///   - termToIndex: Term to index mapping.
    ///   - indexToTerm: Index to term mapping.
    ///   - documentFrequencies: Document frequency per term.
    ///   - totalFrequencies: Total frequency per term.
    internal init(
        configuration: VocabularyConfiguration,
        documentCount: Int,
        termToIndex: [String: Int],
        indexToTerm: [String],
        documentFrequencies: [Int],
        totalFrequencies: [Int]
    ) {
        self.configuration = configuration
        self.documentCount = documentCount
        self.termToIndex = termToIndex
        self.indexToTerm = indexToTerm
        self.documentFrequencies = documentFrequencies
        self.totalFrequencies = totalFrequencies
    }

    // MARK: - Properties

    /// Number of unique terms in the vocabulary.
    public var size: Int {
        indexToTerm.count
    }

    /// Whether the vocabulary is empty.
    public var isEmpty: Bool {
        indexToTerm.isEmpty
    }

    /// All terms in the vocabulary (in index order).
    public var terms: [String] {
        indexToTerm
    }

    // MARK: - Term Lookup

    /// Gets the index for a term.
    ///
    /// - Parameter term: The term to look up.
    /// - Returns: The term index, or nil if not in vocabulary.
    public func index(for term: String) -> Int? {
        termToIndex[term]
    }

    /// Gets the term for an index.
    ///
    /// - Parameter index: The term index.
    /// - Returns: The term string, or nil if out of bounds.
    public func term(at index: Int) -> String? {
        guard index >= 0 && index < indexToTerm.count else { return nil }
        return indexToTerm[index]
    }

    /// Whether the vocabulary contains a term.
    ///
    /// - Parameter term: The term to check.
    /// - Returns: True if the term is in the vocabulary.
    public func contains(_ term: String) -> Bool {
        termToIndex[term] != nil
    }

    // MARK: - Frequency Lookup

    /// Gets the document frequency for a term.
    ///
    /// - Parameter term: The term.
    /// - Returns: Number of documents containing the term, or 0 if not found.
    public func documentFrequency(for term: String) -> Int {
        guard let index = termToIndex[term] else { return 0 }
        return documentFrequencies[index]
    }

    /// Gets the total frequency for a term.
    ///
    /// - Parameter term: The term.
    /// - Returns: Total occurrences of the term, or 0 if not found.
    public func totalFrequency(for term: String) -> Int {
        guard let index = termToIndex[term] else { return 0 }
        return totalFrequencies[index]
    }

    /// Gets the document frequency ratio for a term.
    ///
    /// - Parameter term: The term.
    /// - Returns: Fraction of documents containing the term, or 0 if not found.
    public func documentFrequencyRatio(for term: String) -> Float {
        guard documentCount > 0 else { return 0 }
        return Float(documentFrequency(for: term)) / Float(documentCount)
    }

    /// Gets statistics for a term.
    ///
    /// - Parameter term: The term.
    /// - Returns: Term statistics, or nil if not in vocabulary.
    public func statistics(for term: String) -> TermStatistics? {
        guard let index = termToIndex[term] else { return nil }
        return TermStatistics(
            term: term,
            index: index,
            documentFrequency: documentFrequencies[index],
            totalFrequency: totalFrequencies[index],
            documentFrequencyRatio: Float(documentFrequencies[index]) / Float(max(1, documentCount))
        )
    }

    // MARK: - Vectorization

    /// Converts tokenized text to a term frequency vector.
    ///
    /// - Parameter tokens: Array of tokens.
    /// - Returns: Array of term frequencies indexed by vocabulary index.
    public func termFrequencyVector(_ tokens: [String]) -> [Int] {
        var vector = [Int](repeating: 0, count: size)
        for token in tokens {
            if let index = termToIndex[token] {
                vector[index] += 1
            }
        }
        return vector
    }

    /// Converts tokenized text to a term frequency dictionary.
    ///
    /// - Parameter tokens: Array of tokens.
    /// - Returns: Dictionary mapping term indices to frequencies.
    public func termFrequencyDict(_ tokens: [String]) -> [Int: Int] {
        var dict = [Int: Int]()
        for token in tokens {
            if let index = termToIndex[token] {
                dict[index, default: 0] += 1
            }
        }
        return dict
    }

    /// Gets terms sorted by document frequency (descending).
    ///
    /// - Parameter limit: Maximum number of terms to return (nil = all).
    /// - Returns: Terms sorted by frequency.
    public func termsByFrequency(limit: Int? = nil) -> [String] {
        let sorted = indexToTerm.indices.sorted { documentFrequencies[$0] > documentFrequencies[$1] }
        let limited = limit.map { sorted.prefix($0) } ?? sorted[...]
        return limited.map { indexToTerm[$0] }
    }
}

// MARK: - Vocabulary Builder

/// Builds vocabularies from tokenized documents.
///
/// ## Usage
/// ```swift
/// let builder = VocabularyBuilder(configuration: .default)
/// let vocab = builder.build(from: tokenizedDocs)
/// ```
public struct VocabularyBuilder: Sendable {

    /// Configuration for vocabulary building.
    public let configuration: VocabularyConfiguration

    /// Creates a vocabulary builder.
    ///
    /// - Parameter configuration: Configuration for vocabulary building.
    public init(configuration: VocabularyConfiguration = .default) {
        self.configuration = configuration
    }

    /// Builds a vocabulary from tokenized documents.
    ///
    /// - Parameter tokenizedDocuments: Array of token arrays (one per document).
    /// - Returns: The built vocabulary.
    public func build(from tokenizedDocuments: [[String]]) -> Vocabulary {
        let documentCount = tokenizedDocuments.count
        guard documentCount > 0 else {
            return Vocabulary(
                configuration: configuration,
                documentCount: 0,
                termToIndex: [:],
                indexToTerm: [],
                documentFrequencies: [],
                totalFrequencies: []
            )
        }

        // Step 1: Count document frequencies and total frequencies
        var documentFrequency: [String: Int] = [:]
        var totalFrequency: [String: Int] = [:]

        for tokens in tokenizedDocuments {
            let uniqueTokens = Set(tokens)
            for token in uniqueTokens {
                documentFrequency[token, default: 0] += 1
            }
            for token in tokens {
                totalFrequency[token, default: 0] += 1
            }
        }

        // Step 2: Filter by document frequency
        // Ensure maxDf is at least 1 to avoid filtering everything with small corpora
        let maxDf = max(1, Int(Float(documentCount) * configuration.maxDocumentFrequencyRatio))
        let minDf = configuration.minDocumentFrequency

        let validTerms = documentFrequency.filter { term, df in
            df >= minDf && df <= maxDf
        }

        // Step 3: Limit vocabulary size if configured
        var sortedTerms: [String]
        if let maxSize = configuration.maxVocabularySize, validTerms.count > maxSize {
            // Keep the most frequent terms
            sortedTerms = validTerms.keys.sorted {
                totalFrequency[$0, default: 0] > totalFrequency[$1, default: 0]
            }
            sortedTerms = Array(sortedTerms.prefix(maxSize))
        } else {
            // Sort alphabetically for deterministic ordering
            sortedTerms = validTerms.keys.sorted()
        }

        // Step 4: Build index mappings
        var termToIndex: [String: Int] = [:]
        var indexToTerm: [String] = []
        var dfArray: [Int] = []
        var tfArray: [Int] = []

        indexToTerm.reserveCapacity(sortedTerms.count)
        dfArray.reserveCapacity(sortedTerms.count)
        tfArray.reserveCapacity(sortedTerms.count)

        for (index, term) in sortedTerms.enumerated() {
            termToIndex[term] = index
            indexToTerm.append(term)
            dfArray.append(documentFrequency[term, default: 0])
            tfArray.append(totalFrequency[term, default: 0])
        }

        return Vocabulary(
            configuration: configuration,
            documentCount: documentCount,
            termToIndex: termToIndex,
            indexToTerm: indexToTerm,
            documentFrequencies: dfArray,
            totalFrequencies: tfArray
        )
    }

    /// Builds a vocabulary from documents using a tokenizer.
    ///
    /// - Parameters:
    ///   - documents: Documents to process.
    ///   - tokenizer: Tokenizer to use.
    /// - Returns: The built vocabulary.
    public func build(from documents: [Document], using tokenizer: Tokenizer) -> Vocabulary {
        let tokenizedDocs = tokenizer.tokenize(documents: documents)
        return build(from: tokenizedDocs)
    }
}

// MARK: - Convenience Extensions

extension Vocabulary {

    /// Creates an empty vocabulary.
    public static var empty: Vocabulary {
        Vocabulary(
            configuration: .default,
            documentCount: 0,
            termToIndex: [:],
            indexToTerm: [],
            documentFrequencies: [],
            totalFrequencies: []
        )
    }
}

extension Array where Element == [String] {

    /// Builds a vocabulary from tokenized documents.
    ///
    /// - Parameter configuration: Vocabulary configuration.
    /// - Returns: The built vocabulary.
    public func buildVocabulary(
        configuration: VocabularyConfiguration = .default
    ) -> Vocabulary {
        VocabularyBuilder(configuration: configuration).build(from: self)
    }
}
