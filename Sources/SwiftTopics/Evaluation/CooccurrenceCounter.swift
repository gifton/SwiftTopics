// CooccurrenceCounter.swift
// SwiftTopics
//
// Word pair co-occurrence counting for coherence evaluation

import Foundation

// MARK: - Co-occurrence Mode

/// Mode for counting word co-occurrences.
public enum CooccurrenceMode: Sendable, Codable, Hashable {

    /// Sliding window mode: count pairs within a fixed-size window.
    ///
    /// - Parameter size: Window size (number of tokens).
    case slidingWindow(size: Int)

    /// Document mode: count pairs that appear in the same document (boolean).
    case document

    /// Default mode: sliding window of 10 tokens.
    public static let `default`: CooccurrenceMode = .slidingWindow(size: 10)
}

// MARK: - Word Pair

/// A pair of words for co-occurrence counting.
///
/// Word pairs are stored with the lexicographically smaller word first
/// to ensure symmetric counting: P(w1, w2) = P(w2, w1).
public struct WordPair: Sendable, Hashable {

    /// First word (lexicographically smaller).
    public let word1: String

    /// Second word (lexicographically larger).
    public let word2: String

    /// Creates a word pair, ordering words lexicographically.
    ///
    /// - Parameters:
    ///   - a: First word.
    ///   - b: Second word.
    public init(_ a: String, _ b: String) {
        if a <= b {
            self.word1 = a
            self.word2 = b
        } else {
            self.word1 = b
            self.word2 = a
        }
    }

    /// Whether this pair contains two different words.
    public var isValid: Bool {
        word1 != word2
    }
}

// MARK: - Co-occurrence Counts

/// Counts of word and word-pair occurrences in a corpus.
///
/// This structure holds the raw counts needed for NPMI calculation:
/// - Individual word frequencies
/// - Word pair co-occurrence frequencies
/// - Total count of windows/documents
///
/// ## Thread Safety
/// `CooccurrenceCounts` is `Sendable` and safe to share across threads.
public struct CooccurrenceCounts: Sendable {

    /// Number of windows (or documents) in which each word appears.
    ///
    /// For sliding window mode: count of windows containing the word.
    /// For document mode: count of documents containing the word.
    public let wordCounts: [String: Int]

    /// Number of windows (or documents) in which each word pair appears together.
    ///
    /// Keys are `WordPair` (symmetric: pair(a,b) == pair(b,a)).
    public let pairCounts: [WordPair: Int]

    /// Total number of windows (or documents) counted.
    public let totalWindows: Int

    /// Creates co-occurrence counts.
    ///
    /// - Parameters:
    ///   - wordCounts: Word frequency counts.
    ///   - pairCounts: Word pair frequency counts.
    ///   - totalWindows: Total window/document count.
    public init(
        wordCounts: [String: Int],
        pairCounts: [WordPair: Int],
        totalWindows: Int
    ) {
        self.wordCounts = wordCounts
        self.pairCounts = pairCounts
        self.totalWindows = totalWindows
    }

    /// Gets the count for a word.
    ///
    /// - Parameter word: The word to look up.
    /// - Returns: Number of windows/documents containing the word.
    public func count(for word: String) -> Int {
        wordCounts[word, default: 0]
    }

    /// Gets the count for a word pair.
    ///
    /// - Parameters:
    ///   - word1: First word.
    ///   - word2: Second word.
    /// - Returns: Number of windows/documents where both words appear.
    public func count(for word1: String, _ word2: String) -> Int {
        let pair = WordPair(word1, word2)
        return pairCounts[pair, default: 0]
    }

    /// Number of unique words in the corpus.
    public var vocabularySize: Int {
        wordCounts.count
    }

    /// Number of unique word pairs observed.
    public var pairCount: Int {
        pairCounts.count
    }

    /// Whether the counts are empty.
    public var isEmpty: Bool {
        totalWindows == 0
    }
}

// MARK: - Co-occurrence Counter

/// Counts word and word-pair co-occurrences in a corpus.
///
/// ## Algorithm
///
/// For **sliding window** mode:
/// 1. For each document, extract tokens
/// 2. Slide a window of size W across the tokens
/// 3. For each window, record which words appear (deduped)
/// 4. Count each word and each pair of distinct words
///
/// For **document** mode:
/// 1. For each document, extract unique tokens
/// 2. Count each word and each pair of distinct words
///
/// ## Usage
///
/// ```swift
/// let counter = CooccurrenceCounter(mode: .slidingWindow(size: 10))
/// let counts = counter.count(tokenizedDocuments: docs)
/// ```
///
/// ## Thread Safety
///
/// `CooccurrenceCounter` is `Sendable` and safe to use from any thread.
public struct CooccurrenceCounter: Sendable {

    /// The co-occurrence mode.
    public let mode: CooccurrenceMode

    /// Creates a co-occurrence counter.
    ///
    /// - Parameter mode: The counting mode (default: sliding window of 10).
    public init(mode: CooccurrenceMode = .default) {
        self.mode = mode
    }

    // MARK: - Counting

    /// Counts co-occurrences from tokenized documents.
    ///
    /// - Parameter tokenizedDocuments: Array of token arrays (one per document).
    /// - Returns: Co-occurrence counts.
    public func count(tokenizedDocuments: [[String]]) -> CooccurrenceCounts {
        switch mode {
        case .slidingWindow(let size):
            return countSlidingWindow(tokenizedDocuments, windowSize: size)
        case .document:
            return countDocument(tokenizedDocuments)
        }
    }

    /// Counts co-occurrences from documents using a tokenizer.
    ///
    /// - Parameters:
    ///   - documents: Documents to process.
    ///   - tokenizer: Tokenizer to use.
    /// - Returns: Co-occurrence counts.
    public func count(documents: [Document], tokenizer: Tokenizer) -> CooccurrenceCounts {
        let tokenizedDocs = tokenizer.tokenize(documents: documents)
        return count(tokenizedDocuments: tokenizedDocs)
    }

    // MARK: - Sliding Window Counting

    private func countSlidingWindow(
        _ tokenizedDocuments: [[String]],
        windowSize: Int
    ) -> CooccurrenceCounts {
        var wordCounts = [String: Int]()
        var pairCounts = [WordPair: Int]()
        var totalWindows = 0

        for tokens in tokenizedDocuments {
            guard !tokens.isEmpty else { continue }

            // Slide window across document
            let numWindows = max(1, tokens.count - windowSize + 1)

            for windowStart in 0..<numWindows {
                let windowEnd = min(windowStart + windowSize, tokens.count)
                let window = tokens[windowStart..<windowEnd]

                // Get unique words in this window
                let uniqueWords = Set(window)
                let wordArray = Array(uniqueWords)

                // Count each word
                for word in uniqueWords {
                    wordCounts[word, default: 0] += 1
                }

                // Count each pair of distinct words
                for i in 0..<wordArray.count {
                    for j in (i + 1)..<wordArray.count {
                        let pair = WordPair(wordArray[i], wordArray[j])
                        pairCounts[pair, default: 0] += 1
                    }
                }

                totalWindows += 1
            }
        }

        return CooccurrenceCounts(
            wordCounts: wordCounts,
            pairCounts: pairCounts,
            totalWindows: totalWindows
        )
    }

    // MARK: - Document-Level Counting

    private func countDocument(_ tokenizedDocuments: [[String]]) -> CooccurrenceCounts {
        var wordCounts = [String: Int]()
        var pairCounts = [WordPair: Int]()
        var totalDocuments = 0

        for tokens in tokenizedDocuments {
            guard !tokens.isEmpty else { continue }

            // Get unique words in this document
            let uniqueWords = Set(tokens)
            let wordArray = Array(uniqueWords)

            // Count each word
            for word in uniqueWords {
                wordCounts[word, default: 0] += 1
            }

            // Count each pair of distinct words
            for i in 0..<wordArray.count {
                for j in (i + 1)..<wordArray.count {
                    let pair = WordPair(wordArray[i], wordArray[j])
                    pairCounts[pair, default: 0] += 1
                }
            }

            totalDocuments += 1
        }

        return CooccurrenceCounts(
            wordCounts: wordCounts,
            pairCounts: pairCounts,
            totalWindows: totalDocuments
        )
    }
}

// MARK: - Convenience Extensions

extension Array where Element == [String] {

    /// Counts word co-occurrences in tokenized documents.
    ///
    /// - Parameter mode: Co-occurrence counting mode.
    /// - Returns: Co-occurrence counts.
    public func countCooccurrences(mode: CooccurrenceMode = .default) -> CooccurrenceCounts {
        CooccurrenceCounter(mode: mode).count(tokenizedDocuments: self)
    }
}

extension Array where Element == Document {

    /// Counts word co-occurrences in documents.
    ///
    /// - Parameters:
    ///   - mode: Co-occurrence counting mode.
    ///   - tokenizer: Tokenizer to use.
    /// - Returns: Co-occurrence counts.
    public func countCooccurrences(
        mode: CooccurrenceMode = .default,
        tokenizer: Tokenizer = Tokenizer()
    ) -> CooccurrenceCounts {
        CooccurrenceCounter(mode: mode).count(documents: self, tokenizer: tokenizer)
    }
}
