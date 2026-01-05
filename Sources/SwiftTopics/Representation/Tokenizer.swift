// Tokenizer.swift
// SwiftTopics
//
// Text tokenization for topic representation

import Foundation

// MARK: - Tokenizer Configuration

/// Configuration for text tokenization.
public struct TokenizerConfiguration: Sendable, Codable, Hashable {

    /// Stop words to filter out.
    public let stopWords: Set<String>

    /// Minimum token length (shorter tokens are discarded).
    public let minTokenLength: Int

    /// Maximum token length (longer tokens are discarded).
    public let maxTokenLength: Int

    /// Whether to lowercase all tokens.
    public let lowercase: Bool

    /// Whether to remove tokens that are purely numeric.
    public let removeNumbers: Bool

    /// Whether to generate bigrams in addition to unigrams.
    public let useBigrams: Bool

    /// Creates a tokenizer configuration.
    ///
    /// - Parameters:
    ///   - stopWords: Set of stop words to filter.
    ///   - minTokenLength: Minimum token length (default: 2).
    ///   - maxTokenLength: Maximum token length (default: 50).
    ///   - lowercase: Whether to lowercase tokens (default: true).
    ///   - removeNumbers: Whether to remove numeric tokens (default: false).
    ///   - useBigrams: Whether to generate bigrams (default: false).
    public init(
        stopWords: Set<String> = [],
        minTokenLength: Int = 2,
        maxTokenLength: Int = 50,
        lowercase: Bool = true,
        removeNumbers: Bool = false,
        useBigrams: Bool = false
    ) {
        precondition(minTokenLength >= 1, "minTokenLength must be at least 1")
        precondition(maxTokenLength >= minTokenLength, "maxTokenLength must be >= minTokenLength")

        self.stopWords = stopWords
        self.minTokenLength = minTokenLength
        self.maxTokenLength = maxTokenLength
        self.lowercase = lowercase
        self.removeNumbers = removeNumbers
        self.useBigrams = useBigrams
    }

    /// Default configuration without stop words.
    public static let `default` = TokenizerConfiguration()

    /// Configuration with English stop words.
    public static let english = TokenizerConfiguration(
        stopWords: EnglishStopWords.standard
    )

    /// Configuration with extended English stop words.
    public static let englishExtended = TokenizerConfiguration(
        stopWords: EnglishStopWords.extended
    )

    /// Configuration for technical/code-like text.
    public static let technical = TokenizerConfiguration(
        stopWords: EnglishStopWords.standard,
        minTokenLength: 2,
        removeNumbers: false
    )
}

// MARK: - Tokenizer

/// Tokenizes text into normalized tokens for vocabulary building.
///
/// ## Algorithm
/// 1. Split on whitespace and punctuation
/// 2. Optionally lowercase
/// 3. Filter by length
/// 4. Filter stop words
/// 5. Optionally generate bigrams
///
/// ## Thread Safety
/// `Tokenizer` is immutable and safe to use from any thread.
public struct Tokenizer: Sendable {

    /// The configuration for this tokenizer.
    public let configuration: TokenizerConfiguration

    /// Creates a tokenizer with the specified configuration.
    ///
    /// - Parameter configuration: Tokenizer configuration.
    public init(configuration: TokenizerConfiguration = .english) {
        self.configuration = configuration
    }

    /// Creates a tokenizer with English stop words.
    public init() {
        self.configuration = .english
    }

    /// Tokenizes a single text string.
    ///
    /// - Parameter text: The text to tokenize.
    /// - Returns: Array of normalized tokens.
    public func tokenize(_ text: String) -> [String] {
        // Step 1: Split on non-alphanumeric characters
        let rawTokens = text.components(separatedBy: CharacterSet.alphanumerics.inverted)

        // Step 2: Process each token
        var tokens: [String] = []
        tokens.reserveCapacity(rawTokens.count)

        for rawToken in rawTokens {
            let token = configuration.lowercase ? rawToken.lowercased() : rawToken

            // Skip empty tokens
            guard !token.isEmpty else { continue }

            // Filter by length
            guard token.count >= configuration.minTokenLength,
                  token.count <= configuration.maxTokenLength else { continue }

            // Filter stop words (check lowercase version for case-insensitive matching)
            guard !configuration.stopWords.contains(token.lowercased()) else { continue }

            // Filter numbers if configured
            if configuration.removeNumbers && token.allSatisfy(\.isNumber) {
                continue
            }

            tokens.append(token)
        }

        // Step 3: Generate bigrams if configured
        if configuration.useBigrams && tokens.count >= 2 {
            var bigrams: [String] = []
            bigrams.reserveCapacity(tokens.count - 1)
            for i in 0..<(tokens.count - 1) {
                bigrams.append("\(tokens[i])_\(tokens[i + 1])")
            }
            tokens.append(contentsOf: bigrams)
        }

        return tokens
    }

    /// Tokenizes multiple texts.
    ///
    /// - Parameter texts: Array of texts to tokenize.
    /// - Returns: Array of token arrays (one per text).
    public func tokenize(_ texts: [String]) -> [[String]] {
        texts.map { tokenize($0) }
    }

    /// Tokenizes documents and returns tokens per document.
    ///
    /// - Parameter documents: Documents to tokenize.
    /// - Returns: Array of token arrays (one per document).
    public func tokenize(documents: [Document]) -> [[String]] {
        documents.map { tokenize($0.content) }
    }

    /// Gets unique tokens from a text.
    ///
    /// - Parameter text: The text to tokenize.
    /// - Returns: Set of unique tokens.
    public func uniqueTokens(_ text: String) -> Set<String> {
        Set(tokenize(text))
    }
}

// MARK: - English Stop Words

/// Common English stop words for filtering.
public enum EnglishStopWords {

    /// Standard English stop words (most common function words).
    public static let standard: Set<String> = [
        // Articles
        "a", "an", "the",
        // Pronouns
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        // Verbs (be, have, do)
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        // Prepositions
        "at", "by", "for", "from", "in", "into", "of", "on", "to", "with",
        "about", "against", "between", "through", "during", "before", "after",
        "above", "below", "up", "down", "out", "off", "over", "under",
        // Conjunctions
        "and", "but", "if", "or", "because", "as", "until", "while",
        // Other common words
        "so", "than", "too", "very", "just", "can", "will", "should",
        "now", "also", "only", "then", "more", "no", "not", "such",
        "here", "there", "when", "where", "why", "how", "all", "each",
        "both", "few", "more", "most", "other", "some", "any", "same",
        "own", "s", "t", "ll", "re", "ve", "d", "m"
    ]

    /// Extended English stop words (includes more common words).
    public static let extended: Set<String> = standard.union([
        // Additional common words
        "would", "could", "may", "might", "must", "shall",
        "get", "got", "getting", "go", "going", "gone", "went",
        "come", "came", "coming", "make", "made", "making",
        "take", "took", "taking", "see", "saw", "seen", "seeing",
        "know", "knew", "known", "knowing", "think", "thought", "thinking",
        "say", "said", "saying", "want", "wanted", "wanting",
        "use", "used", "using", "find", "found", "finding",
        "give", "gave", "given", "giving", "tell", "told", "telling",
        "work", "worked", "working", "seem", "seemed", "seeming",
        "feel", "felt", "feeling", "try", "tried", "trying",
        "leave", "left", "leaving", "call", "called", "calling",
        "keep", "kept", "keeping", "let", "need", "needed", "needing",
        "become", "became", "becoming", "begin", "began", "begun", "beginning",
        "show", "showed", "shown", "showing", "hear", "heard", "hearing",
        "play", "played", "playing", "run", "ran", "running",
        "move", "moved", "moving", "live", "lived", "living",
        "believe", "believed", "believing", "bring", "brought", "bringing",
        "happen", "happened", "happening", "write", "wrote", "written", "writing",
        "provide", "provided", "providing", "sit", "sat", "sitting",
        "stand", "stood", "standing", "lose", "lost", "losing",
        "pay", "paid", "paying", "meet", "met", "meeting",
        "include", "included", "including", "continue", "continued", "continuing",
        "set", "learn", "learned", "learning", "change", "changed", "changing",
        "lead", "led", "leading", "understand", "understood", "understanding",
        "watch", "watched", "watching", "follow", "followed", "following",
        "stop", "stopped", "stopping", "create", "created", "creating",
        "speak", "spoke", "spoken", "speaking", "read", "reading",
        "allow", "allowed", "allowing", "add", "added", "adding",
        "spend", "spent", "spending", "grow", "grew", "grown", "growing",
        "open", "opened", "opening", "walk", "walked", "walking",
        "win", "won", "winning", "offer", "offered", "offering",
        "remember", "remembered", "remembering", "love", "loved", "loving",
        "consider", "considered", "considering", "appear", "appeared", "appearing",
        "buy", "bought", "buying", "wait", "waited", "waiting",
        "serve", "served", "serving", "die", "died", "dying",
        "send", "sent", "sending", "expect", "expected", "expecting",
        "build", "built", "building", "stay", "stayed", "staying",
        // Common adverbs
        "already", "always", "never", "often", "still", "ever", "even",
        "really", "actually", "probably", "certainly", "definitely",
        "almost", "enough", "quite", "rather", "well", "much",
        "yet", "usually", "sometimes", "maybe", "perhaps",
        // Common adjectives
        "new", "old", "good", "bad", "great", "little", "big", "small",
        "long", "short", "high", "low", "right", "wrong", "first", "last",
        "next", "different", "important", "many", "few", "several",
        // Numbers as words
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        // Other
        "like", "way", "thing", "things", "something", "nothing", "everything",
        "someone", "anyone", "everyone", "nobody", "anybody", "everybody",
        "place", "places", "time", "times", "day", "days", "year", "years",
        "people", "person", "part", "parts", "case", "cases", "point", "points",
        "fact", "facts", "example", "examples", "number", "numbers"
    ])
}

// MARK: - Tokenizer Builder

/// Builder for creating tokenizer configurations.
public struct TokenizerBuilder: Sendable {

    private var stopWords: Set<String> = EnglishStopWords.standard
    private var minTokenLength: Int = 2
    private var maxTokenLength: Int = 50
    private var lowercase: Bool = true
    private var removeNumbers: Bool = false
    private var useBigrams: Bool = false

    /// Creates a new tokenizer builder with default settings.
    public init() {}

    /// Sets the stop words.
    public func stopWords(_ words: Set<String>) -> TokenizerBuilder {
        var copy = self
        copy.stopWords = words
        return copy
    }

    /// Adds additional stop words.
    public func addStopWords(_ words: Set<String>) -> TokenizerBuilder {
        var copy = self
        copy.stopWords = copy.stopWords.union(words)
        return copy
    }

    /// Sets the minimum token length.
    public func minLength(_ length: Int) -> TokenizerBuilder {
        var copy = self
        copy.minTokenLength = length
        return copy
    }

    /// Sets the maximum token length.
    public func maxLength(_ length: Int) -> TokenizerBuilder {
        var copy = self
        copy.maxTokenLength = length
        return copy
    }

    /// Enables or disables lowercasing.
    public func lowercase(_ enable: Bool) -> TokenizerBuilder {
        var copy = self
        copy.lowercase = enable
        return copy
    }

    /// Enables or disables number removal.
    public func removeNumbers(_ enable: Bool) -> TokenizerBuilder {
        var copy = self
        copy.removeNumbers = enable
        return copy
    }

    /// Enables or disables bigram generation.
    public func useBigrams(_ enable: Bool) -> TokenizerBuilder {
        var copy = self
        copy.useBigrams = enable
        return copy
    }

    /// Builds the tokenizer configuration.
    public func buildConfiguration() -> TokenizerConfiguration {
        TokenizerConfiguration(
            stopWords: stopWords,
            minTokenLength: minTokenLength,
            maxTokenLength: maxTokenLength,
            lowercase: lowercase,
            removeNumbers: removeNumbers,
            useBigrams: useBigrams
        )
    }

    /// Builds the tokenizer.
    public func build() -> Tokenizer {
        Tokenizer(configuration: buildConfiguration())
    }
}
