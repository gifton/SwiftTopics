// BufferedEntry.swift
// SwiftTopics
//
// Entry awaiting incorporation into the topic model

import Foundation

// MARK: - Buffered Entry

/// An entry waiting to be incorporated into the topic model.
///
/// When a new document is processed, it receives an immediate topic assignment
/// via transform (centroid-nearest), but is also buffered for the next
/// micro-retrain operation. The buffer accumulates entries until a threshold
/// is reached (default: 30 entries), triggering a micro-retrain.
///
/// ## Contents
///
/// Each buffered entry contains:
/// - **Document ID**: Links back to the original document
/// - **Embedding**: Pre-computed vector representation
/// - **Tokenized Content**: For c-TF-IDF computation during retrain
/// - **Timestamp**: For ordering and staleness detection
///
/// ## Thread Safety
///
/// `BufferedEntry` is `Sendable` and `Codable` for safe storage and transfer.
public struct BufferedEntry: Sendable, Codable, Hashable, Identifiable {

    // MARK: - Properties

    /// The ID of the associated document.
    public let documentID: DocumentID

    /// The pre-computed embedding for this document.
    ///
    /// Stored here to avoid re-computing during micro-retrain.
    public let embedding: Embedding

    /// Tokenized content for c-TF-IDF computation.
    ///
    /// These are the preprocessed tokens extracted from the document.
    /// Storing them avoids re-tokenizing during micro-retrain.
    public let tokenizedContent: [String]

    /// When this entry was added to the buffer.
    ///
    /// Used for:
    /// - Ordering entries chronologically
    /// - Detecting stale entries that may need special handling
    /// - Debugging and monitoring
    public let addedAt: Date

    /// Identifiable conformance.
    public var id: DocumentID { documentID }

    // MARK: - Initialization

    /// Creates a buffered entry.
    ///
    /// - Parameters:
    ///   - documentID: The ID of the associated document.
    ///   - embedding: Pre-computed embedding.
    ///   - tokenizedContent: Preprocessed tokens from the document.
    ///   - addedAt: Timestamp (defaults to now).
    public init(
        documentID: DocumentID,
        embedding: Embedding,
        tokenizedContent: [String],
        addedAt: Date = Date()
    ) {
        self.documentID = documentID
        self.embedding = embedding
        self.tokenizedContent = tokenizedContent
        self.addedAt = addedAt
    }

    /// Creates a buffered entry from a document and its embedding.
    ///
    /// - Parameters:
    ///   - document: The source document.
    ///   - embedding: Pre-computed embedding.
    ///   - tokenizer: Tokenizer to extract tokens from content.
    ///   - addedAt: Timestamp (defaults to now).
    public init(
        document: Document,
        embedding: Embedding,
        tokenizedContent: [String],
        addedAt: Date = Date()
    ) {
        self.documentID = document.id
        self.embedding = embedding
        self.tokenizedContent = tokenizedContent
        self.addedAt = addedAt
    }
}

// MARK: - Buffer Statistics

/// Statistics about the pending buffer.
public struct BufferStatistics: Sendable {

    /// Number of entries in the buffer.
    public let count: Int

    /// Oldest entry timestamp (nil if buffer is empty).
    public let oldestEntry: Date?

    /// Newest entry timestamp (nil if buffer is empty).
    public let newestEntry: Date?

    /// Total embedding storage size estimate in bytes.
    ///
    /// Calculated as: count × embeddingDimension × 4 (Float size)
    public let estimatedEmbeddingBytes: Int

    /// Whether the buffer has reached the retrain threshold.
    public let thresholdReached: Bool

    /// Creates buffer statistics.
    public init(
        count: Int,
        oldestEntry: Date?,
        newestEntry: Date?,
        embeddingDimension: Int,
        threshold: Int
    ) {
        self.count = count
        self.oldestEntry = oldestEntry
        self.newestEntry = newestEntry
        self.estimatedEmbeddingBytes = count * embeddingDimension * MemoryLayout<Float>.size
        self.thresholdReached = count >= threshold
    }

    /// Creates statistics from a collection of buffered entries.
    public static func from(
        entries: [BufferedEntry],
        threshold: Int
    ) -> BufferStatistics {
        let dimension = entries.first?.embedding.dimension ?? 0
        let dates = entries.map { $0.addedAt }

        return BufferStatistics(
            count: entries.count,
            oldestEntry: dates.min(),
            newestEntry: dates.max(),
            embeddingDimension: dimension,
            threshold: threshold
        )
    }
}

// MARK: - Comparable Extension

extension BufferedEntry: Comparable {
    /// Entries are ordered by their timestamp.
    public static func < (lhs: BufferedEntry, rhs: BufferedEntry) -> Bool {
        lhs.addedAt < rhs.addedAt
    }
}

// MARK: - CustomStringConvertible

extension BufferedEntry: CustomStringConvertible {
    public var description: String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return "BufferedEntry(id=\(documentID), dim=\(embedding.dimension), added=\(formatter.string(from: addedAt)))"
    }
}
