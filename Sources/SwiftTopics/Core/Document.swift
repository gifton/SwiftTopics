// Document.swift
// SwiftTopics
//
// Foundational document type for topic modeling pipeline

import Foundation

// MARK: - Document

/// A document in the topic modeling corpus.
///
/// Documents are the primary input to the topic modeling pipeline. Each document
/// has a unique identifier, text content, and optional metadata for domain-specific
/// information.
///
/// ## Thread Safety
/// `Document` is `Sendable` and can be safely shared across concurrency domains.
///
/// ## Serialization
/// `Document` conforms to `Codable` for persistence and transport.
public struct Document: Sendable, Codable, Hashable, Identifiable {

    /// Unique identifier for this document.
    public let id: DocumentID

    /// The text content of the document.
    public let content: String

    /// Optional metadata associated with this document.
    ///
    /// Metadata can store domain-specific information such as:
    /// - Creation date
    /// - Author
    /// - Tags or categories
    /// - Source information
    public let metadata: DocumentMetadata?

    /// Creates a new document.
    ///
    /// - Parameters:
    ///   - id: Unique identifier. If nil, a new UUID is generated.
    ///   - content: The text content of the document.
    ///   - metadata: Optional metadata dictionary.
    public init(
        id: DocumentID? = nil,
        content: String,
        metadata: DocumentMetadata? = nil
    ) {
        self.id = id ?? DocumentID()
        self.content = content
        self.metadata = metadata
    }

    /// Creates a document with a string identifier.
    ///
    /// Convenience initializer for creating documents with human-readable IDs.
    ///
    /// - Parameters:
    ///   - stringID: A string to use as the document identifier.
    ///   - content: The text content of the document.
    ///   - metadata: Optional metadata dictionary.
    public init(
        stringID: String,
        content: String,
        metadata: DocumentMetadata? = nil
    ) {
        self.id = DocumentID(string: stringID)
        self.content = content
        self.metadata = metadata
    }
}

// MARK: - Document ID

/// A unique identifier for a document.
///
/// DocumentID wraps a UUID to provide type safety and prevent accidental mixing
/// of document IDs with other UUID-based identifiers.
public struct DocumentID: Sendable, Codable, Hashable, CustomStringConvertible {

    /// The underlying UUID value.
    public let value: UUID

    /// Creates a new random document ID.
    public init() {
        self.value = UUID()
    }

    /// Creates a document ID from an existing UUID.
    ///
    /// - Parameter uuid: The UUID to wrap.
    public init(uuid: UUID) {
        self.value = uuid
    }

    /// Creates a document ID from a string.
    ///
    /// If the string is a valid UUID, it will be parsed. Otherwise, a deterministic
    /// UUID will be generated from the string using UUID v5 (namespace-based).
    ///
    /// - Parameter string: The string to convert to a document ID.
    public init(string: String) {
        if let uuid = UUID(uuidString: string) {
            self.value = uuid
        } else {
            // Create deterministic UUID from string using namespace hashing
            self.value = Self.deterministicUUID(from: string)
        }
    }

    public var description: String {
        value.uuidString
    }

    // MARK: - Private

    /// Generates a deterministic UUID from a string.
    ///
    /// Uses a simple hash-based approach to create reproducible UUIDs from strings.
    private static func deterministicUUID(from string: String) -> UUID {
        var hasher = Hasher()
        hasher.combine(string)
        let hash = hasher.finalize()

        // Create UUID bytes from hash
        var bytes = [UInt8](repeating: 0, count: 16)
        withUnsafeBytes(of: hash) { hashBytes in
            for i in 0..<min(hashBytes.count, 8) {
                bytes[i] = hashBytes[i]
            }
        }

        // Second hash for remaining bytes
        var hasher2 = Hasher()
        hasher2.combine(string)
        hasher2.combine(42) // Different seed
        let hash2 = hasher2.finalize()

        withUnsafeBytes(of: hash2) { hashBytes in
            for i in 0..<min(hashBytes.count, 8) {
                bytes[i + 8] = hashBytes[i]
            }
        }

        // Set version (4) and variant (RFC 4122) bits
        bytes[6] = (bytes[6] & 0x0F) | 0x40  // Version 4
        bytes[8] = (bytes[8] & 0x3F) | 0x80  // Variant RFC 4122

        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }
}

// MARK: - Document Metadata

/// Metadata associated with a document.
///
/// Provides a flexible key-value store for domain-specific document information.
/// All values must be `Codable` and `Sendable`.
public struct DocumentMetadata: Sendable, Codable, Hashable {

    /// The underlying storage for metadata values.
    private var storage: [String: MetadataValue]

    /// Creates empty metadata.
    public init() {
        self.storage = [:]
    }

    /// Creates metadata with initial key-value pairs.
    ///
    /// - Parameter values: Dictionary of metadata values.
    public init(_ values: [String: MetadataValue]) {
        self.storage = values
    }

    /// Accesses the value for a given key.
    public subscript(key: String) -> MetadataValue? {
        get { storage[key] }
        set { storage[key] = newValue }
    }

    /// All keys in the metadata.
    public var keys: Dictionary<String, MetadataValue>.Keys {
        storage.keys
    }

    /// The number of key-value pairs.
    public var count: Int {
        storage.count
    }

    /// Whether the metadata is empty.
    public var isEmpty: Bool {
        storage.isEmpty
    }
}

// MARK: - Metadata Value

/// A value that can be stored in document metadata.
///
/// Supports common types that can be serialized and compared.
public enum MetadataValue: Sendable, Codable, Hashable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case date(Date)
    case strings([String])

    /// Gets the value as a String, if applicable.
    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    /// Gets the value as an Int, if applicable.
    public var intValue: Int? {
        if case .int(let value) = self { return value }
        return nil
    }

    /// Gets the value as a Double, if applicable.
    public var doubleValue: Double? {
        if case .double(let value) = self { return value }
        return nil
    }

    /// Gets the value as a Bool, if applicable.
    public var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    /// Gets the value as a Date, if applicable.
    public var dateValue: Date? {
        if case .date(let value) = self { return value }
        return nil
    }

    /// Gets the value as a String array, if applicable.
    public var stringsValue: [String]? {
        if case .strings(let value) = self { return value }
        return nil
    }
}

// MARK: - ExpressibleBy Literals

extension MetadataValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self = .string(value)
    }
}

extension MetadataValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self = .int(value)
    }
}

extension MetadataValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self = .double(value)
    }
}

extension MetadataValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }
}

extension MetadataValue: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: String...) {
        self = .strings(elements)
    }
}
