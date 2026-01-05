// TopicModelStorage.swift
// SwiftTopics
//
// Protocol for persisting incremental topic model state

import Foundation

// MARK: - Storage Protocol

/// Protocol for persisting topic model state.
///
/// Implement this protocol to integrate SwiftTopics with your app's persistence layer.
/// A file-based default implementation is provided via `FileBasedTopicModelStorage`.
///
/// ## Storage Responsibilities
///
/// The storage layer handles three categories of data:
///
/// 1. **Model State**: Small, frequently updated metadata (topics, config, statistics)
/// 2. **Embeddings**: Large, append-only vector data (~2KB per document)
/// 3. **Training Data**: Temporary checkpoint and graph data during training
///
/// ## Thread Safety
///
/// All methods are async and implementations must be thread-safe.
/// The storage may be accessed concurrently from the `IncrementalTopicUpdater` actor.
///
/// ## Atomicity
///
/// Implementations should ensure atomic writes where possible to prevent corruption
/// if the app terminates during a write operation.
public protocol TopicModelStorage: Sendable {

    // MARK: - Model State

    /// Saves the fitted model state.
    ///
    /// This is called after every micro-retrain and full refresh.
    /// The state is relatively small (typically < 1MB) and should be
    /// written atomically.
    ///
    /// - Parameter state: The model state to save.
    /// - Throws: Storage errors.
    func saveModelState(_ state: IncrementalTopicModelState) async throws

    /// Loads the fitted model state.
    ///
    /// Returns nil if no model has been fitted yet.
    ///
    /// - Returns: The saved model state, or nil if none exists.
    /// - Throws: Storage errors (but not for missing data).
    func loadModelState() async throws -> IncrementalTopicModelState?

    // MARK: - Embeddings (Large Data)

    /// Appends embeddings for new documents.
    ///
    /// Embeddings are append-only; once written, they are not modified.
    /// This enables efficient binary storage with memory-mapped access.
    ///
    /// - Parameter embeddings: Document ID and embedding pairs to append.
    /// - Throws: Storage errors.
    func appendEmbeddings(_ embeddings: [(DocumentID, Embedding)]) async throws

    /// Loads embeddings for specific documents.
    ///
    /// Used during transform and micro-retrain operations.
    ///
    /// - Parameter documentIDs: IDs of documents to load embeddings for.
    /// - Returns: Embeddings for the requested documents (in order).
    /// - Throws: Storage errors or if documents not found.
    func loadEmbeddings(for documentIDs: [DocumentID]) async throws -> [Embedding]

    /// Loads all embeddings for full retraining.
    ///
    /// This may load significant data into memory. For very large corpora,
    /// consider implementing streaming access.
    ///
    /// - Returns: All document ID and embedding pairs.
    /// - Throws: Storage errors.
    func loadAllEmbeddings() async throws -> [(DocumentID, Embedding)]

    /// Returns the number of stored embeddings without loading them.
    ///
    /// - Returns: Count of stored embeddings.
    /// - Throws: Storage errors.
    func embeddingCount() async throws -> Int

    // MARK: - Reduced Embeddings (Optional)

    /// Saves reduced (UMAP-transformed) embeddings after training.
    ///
    /// These are used for high-quality transform of new documents.
    /// Storing them is optional but improves transform quality.
    ///
    /// - Parameter embeddings: Document ID and reduced vector pairs.
    /// - Throws: Storage errors.
    func saveReducedEmbeddings(_ embeddings: [(DocumentID, [Float])]) async throws

    /// Loads reduced embeddings.
    ///
    /// - Returns: Document ID and reduced vector pairs, or nil if not stored.
    /// - Throws: Storage errors.
    func loadReducedEmbeddings() async throws -> [(DocumentID, [Float])]?

    // MARK: - k-NN Graph (Optional)

    /// Saves the k-NN graph for UMAP transform.
    ///
    /// Storing the k-NN graph enables higher-quality transform of new documents
    /// but requires significant storage (~1.2MB for 10K docs with k=15).
    ///
    /// - Parameter graph: The nearest neighbor graph.
    /// - Throws: Storage errors.
    func saveKNNGraph(_ graph: NearestNeighborGraph) async throws

    /// Loads the k-NN graph.
    ///
    /// - Returns: The saved k-NN graph, or nil if not stored.
    /// - Throws: Storage errors.
    func loadKNNGraph() async throws -> NearestNeighborGraph?

    // MARK: - Training Checkpoint

    /// Saves a training checkpoint for resumption after interruption.
    ///
    /// Checkpoints are saved periodically during long-running training
    /// operations to enable resume after app termination.
    ///
    /// - Parameter checkpoint: The checkpoint to save.
    /// - Throws: Storage errors.
    func saveCheckpoint(_ checkpoint: TrainingCheckpoint) async throws

    /// Loads the most recent training checkpoint.
    ///
    /// Returns nil if no checkpoint exists or training completed successfully.
    ///
    /// - Returns: The saved checkpoint, or nil if none exists.
    /// - Throws: Storage errors.
    func loadCheckpoint() async throws -> TrainingCheckpoint?

    /// Clears the checkpoint after successful training completion.
    ///
    /// - Throws: Storage errors.
    func clearCheckpoint() async throws

    // MARK: - Pending Buffer

    /// Adds entries to the pending buffer awaiting incorporation into the model.
    ///
    /// Buffered entries are incorporated during the next micro-retrain.
    ///
    /// - Parameter entries: Entries to add to the buffer.
    /// - Throws: Storage errors.
    func appendToPendingBuffer(_ entries: [BufferedEntry]) async throws

    /// Loads and clears the pending buffer.
    ///
    /// This is called when triggering a micro-retrain. The buffer is cleared
    /// atomically to prevent duplicate processing.
    ///
    /// - Returns: All buffered entries.
    /// - Throws: Storage errors.
    func drainPendingBuffer() async throws -> [BufferedEntry]

    /// Returns the number of pending entries without loading them.
    ///
    /// - Returns: Count of pending entries.
    /// - Throws: Storage errors.
    func pendingBufferCount() async throws -> Int

    // MARK: - Maintenance

    /// Removes all stored data.
    ///
    /// Use with caution. This is useful for testing or when the user
    /// wants to start fresh.
    ///
    /// - Throws: Storage errors.
    func clear() async throws

    /// Returns the total storage size in bytes.
    ///
    /// Useful for displaying storage usage to users.
    ///
    /// - Returns: Total bytes used by all stored data.
    /// - Throws: Storage errors.
    func storageSizeBytes() async throws -> UInt64
}

// MARK: - Storage Error

/// Errors that can occur during storage operations.
public enum TopicModelStorageError: Error, Sendable {

    /// The requested document was not found in storage.
    case documentNotFound(DocumentID)

    /// Multiple documents were not found.
    case documentsNotFound([DocumentID])

    /// The storage directory could not be created or accessed.
    case directoryAccessError(path: String, underlying: Error?)

    /// Failed to write data to storage.
    case writeError(path: String, underlying: Error)

    /// Failed to read data from storage.
    case readError(path: String, underlying: Error)

    /// Data corruption detected (checksum mismatch, invalid format, etc.).
    case dataCorrupted(path: String, reason: String)

    /// The storage is in an inconsistent state.
    case inconsistentState(reason: String)

    /// Storage capacity exceeded.
    case capacityExceeded(limit: UInt64, required: UInt64)

    /// An unknown error occurred.
    case unknown(Error)
}

// MARK: - Error Descriptions

extension TopicModelStorageError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .documentNotFound(let id):
            return "Document not found: \(id)"
        case .documentsNotFound(let ids):
            return "Documents not found: \(ids.count) documents"
        case .directoryAccessError(let path, let underlying):
            let base = "Cannot access storage directory: \(path)"
            if let underlying = underlying {
                return "\(base) - \(underlying.localizedDescription)"
            }
            return base
        case .writeError(let path, let underlying):
            return "Failed to write to \(path): \(underlying.localizedDescription)"
        case .readError(let path, let underlying):
            return "Failed to read from \(path): \(underlying.localizedDescription)"
        case .dataCorrupted(let path, let reason):
            return "Data corrupted at \(path): \(reason)"
        case .inconsistentState(let reason):
            return "Storage in inconsistent state: \(reason)"
        case .capacityExceeded(let limit, let required):
            return "Storage capacity exceeded: requires \(required) bytes, limit is \(limit) bytes"
        case .unknown(let error):
            return "Unknown storage error: \(error.localizedDescription)"
        }
    }
}
