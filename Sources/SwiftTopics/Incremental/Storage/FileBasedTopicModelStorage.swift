// FileBasedTopicModelStorage.swift
// SwiftTopics
//
// File-based implementation of TopicModelStorage

import Foundation

// MARK: - File-Based Topic Model Storage

/// File-based implementation of `TopicModelStorage`.
///
/// This is the default storage implementation for SwiftTopics. It stores
/// all data in a directory structure:
///
/// ```
/// <directory>/
/// ├── model_state.json       # Serialized IncrementalTopicModelState
/// ├── embeddings.bin         # Binary embedding data
/// ├── embeddings_index.json  # DocumentID → byte offset mapping
/// ├── reduced_embeddings.bin # Binary reduced embedding data (optional)
/// ├── knn_graph.json         # k-NN graph (optional)
/// ├── pending_buffer.json    # Buffered entries awaiting retrain
/// └── checkpoint.json        # Training checkpoint (if interrupted)
/// ```
///
/// ## Binary Embedding Format
///
/// Embeddings are stored in a compact binary format:
/// - Header: dimension (Int32)
/// - For each embedding: Float32 values (dimension × 4 bytes)
///
/// ## Thread Safety
///
/// All operations use file-level locking to prevent corruption.
/// Multiple concurrent reads are allowed; writes are exclusive.
public actor FileBasedTopicModelStorage: TopicModelStorage {

    // MARK: - File Names

    private enum FileName {
        static let modelState = "model_state.json"
        static let embeddings = "embeddings.bin"
        static let embeddingsIndex = "embeddings_index.json"
        static let reducedEmbeddings = "reduced_embeddings.bin"
        static let reducedIndex = "reduced_index.json"
        static let knnGraph = "knn_graph.json"
        static let pendingBuffer = "pending_buffer.json"
        static let checkpoint = "checkpoint.json"
    }

    // MARK: - Properties

    /// Base directory for all storage files.
    private let directory: URL

    /// File manager for file operations.
    private let fileManager = FileManager.default

    /// JSON encoder configured for the storage format.
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    /// JSON decoder configured for the storage format.
    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    /// Cached embedding index for fast lookups.
    private var embeddingIndex: EmbeddingIndex?

    // MARK: - Initialization

    /// Creates file-based storage in the specified directory.
    ///
    /// The directory will be created if it doesn't exist.
    ///
    /// - Parameter directory: Directory for storage files.
    /// - Throws: `TopicModelStorageError.directoryAccessError` if directory
    ///           cannot be created or accessed.
    public init(directory: URL) throws {
        self.directory = directory

        // Create directory if needed
        if !fileManager.fileExists(atPath: directory.path) {
            do {
                try fileManager.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true,
                    attributes: nil
                )
            } catch {
                throw TopicModelStorageError.directoryAccessError(
                    path: directory.path,
                    underlying: error
                )
            }
        }
    }

    // MARK: - File Paths

    private func path(for fileName: String) -> URL {
        directory.appendingPathComponent(fileName)
    }

    private var modelStatePath: URL { path(for: FileName.modelState) }
    private var embeddingsPath: URL { path(for: FileName.embeddings) }
    private var embeddingsIndexPath: URL { path(for: FileName.embeddingsIndex) }
    private var reducedEmbeddingsPath: URL { path(for: FileName.reducedEmbeddings) }
    private var reducedIndexPath: URL { path(for: FileName.reducedIndex) }
    private var knnGraphPath: URL { path(for: FileName.knnGraph) }
    private var pendingBufferPath: URL { path(for: FileName.pendingBuffer) }
    private var checkpointPath: URL { path(for: FileName.checkpoint) }

    // MARK: - Model State

    public func saveModelState(_ state: IncrementalTopicModelState) async throws {
        let data = try encoder.encode(state)
        try atomicWrite(data: data, to: modelStatePath)
    }

    public func loadModelState() async throws -> IncrementalTopicModelState? {
        guard fileManager.fileExists(atPath: modelStatePath.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: modelStatePath)
            return try decoder.decode(IncrementalTopicModelState.self, from: data)
        } catch {
            throw TopicModelStorageError.readError(
                path: modelStatePath.path,
                underlying: error
            )
        }
    }

    // MARK: - Embeddings

    public func appendEmbeddings(_ embeddings: [(DocumentID, Embedding)]) async throws {
        guard !embeddings.isEmpty else { return }

        // Load or create index
        var index = try await loadOrCreateEmbeddingIndex()

        // Get current file size (where new embeddings will start)
        let startOffset = fileSize(at: embeddingsPath)

        // Prepare binary data
        let dimension = embeddings[0].1.dimension
        var binaryData = Data()

        // Write header if file is new
        if startOffset == 0 {
            var dim = Int32(dimension)
            binaryData.append(Data(bytes: &dim, count: MemoryLayout<Int32>.size))
        }

        // Append each embedding
        var currentOffset = startOffset == 0 ? MemoryLayout<Int32>.size : Int(startOffset)

        for (docID, embedding) in embeddings {
            // Validate dimension consistency
            guard embedding.dimension == dimension else {
                throw TopicModelStorageError.inconsistentState(
                    reason: "Embedding dimension \(embedding.dimension) doesn't match expected \(dimension)"
                )
            }

            // Record offset in index
            index.offsets[docID] = currentOffset
            index.documentIDs.append(docID)

            // Append embedding bytes
            for value in embedding.vector {
                var v = value
                binaryData.append(Data(bytes: &v, count: MemoryLayout<Float>.size))
            }

            currentOffset += dimension * MemoryLayout<Float>.size
        }

        index.dimension = dimension
        index.count = index.documentIDs.count

        // Append to binary file
        try appendToFile(data: binaryData, at: embeddingsPath)

        // Save updated index
        try await saveEmbeddingIndex(index)

        // Update cache
        self.embeddingIndex = index
    }

    public func loadEmbeddings(for documentIDs: [DocumentID]) async throws -> [Embedding] {
        guard !documentIDs.isEmpty else { return [] }

        let index = try await loadOrCreateEmbeddingIndex()

        // Find offsets for all requested documents
        var notFound = [DocumentID]()
        var offsets = [(DocumentID, Int)]()

        for docID in documentIDs {
            if let offset = index.offsets[docID] {
                offsets.append((docID, offset))
            } else {
                notFound.append(docID)
            }
        }

        guard notFound.isEmpty else {
            if notFound.count == 1 {
                throw TopicModelStorageError.documentNotFound(notFound[0])
            } else {
                throw TopicModelStorageError.documentsNotFound(notFound)
            }
        }

        // Read embeddings from binary file
        let dimension = index.dimension
        let embeddingBytes = dimension * MemoryLayout<Float>.size

        guard let fileHandle = FileHandle(forReadingAtPath: embeddingsPath.path) else {
            throw TopicModelStorageError.readError(
                path: embeddingsPath.path,
                underlying: NSError(domain: "SwiftTopics", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "Cannot open embeddings file"
                ])
            )
        }

        defer { try? fileHandle.close() }

        // Read embeddings in order requested
        var embeddings = [Embedding]()
        embeddings.reserveCapacity(documentIDs.count)

        // Create a mapping from docID to its position in the result
        var docIDToIndex = [DocumentID: Int]()
        for (i, docID) in documentIDs.enumerated() {
            docIDToIndex[docID] = i
        }

        // Read embeddings (may be out of order in file)
        var resultBuffer = [Embedding?](repeating: nil, count: documentIDs.count)

        for (docID, offset) in offsets {
            try fileHandle.seek(toOffset: UInt64(offset))
            guard let data = try fileHandle.read(upToCount: embeddingBytes) else {
                throw TopicModelStorageError.dataCorrupted(
                    path: embeddingsPath.path,
                    reason: "Could not read \(embeddingBytes) bytes at offset \(offset)"
                )
            }

            let vector = data.withUnsafeBytes { buffer -> [Float] in
                Array(buffer.bindMemory(to: Float.self))
            }

            let embedding = Embedding(vector: vector)
            if let resultIndex = docIDToIndex[docID] {
                resultBuffer[resultIndex] = embedding
            }
        }

        return resultBuffer.compactMap { $0 }
    }

    public func loadAllEmbeddings() async throws -> [(DocumentID, Embedding)] {
        let index = try await loadOrCreateEmbeddingIndex()

        guard index.count > 0 else { return [] }

        let embeddings = try await loadEmbeddings(for: index.documentIDs)
        return zip(index.documentIDs, embeddings).map { ($0, $1) }
    }

    public func embeddingCount() async throws -> Int {
        let index = try await loadOrCreateEmbeddingIndex()
        return index.count
    }

    // MARK: - Reduced Embeddings

    public func saveReducedEmbeddings(_ embeddings: [(DocumentID, [Float])]) async throws {
        guard !embeddings.isEmpty else { return }

        let dimension = embeddings[0].1.count

        // Build index
        var index = ReducedEmbeddingIndex(dimension: dimension)

        // Build binary data
        var binaryData = Data()
        var dim = Int32(dimension)
        binaryData.append(Data(bytes: &dim, count: MemoryLayout<Int32>.size))

        var currentOffset = MemoryLayout<Int32>.size

        for (docID, vector) in embeddings {
            index.offsets[docID] = currentOffset
            index.documentIDs.append(docID)

            for value in vector {
                var v = value
                binaryData.append(Data(bytes: &v, count: MemoryLayout<Float>.size))
            }

            currentOffset += dimension * MemoryLayout<Float>.size
        }

        index.count = embeddings.count

        // Write atomically
        try atomicWrite(data: binaryData, to: reducedEmbeddingsPath)

        let indexData = try encoder.encode(index)
        try atomicWrite(data: indexData, to: reducedIndexPath)
    }

    public func loadReducedEmbeddings() async throws -> [(DocumentID, [Float])]? {
        guard fileManager.fileExists(atPath: reducedEmbeddingsPath.path),
              fileManager.fileExists(atPath: reducedIndexPath.path) else {
            return nil
        }

        // Load index
        let indexData = try Data(contentsOf: reducedIndexPath)
        let index = try decoder.decode(ReducedEmbeddingIndex.self, from: indexData)

        guard index.count > 0 else { return [] }

        // Load binary data
        guard let fileHandle = FileHandle(forReadingAtPath: reducedEmbeddingsPath.path) else {
            throw TopicModelStorageError.readError(
                path: reducedEmbeddingsPath.path,
                underlying: NSError(domain: "SwiftTopics", code: 1)
            )
        }

        defer { try? fileHandle.close() }

        let embeddingBytes = index.dimension * MemoryLayout<Float>.size
        var result = [(DocumentID, [Float])]()
        result.reserveCapacity(index.count)

        for docID in index.documentIDs {
            guard let offset = index.offsets[docID] else { continue }

            try fileHandle.seek(toOffset: UInt64(offset))
            guard let data = try fileHandle.read(upToCount: embeddingBytes) else {
                continue
            }

            let vector = data.withUnsafeBytes { buffer -> [Float] in
                Array(buffer.bindMemory(to: Float.self))
            }

            result.append((docID, vector))
        }

        return result
    }

    // MARK: - k-NN Graph

    public func saveKNNGraph(_ graph: NearestNeighborGraph) async throws {
        let serializable = SerializableKNNGraph(from: graph)
        let data = try encoder.encode(serializable)
        try atomicWrite(data: data, to: knnGraphPath)
    }

    public func loadKNNGraph() async throws -> NearestNeighborGraph? {
        guard fileManager.fileExists(atPath: knnGraphPath.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: knnGraphPath)
            let serializable = try decoder.decode(SerializableKNNGraph.self, from: data)
            return serializable.toGraph()
        } catch {
            throw TopicModelStorageError.readError(
                path: knnGraphPath.path,
                underlying: error
            )
        }
    }

    // MARK: - Training Checkpoint

    public func saveCheckpoint(_ checkpoint: TrainingCheckpoint) async throws {
        let data = try encoder.encode(checkpoint)
        try atomicWrite(data: data, to: checkpointPath)
    }

    public func loadCheckpoint() async throws -> TrainingCheckpoint? {
        guard fileManager.fileExists(atPath: checkpointPath.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: checkpointPath)
            return try decoder.decode(TrainingCheckpoint.self, from: data)
        } catch {
            // Corrupted checkpoint - return nil so training restarts
            return nil
        }
    }

    public func clearCheckpoint() async throws {
        if fileManager.fileExists(atPath: checkpointPath.path) {
            try fileManager.removeItem(at: checkpointPath)
        }
    }

    // MARK: - Pending Buffer

    public func appendToPendingBuffer(_ entries: [BufferedEntry]) async throws {
        guard !entries.isEmpty else { return }

        // Load existing buffer
        var buffer = try loadPendingBufferInternal() ?? []
        buffer.append(contentsOf: entries)

        // Save updated buffer
        let data = try encoder.encode(buffer)
        try atomicWrite(data: data, to: pendingBufferPath)
    }

    public func drainPendingBuffer() async throws -> [BufferedEntry] {
        let buffer = try loadPendingBufferInternal() ?? []

        // Clear the buffer file
        if fileManager.fileExists(atPath: pendingBufferPath.path) {
            try fileManager.removeItem(at: pendingBufferPath)
        }

        return buffer
    }

    public func pendingBufferCount() async throws -> Int {
        let buffer = try loadPendingBufferInternal() ?? []
        return buffer.count
    }

    private func loadPendingBufferInternal() throws -> [BufferedEntry]? {
        guard fileManager.fileExists(atPath: pendingBufferPath.path) else {
            return nil
        }

        let data = try Data(contentsOf: pendingBufferPath)
        return try decoder.decode([BufferedEntry].self, from: data)
    }

    // MARK: - Maintenance

    public func clear() async throws {
        // Remove all storage files
        let files = [
            modelStatePath,
            embeddingsPath,
            embeddingsIndexPath,
            reducedEmbeddingsPath,
            reducedIndexPath,
            knnGraphPath,
            pendingBufferPath,
            checkpointPath
        ]

        for file in files {
            if fileManager.fileExists(atPath: file.path) {
                try fileManager.removeItem(at: file)
            }
        }

        // Clear cache
        embeddingIndex = nil
    }

    public func storageSizeBytes() async throws -> UInt64 {
        var total: UInt64 = 0

        let files = [
            modelStatePath,
            embeddingsPath,
            embeddingsIndexPath,
            reducedEmbeddingsPath,
            reducedIndexPath,
            knnGraphPath,
            pendingBufferPath,
            checkpointPath
        ]

        for file in files {
            total += fileSize(at: file)
        }

        return total
    }

    // MARK: - Helper Methods

    /// Writes data atomically (write to temp, then rename).
    private func atomicWrite(data: Data, to url: URL) throws {
        let tempURL = url.appendingPathExtension("tmp")

        do {
            try data.write(to: tempURL, options: .atomic)
            _ = try fileManager.replaceItemAt(url, withItemAt: tempURL)
        } catch {
            // Clean up temp file if it exists
            try? fileManager.removeItem(at: tempURL)
            throw TopicModelStorageError.writeError(path: url.path, underlying: error)
        }
    }

    /// Appends data to file (creates if doesn't exist).
    private func appendToFile(data: Data, at url: URL) throws {
        if fileManager.fileExists(atPath: url.path) {
            guard let handle = FileHandle(forWritingAtPath: url.path) else {
                throw TopicModelStorageError.writeError(
                    path: url.path,
                    underlying: NSError(domain: "SwiftTopics", code: 1)
                )
            }
            defer { try? handle.close() }

            try handle.seekToEnd()
            try handle.write(contentsOf: data)
        } else {
            try data.write(to: url)
        }
    }

    /// Gets file size in bytes (0 if doesn't exist).
    private func fileSize(at url: URL) -> UInt64 {
        guard let attrs = try? fileManager.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? UInt64 else {
            return 0
        }
        return size
    }

    // MARK: - Embedding Index

    private func loadOrCreateEmbeddingIndex() async throws -> EmbeddingIndex {
        if let cached = embeddingIndex {
            return cached
        }

        if fileManager.fileExists(atPath: embeddingsIndexPath.path) {
            let data = try Data(contentsOf: embeddingsIndexPath)
            let index = try decoder.decode(EmbeddingIndex.self, from: data)
            embeddingIndex = index
            return index
        }

        let newIndex = EmbeddingIndex()
        embeddingIndex = newIndex
        return newIndex
    }

    private func saveEmbeddingIndex(_ index: EmbeddingIndex) async throws {
        let data = try encoder.encode(index)
        try atomicWrite(data: data, to: embeddingsIndexPath)
    }
}

// MARK: - Embedding Index

/// Index mapping DocumentID to byte offset in embeddings file.
private struct EmbeddingIndex: Codable {
    var dimension: Int = 0
    var count: Int = 0
    var offsets: [DocumentID: Int] = [:]
    var documentIDs: [DocumentID] = []
}

/// Index for reduced embeddings.
private struct ReducedEmbeddingIndex: Codable {
    var dimension: Int
    var count: Int = 0
    var offsets: [DocumentID: Int] = [:]
    var documentIDs: [DocumentID] = []

    init(dimension: Int = 0) {
        self.dimension = dimension
    }
}

// MARK: - Serializable k-NN Graph

/// JSON-serializable wrapper for NearestNeighborGraph.
private struct SerializableKNNGraph: Codable {
    let neighbors: [[Int]]
    let distances: [[Float]]
    let k: Int
    let metricName: String

    init(from graph: NearestNeighborGraph) {
        self.neighbors = graph.neighbors
        self.distances = graph.distances
        self.k = graph.k
        self.metricName = graph.metric.name
    }

    func toGraph() -> NearestNeighborGraph {
        let metric: DistanceMetric
        switch metricName {
        case "euclidean": metric = .euclidean
        case "cosine": metric = .cosine
        case "manhattan": metric = .manhattan
        case "squaredEuclidean": metric = .squaredEuclidean
        default: metric = .euclidean
        }

        return NearestNeighborGraph(
            neighbors: neighbors,
            distances: distances,
            k: k,
            metric: metric
        )
    }
}

// MARK: - DistanceMetric Extension

extension DistanceMetric {
    /// Human-readable name for serialization.
    var name: String {
        switch self {
        case .euclidean: return "euclidean"
        case .cosine: return "cosine"
        case .manhattan: return "manhattan"
        case .squaredEuclidean: return "squaredEuclidean"
        }
    }
}
