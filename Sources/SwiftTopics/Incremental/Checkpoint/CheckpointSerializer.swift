// CheckpointSerializer.swift
// SwiftTopics
//
// Binary serialization helpers for training checkpoint data

import Foundation

// MARK: - Checkpoint Serializer

/// Efficient binary serialization for large checkpoint data.
///
/// Provides optimized binary formats for:
/// - **Embeddings**: 2D Float32 arrays in row-major order
/// - **MST Edges**: Compact edge representation
/// - **Boolean Arrays**: Bit-packed for space efficiency
/// - **Float Arrays**: Raw Float32 buffers
///
/// ## Binary Formats
///
/// ### Embedding Format
/// ```
/// Header: nPoints (Int32), nDimensions (Int32)
/// Data: Float32 values in row-major order [n × d]
/// ```
///
/// ### MST Edge Format
/// ```
/// Header: edgeCount (Int32)
/// Per edge: source (Int32), target (Int32), weight (Float32)
/// ```
///
/// ### Boolean Array Format
/// ```
/// Header: count (Int32)
/// Data: Bit-packed bytes (8 booleans per byte)
/// ```
///
/// ## Thread Safety
/// All methods are stateless and thread-safe.
public enum CheckpointSerializer {

    // MARK: - Embedding Serialization

    /// Serializes a 2D embedding matrix to binary format.
    ///
    /// Format: [nPoints: Int32] [nDimensions: Int32] [values: Float32...]
    ///
    /// - Parameter embedding: The embedding matrix [n × d].
    /// - Returns: Binary data representation.
    public static func serializeEmbedding(_ embedding: [[Float]]) -> Data {
        guard !embedding.isEmpty, !embedding[0].isEmpty else {
            return Data()
        }

        let nPoints = Int32(embedding.count)
        let nDimensions = Int32(embedding[0].count)
        let totalFloats = embedding.count * embedding[0].count

        // Calculate size: 2 Int32s for header + Float32s for data
        var data = Data(capacity: 8 + totalFloats * 4)

        // Write header
        withUnsafeBytes(of: nPoints) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: nDimensions) { data.append(contentsOf: $0) }

        // Write data in row-major order
        for row in embedding {
            for value in row {
                withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
            }
        }

        return data
    }

    /// Deserializes a 2D embedding matrix from binary format.
    ///
    /// - Parameter data: Binary data from `serializeEmbedding`.
    /// - Returns: The embedding matrix, or nil if data is invalid.
    public static func deserializeEmbedding(_ data: Data) -> [[Float]]? {
        guard data.count >= 8 else { return nil }

        // Read header with unaligned-safe copies
        var nPoints: Int32 = 0
        withUnsafeMutableBytes(of: &nPoints) { dest in
            _ = data.copyBytes(to: dest, from: 0..<4)
        }

        var nDimensions: Int32 = 0
        withUnsafeMutableBytes(of: &nDimensions) { dest in
            _ = data.copyBytes(to: dest, from: 4..<(4 + 4))
        }

        guard nPoints > 0, nDimensions > 0 else { return nil }

        let n = Int(nPoints)
        let d = Int(nDimensions)
        let expectedSize = 8 + n * d * MemoryLayout<Float>.size
        guard data.count >= expectedSize else { return nil }

        // Copy floats into a properly aligned buffer
        var flat = [Float](repeating: 0, count: n * d)
        flat.withUnsafeMutableBytes { dest in
            let range = 8..<(8 + n * d * MemoryLayout<Float>.size)
            _ = data.copyBytes(to: dest, from: range)
        }

        // Rebuild rows without additional copies
        var embedding = Array(repeating: [Float](), count: n)
        for i in 0..<n {
            let start = i * d
            let end = start + d
            embedding[i] = Array(flat[start..<end])
        }
        return embedding
    }

    // MARK: - MST Edge Serialization

    /// Serializes MST edges to binary format.
    ///
    /// Format: [edgeCount: Int32] [source: Int32, target: Int32, weight: Float32]...
    ///
    /// - Parameter edges: The MST edges.
    /// - Returns: Binary data representation.
    public static func serializeMSTEdges(_ edges: [MSTEdge]) -> Data {
        let edgeCount = Int32(edges.count)

        // Each edge: 2 Int32s + 1 Float32 = 12 bytes
        var data = Data(capacity: 4 + edges.count * 12)

        // Write header
        withUnsafeBytes(of: edgeCount) { data.append(contentsOf: $0) }

        // Write edges
        for edge in edges {
            let source = Int32(edge.source)
            let target = Int32(edge.target)
            let weight = edge.weight

            withUnsafeBytes(of: source) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: target) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: weight) { data.append(contentsOf: $0) }
        }

        return data
    }

    /// Deserializes MST edges from binary format.
    ///
    /// - Parameter data: Binary data from `serializeMSTEdges`.
    /// - Returns: The MST edges, or nil if data is invalid.
    public static func deserializeMSTEdges(_ data: Data) -> [MSTEdge]? {
        guard data.count >= 4 else { return nil }

        var edgeCount: Int32 = 0
        withUnsafeMutableBytes(of: &edgeCount) { dest in
            _ = data.copyBytes(to: dest, from: 0..<4)
        }
        guard edgeCount >= 0 else { return nil }
        let count = Int(edgeCount)
        let expectedSize = 4 + count * (2 * MemoryLayout<Int32>.size + MemoryLayout<Float>.size)
        guard data.count >= expectedSize else { return nil }

        var edges: [MSTEdge] = []
        edges.reserveCapacity(count)

        var offset = 4
        for _ in 0..<count {
            let sRange = offset..<(offset + 4)
            let tRange = (offset + 4)..<(offset + 8)
            let wRange = (offset + 8)..<(offset + 12)

            var s: Int32 = 0
            withUnsafeMutableBytes(of: &s) { dest in
                _ = data.copyBytes(to: dest, from: sRange)
            }
            var t: Int32 = 0
            withUnsafeMutableBytes(of: &t) { dest in
                _ = data.copyBytes(to: dest, from: tRange)
            }
            var w: Float = 0
            withUnsafeMutableBytes(of: &w) { dest in
                _ = data.copyBytes(to: dest, from: wRange)
            }

            edges.append(MSTEdge(source: Int(s), target: Int(t), weight: w))
            offset += 12
        }

        return edges
    }

    // MARK: - Boolean Array Serialization

    /// Serializes a boolean array to bit-packed format.
    ///
    /// Format: [count: Int32] [bit-packed bytes]
    ///
    /// - Parameter booleans: The boolean array.
    /// - Returns: Binary data representation.
    public static func serializeBoolArray(_ booleans: [Bool]) -> Data {
        let count = Int32(booleans.count)
        let byteCount = (booleans.count + 7) / 8

        var data = Data(capacity: 4 + byteCount)

        // Write header
        withUnsafeBytes(of: count) { data.append(contentsOf: $0) }

        // Bit-pack booleans
        for byteIdx in 0..<byteCount {
            var byte: UInt8 = 0
            for bitIdx in 0..<8 {
                let boolIdx = byteIdx * 8 + bitIdx
                if boolIdx < booleans.count && booleans[boolIdx] {
                    byte |= (1 << bitIdx)
                }
            }
            data.append(byte)
        }

        return data
    }

    /// Deserializes a boolean array from bit-packed format.
    ///
    /// - Parameter data: Binary data from `serializeBoolArray`.
    /// - Returns: The boolean array, or nil if data is invalid.
    public static func deserializeBoolArray(_ data: Data) -> [Bool]? {
        guard data.count >= 4 else { return nil }

        var count: Int32 = 0
        withUnsafeMutableBytes(of: &count) { dest in
            _ = data.copyBytes(to: dest, from: 0..<4)
        }
        guard count >= 0 else { return nil }
        let c = Int(count)
        let byteCount = (c + 7) / 8
        guard data.count >= 4 + byteCount else { return nil }

        var booleans = [Bool]()
        booleans.reserveCapacity(c)
        for byteIdx in 0..<byteCount {
            let byte = data[4 + byteIdx]
            for bitIdx in 0..<8 {
                let idx = byteIdx * 8 + bitIdx
                if idx >= c { break }
                booleans.append((byte & (1 << bitIdx)) != 0)
            }
        }
        return booleans
    }

    // MARK: - Float Array Serialization

    /// Serializes a float array to binary format.
    ///
    /// Format: [count: Int32] [values: Float32...]
    ///
    /// - Parameter floats: The float array.
    /// - Returns: Binary data representation.
    public static func serializeFloatArray(_ floats: [Float]) -> Data {
        let count = Int32(floats.count)

        var data = Data(capacity: 4 + floats.count * 4)

        // Write header
        withUnsafeBytes(of: count) { data.append(contentsOf: $0) }

        // Write values
        for value in floats {
            withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
        }

        return data
    }

    /// Deserializes a float array from binary format.
    ///
    /// - Parameter data: Binary data from `serializeFloatArray`.
    /// - Returns: The float array, or nil if data is invalid.
    public static func deserializeFloatArray(_ data: Data) -> [Float]? {
        guard data.count >= 4 else { return nil }

        var count: Int32 = 0
        withUnsafeMutableBytes(of: &count) { dest in
            _ = data.copyBytes(to: dest, from: 0..<4)
        }
        guard count >= 0 else { return nil }
        let c = Int(count)
        let expectedSize = 4 + c * MemoryLayout<Float>.size
        guard data.count >= expectedSize else { return nil }

        var floats = [Float](repeating: 0, count: c)
        floats.withUnsafeMutableBytes { dest in
            let range = 4..<(4 + c * MemoryLayout<Float>.size)
            _ = data.copyBytes(to: dest, from: range)
        }
        return floats
    }

    // MARK: - Core Distance Serialization

    /// Serializes core distances (same as float array with semantic naming).
    public static func serializeCoreDistances(_ distances: [Float]) -> Data {
        serializeFloatArray(distances)
    }

    /// Deserializes core distances.
    public static func deserializeCoreDistances(_ data: Data) -> [Float]? {
        deserializeFloatArray(data)
    }

    // MARK: - UMAP State Serialization

    /// Complete UMAP checkpoint state for serialization.
    public struct UMAPCheckpointState: Codable, Sendable {
        /// Current epoch (0-indexed).
        public let currentEpoch: Int

        /// Total epochs.
        public let totalEpochs: Int

        /// Sampling schedule state for accurate resumption.
        public let samplingScheduleState: [Float]

        /// Random seed used (for reproducibility).
        public let randomSeed: UInt64?

        /// Creates UMAP checkpoint state.
        public init(
            currentEpoch: Int,
            totalEpochs: Int,
            samplingScheduleState: [Float],
            randomSeed: UInt64? = nil
        ) {
            self.currentEpoch = currentEpoch
            self.totalEpochs = totalEpochs
            self.samplingScheduleState = samplingScheduleState
            self.randomSeed = randomSeed
        }
    }

    /// Serializes UMAP state metadata to JSON.
    ///
    /// Note: The embedding itself is stored separately in binary format.
    public static func serializeUMAPState(_ state: UMAPCheckpointState) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        return try encoder.encode(state)
    }

    /// Deserializes UMAP state metadata from JSON.
    public static func deserializeUMAPState(_ data: Data) throws -> UMAPCheckpointState {
        let decoder = JSONDecoder()
        return try decoder.decode(UMAPCheckpointState.self, from: data)
    }

    // MARK: - MST State Serialization

    /// Complete MST checkpoint state for serialization.
    public struct MSTCheckpointState: Codable, Sendable {
        /// Number of points in the graph.
        public let pointCount: Int

        /// Edges completed so far (stored separately in binary).
        public let edgesCompleted: Int

        /// Creates MST checkpoint state.
        public init(pointCount: Int, edgesCompleted: Int) {
            self.pointCount = pointCount
            self.edgesCompleted = edgesCompleted
        }
    }

    /// Serializes MST state metadata to JSON.
    ///
    /// Note: The edges and inMST arrays are stored separately in binary format.
    public static func serializeMSTState(_ state: MSTCheckpointState) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        return try encoder.encode(state)
    }

    /// Deserializes MST state metadata from JSON.
    public static func deserializeMSTState(_ data: Data) throws -> MSTCheckpointState {
        let decoder = JSONDecoder()
        return try decoder.decode(MSTCheckpointState.self, from: data)
    }
}

// MARK: - File-Based Helpers

extension CheckpointSerializer {

    /// Saves embedding to a file.
    ///
    /// - Parameters:
    ///   - embedding: The embedding to save.
    ///   - url: The file URL.
    /// - Throws: File system errors.
    public static func saveEmbedding(_ embedding: [[Float]], to url: URL) throws {
        let data = serializeEmbedding(embedding)
        try data.write(to: url, options: .atomic)
    }

    /// Loads embedding from a file.
    ///
    /// - Parameter url: The file URL.
    /// - Returns: The embedding, or nil if file doesn't exist or is invalid.
    public static func loadEmbedding(from url: URL) throws -> [[Float]]? {
        let data = try Data(contentsOf: url)
        return deserializeEmbedding(data)
    }

    /// Saves MST edges to a file.
    ///
    /// - Parameters:
    ///   - edges: The edges to save.
    ///   - url: The file URL.
    /// - Throws: File system errors.
    public static func saveMSTEdges(_ edges: [MSTEdge], to url: URL) throws {
        let data = serializeMSTEdges(edges)
        try data.write(to: url, options: .atomic)
    }

    /// Loads MST edges from a file.
    ///
    /// - Parameter url: The file URL.
    /// - Returns: The edges, or nil if file doesn't exist or is invalid.
    public static func loadMSTEdges(from url: URL) throws -> [MSTEdge]? {
        let data = try Data(contentsOf: url)
        return deserializeMSTEdges(data)
    }

    /// Saves boolean array to a file.
    ///
    /// - Parameters:
    ///   - booleans: The boolean array to save.
    ///   - url: The file URL.
    /// - Throws: File system errors.
    public static func saveBoolArray(_ booleans: [Bool], to url: URL) throws {
        let data = serializeBoolArray(booleans)
        try data.write(to: url, options: .atomic)
    }

    /// Loads boolean array from a file.
    ///
    /// - Parameter url: The file URL.
    /// - Returns: The boolean array, or nil if file doesn't exist or is invalid.
    public static func loadBoolArray(from url: URL) throws -> [Bool]? {
        let data = try Data(contentsOf: url)
        return deserializeBoolArray(data)
    }

    /// Saves float array to a file.
    ///
    /// - Parameters:
    ///   - floats: The float array to save.
    ///   - url: The file URL.
    /// - Throws: File system errors.
    public static func saveFloatArray(_ floats: [Float], to url: URL) throws {
        let data = serializeFloatArray(floats)
        try data.write(to: url, options: .atomic)
    }

    /// Loads float array from a file.
    ///
    /// - Parameter url: The file URL.
    /// - Returns: The float array, or nil if file doesn't exist or is invalid.
    public static func loadFloatArray(from url: URL) throws -> [Float]? {
        let data = try Data(contentsOf: url)
        return deserializeFloatArray(data)
    }
}
