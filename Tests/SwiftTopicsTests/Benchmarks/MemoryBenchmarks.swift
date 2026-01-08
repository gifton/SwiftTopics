// MemoryBenchmarks.swift
// SwiftTopicsTests
//
// Memory profiling benchmarks for HDBSCAN and UMAP
// Part of Phase 6: Memory Profiling

import XCTest
import Darwin
@testable import SwiftTopics

/// Memory profiling benchmarks for CPU vs GPU paths.
///
/// Measures memory usage during HDBSCAN clustering and UMAP reduction,
/// comparing CPU and GPU memory footprints.
///
/// ## Memory Measurements
///
/// - **Process Memory**: Measured via Darwin's `mach_task_basic_info`
/// - **GPU Buffer Estimation**: Calculated based on data dimensions
/// - **Peak Memory**: Delta between pre/post operation memory
///
/// ## Important Notes
///
/// Process memory measurements include:
/// - Heap allocations (Swift objects, arrays)
/// - Stack memory (thread-local)
/// - Mapped memory (including GPU shared buffers on Apple Silicon)
///
/// GPU-specific memory is estimated based on buffer sizes required
/// for kernels (distance matrices, k-NN results, etc.).
final class MemoryBenchmarks: XCTestCase {

    // MARK: - Configuration

    /// Memory benchmark sizes to test.
    static let memorySizes: [Int] = [500, 1000, 2000, 5000]

    /// HDBSCAN configuration for benchmarks.
    let hdbscanConfig = HDBSCANConfiguration(
        minClusterSize: 5,
        minSamples: 3,
        logTiming: false
    )

    // MARK: - HDBSCAN Memory Tests

    /// Tests memory usage during HDBSCAN clustering.
    func testHDBSCANMemoryUsage() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  HDBSCAN Memory Profiling")
        print("═══════════════════════════════════════════════════════════════════")

        for size in Self.memorySizes.prefix(3) {  // Limit to avoid very long tests
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, size / 100),
                pointsPerCluster: 100,
                seed: UInt64(size)
            ).prefix(size)

            // CPU path memory
            let cpuMemory = try await measurePeakMemory {
                let engine = try await HDBSCANEngine(configuration: self.hdbscanConfig, gpuContext: nil)
                _ = try await engine.fit(Array(embeddings))
            }

            // GPU path memory
            let gpuMemory = try await measurePeakMemory {
                let engine = try await HDBSCANEngine(configuration: self.hdbscanConfig, gpuContext: gpuContext)
                _ = try await engine.fit(Array(embeddings))
            }

            // Estimated GPU buffer memory
            let gpuBufferEstimate = estimateHDBSCANGPUBuffers(pointCount: size, dimension: 384)

            print("  \(size) points:")
            print("    CPU: Baseline=\(formatMemory(cpuMemory.baseline)), Peak=\(formatMemory(cpuMemory.peak)), Delta=\(formatMemory(cpuMemory.delta))")
            print("    GPU: Baseline=\(formatMemory(gpuMemory.baseline)), Peak=\(formatMemory(gpuMemory.peak)), Delta=\(formatMemory(gpuMemory.delta))")
            print("    GPU Buffer Estimate: \(formatMemory(gpuBufferEstimate))")
        }
    }

    // MARK: - UMAP Memory Tests

    /// Tests memory usage during UMAP reduction.
    func testUMAPMemoryUsage() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard gpuContext != nil else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  UMAP Memory Profiling")
        print("═══════════════════════════════════════════════════════════════════")

        let sizes = [500, 1000, 2000]

        for size in sizes {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, size / 100),
                pointsPerCluster: 100,
                seed: UInt64(size)
            ).prefix(size)

            // CPU path memory
            let cpuConfig = UMAPConfiguration(
                nNeighbors: 15,
                minDist: 0.1,
                nEpochs: 50  // Reduced for benchmarking
            )

            let cpuMemory = try await measurePeakMemory {
                let reducer = UMAPReducer(configuration: cpuConfig, nComponents: 15, gpuContext: nil)
                _ = try await reducer.fitTransform(Array(embeddings))
            }

            // GPU path memory
            let gpuMemory = try await measurePeakMemory {
                let reducer = UMAPReducer(configuration: cpuConfig, nComponents: 15, gpuContext: gpuContext)
                _ = try await reducer.fitTransform(Array(embeddings))
            }

            // Estimated GPU buffer memory
            let gpuBufferEstimate = estimateUMAPGPUBuffers(
                pointCount: size,
                dimension: 384,
                nNeighbors: 15
            )

            print("  \(size) points:")
            print("    CPU: Baseline=\(formatMemory(cpuMemory.baseline)), Peak=\(formatMemory(cpuMemory.peak)), Delta=\(formatMemory(cpuMemory.delta))")
            print("    GPU: Baseline=\(formatMemory(gpuMemory.baseline)), Peak=\(formatMemory(gpuMemory.peak)), Delta=\(formatMemory(gpuMemory.delta))")
            print("    GPU Buffer Estimate: \(formatMemory(gpuBufferEstimate))")
        }
    }

    // MARK: - GPU Buffer Allocation Tests

    /// Tests GPU buffer allocation patterns.
    func testGPUBufferAllocation() async throws {
        let gpuContext = await TopicsGPUContext.create(allowCPUFallback: false)
        guard let gpu = gpuContext else {
            throw XCTSkip("GPU not available")
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  GPU Buffer Allocation Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        let sizes = [500, 1000, 2000]

        for size in sizes {
            let embeddings = TestDataGenerator.randomEmbeddings(
                count: size,
                dimension: 384,
                seed: UInt64(size)
            )

            // Measure memory before GPU operations
            let beforeMemory = currentMemoryUsage()

            // Perform k-NN computation (primary GPU operation)
            _ = try await gpu.computeBatchKNN(embeddings, k: 15)

            // Measure memory after GPU operations
            let afterMemory = currentMemoryUsage()

            // Theoretical buffer sizes
            let distanceMatrixSize = UInt64(size * size) * 4  // n² × sizeof(Float)
            let knnIndicesSize = UInt64(size * 15) * 4  // n × k × sizeof(Int32)
            let knnDistancesSize = UInt64(size * 15) * 4  // n × k × sizeof(Float)
            let inputBufferSize = UInt64(size * 384) * 4  // n × d × sizeof(Float)

            let totalEstimated = distanceMatrixSize + knnIndicesSize + knnDistancesSize + inputBufferSize

            print("  \(size) points:")
            print("    Process memory delta: \(formatMemory(afterMemory > beforeMemory ? afterMemory - beforeMemory : 0))")
            print("    Estimated GPU buffers:")
            print("      Distance matrix (n²): \(formatMemory(distanceMatrixSize))")
            print("      k-NN indices (n×k):   \(formatMemory(knnIndicesSize))")
            print("      k-NN distances (n×k): \(formatMemory(knnDistancesSize))")
            print("      Input buffer (n×d):   \(formatMemory(inputBufferSize))")
            print("      Total estimated:      \(formatMemory(totalEstimated))")
        }
    }

    // MARK: - Memory Scaling Test

    /// Tests how memory usage scales with dataset size.
    func testMemoryScaling() async throws {
        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Memory Scaling Analysis")
        print("═══════════════════════════════════════════════════════════════════")

        let scales = [100, 250, 500, 750, 1000]
        var memoryData: [(points: Int, memory: UInt64)] = []

        for size in scales {
            let embeddings = TestDataGenerator.clusteredEmbeddings(
                clusterCount: max(5, size / 50),
                pointsPerCluster: 50,
                seed: UInt64(size)
            ).prefix(size)

            let memory = try await measurePeakMemory {
                let engine = try await HDBSCANEngine(configuration: self.hdbscanConfig, gpuContext: nil)
                _ = try await engine.fit(Array(embeddings))
            }

            memoryData.append((size, memory.delta))
            print("  \(size) points: \(formatMemory(memory.delta))")
        }

        // Calculate scaling exponent
        let scalingExponent = calculateMemoryScalingExponent(memoryData)
        print("\n  Memory scaling exponent: \(String(format: "%.2f", scalingExponent))")
        print("  (Expected ~2.0 for O(n²) distance matrix, ~1.0 for O(n) output)")
    }

    // MARK: - Quick Memory Check

    /// Quick memory check for CI/development.
    func testQuickMemoryCheck() async throws {
        let size = 500
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 5,
            pointsPerCluster: 100,
            seed: 42
        )

        let memory = try await measurePeakMemory {
            let engine = try await HDBSCANEngine(configuration: self.hdbscanConfig, gpuContext: nil)
            _ = try await engine.fit(embeddings)
        }

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Quick Memory Check (\(size) points)")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Baseline: \(formatMemory(memory.baseline))")
        print("  Peak:     \(formatMemory(memory.peak))")
        print("  Delta:    \(formatMemory(memory.delta))")

        // Sanity check: memory usage should be reasonable
        let expectedMaxMemory: UInt64 = 500 * 1024 * 1024  // 500 MB
        XCTAssertLessThan(memory.delta, expectedMaxMemory, "Memory usage should be under 500MB for 500 points")
    }

    // MARK: - Memory Measurement Helpers

    /// Memory measurement result.
    private struct MemoryMeasurement {
        let baseline: UInt64
        let peak: UInt64
        let delta: UInt64
    }

    /// Returns current process memory usage in bytes.
    private func currentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Measures peak memory during an operation.
    private func measurePeakMemory(
        operation: () async throws -> Void
    ) async throws -> MemoryMeasurement {
        // Trigger GC before measurement
        autoreleasepool { }

        let baseline = currentMemoryUsage()
        try await operation()
        let peak = currentMemoryUsage()

        return MemoryMeasurement(
            baseline: baseline,
            peak: peak,
            delta: peak > baseline ? peak - baseline : 0
        )
    }

    // MARK: - GPU Buffer Estimation

    /// Estimates GPU buffer memory for HDBSCAN operations.
    private func estimateHDBSCANGPUBuffers(pointCount: Int, dimension: Int) -> UInt64 {
        let n = UInt64(pointCount)
        let d = UInt64(dimension)
        let floatSize: UInt64 = 4

        // Distance matrix: n × n × 4 bytes
        let distanceMatrix = n * n * floatSize

        // Core distances: n × 4 bytes
        let coreDistances = n * floatSize

        // MST edges: (n-1) × 12 bytes (source, target, weight)
        let mstEdges = (n - 1) * 12

        // Input embeddings: n × d × 4 bytes
        let inputBuffer = n * d * floatSize

        // k-NN intermediate: n × k × 8 bytes (index + distance)
        let knnBuffer = n * 15 * 8  // k=15 typical

        return distanceMatrix + coreDistances + mstEdges + inputBuffer + knnBuffer
    }

    /// Estimates GPU buffer memory for UMAP operations.
    private func estimateUMAPGPUBuffers(pointCount: Int, dimension: Int, nNeighbors: Int) -> UInt64 {
        let n = UInt64(pointCount)
        let d = UInt64(dimension)
        let k = UInt64(nNeighbors)
        let outputDim: UInt64 = 15  // Typical for topic modeling
        let floatSize: UInt64 = 4

        // k-NN graph: n × k × 8 bytes (index + distance)
        let knnGraph = n * k * 8

        // Edge list: ~n × k edges × 12 bytes (source, target, weight)
        let edgeList = n * k * 12

        // Embedding buffer: n × outputDim × 4 bytes
        let embeddingBuffer = n * outputDim * floatSize

        // Gradient buffer: same as embedding
        let gradientBuffer = n * outputDim * floatSize

        // Input buffer: n × d × 4 bytes
        let inputBuffer = n * d * floatSize

        return knnGraph + edgeList + embeddingBuffer + gradientBuffer + inputBuffer
    }

    // MARK: - Formatting Helpers

    /// Formats memory size for display.
    private func formatMemory(_ bytes: UInt64) -> String {
        let kb = Double(bytes) / 1024
        let mb = kb / 1024
        let gb = mb / 1024

        if gb >= 1.0 {
            return String(format: "%.2f GB", gb)
        } else if mb >= 1.0 {
            return String(format: "%.1f MB", mb)
        } else if kb >= 1.0 {
            return String(format: "%.0f KB", kb)
        } else {
            return "\(bytes) B"
        }
    }

    /// Calculates memory scaling exponent using log-log regression.
    private func calculateMemoryScalingExponent(_ data: [(points: Int, memory: UInt64)]) -> Double {
        guard data.count >= 2 else { return 0 }

        // Filter out zero memory values
        let validData = data.filter { $0.memory > 0 }
        guard validData.count >= 2 else { return 0 }

        let logData = validData.map { (log(Double($0.points)), log(Double($0.memory))) }

        let n = Double(logData.count)
        let sumX = logData.reduce(0) { $0 + $1.0 }
        let sumY = logData.reduce(0) { $0 + $1.1 }
        let sumXY = logData.reduce(0) { $0 + $1.0 * $1.1 }
        let sumX2 = logData.reduce(0) { $0 + $1.0 * $1.0 }

        let denominator = n * sumX2 - sumX * sumX
        guard denominator != 0 else { return 0 }

        return (n * sumXY - sumX * sumY) / denominator
    }
}

// MARK: - Per-Component Memory Extension

extension MemoryBenchmarks {

    /// Tests memory usage of individual pipeline components.
    func testComponentMemoryBreakdown() async throws {
        let size = 1000
        let embeddings = TestDataGenerator.clusteredEmbeddings(
            clusterCount: 10,
            pointsPerCluster: 100,
            seed: 42
        )

        print("\n")
        print("═══════════════════════════════════════════════════════════════════")
        print("  Component Memory Breakdown (\(size) points)")
        print("═══════════════════════════════════════════════════════════════════")

        // PCA Reduction
        let pcaMemory = try await measurePeakMemory {
            var pca = PCAReducer(components: 15)
            try await pca.fit(embeddings)
            _ = try await pca.transform(embeddings)
        }
        print("  PCA Reduction:    \(formatMemory(pcaMemory.delta))")

        // HDBSCAN Clustering
        let hdbscanMemory = try await measurePeakMemory {
            let engine = try await HDBSCANEngine(configuration: self.hdbscanConfig, gpuContext: nil)
            _ = try await engine.fit(embeddings)
        }
        print("  HDBSCAN Cluster:  \(formatMemory(hdbscanMemory.delta))")

        // Create reduced embeddings for c-TF-IDF
        var pca = PCAReducer(components: 15)
        try await pca.fit(embeddings)
        let reducedEmbeddings = try await pca.transform(embeddings)

        // Get cluster assignment for c-TF-IDF
        let engine = try await HDBSCANEngine(configuration: hdbscanConfig, gpuContext: nil)
        let assignment = try await engine.fit(reducedEmbeddings)

        // Create simple test documents
        let documents = (0..<size).map { i in
            Document(
                id: DocumentID(),
                content: "Document \(i) with sample text content for topic modeling.",
                metadata: nil
            )
        }

        // c-TF-IDF Representation
        let ctfidfMemory = try await measurePeakMemory {
            let representer = CTFIDFRepresenter(configuration: .default)
            _ = try await representer.represent(documents: documents, assignment: assignment)
        }
        print("  c-TF-IDF:         \(formatMemory(ctfidfMemory.delta))")

        print("\n  Total (sum):      \(formatMemory(pcaMemory.delta + hdbscanMemory.delta + ctfidfMemory.delta))")
    }
}
