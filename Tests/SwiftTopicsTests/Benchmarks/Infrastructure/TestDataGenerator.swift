// TestDataGenerator.swift
// SwiftTopicsTests
//
// Seeded RNG, clustered embeddings, and fixture caching for benchmarks.
// Part of Day 3: Polish

import Foundation
@testable import SwiftTopics

// MARK: - Test Data Generator

/// Generators for reproducible test data.
///
/// Provides methods for generating embeddings with controlled random seeds,
/// enabling reproducible benchmark runs. Also supports fixture caching for
/// large datasets to avoid regeneration overhead.
///
/// ## Usage
/// ```swift
/// // Random embeddings
/// let embeddings = TestDataGenerator.randomEmbeddings(count: 1000, seed: 42)
///
/// // Clustered embeddings (for testing clustering algorithms)
/// let clustered = TestDataGenerator.clusteredEmbeddings(
///     clusterCount: 5,
///     pointsPerCluster: 100,
///     seed: 42
/// )
///
/// // Cached fixtures for large benchmarks
/// let fixture = try await TestDataGenerator.fixture(
///     name: "large_benchmark",
///     count: 10000
/// )
/// ```
public enum TestDataGenerator {

    // MARK: - Fixture Directory

    /// Directory for cached fixture files.
    public static let fixtureDirectory: URL = {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return cwd.appendingPathComponent("BenchmarkFixtures")
    }()

    // MARK: - Random Embeddings

    /// Generates random embeddings with fixed seed.
    ///
    /// - Parameters:
    ///   - count: Number of embeddings to generate.
    ///   - dimension: Embedding dimension (default: 384).
    ///   - seed: Random seed for reproducibility.
    /// - Returns: Array of random embeddings.
    public static func randomEmbeddings(
        count: Int,
        dimension: Int = 384,
        seed: UInt64 = 42
    ) -> [Embedding] {
        var generator = SeededRNG(seed: seed)
        return (0..<count).map { _ in
            let vector = (0..<dimension).map { _ in
                Float.random(in: -1...1, using: &generator)
            }
            return Embedding(vector: vector)
        }
    }

    // MARK: - Clustered Embeddings

    /// Generates clustered embeddings with known structure.
    ///
    /// Creates embeddings grouped around cluster centers, useful for
    /// testing clustering algorithms where ground truth is known.
    ///
    /// - Parameters:
    ///   - clusterCount: Number of clusters.
    ///   - pointsPerCluster: Points in each cluster.
    ///   - dimension: Embedding dimension (default: 384).
    ///   - clusterSpread: Standard deviation within clusters (default: 0.05).
    ///   - seed: Random seed for reproducibility.
    /// - Returns: Array of clustered embeddings.
    public static func clusteredEmbeddings(
        clusterCount: Int,
        pointsPerCluster: Int,
        dimension: Int = 384,
        clusterSpread: Float = 0.05,
        seed: UInt64 = 42
    ) -> [Embedding] {
        var generator = SeededRNG(seed: seed)
        var embeddings: [Embedding] = []
        embeddings.reserveCapacity(clusterCount * pointsPerCluster)

        // Generate cluster centers
        let centers = (0..<clusterCount).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1, using: &generator) }
        }

        // Generate points around each center
        for center in centers {
            for _ in 0..<pointsPerCluster {
                let point = center.map { value in
                    value + gaussianRandom(stddev: clusterSpread, using: &generator)
                }
                embeddings.append(Embedding(vector: point))
            }
        }

        return embeddings
    }

    // MARK: - Fixtures

    /// Loads or generates fixture data for large benchmarks.
    ///
    /// Caches to disk for fast reloading on subsequent runs.
    /// Fixture files are stored in `./BenchmarkFixtures/`.
    ///
    /// - Parameters:
    ///   - name: Base name for the fixture file.
    ///   - count: Number of embeddings.
    ///   - dimension: Embedding dimension (default: 384).
    ///   - generator: Generator function if fixture doesn't exist.
    /// - Returns: Array of embeddings.
    /// - Throws: If fixture loading fails.
    public static func fixture(
        name: String,
        count: Int,
        dimension: Int = 384,
        generator: (() -> [Embedding])? = nil
    ) async throws -> [Embedding] {
        let filename = "\(name)_\(count)_\(dimension).json"
        let fileURL = fixtureDirectory.appendingPathComponent(filename)

        // Try to load existing fixture
        if FileManager.default.fileExists(atPath: fileURL.path) {
            let data = try Data(contentsOf: fileURL)
            let fixture = try JSONDecoder().decode(EmbeddingFixture.self, from: data)
            return fixture.embeddings
        }

        // Generate new fixture
        let embeddings: [Embedding]
        if let gen = generator {
            embeddings = gen()
        } else {
            // Default to random embeddings
            embeddings = randomEmbeddings(count: count, dimension: dimension)
        }

        // Save fixture for future use
        try await saveFixture(embeddings: embeddings, to: fileURL)

        return embeddings
    }

    /// Saves embeddings to a fixture file.
    private static func saveFixture(embeddings: [Embedding], to url: URL) async throws {
        // Ensure directory exists
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Save as JSON
        let fixture = EmbeddingFixture(embeddings: embeddings)
        let data = try JSONEncoder().encode(fixture)
        try data.write(to: url)
    }

    /// Clears all cached fixtures.
    public static func clearFixtures() throws {
        if FileManager.default.fileExists(atPath: fixtureDirectory.path) {
            try FileManager.default.removeItem(at: fixtureDirectory)
        }
    }
}

// MARK: - Fixture Storage

/// JSON-serializable wrapper for embedding fixtures.
private struct EmbeddingFixture: Codable {
    let embeddings: [Embedding]
}

// MARK: - Seeded Random Number Generator

/// A seeded random number generator for reproducible results.
///
/// Uses xorshift64* algorithm for fast, high-quality random numbers.
public struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    /// Creates a seeded random generator.
    public init(seed: UInt64) {
        // Ensure non-zero seed
        self.state = seed == 0 ? 0xDEADBEEF : seed
    }

    /// Generates the next random value.
    public mutating func next() -> UInt64 {
        // xorshift64* algorithm
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return state &* 0x2545F4914F6CDD1D
    }
}

// MARK: - Gaussian Random

/// Generates a Gaussian (normal) random number using Box-Muller transform.
private func gaussianRandom<G: RandomNumberGenerator>(
    mean: Float = 0,
    stddev: Float = 1,
    using generator: inout G
) -> Float {
    // Box-Muller transform
    let u1 = Float.random(in: Float.ulpOfOne...1, using: &generator)
    let u2 = Float.random(in: 0...1, using: &generator)

    let z = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    return mean + stddev * z
}
