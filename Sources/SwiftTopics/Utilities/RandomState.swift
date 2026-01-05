// RandomState.swift
// SwiftTopics
//
// Seedable random number generation for reproducibility

import Foundation

// MARK: - Random State

/// A seedable random number generator for reproducible results.
///
/// SwiftTopics algorithms use `RandomState` for all random operations to ensure
/// reproducibility when a seed is provided. This is critical for:
/// - **Testing**: Deterministic test results
/// - **Research**: Reproducible experiments
/// - **Debugging**: Recreating specific scenarios
///
/// ## Usage
/// ```swift
/// var rng = RandomState(seed: 42)
///
/// // Generate random floats
/// let value = rng.nextFloat()                    // [0, 1)
/// let scaled = rng.nextFloat(in: -1.0...1.0)    // [-1, 1]
///
/// // Generate random integers
/// let index = rng.nextInt(upperBound: 100)      // [0, 100)
///
/// // Shuffle arrays
/// var items = [1, 2, 3, 4, 5]
/// rng.shuffle(&items)
///
/// // Sample without replacement
/// let sample = rng.sample(from: items, count: 3)
/// ```
///
/// ## Algorithm
/// Uses the xorshift128+ algorithm which provides:
/// - Period of 2^128 - 1
/// - Good statistical properties
/// - Fast generation (single xor/shift operations)
///
/// ## Thread Safety
/// `RandomState` is NOT thread-safe. Each thread should have its own instance.
/// For concurrent use, wrap in an actor or use separate instances.
public struct RandomState: Sendable {

    // MARK: - State

    /// Internal state (128 bits).
    private var state: (UInt64, UInt64)

    /// The original seed used to initialize this state.
    public let seed: UInt64

    // MARK: - Initialization

    /// Creates a random state with the given seed.
    ///
    /// - Parameter seed: The seed value for reproducibility.
    public init(seed: UInt64) {
        self.seed = seed
        // Initialize state using SplitMix64 to avoid zero states
        var sm = seed
        self.state.0 = Self.splitMix64(&sm)
        self.state.1 = Self.splitMix64(&sm)
    }

    /// Creates a random state with a random seed.
    public init() {
        self.init(seed: UInt64.random(in: 0...UInt64.max))
    }

    /// Creates a random state from a seed or generates a random seed.
    ///
    /// - Parameter seed: Optional seed. If nil, uses a random seed.
    public init(seed: UInt64?) {
        if let seed = seed {
            self.init(seed: seed)
        } else {
            self.init()
        }
    }

    // MARK: - Core Generation

    /// Generates the next random UInt64.
    ///
    /// Uses xorshift128+ algorithm.
    public mutating func next() -> UInt64 {
        var s1 = state.0
        let s0 = state.1
        let result = s0 &+ s1

        state.0 = s0
        s1 ^= s1 << 23
        state.1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5)

        return result
    }

    // MARK: - Float Generation

    /// Generates a random Float in [0, 1).
    @inlinable
    public mutating func nextFloat() -> Float {
        // Use high 24 bits for float mantissa (float has 23 mantissa bits)
        let bits = next() >> 40
        return Float(bits) / Float(1 << 24)
    }

    /// Generates a random Float in the given range.
    ///
    /// - Parameter range: The range (closed).
    /// - Returns: Random value in [range.lowerBound, range.upperBound].
    @inlinable
    public mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        range.lowerBound + nextFloat() * (range.upperBound - range.lowerBound)
    }

    /// Generates a random Float in the given half-open range.
    ///
    /// - Parameter range: The range (half-open).
    /// - Returns: Random value in [range.lowerBound, range.upperBound).
    @inlinable
    public mutating func nextFloat(in range: Range<Float>) -> Float {
        range.lowerBound + nextFloat() * (range.upperBound - range.lowerBound)
    }

    /// Generates a random Double in [0, 1).
    @inlinable
    public mutating func nextDouble() -> Double {
        // Use high 53 bits for double mantissa
        let bits = next() >> 11
        return Double(bits) / Double(1 << 53)
    }

    // MARK: - Integer Generation

    /// Generates a random Int in [0, upperBound).
    ///
    /// - Parameter upperBound: Exclusive upper bound (must be positive).
    /// - Returns: Random integer in [0, upperBound).
    @inlinable
    public mutating func nextInt(upperBound: Int) -> Int {
        precondition(upperBound > 0, "upperBound must be positive")
        // Uniform distribution using rejection sampling
        let bound = UInt64(upperBound)
        let threshold = (UInt64.max - bound + 1) % bound

        var result: UInt64
        repeat {
            result = next()
        } while result < threshold

        return Int(result % bound)
    }

    /// Generates a random Int in the given range.
    ///
    /// - Parameter range: The range (closed).
    /// - Returns: Random value in [range.lowerBound, range.upperBound].
    @inlinable
    public mutating func nextInt(in range: ClosedRange<Int>) -> Int {
        range.lowerBound + nextInt(upperBound: range.count)
    }

    /// Generates a random Int in the given half-open range.
    ///
    /// - Parameter range: The range (half-open).
    /// - Returns: Random value in [range.lowerBound, range.upperBound).
    @inlinable
    public mutating func nextInt(in range: Range<Int>) -> Int {
        range.lowerBound + nextInt(upperBound: range.count)
    }

    /// Generates a random Bool.
    @inlinable
    public mutating func nextBool() -> Bool {
        next() & 1 == 1
    }

    // MARK: - Gaussian Distribution

    /// Generates a normally distributed Float.
    ///
    /// Uses the Box-Muller transform for conversion from uniform.
    ///
    /// - Parameters:
    ///   - mean: The mean (μ) of the distribution.
    ///   - standardDeviation: The standard deviation (σ).
    /// - Returns: Normally distributed random value.
    public mutating func nextGaussian(
        mean: Float = 0,
        standardDeviation: Float = 1
    ) -> Float {
        // Box-Muller transform
        let u1 = nextFloat()
        let u2 = nextFloat()

        let z0 = sqrt(-2.0 * log(max(u1, Float.ulpOfOne))) * cos(2.0 * .pi * u2)
        return z0 * standardDeviation + mean
    }

    /// Generates two normally distributed Floats (more efficient than calling once).
    ///
    /// The Box-Muller transform naturally produces pairs.
    public mutating func nextGaussianPair(
        mean: Float = 0,
        standardDeviation: Float = 1
    ) -> (Float, Float) {
        let u1 = nextFloat()
        let u2 = nextFloat()

        let r = sqrt(-2.0 * log(max(u1, Float.ulpOfOne)))
        let theta = 2.0 * .pi * u2

        let z0 = r * cos(theta) * standardDeviation + mean
        let z1 = r * sin(theta) * standardDeviation + mean

        return (z0, z1)
    }

    // MARK: - Array Operations

    /// Shuffles an array in-place using Fisher-Yates algorithm.
    ///
    /// - Parameter array: The array to shuffle.
    public mutating func shuffle<T>(_ array: inout [T]) {
        guard array.count > 1 else { return }

        for i in stride(from: array.count - 1, through: 1, by: -1) {
            let j = nextInt(upperBound: i + 1)
            array.swapAt(i, j)
        }
    }

    /// Returns a shuffled copy of the array.
    ///
    /// - Parameter array: The array to shuffle.
    /// - Returns: A new shuffled array.
    public mutating func shuffled<T>(_ array: [T]) -> [T] {
        var copy = array
        shuffle(&copy)
        return copy
    }

    /// Samples elements without replacement.
    ///
    /// - Parameters:
    ///   - collection: The collection to sample from.
    ///   - count: Number of elements to sample.
    /// - Returns: Array of sampled elements.
    public mutating func sample<C: Collection>(
        from collection: C,
        count: Int
    ) -> [C.Element] where C.Index == Int {
        precondition(count >= 0 && count <= collection.count, "Invalid sample count")

        var indices = Array(collection.indices)
        var result: [C.Element] = []
        result.reserveCapacity(count)

        for _ in 0..<count {
            let i = nextInt(upperBound: indices.count)
            result.append(collection[indices[i]])
            indices.swapAt(i, indices.count - 1)
            indices.removeLast()
        }

        return result
    }

    /// Samples indices without replacement.
    ///
    /// - Parameters:
    ///   - count: Number of indices to sample.
    ///   - upperBound: Exclusive upper bound for indices.
    /// - Returns: Array of sampled indices.
    public mutating func sampleIndices(count: Int, upperBound: Int) -> [Int] {
        precondition(count >= 0 && count <= upperBound, "Invalid sample count")

        var available = Array(0..<upperBound)
        var result: [Int] = []
        result.reserveCapacity(count)

        for _ in 0..<count {
            let i = nextInt(upperBound: available.count)
            result.append(available[i])
            available.swapAt(i, available.count - 1)
            available.removeLast()
        }

        return result
    }

    /// Selects a random element from a collection.
    ///
    /// - Parameter collection: The collection to choose from.
    /// - Returns: A random element, or nil if collection is empty.
    public mutating func choice<C: Collection>(
        from collection: C
    ) -> C.Element? where C.Index == Int {
        guard !collection.isEmpty else { return nil }
        let index = nextInt(upperBound: collection.count)
        return collection[index]
    }

    /// Selects elements with the given weights (with replacement).
    ///
    /// - Parameters:
    ///   - collection: The collection to choose from.
    ///   - weights: Probability weights for each element.
    ///   - count: Number of elements to select.
    /// - Returns: Selected elements.
    public mutating func weightedChoice<C: Collection>(
        from collection: C,
        weights: [Float],
        count: Int
    ) -> [C.Element] where C.Index == Int {
        precondition(collection.count == weights.count, "Weights must match collection size")
        precondition(!collection.isEmpty, "Collection cannot be empty")

        // Build cumulative weights
        var cumulative: [Float] = []
        cumulative.reserveCapacity(weights.count)
        var sum: Float = 0
        for w in weights {
            sum += w
            cumulative.append(sum)
        }

        var result: [C.Element] = []
        result.reserveCapacity(count)

        for _ in 0..<count {
            let r = nextFloat() * sum

            // Binary search for the index
            var lo = 0
            var hi = cumulative.count - 1
            while lo < hi {
                let mid = (lo + hi) / 2
                if cumulative[mid] < r {
                    lo = mid + 1
                } else {
                    hi = mid
                }
            }
            result.append(collection[lo])
        }

        return result
    }

    // MARK: - Random Vectors

    /// Generates a random float array.
    ///
    /// - Parameters:
    ///   - count: Number of elements.
    ///   - range: Value range.
    /// - Returns: Random float array.
    public mutating func randomArray(
        count: Int,
        in range: ClosedRange<Float> = 0...1
    ) -> [Float] {
        (0..<count).map { _ in nextFloat(in: range) }
    }

    /// Generates a random unit vector (uniformly distributed on hypersphere).
    ///
    /// Uses Gaussian normalization method.
    ///
    /// - Parameter dimension: Vector dimension.
    /// - Returns: Unit vector on the dimension-sphere.
    public mutating func randomUnitVector(dimension: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: dimension)
        var normSq: Float = 0

        for i in 0..<dimension {
            let g = nextGaussian()
            vector[i] = g
            normSq += g * g
        }

        let norm = sqrt(normSq)
        guard norm > Float.ulpOfOne else {
            // Extremely unlikely, but handle gracefully
            return randomUnitVector(dimension: dimension)
        }

        for i in 0..<dimension {
            vector[i] /= norm
        }

        return vector
    }

    // MARK: - State Management

    /// Creates a copy of this random state.
    ///
    /// Useful for creating independent streams from a checkpoint.
    public func copy() -> RandomState {
        var copy = RandomState(seed: seed)
        copy.state = state
        return copy
    }

    /// Advances the state by the given number of steps.
    ///
    /// Useful for creating non-overlapping sequences.
    public mutating func advance(by steps: Int) {
        for _ in 0..<steps {
            _ = next()
        }
    }

    /// Creates a child random state with a derived seed.
    ///
    /// Useful for creating independent random states for parallel work.
    ///
    /// - Parameter childIndex: Index to incorporate into child seed.
    /// - Returns: A new independent random state.
    public func child(index childIndex: Int) -> RandomState {
        // Combine original seed with child index for unique but deterministic child seed
        var hasher = Hasher()
        hasher.combine(seed)
        hasher.combine(childIndex)
        let childSeed = UInt64(bitPattern: Int64(hasher.finalize()))
        return RandomState(seed: childSeed)
    }

    // MARK: - Private Helpers

    /// SplitMix64 for seeding.
    private static func splitMix64(_ state: inout UInt64) -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

// MARK: - RandomNumberGenerator Conformance

extension RandomState: RandomNumberGenerator {
    // Protocol conformance is satisfied by the existing next() method
}

// MARK: - Thread-Safe Wrapper

/// A thread-safe wrapper around `RandomState`.
///
/// Use when random state needs to be shared across concurrent tasks.
public actor ThreadSafeRandomState {

    private var state: RandomState

    /// Creates a thread-safe random state with the given seed.
    public init(seed: UInt64) {
        self.state = RandomState(seed: seed)
    }

    /// Creates a thread-safe random state with a random seed.
    public init() {
        self.state = RandomState()
    }

    /// The seed used to initialize this state.
    public var seed: UInt64 {
        state.seed
    }

    /// Generates a random Float in [0, 1).
    public func nextFloat() -> Float {
        state.nextFloat()
    }

    /// Generates a random Int in [0, upperBound).
    public func nextInt(upperBound: Int) -> Int {
        state.nextInt(upperBound: upperBound)
    }

    /// Generates a normally distributed Float.
    public func nextGaussian(mean: Float = 0, standardDeviation: Float = 1) -> Float {
        state.nextGaussian(mean: mean, standardDeviation: standardDeviation)
    }

    /// Shuffles an array.
    public func shuffled<T: Sendable>(_ array: [T]) -> [T] {
        state.shuffled(array)
    }

    /// Samples elements without replacement.
    public func sample<T: Sendable>(from array: [T], count: Int) -> [T] {
        state.sample(from: array, count: count)
    }

    /// Creates a child random state for parallel work.
    public func child(index: Int) -> RandomState {
        state.child(index: index)
    }
}
