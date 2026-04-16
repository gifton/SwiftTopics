# SwiftTopics Benchmark Suite - Implementation Plan

## Goal

Create a comprehensive benchmark suite that measures and reports the performance gains from GPU acceleration via VectorAccelerate integration. The suite should provide:

- **CPU vs GPU comparisons** at realistic scales
- **Speedup ratios** (Nx faster) for each operation
- **Scaling analysis** showing how performance changes with dataset size
- **Per-phase timing breakdowns** using the new instrumentation from Phase 3
- **Statistical rigor** with multiple runs and percentile reporting
- **Formatted output** suitable for documentation and CI reporting

---

## Current State

### Existing Benchmarks
- **VectorAccelerate/SwiftTopicsBenchmarks.swift**: Kernel-level tests (50-200 docs)
- **SwiftTopics/GPUHDBSCANTests.swift**: GPU-only timing (100-1000 points)
- **SwiftTopics/GPUUMAPTests.swift**: GPU-only timing (300-1000 points)

### Gaps
- No CPU vs GPU comparison at SwiftTopics level
- Small test scales (max 200 for CPU comparisons)
- No statistical aggregation
- No consolidated benchmark runner
- Timing breakdown not utilized

---

## Phase Overview

| Phase | Name | Description | Deliverable |
|-------|------|-------------|-------------|
| 1 | Infrastructure | Benchmark harness, utilities, output formatting | `BenchmarkHarness.swift` |
| 2 | HDBSCAN Benchmarks | CPU vs GPU for clustering | `HDBSCANBenchmarks.swift` |
| 3 | UMAP Benchmarks | CPU vs GPU for dimensionality reduction | `UMAPBenchmarks.swift` |
| 4 | End-to-End Pipeline | Full TopicModel CPU vs GPU | `PipelineBenchmarks.swift` |
| 5 | Scaling Analysis | Performance vs dataset size curves | `ScalingBenchmarks.swift` |
| 6 | Memory Profiling | Memory usage comparison | `MemoryBenchmarks.swift` |
| 7 | Threshold Tuning | Find optimal GPU threshold | `ThresholdAnalysis.swift` |
| 8 | CI Integration | Automated benchmark runs | GitHub Actions workflow |
| 9 | Documentation | README updates, benchmark results | `BENCHMARK_RESULTS.md` |
| 10 | Validation | Cross-check against VectorAccelerate benchmarks | Validation tests |

---

## Phase Details

### Phase 1: Infrastructure

**Goal**: Create reusable benchmark harness with statistical measurement and formatted output.

---

#### Implementation Strategy (3-Day Plan)

| Day | Focus | Files | Lines | Deliverable |
|-----|-------|-------|-------|-------------|
| **Day 1** | Foundation | `BenchmarkTypes.swift`, `StatisticsCalculator.swift`, `BenchmarkHarness.swift` | ~500 | Core measurement working |
| **Day 2** | Core Features | `Benchmark.swift`, `BenchmarkReporter.swift`, basic test | ~450 | DSL + console output |
| **Day 3** | Polish | `HardwareInfo.swift`, `TestDataGenerator.swift`, `BenchmarkStorage.swift`, full test | ~500 | Complete infrastructure |

**Total**: ~1,450-1,600 lines across 8 files + tests

##### Day 1: Foundation (~500 lines)

**Goal**: Get basic timing measurement working end-to-end.

| Step | File | Lines | Complexity | Description |
|------|------|-------|------------|-------------|
| 1 | `BenchmarkTypes.swift` | ~200 | 🟢 Low | Core data structures: `BenchmarkConfiguration`, `TimingStatistics`, `BenchmarkResult`, `ComparisonResult`, `BenchmarkError` |
| 2 | `StatisticsCalculator.swift` | ~120 | 🟡 Medium | Statistical computations: mean, median, stddev, variance, percentiles (p25, p75, p95, p99), min, max |
| 3 | `BenchmarkHarness.swift` | ~250 | 🟡 Medium | Core measurement engine using `ContinuousClock`, interleaved execution pattern, warmup handling |

**Day 1 Exit Criteria**:
```swift
// This should work at end of Day 1:
let result = try await BenchmarkHarness.compare(
    name: "Test",
    scale: "100 points",
    configuration: .default,
    baseline: ("A", { try await Task.sleep(for: .milliseconds(10)) }),
    variant: ("B", { try await Task.sleep(for: .milliseconds(5)) })
)
print(result.speedup)  // ~2.0x
```

##### Day 2: Core Features (~450 lines)

**Goal**: Add DSL builder and formatted console output.

| Step | File | Lines | Complexity | Description |
|------|------|-------|------------|-------------|
| 4 | `Benchmark.swift` | ~150 | 🟢 Low | Fluent DSL builder: `.scale()`, `.iterations()`, `.baseline()`, `.variant()`, `.run()` |
| 5 | `BenchmarkReporter.swift` | ~250 | 🟡 Medium | Console tables with box-drawing, JSON output, detailed statistics display |
| 6 | Basic integration test | ~50 | 🟢 Low | Verify DSL → Harness → Reporter pipeline |

**Day 2 Exit Criteria**:
```swift
// This should work at end of Day 2:
let result = try await Benchmark("Test Benchmark")
    .scale("100 points")
    .iterations(10)
    .baseline("Slow") { /* ... */ }
    .variant("Fast") { /* ... */ }
    .runAndReport()

// Console output:
// ╔═══════════════════════════════════════════════════════════╗
// ║  Test Benchmark                                           ║
// ╠═══════════╦══════════════╦══════════════╦════════════════╣
// ║ Scale     ║ Slow (ms)    ║ Fast (ms)    ║ Speedup        ║
// ...
```

##### Day 3: Polish (~500 lines)

**Goal**: Add hardware detection, test data generation, and persistence.

| Step | File | Lines | Complexity | Description |
|------|------|-------|------------|-------------|
| 7 | `HardwareInfo.swift` | ~250 | 🟠 Medium-High | System introspection via `sysctl` and Metal, chip-specific bandwidth lookup |
| 8 | `TestDataGenerator.swift` | ~150 | 🟢 Low | Seeded RNG, clustered embeddings, fixture caching to project directory |
| 9 | `BenchmarkStorage.swift` | ~130 | 🟢 Low | JSON persistence with timestamps, regression detection |
| 10 | Full integration test | ~80 | 🟢 Low | End-to-end test with real embeddings |

**Day 3 Exit Criteria**:
```swift
// Full pipeline working:
let embeddings = TestDataGenerator.clusteredEmbeddings(
    clusterCount: 5, pointsPerCluster: 100, seed: 42
)

let result = try await Benchmark("HDBSCAN")
    .scale("500 points")
    .baseline("CPU") { try await cpuEngine.fit(embeddings) }
    .variant("GPU") { try await gpuEngine.fit(embeddings) }
    .runAndReport()

// Result includes hardware info, saves to JSON
try BenchmarkStorage().save(result)
```

##### Implementation Notes

**Hardware Detection Priority**:
- Primary targets: **M3 Max**, **A18 Pro** (iPhone 16 Pro Max), **M4**
- Secondary: M1/M2 variants for broader compatibility
- Graceful fallback to "unknown" for unrecognized chips

**Memory Bandwidth Lookup Table** (primary chips):
```swift
let bandwidthTable: [String: Double] = [
    // Primary targets
    "Apple M3 Max": 400.0,      // GB/s - 48GB config
    "Apple A18 Pro": 75.0,      // GB/s - iPhone 16 Pro Max
    "Apple M4": 120.0,          // GB/s - base M4
    "Apple M4 Pro": 273.0,      // GB/s
    "Apple M4 Max": 546.0,      // GB/s

    // Secondary (for compatibility)
    "Apple M1": 68.25,
    "Apple M2": 100.0,
    "Apple M3": 100.0,
]
```

**Fixture Storage**:
- Location: `./BenchmarkFixtures/` (project directory)
- Format: `{name}_{count}_{dimension}.json`
- Example: `./BenchmarkFixtures/clustered_1000_384.json`

---

**Design Decisions**:
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Timing | `ContinuousClock` | Modern Swift, type-safe `Duration`, nanosecond precision |
| Statistics | Full suite | Research-grade analysis with all metrics |
| Output | Console + JSON | Human-readable + machine-parseable |
| Architecture | DSL/Builder | Clean syntax, zero runtime overhead |
| Comparison | Paired | Same data, interleaved runs, automatic speedup |
| Test Data | Seeded + Fixtures | Reproducible, fast for large datasets |
| Failure | Exclude + report | No CPU timeout, GPU unavail = fail test |
| Hardware | Detailed | Theoretical bandwidth for bottleneck analysis |
| Persistence | JSON + timestamp | Local history for regression detection |

---

#### 1.1 Core Types

##### BenchmarkConfiguration

```swift
/// Configuration for a benchmark run.
public struct BenchmarkConfiguration: Sendable {
    /// Number of measured iterations (after warmup).
    public let iterations: Int

    /// Number of warmup iterations (excluded from stats).
    public let warmupIterations: Int

    /// Optional timeout per iteration (nil = no timeout).
    public let timeout: Duration?

    /// Whether to run baseline and variant interleaved.
    public let interleaved: Bool

    /// Random seed for reproducible test data.
    public let seed: UInt64

    public static let `default` = BenchmarkConfiguration(
        iterations: 10,
        warmupIterations: 3,
        timeout: nil,
        interleaved: true,
        seed: 42
    )

    public static let quick = BenchmarkConfiguration(
        iterations: 5,
        warmupIterations: 2,
        timeout: nil,
        interleaved: true,
        seed: 42
    )

    public static let thorough = BenchmarkConfiguration(
        iterations: 25,
        warmupIterations: 5,
        timeout: nil,
        interleaved: true,
        seed: 42
    )
}
```

##### TimingStatistics (Full Suite)

```swift
/// Comprehensive timing statistics for a benchmark.
public struct TimingStatistics: Sendable, Codable {
    // MARK: - Core Metrics
    public let sampleCount: Int
    public let failedCount: Int

    // MARK: - Central Tendency
    public let mean: Duration
    public let median: Duration

    // MARK: - Spread
    public let min: Duration
    public let max: Duration
    public let standardDeviation: Duration
    public let variance: Double  // in nanoseconds²

    // MARK: - Percentiles
    public let p25: Duration     // First quartile
    public let p75: Duration     // Third quartile
    public let p95: Duration
    public let p99: Duration

    // MARK: - Derived Metrics
    public var iqr: Duration { p75 - p25 }  // Interquartile range
    public var coefficientOfVariation: Double {
        mean.nanoseconds > 0 ? standardDeviation.nanoseconds / mean.nanoseconds : 0
    }

    // MARK: - Convenience
    public var medianMilliseconds: Double {
        Double(median.components.attoseconds) / 1_000_000_000_000_000
    }
}
```

##### BenchmarkResult

```swift
/// Result of a single benchmark (one operation, one configuration).
public struct BenchmarkResult: Sendable, Codable {
    public let name: String
    public let scale: String  // e.g., "1000 points"
    public let statistics: TimingStatistics
    public let timestamp: Date
    public let hardwareInfo: HardwareInfo

    /// Individual timing samples (for detailed analysis).
    public let samples: [Duration]

    /// Any failures that occurred.
    public let failures: [BenchmarkFailure]
}
```

##### ComparisonResult (Paired Comparison)

```swift
/// Result of comparing baseline vs variant (e.g., CPU vs GPU).
public struct ComparisonResult: Sendable, Codable {
    public let name: String
    public let scale: String

    public let baseline: BenchmarkResult      // e.g., CPU
    public let variant: BenchmarkResult       // e.g., GPU

    // MARK: - Speedup Metrics

    /// Speedup ratio (baseline.median / variant.median).
    /// > 1.0 means variant is faster.
    public var speedup: Double {
        guard variant.statistics.median.nanoseconds > 0 else { return .infinity }
        return Double(baseline.statistics.median.components.attoseconds) /
               Double(variant.statistics.median.components.attoseconds)
    }

    /// Speedup using mean instead of median.
    public var speedupMean: Double {
        guard variant.statistics.mean.nanoseconds > 0 else { return .infinity }
        return Double(baseline.statistics.mean.components.attoseconds) /
               Double(variant.statistics.mean.components.attoseconds)
    }

    /// Whether the variant is statistically significantly faster.
    /// Uses coefficient of variation to determine significance.
    public var isSignificant: Bool {
        // Speedup > 1.2x AND low variance
        speedup > 1.2 &&
        baseline.statistics.coefficientOfVariation < 0.3 &&
        variant.statistics.coefficientOfVariation < 0.3
    }

    public let timestamp: Date
    public let hardwareInfo: HardwareInfo
}
```

---

#### 1.2 Hardware Detection

```swift
/// Detailed hardware information for benchmark context.
public struct HardwareInfo: Sendable, Codable {
    // MARK: - System
    public let osVersion: String           // "macOS 26.0"
    public let swiftVersion: String        // "6.2"
    public let architecture: String        // "arm64"

    // MARK: - CPU
    public let cpuModel: String            // "Apple M2 Pro"
    public let cpuCoreCount: Int           // 12
    public let cpuPerformanceCores: Int    // 8
    public let cpuEfficiencyCores: Int     // 4

    // MARK: - Memory
    public let totalRAM: UInt64            // bytes
    public let availableRAM: UInt64        // bytes at benchmark start

    // MARK: - GPU
    public let gpuName: String             // "Apple M2 Pro"
    public let gpuCoreCount: Int           // 19
    public let gpuMemory: UInt64           // bytes (unified memory)
    public let metalVersion: String        // "Metal 4"

    // MARK: - Theoretical Performance
    public let gpuMemoryBandwidth: Double? // GB/s (estimated)
    public let gpuPeakTFLOPS: Double?      // TFLOPS (estimated)

    /// Captures current hardware info.
    public static func capture() -> HardwareInfo

    /// Formatted summary string.
    public var summary: String {
        """
        Hardware: \(cpuModel), \(totalRAM / 1_073_741_824)GB RAM
        GPU: \(gpuName) (\(gpuCoreCount) cores, \(metalVersion))
        \(osVersion), Swift \(swiftVersion)
        Memory Bandwidth: \(gpuMemoryBandwidth.map { "\($0) GB/s" } ?? "unknown")
        """
    }
}
```

---

#### 1.3 DSL/Builder API

```swift
/// Fluent builder for constructing benchmarks.
public final class Benchmark: @unchecked Sendable {
    private var name: String
    private var scale: String = ""
    private var config: BenchmarkConfiguration = .default
    private var baselineOperation: (@Sendable () async throws -> Void)?
    private var variantOperation: (@Sendable () async throws -> Void)?
    private var baselineLabel: String = "Baseline"
    private var variantLabel: String = "Variant"

    public init(_ name: String) {
        self.name = name
    }

    // MARK: - Configuration

    @discardableResult
    public func scale(_ scale: String) -> Self {
        self.scale = scale
        return self
    }

    @discardableResult
    public func iterations(_ count: Int) -> Self {
        self.config = BenchmarkConfiguration(
            iterations: count,
            warmupIterations: config.warmupIterations,
            timeout: config.timeout,
            interleaved: config.interleaved,
            seed: config.seed
        )
        return self
    }

    @discardableResult
    public func warmup(_ count: Int) -> Self {
        self.config = BenchmarkConfiguration(
            iterations: config.iterations,
            warmupIterations: count,
            timeout: config.timeout,
            interleaved: config.interleaved,
            seed: config.seed
        )
        return self
    }

    @discardableResult
    public func seed(_ seed: UInt64) -> Self {
        self.config = BenchmarkConfiguration(
            iterations: config.iterations,
            warmupIterations: config.warmupIterations,
            timeout: config.timeout,
            interleaved: config.interleaved,
            seed: seed
        )
        return self
    }

    @discardableResult
    public func configuration(_ config: BenchmarkConfiguration) -> Self {
        self.config = config
        return self
    }

    // MARK: - Operations (Paired Comparison)

    @discardableResult
    public func baseline(
        _ label: String = "CPU",
        _ operation: @escaping @Sendable () async throws -> Void
    ) -> Self {
        self.baselineLabel = label
        self.baselineOperation = operation
        return self
    }

    @discardableResult
    public func variant(
        _ label: String = "GPU",
        _ operation: @escaping @Sendable () async throws -> Void
    ) -> Self {
        self.variantLabel = label
        self.variantOperation = operation
        return self
    }

    // MARK: - Execution

    /// Runs the benchmark and returns comparison result.
    public func run() async throws -> ComparisonResult {
        guard let baseline = baselineOperation,
              let variant = variantOperation else {
            throw BenchmarkError.missingOperation
        }

        return try await BenchmarkHarness.compare(
            name: name,
            scale: scale,
            configuration: config,
            baseline: (baselineLabel, baseline),
            variant: (variantLabel, variant)
        )
    }

    /// Runs and prints formatted results.
    public func runAndReport(to reporter: BenchmarkReporter = .console) async throws -> ComparisonResult {
        let result = try await run()
        reporter.report(result)
        return result
    }
}
```

**Usage Example**:

```swift
let result = try await Benchmark("HDBSCAN Clustering")
    .scale("1000 points")
    .iterations(10)
    .warmup(3)
    .baseline("CPU") {
        try await cpuEngine.fit(embeddings)
    }
    .variant("GPU") {
        try await gpuEngine.fit(embeddings)
    }
    .runAndReport()

// result.speedup = 42.3
// result.baseline.statistics.median = 2340ms
// result.variant.statistics.median = 55ms
```

---

#### 1.4 Benchmark Harness (Core Engine)

```swift
/// Core benchmark execution engine.
public enum BenchmarkHarness {

    /// Measures a single operation.
    public static func measure(
        name: String,
        scale: String,
        configuration: BenchmarkConfiguration,
        operation: @escaping @Sendable () async throws -> Void
    ) async throws -> BenchmarkResult

    /// Compares baseline vs variant with paired/interleaved execution.
    public static func compare(
        name: String,
        scale: String,
        configuration: BenchmarkConfiguration,
        baseline: (label: String, operation: @Sendable () async throws -> Void),
        variant: (label: String, operation: @Sendable () async throws -> Void)
    ) async throws -> ComparisonResult

    /// Runs multiple comparisons at different scales.
    public static func scalingSuite(
        name: String,
        scales: [(label: String, setup: @Sendable () async throws -> (baseline: @Sendable () async throws -> Void, variant: @Sendable () async throws -> Void))],
        configuration: BenchmarkConfiguration
    ) async throws -> [ComparisonResult]
}
```

**Paired Execution Strategy**:

```
Interleaved execution pattern for 10 iterations:
  Warmup: B V B V B (3 warmup each)
  Measured: B V B V B V B V B V B V B V B V B V B V (10 each)

Where B = Baseline, V = Variant

This ensures:
- Same thermal conditions for both
- Reduced systematic bias
- Better variance estimation
```

---

#### 1.5 Test Data Generators

```swift
/// Generators for reproducible test data.
public enum TestDataGenerator {

    /// Generates random embeddings with fixed seed.
    /// - Parameters:
    ///   - count: Number of embeddings
    ///   - dimension: Embedding dimension (default: 384)
    ///   - seed: Random seed for reproducibility
    public static func randomEmbeddings(
        count: Int,
        dimension: Int = 384,
        seed: UInt64 = 42
    ) -> [Embedding]

    /// Generates clustered embeddings with known structure.
    /// - Parameters:
    ///   - clusterCount: Number of clusters
    ///   - pointsPerCluster: Points in each cluster
    ///   - dimension: Embedding dimension
    ///   - clusterSpread: Standard deviation within clusters
    ///   - seed: Random seed
    public static func clusteredEmbeddings(
        clusterCount: Int,
        pointsPerCluster: Int,
        dimension: Int = 384,
        clusterSpread: Float = 0.05,
        seed: UInt64 = 42
    ) -> [Embedding]

    /// Loads or generates fixture data for large benchmarks.
    /// Caches to disk for fast reloading.
    public static func fixture(
        name: String,
        count: Int,
        dimension: Int = 384,
        generator: () -> [Embedding]
    ) async throws -> [Embedding]
}
```

---

#### 1.6 Output & Reporting

##### Console Reporter

```swift
/// Formats and prints benchmark results.
public final class BenchmarkReporter {
    public static let console = BenchmarkReporter(format: .console)
    public static let json = BenchmarkReporter(format: .json)

    public enum Format {
        case console  // Pretty tables
        case json     // Machine-readable
    }

    /// Reports a single comparison result.
    public func report(_ result: ComparisonResult)

    /// Reports multiple comparison results as a table.
    public func report(_ results: [ComparisonResult], title: String)

    /// Reports with full statistical breakdown.
    public func reportDetailed(_ result: ComparisonResult)
}
```

**Console Output Format**:

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  HDBSCAN Clustering Benchmark                                                     ║
║  Hardware: Apple M2 Pro, 16GB RAM, Metal 4                                        ║
║  GPU: 19 cores, ~200 GB/s bandwidth                                               ║
╠═══════════════╦═══════════════╦═══════════════╦════════════╦══════════════════════╣
║ Scale         ║ CPU (median)  ║ GPU (median)  ║ Speedup    ║ Status               ║
╠═══════════════╬═══════════════╬═══════════════╬════════════╬══════════════════════╣
║ 500 points    ║     2,340 ms  ║        58 ms  ║    40.3x   ║ ✓ Significant        ║
║ 1,000 points  ║     9,120 ms  ║       142 ms  ║    64.2x   ║ ✓ Significant        ║
║ 2,000 points  ║    38,400 ms  ║       380 ms  ║   101.1x   ║ ✓ Significant        ║
╚═══════════════╩═══════════════╩═══════════════╩════════════╩══════════════════════╝

Detailed Statistics (1,000 points):
  CPU: mean=9,234ms, stddev=312ms, p95=9,680ms, CV=3.4%
  GPU: mean=145ms, stddev=8ms, p95=158ms, CV=5.5%
```

##### JSON Output Format

```json
{
  "benchmark": "HDBSCAN Clustering",
  "timestamp": "2026-01-06T14:32:00Z",
  "hardware": {
    "cpuModel": "Apple M2 Pro",
    "gpuName": "Apple M2 Pro",
    "gpuCoreCount": 19,
    "gpuMemoryBandwidth": 200.0,
    "totalRAM": 17179869184
  },
  "results": [
    {
      "scale": "1000 points",
      "baseline": {
        "label": "CPU",
        "statistics": {
          "median": 9120000000,
          "mean": 9234000000,
          "stddev": 312000000,
          "p95": 9680000000,
          "min": 8890000000,
          "max": 9780000000
        }
      },
      "variant": {
        "label": "GPU",
        "statistics": {
          "median": 142000000,
          "mean": 145000000,
          "stddev": 8000000,
          "p95": 158000000,
          "min": 135000000,
          "max": 162000000
        }
      },
      "speedup": 64.2,
      "isSignificant": true
    }
  ]
}
```

---

#### 1.7 Result Persistence

```swift
/// Persists benchmark results to JSON files.
public final class BenchmarkStorage {
    private let directory: URL

    public init(directory: URL = .benchmarkResults)

    /// Saves a comparison result with timestamp.
    public func save(_ result: ComparisonResult) throws -> URL

    /// Saves multiple results as a benchmark run.
    public func saveRun(_ results: [ComparisonResult], name: String) throws -> URL

    /// Loads previous results for comparison.
    public func loadPrevious(name: String, count: Int = 5) throws -> [ComparisonResult]

    /// Compares current result to previous baseline.
    public func detectRegression(
        current: ComparisonResult,
        threshold: Double = 0.10  // 10% regression threshold
    ) throws -> RegressionStatus
}

public enum RegressionStatus {
    case noBaseline
    case improved(percentage: Double)
    case stable
    case regressed(percentage: Double)
}
```

**File naming convention**:
```
SwiftTopics/
└── BenchmarkResults/
    ├── HDBSCAN_2026-01-06T14-32-00.json
    ├── HDBSCAN_2026-01-05T10-15-30.json
    ├── UMAP_2026-01-06T14-35-00.json
    └── FullPipeline_2026-01-06T15-00-00.json
```

---

#### 1.8 Error Handling

```swift
/// Errors that can occur during benchmarking.
public enum BenchmarkError: Error {
    /// No operation provided to benchmark.
    case missingOperation

    /// GPU is not available (test should fail).
    case gpuUnavailable

    /// All iterations failed.
    case allIterationsFailed([Error])

    /// Fixture file not found and cannot be generated.
    case fixtureNotFound(String)
}

/// Records a failed iteration without stopping the benchmark.
public struct BenchmarkFailure: Sendable, Codable {
    public let iteration: Int
    public let error: String
    public let timestamp: Date
}
```

---

#### 1.9 File Structure (Phase 1)

```
SwiftTopics/
├── Sources/
│   └── SwiftTopics/
│       └── Benchmarking/                    # Public API (if we want it usable outside tests)
│           ├── BenchmarkTypes.swift         # Core types (Configuration, Statistics, Results)
│           ├── HardwareInfo.swift           # Hardware detection
│           └── BenchmarkExports.swift       # Public exports
└── Tests/
    └── SwiftTopicsTests/
        └── Benchmarks/
            ├── Infrastructure/
            │   ├── Benchmark.swift          # DSL builder
            │   ├── BenchmarkHarness.swift   # Core execution engine
            │   ├── BenchmarkReporter.swift  # Console + JSON output
            │   ├── BenchmarkStorage.swift   # Result persistence
            │   ├── TestDataGenerator.swift  # Seeded data generation
            │   └── StatisticsCalculator.swift # Statistical computations
            └── [Future phases...]
```

---

#### 1.10 Implementation Order

| Step | File | Description |
|------|------|-------------|
| 1 | `BenchmarkTypes.swift` | Core types: Configuration, TimingStatistics, Results |
| 2 | `StatisticsCalculator.swift` | Statistical computations from samples |
| 3 | `HardwareInfo.swift` | Hardware detection and reporting |
| 4 | `BenchmarkHarness.swift` | Core measurement and comparison logic |
| 5 | `Benchmark.swift` | DSL builder API |
| 6 | `BenchmarkReporter.swift` | Console and JSON output |
| 7 | `TestDataGenerator.swift` | Seeded generators and fixtures |
| 8 | `BenchmarkStorage.swift` | JSON persistence |
| 9 | Integration test | Verify full pipeline works |

**Dependencies**: None

---

**Output Format Example**:
```
╔═══════════════════════════════════════════════════════════════════╗
║  HDBSCAN Clustering Benchmark                                     ║
╠═══════════╦══════════════╦══════════════╦════════════╦════════════╣
║ Scale     ║ CPU (ms)     ║ GPU (ms)     ║ Speedup    ║ Status     ║
╠═══════════╬══════════════╬══════════════╬════════════╬════════════╣
║ 500 pts   ║    2,340     ║      58      ║   40.3x    ║ ✓ PASS     ║
║ 1,000 pts ║    9,120     ║     142      ║   64.2x    ║ ✓ PASS     ║
║ 2,000 pts ║   38,400     ║     380      ║  101.1x    ║ ✓ PASS     ║
╚═══════════╩══════════════╩══════════════╩════════════╩════════════╝
```

**Dependencies**: None

---

### Phase 2: HDBSCAN Benchmarks

**Goal**: Measure CPU vs GPU speedup for HDBSCAN clustering at various scales.

**Test Points**:
- 100, 250, 500, 1000, 2000, 5000 points
- 384-dimensional embeddings (typical sentence embedding size)

**Measurements**:
- Total clustering time (CPU vs GPU)
- Per-phase breakdown (using `HDBSCANTimingBreakdown`)
  - Core distances
  - Mutual reachability
  - MST construction
  - Hierarchy building
  - Cluster extraction

**Expected Results** (from implementation plan):
- 500 points: ~25x speedup
- 1,000 points: ~40x speedup
- 5,000 points: ~100x speedup

**Dependencies**: Phase 1

---

### Phase 3: UMAP Benchmarks

**Goal**: Measure CPU vs GPU speedup for UMAP dimensionality reduction.

**Test Points**:
- 100, 250, 500, 1000, 2000 points
- Input: 384D → Output: 15D (typical for topic modeling)

**Measurements**:
- Total reduction time (CPU vs GPU)
- Per-phase breakdown (using `UMAPTimingBreakdown`)
  - k-NN graph construction
  - Fuzzy simplicial set
  - Spectral initialization
  - Optimization epochs
- Per-epoch timing

**Expected Results**:
- Single epoch: 5-10x speedup
- Full pipeline: 6-15x speedup (CPU-bound k-NN limits overall gain)

**Dependencies**: Phase 1

---

### Phase 4: End-to-End Pipeline Benchmarks

**Goal**: Measure full `TopicModel.fit()` performance with GPU vs without.

**Test Configuration**:
- Document counts: 100, 500, 1000, 2000
- Realistic document content (varied lengths)
- Pre-computed embeddings (to isolate topic modeling from embedding time)

**Measurements**:
- Total `fit()` time
- Per-stage breakdown via `TopicModelProgress`
  - Embedding (if applicable)
  - Reduction
  - Clustering
  - Representation (c-TF-IDF)
  - Evaluation (coherence)

**Dependencies**: Phases 1, 2, 3

---

### Phase 5: Scaling Analysis

**Goal**: Generate performance scaling curves to understand algorithmic complexity.

**Analysis**:
- Plot CPU time vs N (expect O(N²) for HDBSCAN)
- Plot GPU time vs N (expect better scaling due to parallelism)
- Calculate scaling exponent (log-log regression)
- Identify crossover point (where GPU becomes faster than CPU)

**Output**:
- Scaling data in CSV format
- Markdown table with scaling exponents
- Recommendations for threshold tuning

**Dependencies**: Phases 2, 3, 4

---

### Phase 6: Memory Profiling

**Goal**: Measure memory usage for CPU vs GPU paths.

**Measurements**:
- Peak memory usage during HDBSCAN
- Peak memory usage during UMAP
- GPU buffer memory allocation
- Memory scaling with dataset size

**Tools**:
- Swift `MemoryLayout` for estimation
- GPU buffer pool statistics from `TopicsGPUContext`

**Dependencies**: Phase 1

---

### Phase 7: Threshold Tuning

**Goal**: Empirically determine optimal `gpuMinPointsThreshold`.

**Approach**:
- Run benchmarks at N = 25, 50, 75, 100, 150, 200, 300
- Find crossover point where GPU becomes faster
- Account for GPU initialization overhead
- Test with different embedding dimensions

**Output**:
- Recommended default threshold
- Guidance for different hardware profiles
- Configuration presets

**Dependencies**: Phases 2, 3

---

### Phase 8: CI Integration

**Goal**: Automated benchmark runs on commits/PRs.

**Components**:
- GitHub Actions workflow
- Performance regression detection
- Benchmark result artifacts
- Optional: Benchmark dashboard

**Considerations**:
- CI runners may not have GPU (handle gracefully)
- Cache baseline results for comparison
- Alert on >10% regression

**Dependencies**: All previous phases

---

### Phase 9: Documentation

**Goal**: Update documentation with benchmark results and usage guidance.

**Deliverables**:
- `BENCHMARK_RESULTS.md` - Full benchmark report
- README.md updates - Performance section with real numbers
- GPU tuning guide - How to optimize for different hardware
- Benchmark runner usage - How users can run benchmarks themselves

**Dependencies**: Phases 1-7

---

### Phase 10: Validation

**Goal**: Cross-validate SwiftTopics benchmarks against VectorAccelerate benchmarks.

**Tasks**:
- Compare kernel-level vs engine-level timings
- Verify overhead is reasonable
- Ensure consistency across runs
- Document any discrepancies

**Dependencies**: All previous phases

---

## Success Criteria

### Minimum Viable Benchmarks (Phases 1-4)
- [ ] CPU vs GPU comparison for HDBSCAN at 500, 1000, 2000 points
- [ ] CPU vs GPU comparison for UMAP at 500, 1000, 2000 points
- [ ] Full pipeline benchmark at 500, 1000 documents
- [ ] Formatted output showing speedup ratios

### Complete Benchmark Suite (Phases 1-10)
- [ ] All minimum criteria plus:
- [ ] Scaling analysis with curves
- [ ] Memory profiling
- [ ] Optimized threshold recommendations
- [ ] CI integration
- [ ] Comprehensive documentation

---

## Estimated Effort

| Phase | Complexity | Estimated Time |
|-------|------------|----------------|
| 1 | Medium | Core infrastructure |
| 2 | Medium | HDBSCAN benchmarks |
| 3 | Medium | UMAP benchmarks |
| 4 | Low | Pipeline benchmarks |
| 5 | Medium | Analysis code |
| 6 | Low | Memory profiling |
| 7 | Medium | Threshold tuning |
| 8 | Medium | CI setup |
| 9 | Low | Documentation |
| 10 | Low | Validation |

---

## File Structure

```
SwiftTopics/
├── Tests/
│   └── SwiftTopicsTests/
│       └── Benchmarks/
│           ├── Infrastructure/
│           │   ├── BenchmarkHarness.swift
│           │   ├── BenchmarkResult.swift
│           │   ├── BenchmarkReporter.swift
│           │   └── TestDataGenerators.swift
│           ├── HDBSCANBenchmarks.swift
│           ├── UMAPBenchmarks.swift
│           ├── PipelineBenchmarks.swift
│           ├── ScalingBenchmarks.swift
│           ├── MemoryBenchmarks.swift
│           └── ThresholdAnalysis.swift
├── .github/
│   └── workflows/
│       └── benchmarks.yml
└── Docs/
    └── BENCHMARK_RESULTS.md
```

---

## Notes

- CPU benchmarks may be slow at large scales (>1000 points) - consider timeouts
- GPU availability varies - all tests must handle graceful fallback
- Results will vary by hardware - document test machine specs
- Consider warm-up runs to avoid JIT/cache effects

---

## Next Steps

1. Review this plan and identify any missing phases
2. Prioritize phases (MVP vs nice-to-have)
3. Create detailed specifications for Phase 1
4. Begin implementation
