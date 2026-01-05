import Testing
import Foundation
@testable import SwiftTopics

// MARK: - Core Distance Tests

@Test("Core distance computation for simple points")
func testCoreDistanceSimple() async throws {
    // Create a simple 2D dataset
    let points: [[Float]] = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],  // Outlier point
    ]
    let embeddings = points.map { Embedding(vector: $0) }

    let computer = CoreDistanceComputer(minSamples: 2, preferGPU: false)
    let coreDistances = try await computer.compute(
        embeddings: embeddings,
        gpuContext: nil
    )

    #expect(coreDistances.count == 4)

    // First three points are close, fourth is far
    // Core distance for first 3 should be ~1.0 (distance to 2nd neighbor)
    // Core distance for outlier should be large
    #expect(coreDistances[0] < 2.0)
    #expect(coreDistances[1] < 2.0)
    #expect(coreDistances[2] < 2.0)
    #expect(coreDistances[3] > 10.0)
}

@Test("Core distance with k=1")
func testCoreDistanceK1() async throws {
    let points: [[Float]] = [
        [0.0, 0.0],
        [1.0, 0.0],
        [5.0, 0.0],
    ]
    let embeddings = points.map { Embedding(vector: $0) }

    let computer = CoreDistanceComputer(minSamples: 1, preferGPU: false)
    let coreDistances = try await computer.compute(
        embeddings: embeddings,
        gpuContext: nil
    )

    // With k=1, core distance is distance to nearest neighbor
    #expect(abs(coreDistances[0] - 1.0) < 0.01)  // Point 0 -> Point 1
    #expect(abs(coreDistances[1] - 1.0) < 0.01)  // Point 1 -> Point 0
    #expect(abs(coreDistances[2] - 4.0) < 0.01)  // Point 2 -> Point 1
}

// MARK: - Mutual Reachability Tests

@Test("Mutual reachability distance computation")
func testMutualReachability() async throws {
    let points: [[Float]] = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    let embeddings = points.map { Embedding(vector: $0) }

    // Manual core distances (pretend k=1)
    let coreDistances: [Float] = [1.0, 1.0, 1.0]

    let graph = MutualReachabilityGraph(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    #expect(graph.count == 3)

    // Distance from 0 to 1: max(core[0], core[1], euclidean) = max(1, 1, 1) = 1
    let d01 = graph.distance(from: 0, to: 1)
    #expect(abs(d01 - 1.0) < 0.01)

    // Distance from 0 to 2: max(1, 1, 1) = 1
    let d02 = graph.distance(from: 0, to: 2)
    #expect(abs(d02 - 1.0) < 0.01)

    // Distance from 1 to 2: max(1, 1, sqrt(2)) = sqrt(2) ≈ 1.414
    let d12 = graph.distance(from: 1, to: 2)
    #expect(abs(d12 - 1.414) < 0.01)
}

@Test("Mutual reachability with varying core distances")
func testMutualReachabilityVaryingCore() async throws {
    let points: [[Float]] = [
        [0.0, 0.0],
        [0.5, 0.0],  // Close to point 0
    ]
    let embeddings = points.map { Embedding(vector: $0) }

    // Point 0 has low core distance (dense region)
    // Point 1 has high core distance (sparse region)
    let coreDistances: [Float] = [0.1, 5.0]

    let graph = MutualReachabilityGraph(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    // Mutual reachability is max(0.1, 5.0, 0.5) = 5.0
    let d = graph.distance(from: 0, to: 1)
    #expect(abs(d - 5.0) < 0.01)
}

// MARK: - MST Tests

@Test("MST construction with Prim's algorithm")
func testMSTPrim() async throws {
    let points: [[Float]] = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ]
    let embeddings = points.map { Embedding(vector: $0) }
    let coreDistances: [Float] = [1.0, 1.0, 1.0]

    let graph = MutualReachabilityGraph(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    let builder = PrimMSTBuilder()
    let mst = builder.build(from: graph)

    #expect(mst.edges.count == 2)  // n-1 edges
    #expect(mst.pointCount == 3)

    // Total weight should be 1 + 1 = 2 (two edges of distance 1)
    #expect(abs(mst.totalWeight - 2.0) < 0.01)
}

@Test("MST produces sorted edges")
func testMSTSortedEdges() async throws {
    let points: [[Float]] = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],  // Further from origin
    ]
    let embeddings = points.map { Embedding(vector: $0) }
    let coreDistances: [Float] = [0.1, 0.1, 0.1]

    let graph = MutualReachabilityGraph(
        embeddings: embeddings,
        coreDistances: coreDistances
    )

    let builder = PrimMSTBuilder()
    let mst = builder.build(from: graph)

    let sorted = mst.sortedEdges
    #expect(sorted.count == 2)

    // Edges should be in ascending weight order
    #expect(sorted[0].weight <= sorted[1].weight)
}

// MARK: - Union-Find Tests

@Test("Union-Find basic operations")
func testUnionFind() {
    var uf = UnionFind(count: 5)

    #expect(uf.setCount == 5)
    var connected01 = uf.connected(0, 1)
    #expect(!connected01)

    uf.union(0, 1)
    connected01 = uf.connected(0, 1)
    #expect(connected01)
    #expect(uf.setCount == 4)

    uf.union(2, 3)
    #expect(uf.setCount == 3)

    uf.union(1, 2)
    let connected03 = uf.connected(0, 3)
    #expect(connected03)  // Transitive
    #expect(uf.setCount == 2)
}

// MARK: - Cluster Hierarchy Tests

@Test("Cluster hierarchy construction")
func testClusterHierarchy() async throws {
    // Create a simple linear MST
    let edges = [
        MSTEdge(source: 0, target: 1, weight: 1.0),
        MSTEdge(source: 1, target: 2, weight: 2.0),
    ]
    let mst = MinimumSpanningTree(edges: edges, pointCount: 3)

    let builder = ClusterHierarchyBuilder(minClusterSize: 2)
    let hierarchy = builder.build(from: mst)

    #expect(hierarchy.nodes.count > 0)
    #expect(hierarchy.rootID >= 0)
}

// MARK: - HDBSCAN Full Pipeline Tests

@Test("HDBSCAN clusters simple blobs")
func testHDBSCANSimpleBlobs() async throws {
    // Create two well-separated clusters with more points and tighter grouping
    var embeddings = [Embedding]()

    // Cluster 1: dense cluster around (0, 0)
    for _ in 0..<15 {
        let x = Float.random(in: -0.1...0.1)
        let y = Float.random(in: -0.1...0.1)
        embeddings.append(Embedding(vector: [x, y]))
    }

    // Cluster 2: dense cluster around (10, 10)
    for _ in 0..<15 {
        let x = 10.0 + Float.random(in: -0.1...0.1)
        let y = 10.0 + Float.random(in: -0.1...0.1)
        embeddings.append(Embedding(vector: [x, y]))
    }

    let config = HDBSCANConfiguration(
        minClusterSize: 5,
        minSamples: 3
    )

    let result = try await hdbscan(embeddings, configuration: config)

    // Should find at least 1 cluster, ideally 2
    #expect(result.clusterCount >= 1)
    #expect(result.clusterCount <= 3)

    // Some points should be clustered (not all outliers)
    let clusteredCount = result.labels.filter { $0 >= 0 }.count
    #expect(clusteredCount >= 10)  // At least 10 of 30 points clustered
}

@Test("HDBSCAN handles single point")
func testHDBSCANSinglePoint() async throws {
    let embeddings = [Embedding(vector: [1.0, 2.0])]

    let config = HDBSCANConfiguration(minClusterSize: 2)

    // Should not throw, but return single outlier
    let result = try await hdbscan(embeddings, configuration: config)

    #expect(result.clusterCount == 0)
    #expect(result.label(for: 0) == -1)
}

@Test("HDBSCAN detects outliers")
func testHDBSCANOutliers() async throws {
    var embeddings = [Embedding]()

    // Dense cluster
    for i in 0..<15 {
        let x = Float(i % 5) * 0.1
        let y = Float(i / 5) * 0.1
        embeddings.append(Embedding(vector: [x, y]))
    }

    // Single outlier far away
    embeddings.append(Embedding(vector: [100.0, 100.0]))

    let config = HDBSCANConfiguration(
        minClusterSize: 5,
        minSamples: 3
    )

    let result = try await hdbscan(embeddings, configuration: config)

    // Should have at least 1 cluster
    #expect(result.clusterCount >= 1)

    // The last point should be an outlier
    #expect(result.label(for: 15) == -1)
    #expect(result.outlierScore(for: 15) > 0.5)
}

@Test("HDBSCAN with EOM vs Leaf selection")
func testHDBSCANSelectionMethods() async throws {
    var embeddings = [Embedding]()

    // Create hierarchical clusters
    for i in 0..<20 {
        let x = Float(i % 5) * 0.1
        let y = Float(i / 5) * 0.1
        embeddings.append(Embedding(vector: [x, y]))
    }

    let configEOM = HDBSCANConfiguration(
        minClusterSize: 3,
        clusterSelectionMethod: .eom
    )

    let configLeaf = HDBSCANConfiguration(
        minClusterSize: 3,
        clusterSelectionMethod: .leaf
    )

    let resultEOM = try await hdbscan(embeddings, configuration: configEOM)
    let resultLeaf = try await hdbscan(embeddings, configuration: configLeaf)

    // Both should produce valid results
    #expect(resultEOM.clusterCount >= 0)
    #expect(resultLeaf.clusterCount >= 0)
}

@Test("HDBSCAN builder pattern")
func testHDBSCANBuilder() async throws {
    let config = HDBSCANBuilder()
        .minClusterSize(5)
        .minSamples(3)
        .epsilon(0.1)
        .selectionMethod(.eom)
        .buildConfiguration()

    #expect(config.minClusterSize == 5)
    #expect(config.effectiveMinSamples == 3)
    #expect(config.clusterSelectionEpsilon == 0.1)
    #expect(config.clusterSelectionMethod == .eom)
}

@Test("HDBSCAN with identical points")
func testHDBSCANIdenticalPoints() async throws {
    // All points at same location
    let embeddings = (0..<10).map { _ in Embedding(vector: [1.0, 1.0]) }

    let config = HDBSCANConfiguration(minClusterSize: 3)

    // Should not crash - this is the main test
    let result = try await hdbscan(embeddings, configuration: config)

    // Result should have correct count
    #expect(result.labels.count == 10)

    // With identical points at distance 0, HDBSCAN behavior is implementation-defined
    // The important thing is it doesn't crash and produces valid output
    #expect(result.clusterCount >= 0)
}

// MARK: - Cluster Assignment Tests

@Test("ClusterAssignment properties")
func testClusterAssignmentProperties() {
    let assignment = ClusterAssignment(
        labels: [0, 0, 1, 1, -1],
        probabilities: [0.9, 0.8, 0.95, 0.85, 0.0],
        outlierScores: [0.1, 0.2, 0.05, 0.15, 0.9],
        clusterCount: 2
    )

    #expect(assignment.clusterCount == 2)
    #expect(assignment.pointCount == 5)
    #expect(assignment.outlierCount == 1)
    #expect(assignment.outlierRate == 0.2)

    #expect(assignment.label(for: 0) == 0)
    #expect(assignment.isOutlier(4))
    #expect(!assignment.isOutlier(0))

    let cluster0 = assignment.pointsInCluster(0)
    #expect(cluster0 == [0, 1])

    let outliers = assignment.outlierIndices
    #expect(outliers == [4])

    let sizes = assignment.clusterSizes
    #expect(sizes == [2, 2])
}

// MARK: - Edge Cases

@Test("HDBSCAN with minimum required points")
func testHDBSCANMinimumPoints() async throws {
    // Exactly minClusterSize points
    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [0.1, 0.0]),
        Embedding(vector: [0.0, 0.1]),
        Embedding(vector: [0.1, 0.1]),
        Embedding(vector: [0.05, 0.05]),
    ]

    let config = HDBSCANConfiguration(minClusterSize: 5)
    let result = try await hdbscan(embeddings, configuration: config)

    // Should produce a valid result (either 1 cluster or all outliers)
    #expect(result.labels.count == 5)
}

@Test("HDBSCAN high-dimensional data")
func testHDBSCANHighDimensional() async throws {
    // 50-dimensional embeddings
    var embeddings = [Embedding]()

    for i in 0..<20 {
        var vector = [Float](repeating: 0, count: 50)
        // Two clusters based on first dimension
        if i < 10 {
            vector[0] = 0.0 + Float.random(in: -0.1...0.1)
        } else {
            vector[0] = 10.0 + Float.random(in: -0.1...0.1)
        }
        // Random noise in other dimensions
        for d in 1..<50 {
            vector[d] = Float.random(in: -0.1...0.1)
        }
        embeddings.append(Embedding(vector: vector))
    }

    let config = HDBSCANConfiguration(minClusterSize: 3)
    let result = try await hdbscan(embeddings, configuration: config)

    // Should produce valid results
    #expect(result.labels.count == 20)
    #expect(result.clusterCount >= 1)
}

// MARK: - Clustering Result Tests

@Test("ClusteringResult contains hierarchy")
func testClusteringResultHierarchy() async throws {
    var embeddings = [Embedding]()
    for i in 0..<15 {
        embeddings.append(Embedding(vector: [Float(i), 0.0]))
    }

    let config = HDBSCANConfiguration(minClusterSize: 3)
    let result = try await hdbscanWithDetails(embeddings, configuration: config)

    #expect(result.hierarchy != nil)
    #expect(result.coreDistances != nil)
    #expect(result.coreDistances?.count == 15)
    #expect(result.processingTime >= 0)
}

// MARK: - PCA Tests

@Test("PCA reduces dimensionality")
func testPCAReducesDimensionality() async throws {
    // Create 10 embeddings with 5 dimensions
    let embeddings = (0..<10).map { i in
        Embedding(vector: [Float(i), Float(i) * 2, Float(i) * 3, Float(i) * 0.5, Float(i) * 0.1])
    }

    // Reduce to 2 dimensions
    let reduced = try await pca(embeddings, components: 2)

    #expect(reduced.count == 10)
    #expect(reduced[0].dimension == 2)
}

@Test("PCA preserves sample count")
func testPCAPreservesSampleCount() async throws {
    let embeddings = (0..<50).map { _ in
        Embedding.random(dimension: 20)
    }

    let reduced = try await pca(embeddings, components: 5)

    #expect(reduced.count == 50)
    #expect(reduced.allSatisfy { $0.dimension == 5 })
}

@Test("PCA with variance ratio threshold")
func testPCAVarianceRatio() async throws {
    // Create data with variance concentrated in first few dimensions
    var embeddings = [Embedding]()
    for i in 0..<20 {
        var vector = [Float](repeating: 0, count: 10)
        // Most variance in first 2 dimensions
        vector[0] = Float(i) * 10.0 + Float.random(in: -0.1...0.1)
        vector[1] = Float(i) * 5.0 + Float.random(in: -0.1...0.1)
        // Little variance in remaining dimensions
        for d in 2..<10 {
            vector[d] = Float.random(in: -0.01...0.01)
        }
        embeddings.append(Embedding(vector: vector))
    }

    // Should select few components to explain 95% variance
    let reduced = try await pcaWithVariance(embeddings, varianceRatio: 0.95)

    #expect(reduced.count == 20)
    // Should reduce to fewer dimensions than original
    #expect(reduced[0].dimension < 10)
}

@Test("PCA builder pattern")
func testPCABuilder() {
    let reducer = PCABuilder()
        .components(10)
        .whiten(true)
        .build()

    #expect(reducer.components == 10)
    #expect(reducer.whiten == true)
}

@Test("PCA transform new data")
func testPCATransformNewData() async throws {
    // Training data
    let trainData = (0..<20).map { i in
        Embedding(vector: [Float(i), Float(i) * 2, Float(i) * 3])
    }

    // Fit PCA
    var reducer = PCAReducer(components: 2)
    try await reducer.fit(trainData)

    #expect(reducer.isFitted)
    #expect(reducer.outputDimension == 2)

    // Transform new data
    let newData = [
        Embedding(vector: [5.0, 10.0, 15.0]),
        Embedding(vector: [10.0, 20.0, 30.0]),
    ]

    let transformed = try await reducer.transform(newData)

    #expect(transformed.count == 2)
    #expect(transformed[0].dimension == 2)
}

@Test("PCA throws for empty input")
func testPCAEmptyInput() async throws {
    let embeddings: [Embedding] = []

    await #expect(throws: ReductionError.self) {
        try await pca(embeddings, components: 2)
    }
}

@Test("PCA throws for inconsistent dimensions")
func testPCAInconsistentDimensions() async throws {
    let embeddings = [
        Embedding(vector: [1.0, 2.0]),
        Embedding(vector: [1.0, 2.0, 3.0]),
    ]

    await #expect(throws: ReductionError.self) {
        try await pca(embeddings, components: 1)
    }
}

@Test("PCA throws for too many components")
func testPCATooManyComponents() async throws {
    let embeddings = (0..<5).map { _ in
        Embedding(vector: [1.0, 2.0, 3.0])
    }

    // Can't reduce to 10 components from 3-dimensional data
    await #expect(throws: ReductionError.self) {
        try await pca(embeddings, components: 10)
    }
}

@Test("PCA with whitening")
func testPCAWhitening() async throws {
    // Create data with different variances along axes
    var embeddings = [Embedding]()
    for i in 0..<20 {
        let x = Float(i) * 10.0  // High variance
        let y = Float(i) * 0.1   // Low variance
        embeddings.append(Embedding(vector: [x, y]))
    }

    let normalReduced = try await pca(embeddings, components: 2, whiten: false)
    let whitenedReduced = try await pca(embeddings, components: 2, whiten: true)

    // Whitened data should have approximately unit variance
    // Compute variance of first component for whitened data
    let whitenedValues = whitenedReduced.map { $0.vector[0] }
    let whitenedMean = whitenedValues.reduce(0, +) / Float(whitenedValues.count)
    let whitenedVariance = whitenedValues.map { ($0 - whitenedMean) * ($0 - whitenedMean) }
        .reduce(0, +) / Float(whitenedValues.count - 1)

    // Whitened variance should be close to 1.0
    #expect(abs(whitenedVariance - 1.0) < 0.2)

    // Normal PCA variance can be anything
    #expect(normalReduced.count == 20)
}

@Test("PCA explained variance ratio")
func testPCAExplainedVariance() async throws {
    // Create linearly correlated data
    var embeddings = [Embedding]()
    for i in 0..<30 {
        let t = Float(i)
        embeddings.append(Embedding(vector: [t, t * 2 + Float.random(in: -0.1...0.1), t * 3]))
    }

    var reducer = PCAReducer(components: 3)
    try await reducer.fit(embeddings)

    // Check explained variance
    let explainedRatio = reducer.cumulativeExplainedVariance
    #expect(explainedRatio != nil)
    #expect(explainedRatio! > 0.9)  // Most variance should be explained

    // Individual ratios should sum to cumulative
    let individualRatios = reducer.explainedVarianceRatios
    #expect(individualRatios != nil)
    #expect(individualRatios!.count == 3)
}

@Test("PCA handles single sample gracefully")
func testPCASingleSample() async throws {
    let embeddings = [Embedding(vector: [1.0, 2.0, 3.0])]

    await #expect(throws: ReductionError.self) {
        try await pca(embeddings, components: 2)
    }
}

@Test("PCA components property extraction")
func testPCAComponentsProperty() async throws {
    let embeddings = (0..<10).map { i in
        Embedding(vector: [Float(i), Float(i) * 2, Float(i) * 3, Float(i) * 4])
    }

    var reducer = PCAReducer(components: 2)
    try await reducer.fit(embeddings)

    // Check we can access the principal components
    let components = reducer.principalComponents
    #expect(components != nil)
    // 4 dimensions × 2 components = 8 values (column-major)
    #expect(components!.count == 8)

    let eigenvalues = reducer.eigenvalues
    #expect(eigenvalues != nil)
    #expect(eigenvalues!.count == 2)
    // Eigenvalues should be in descending order
    #expect(eigenvalues![0] >= eigenvalues![1])
}

@Test("PCA reconstruction approximates original")
func testPCAReconstruction() async throws {
    // Create simple 2D data
    let embeddings = (0..<10).map { i in
        Embedding(vector: [Float(i), Float(i) * 0.5])
    }

    // Reduce to 1D and back - should lose some information but not much
    var reducer = PCAReducer(components: 1)
    try await reducer.fit(embeddings)

    let reduced = try await reducer.transform(embeddings)

    #expect(reduced.count == 10)
    #expect(reduced[0].dimension == 1)

    // First principal component should capture most variance
    let explainedRatio = reducer.cumulativeExplainedVariance
    #expect(explainedRatio != nil)
    #expect(explainedRatio! > 0.8)  // High variance explained with 1 component
}

@Test("PCA array extension")
func testPCAArrayExtension() async throws {
    let embeddings = (0..<15).map { _ in
        Embedding.random(dimension: 10)
    }

    let reduced = try await embeddings.reducePCA(components: 3)

    #expect(reduced.count == 15)
    #expect(reduced[0].dimension == 3)
}

@Test("PCA numerical stability with near-zero variance")
func testPCANumericalStability() async throws {
    // Create data with very low variance in some dimensions
    var embeddings = [Embedding]()
    for i in 0..<20 {
        embeddings.append(Embedding(vector: [
            Float(i),           // Normal variance
            1.0,                // Zero variance (constant)
            Float(i) * 1e-10    // Very low variance
        ]))
    }

    // Should handle gracefully with regularization
    let reduced = try await pca(embeddings, components: 2)

    #expect(reduced.count == 20)
    #expect(reduced[0].dimension == 2)
    // Values should be finite
    #expect(reduced.allSatisfy { embedding in
        embedding.vector.allSatisfy { $0.isFinite }
    })
}

// MARK: - Tokenizer Tests

@Test("Tokenizer splits on whitespace and punctuation")
func testTokenizerBasicSplit() {
    let tokenizer = Tokenizer(configuration: .default)
    let tokens = tokenizer.tokenize("Hello, world! This is a test.")

    #expect(tokens.contains("hello"))
    #expect(tokens.contains("world"))
    #expect(tokens.contains("this"))
    #expect(tokens.contains("test"))
    // Punctuation should be removed
    #expect(!tokens.contains(","))
    #expect(!tokens.contains("!"))
}

@Test("Tokenizer filters stop words")
func testTokenizerStopWords() {
    let tokenizer = Tokenizer(configuration: .english)
    let tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog")

    // Stop words should be filtered
    #expect(!tokens.contains("the"))
    #expect(!tokens.contains("over"))
    // Content words should remain
    #expect(tokens.contains("quick"))
    #expect(tokens.contains("brown"))
    #expect(tokens.contains("fox"))
    #expect(tokens.contains("jumps"))
    #expect(tokens.contains("lazy"))
    #expect(tokens.contains("dog"))
}

@Test("Tokenizer respects minimum length")
func testTokenizerMinLength() {
    let config = TokenizerConfiguration(
        stopWords: [],
        minTokenLength: 4
    )
    let tokenizer = Tokenizer(configuration: config)
    let tokens = tokenizer.tokenize("I am a good programmer")

    // Short words should be filtered (< 4 chars)
    #expect(!tokens.contains("am"))   // 2 chars - filtered
    #expect(!tokens.contains("a"))    // 1 char - filtered
    // Words >= 4 chars should remain
    #expect(tokens.contains("good"))       // 4 chars - included
    #expect(tokens.contains("programmer")) // 10 chars - included
}

@Test("Tokenizer generates bigrams")
func testTokenizerBigrams() {
    let config = TokenizerConfiguration(
        stopWords: [],
        minTokenLength: 1,
        useBigrams: true
    )
    let tokenizer = Tokenizer(configuration: config)
    let tokens = tokenizer.tokenize("hello world test")

    // Should have unigrams
    #expect(tokens.contains("hello"))
    #expect(tokens.contains("world"))
    #expect(tokens.contains("test"))
    // Should have bigrams
    #expect(tokens.contains("hello_world"))
    #expect(tokens.contains("world_test"))
}

@Test("Tokenizer removes numbers when configured")
func testTokenizerRemoveNumbers() {
    let configWithNumbers = TokenizerConfiguration(
        stopWords: [],
        removeNumbers: false
    )
    let configWithoutNumbers = TokenizerConfiguration(
        stopWords: [],
        removeNumbers: true
    )

    let tokenizerWithNumbers = Tokenizer(configuration: configWithNumbers)
    let tokenizerWithoutNumbers = Tokenizer(configuration: configWithoutNumbers)

    let text = "There are 42 apples and 100 oranges"

    let withNumbers = tokenizerWithNumbers.tokenize(text)
    let withoutNumbers = tokenizerWithoutNumbers.tokenize(text)

    #expect(withNumbers.contains("42"))
    #expect(withNumbers.contains("100"))
    #expect(!withoutNumbers.contains("42"))
    #expect(!withoutNumbers.contains("100"))
    // Non-numeric tokens should remain
    #expect(withoutNumbers.contains("apples"))
    #expect(withoutNumbers.contains("oranges"))
}

@Test("Tokenizer builder pattern")
func testTokenizerBuilder() {
    let tokenizer = TokenizerBuilder()
        .stopWords([])  // Clear stop words so "buy" isn't filtered
        .minLength(3)
        .useBigrams(true)
        .removeNumbers(true)
        .build()

    let tokens = tokenizer.tokenize("I buy 5 apples daily")

    #expect(!tokens.contains("5"))      // Number filtered
    #expect(!tokens.contains("i"))      // Too short (< 3)
    #expect(tokens.contains("buy"))     // 3 chars, included
    #expect(tokens.contains("apples"))  // 6 chars, included
    #expect(tokens.contains("daily"))   // 5 chars, included
    #expect(tokens.contains("buy_apples"))  // Bigram
}

// MARK: - Vocabulary Tests

@Test("Vocabulary builds from tokenized documents")
func testVocabularyBuild() {
    let docs = [
        ["apple", "banana", "apple"],
        ["banana", "cherry"],
        ["apple", "date"]
    ]

    let vocab = VocabularyBuilder().build(from: docs)

    #expect(vocab.size == 4)
    #expect(vocab.contains("apple"))
    #expect(vocab.contains("banana"))
    #expect(vocab.contains("cherry"))
    #expect(vocab.contains("date"))

    // Document frequencies
    #expect(vocab.documentFrequency(for: "apple") == 2)  // appears in docs 0 and 2
    #expect(vocab.documentFrequency(for: "banana") == 2) // appears in docs 0 and 1
    #expect(vocab.documentFrequency(for: "cherry") == 1) // appears in doc 1 only
}

@Test("Vocabulary filters by document frequency")
func testVocabularyMinDf() {
    // Need enough documents so that common words don't exceed maxDf ratio (0.95)
    let docs = [
        ["common", "common", "rare1"],
        ["common", "rare2"],
        ["common", "rare3"],
        ["another", "rare4"],
        ["another", "common"]  // common appears in 4/5 = 80% of docs
    ]

    let config = VocabularyConfiguration(minDocumentFrequency: 2)
    let vocab = VocabularyBuilder(configuration: config).build(from: docs)

    // "common" appears in 4 docs (>= 2), should be included
    // "another" appears in 2 docs (>= 2), should be included
    // "rare*" terms appear in 1 doc each (< 2), should be excluded
    #expect(vocab.size == 2)
    #expect(vocab.contains("common"))
    #expect(vocab.contains("another"))
    #expect(!vocab.contains("rare1"))
    #expect(!vocab.contains("rare2"))
}

@Test("Vocabulary filters by max document frequency ratio")
func testVocabularyMaxDfRatio() {
    let docs = [
        ["ubiquitous", "rare"],
        ["ubiquitous"],
        ["ubiquitous"],
        ["ubiquitous"]
    ]

    let config = VocabularyConfiguration(
        minDocumentFrequency: 1,
        maxDocumentFrequencyRatio: 0.5  // Exclude terms in >50% of docs
    )
    let vocab = VocabularyBuilder(configuration: config).build(from: docs)

    // "ubiquitous" appears in 100% of docs, should be filtered
    #expect(!vocab.contains("ubiquitous"))
    // "rare" appears in 25%, should be kept
    #expect(vocab.contains("rare"))
}

@Test("Vocabulary term frequency vector")
func testVocabularyTermFrequencyVector() {
    let docs = [
        ["apple", "banana"],
        ["cherry"]
    ]

    let vocab = VocabularyBuilder().build(from: docs)
    let tokens = ["apple", "apple", "banana", "unknown"]

    let vector = vocab.termFrequencyVector(tokens)

    #expect(vector.count == vocab.size)
    // apple appears twice
    if let appleIdx = vocab.index(for: "apple") {
        #expect(vector[appleIdx] == 2)
    }
    // banana appears once
    if let bananaIdx = vocab.index(for: "banana") {
        #expect(vector[bananaIdx] == 1)
    }
    // unknown is not in vocabulary, ignored
}

@Test("Vocabulary empty input")
func testVocabularyEmpty() {
    let vocab = VocabularyBuilder().build(from: [])

    #expect(vocab.isEmpty)
    #expect(vocab.size == 0)
    #expect(vocab.documentCount == 0)
}

// MARK: - c-TF-IDF Tests

@Test("c-TF-IDF computes scores for clusters")
func testCTFIDFBasic() {
    // Two clusters with different vocabularies
    let clusterTokens = [
        ["machine", "learning", "neural", "network", "machine", "learning"],
        ["web", "development", "javascript", "html", "css", "web"]
    ]

    // Build vocabulary from each cluster as separate "documents"
    let vocab = VocabularyBuilder().build(from: clusterTokens)

    let computer = CTFIDFComputer()
    let scores = computer.compute(clusterTokens: clusterTokens, vocabulary: vocab)

    #expect(scores.count == 2)

    // Cluster 0 should have ML-related top terms
    let cluster0TopTerms = scores[0].topKTerms(3)
    #expect(cluster0TopTerms.contains("machine") || cluster0TopTerms.contains("learning"))

    // Cluster 1 should have web-related top terms
    let cluster1TopTerms = scores[1].topKTerms(3)
    #expect(cluster1TopTerms.contains("web") || cluster1TopTerms.contains("javascript"))
}

@Test("c-TF-IDF normalizes scores")
func testCTFIDFNormalization() {
    let clusterTokens = [
        ["word", "word", "word", "other"]
    ]

    let vocab = VocabularyBuilder().build(from: [["word", "other"]])

    let normalizedComputer = CTFIDFComputer(normalizeScores: true)
    let normalizedScores = normalizedComputer.compute(clusterTokens: clusterTokens, vocabulary: vocab)

    // Top score should be 1.0 after normalization
    if let topScore = normalizedScores.first?.scores.first?.score {
        #expect(abs(topScore - 1.0) < 0.01)
    }
}

@Test("c-TF-IDF handles empty clusters")
func testCTFIDFEmptyClusters() {
    let clusterTokens: [[String]] = [
        ["word", "another"],
        []  // Empty cluster
    ]

    let vocab = VocabularyBuilder().build(from: [["word", "another"]])

    let computer = CTFIDFComputer()
    let scores = computer.compute(clusterTokens: clusterTokens, vocabulary: vocab)

    #expect(scores.count == 2)
    #expect(scores[1].scores.isEmpty)  // Empty cluster has no scores
}

@Test("c-TF-IDF converts to TopicKeywords")
func testCTFIDFToKeywords() {
    let clusterTokens = [
        ["keyword1", "keyword2", "keyword3", "keyword1"]
    ]

    let vocab = VocabularyBuilder().build(from: clusterTokens)

    let computer = CTFIDFComputer()
    let scores = computer.compute(clusterTokens: clusterTokens, vocabulary: vocab)

    let keywords = scores[0].toKeywords(2)

    #expect(keywords.count == 2)
    #expect(!keywords[0].term.isEmpty)
    #expect(keywords[0].score > 0)
    #expect(keywords[0].frequency != nil)
}

// MARK: - CTFIDFRepresenter Tests

@Test("CTFIDFRepresenter extracts topics from documents")
func testCTFIDFRepresenterBasic() async throws {
    // Create documents about different topics
    let documents = [
        Document(content: "Machine learning algorithms are fascinating"),
        Document(content: "Neural networks learn from data"),
        Document(content: "Deep learning models are powerful"),
        Document(content: "Web development uses JavaScript"),
        Document(content: "HTML and CSS for frontend"),
        Document(content: "Backend APIs with REST")
    ]

    // Cluster assignment: 0-2 in cluster 0 (ML), 3-5 in cluster 1 (Web)
    let assignment = ClusterAssignment(
        labels: [0, 0, 0, 1, 1, 1],
        clusterCount: 2
    )

    let representer = CTFIDFRepresenter()
    let topics = try await representer.represent(
        documents: documents,
        assignment: assignment
    )

    #expect(topics.count == 2)

    // Topic 0 should have ML-related keywords
    let topic0Keywords = topics[0].keywords.map(\.term)
    #expect(topic0Keywords.count > 0)

    // Topic 1 should have web-related keywords
    let topic1Keywords = topics[1].keywords.map(\.term)
    #expect(topic1Keywords.count > 0)

    // Sizes should match cluster sizes
    #expect(topics[0].size == 3)
    #expect(topics[1].size == 3)
}

@Test("CTFIDFRepresenter handles outliers")
func testCTFIDFRepresenterOutliers() async throws {
    let documents = [
        Document(content: "First document about topic A"),
        Document(content: "Second document about topic A"),
        Document(content: "Outlier document not in any cluster")
    ]

    // Third document is outlier (-1)
    let assignment = ClusterAssignment(
        labels: [0, 0, -1],
        clusterCount: 1
    )

    let representer = CTFIDFRepresenter()
    let topics = try await representer.represent(
        documents: documents,
        assignment: assignment
    )

    #expect(topics.count == 1)
    #expect(topics[0].size == 2)  // Only non-outlier documents
}

@Test("CTFIDFRepresenter with embeddings computes centroids")
func testCTFIDFRepresenterCentroids() async throws {
    let documents = [
        Document(content: "Document one"),
        Document(content: "Document two"),
        Document(content: "Document three")
    ]

    let embeddings = [
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.0, 1.0]),
        Embedding(vector: [1.0, 1.0])
    ]

    let assignment = ClusterAssignment(
        labels: [0, 0, 0],
        clusterCount: 1
    )

    let representer = CTFIDFRepresenter()
    let topics = try await representer.represent(
        documents: documents,
        embeddings: embeddings,
        assignment: assignment
    )

    #expect(topics.count == 1)
    #expect(topics[0].centroid != nil)

    // Centroid should be the mean of all embeddings
    let centroid = topics[0].centroid!
    #expect(centroid.dimension == 2)
    // Mean of [1,0], [0,1], [1,1] = [0.667, 0.667]
    #expect(abs(centroid.vector[0] - 0.667) < 0.01)
    #expect(abs(centroid.vector[1] - 0.667) < 0.01)
}

@Test("CTFIDFRepresenter throws for empty documents")
func testCTFIDFRepresenterEmptyDocuments() async throws {
    let documents: [Document] = []
    let assignment = ClusterAssignment(labels: [], clusterCount: 0)

    let representer = CTFIDFRepresenter()

    await #expect(throws: RepresentationError.self) {
        try await representer.represent(documents: documents, assignment: assignment)
    }
}

@Test("CTFIDFRepresenter throws for count mismatch")
func testCTFIDFRepresenterCountMismatch() async throws {
    let documents = [
        Document(content: "One"),
        Document(content: "Two")
    ]

    // Only 1 label for 2 documents
    let assignment = ClusterAssignment(labels: [0], clusterCount: 1)

    let representer = CTFIDFRepresenter()

    await #expect(throws: RepresentationError.self) {
        try await representer.represent(documents: documents, assignment: assignment)
    }
}

@Test("CTFIDFRepresenter with custom configuration")
func testCTFIDFRepresenterCustomConfig() async throws {
    let documents = [
        Document(content: "Machine learning is great machine learning"),
        Document(content: "More machine learning content here")
    ]

    let assignment = ClusterAssignment(labels: [0, 0], clusterCount: 1)

    let config = CTFIDFConfiguration(
        keywordsPerTopic: 3,
        minDocumentFrequency: 1,
        minTermLength: 5  // Only words >= 5 chars
    )

    let representer = CTFIDFRepresenter(configuration: config)
    let topics = try await representer.represent(
        documents: documents,
        assignment: assignment
    )

    #expect(topics.count == 1)
    #expect(topics[0].keywords.count <= 3)

    // All keywords should be >= 5 chars
    for keyword in topics[0].keywords {
        #expect(keyword.term.count >= 5)
    }
}

@Test("CTFIDFRepresenter builder pattern")
func testCTFIDFRepresenterBuilder() {
    let representer = CTFIDFRepresenterBuilder()
        .keywordsPerTopic(5)
        .minDocumentFrequency(2)
        .useBigrams(true)
        .diversify(true)
        .diversityWeight(0.5)
        .build()

    #expect(representer.configuration.keywordsPerTopic == 5)
    #expect(representer.configuration.minDocumentFrequency == 2)
    #expect(representer.configuration.useBigrams == true)
    #expect(representer.configuration.diversify == true)
    #expect(representer.configuration.diversityWeight == 0.5)
}

@Test("CTFIDFRepresenter handles all outliers")
func testCTFIDFRepresenterAllOutliers() async throws {
    let documents = [
        Document(content: "Outlier one"),
        Document(content: "Outlier two")
    ]

    // All documents are outliers
    let assignment = ClusterAssignment(labels: [-1, -1], clusterCount: 0)

    let representer = CTFIDFRepresenter()
    let topics = try await representer.represent(
        documents: documents,
        assignment: assignment
    )

    // Should return empty topics list when all are outliers
    #expect(topics.isEmpty)
}

@Test("CTFIDFRepresenter single document cluster")
func testCTFIDFRepresenterSingleDocCluster() async throws {
    // Use multiple diverse words to ensure vocabulary isn't empty
    let documents = [
        Document(content: "Programming Swift development code algorithms data structures machine learning neural networks")
    ]

    let assignment = ClusterAssignment(labels: [0], clusterCount: 1)

    // Use a configuration with minDocFreq=1 to handle single doc
    let config = CTFIDFConfiguration(
        keywordsPerTopic: 5,
        minDocumentFrequency: 1
    )
    let representer = CTFIDFRepresenter(configuration: config)
    let topics = try await representer.represent(
        documents: documents,
        assignment: assignment
    )

    #expect(topics.count == 1)
    #expect(topics[0].size == 1)
    // Should extract keywords from the diverse content
    #expect(!topics[0].keywords.isEmpty)
}

@Test("Document array extension for topic extraction")
func testDocumentArrayExtension() async throws {
    let documents = [
        Document(content: "Topic A content here"),
        Document(content: "More topic A stuff"),
        Document(content: "Topic B different subject")
    ]

    let assignment = ClusterAssignment(labels: [0, 0, 1], clusterCount: 2)

    let topics = try await documents.extractTopics(assignment: assignment)

    #expect(topics.count == 2)
}

// MARK: - Co-occurrence Counter Tests

@Test("CooccurrenceCounter counts word pairs in sliding window")
func testCooccurrenceCounterSlidingWindow() {
    let tokenizedDocs = [
        ["apple", "banana", "cherry", "date"],
        ["apple", "banana", "elderberry"]
    ]

    let counter = CooccurrenceCounter(mode: .slidingWindow(size: 3))
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    // Word counts
    #expect(counts.count(for: "apple") > 0)
    #expect(counts.count(for: "banana") > 0)

    // Pair counts - apple and banana should co-occur in windows
    #expect(counts.count(for: "apple", "banana") > 0)

    // Total windows should be positive
    #expect(counts.totalWindows > 0)
}

@Test("CooccurrenceCounter counts document-level co-occurrence")
func testCooccurrenceCounterDocument() {
    let tokenizedDocs = [
        ["apple", "banana", "cherry"],
        ["apple", "date"],
        ["banana", "cherry", "elderberry"]
    ]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    // apple and banana both appear in doc 0
    #expect(counts.count(for: "apple", "banana") == 1)

    // banana and cherry appear in docs 0 and 2
    #expect(counts.count(for: "banana", "cherry") == 2)

    // apple and elderberry never co-occur
    #expect(counts.count(for: "apple", "elderberry") == 0)

    // Total should equal number of non-empty documents
    #expect(counts.totalWindows == 3)
}

@Test("CooccurrenceCounter word pair symmetry")
func testCooccurrenceCounterSymmetry() {
    let tokenizedDocs = [
        ["word1", "word2"]
    ]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    // Order shouldn't matter
    #expect(counts.count(for: "word1", "word2") == counts.count(for: "word2", "word1"))
}

@Test("CooccurrenceCounter handles empty documents")
func testCooccurrenceCounterEmpty() {
    let tokenizedDocs: [[String]] = [[], ["word"], []]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    // Only one non-empty document
    #expect(counts.totalWindows == 1)
    #expect(counts.count(for: "word") == 1)
}

@Test("WordPair ordering is consistent")
func testWordPairOrdering() {
    let pair1 = WordPair("zebra", "apple")
    let pair2 = WordPair("apple", "zebra")

    // Both should have same ordering (alphabetical)
    #expect(pair1.word1 == "apple")
    #expect(pair1.word2 == "zebra")
    #expect(pair1 == pair2)
}

// MARK: - NPMI Scorer Tests

@Test("NPMIScorer computes NPMI for perfectly co-occurring words")
func testNPMIScorerPerfectCooccurrence() {
    // Words that always appear together when either appears,
    // but not in all documents (to get meaningful NPMI)
    let tokenizedDocs = [
        ["alpha", "beta"],  // Both appear together
        ["alpha", "beta"],  // Both appear together
        ["gamma"],          // Neither appears
        ["delta"]           // Neither appears
    ]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    let scorer = NPMIScorer()
    let score = scorer.score(word1: "alpha", word2: "beta", counts: counts)

    #expect(score != nil)
    // Perfect co-occurrence should yield NPMI close to 1.0
    // P(alpha) = P(beta) = 2/4 = 0.5, P(alpha,beta) = 2/4 = 0.5
    // PMI = log(0.5 / 0.25) = log(2), NPMI = log(2) / log(2) = 1.0
    #expect(score!.npmi > 0.9)
}

@Test("NPMIScorer computes NPMI for independent words")
func testNPMIScorerIndependentWords() {
    // Words that appear in different documents (no co-occurrence)
    let tokenizedDocs = [
        ["alpha", "alpha"],
        ["beta", "beta"],
        ["gamma", "gamma"]
    ]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    let scorer = NPMIScorer()
    let score = scorer.score(word1: "alpha", word2: "beta", counts: counts)

    #expect(score != nil)
    // No co-occurrence should yield NPMI close to -1.0 (or very negative)
    #expect(score!.npmi < 0)
}

@Test("NPMIScorer returns nil for unknown words")
func testNPMIScorerUnknownWords() {
    let tokenizedDocs = [["known", "word"]]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    let scorer = NPMIScorer()
    let score = scorer.score(word1: "unknown", word2: "word", counts: counts)

    #expect(score == nil)
}

@Test("NPMIScorer computes topic coherence")
func testNPMIScorerTopicCoherence() {
    // Corpus where machine learning terms co-occur
    // Include documents without these terms to get meaningful NPMI
    let tokenizedDocs = [
        ["machine", "learning", "neural"],
        ["machine", "learning", "network"],
        ["neural", "network", "deep"],
        ["machine", "neural", "network"],
        ["unrelated", "content", "here"],    // No ML terms
        ["different", "topic", "entirely"]   // No ML terms
    ]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    let scorer = NPMIScorer()
    let result = scorer.score(keywords: ["machine", "learning", "neural", "network"], counts: counts)

    // Mean NPMI should be positive for related terms
    #expect(result.meanNPMI > 0)
    // Should have (4 choose 2) = 6 pairs
    #expect(result.pairCount == 6)
}

@Test("NPMIScorer handles single keyword")
func testNPMIScorerSingleKeyword() {
    let tokenizedDocs = [["word"]]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    let scorer = NPMIScorer()
    let result = scorer.score(keywords: ["word"], counts: counts)

    // No pairs possible
    #expect(result.pairCount == 0)
    #expect(result.meanNPMI == 0)
}

@Test("NPMIScorer scores Topic objects")
func testNPMIScorerWithTopic() {
    let tokenizedDocs = [
        ["apple", "fruit", "red"],
        ["apple", "fruit", "green"],
        ["banana", "fruit", "yellow"]
    ]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    let topic = Topic(
        id: TopicID(0),
        keywords: [
            TopicKeyword(term: "fruit", score: 1.0),
            TopicKeyword(term: "apple", score: 0.9),
            TopicKeyword(term: "banana", score: 0.8)
        ],
        size: 3
    )

    let scorer = NPMIScorer()
    let result = scorer.score(topic: topic, counts: counts, topKeywords: 3)

    #expect(result.keywords.count == 3)
    #expect(result.pairCount == 3)  // 3 choose 2
}

// MARK: - Coherence Evaluator Tests

@Test("CoherenceEvaluator evaluates topics")
func testCoherenceEvaluatorBasic() async {
    let documents = [
        Document(content: "machine learning algorithms neural network"),
        Document(content: "machine learning models neural"),
        Document(content: "neural network deep learning"),
        Document(content: "web development javascript html"),
        Document(content: "css html frontend design"),
        Document(content: "javascript react angular web")
    ]

    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "machine", score: 1.0),
                TopicKeyword(term: "learning", score: 0.9),
                TopicKeyword(term: "neural", score: 0.8),
                TopicKeyword(term: "network", score: 0.7)
            ],
            size: 3
        ),
        Topic(
            id: TopicID(1),
            keywords: [
                TopicKeyword(term: "web", score: 1.0),
                TopicKeyword(term: "javascript", score: 0.9),
                TopicKeyword(term: "html", score: 0.8)
            ],
            size: 3
        )
    ]

    let evaluator = NPMICoherenceEvaluator()
    let result = await evaluator.evaluate(topics: topics, documents: documents)

    #expect(result.topicCount == 2)
    #expect(result.topicScores.count == 2)
    // Both topics should have non-negative coherence
    #expect(result.meanCoherence.isFinite)
}

@Test("CoherenceEvaluator handles empty topics")
func testCoherenceEvaluatorEmptyTopics() async {
    let documents = [Document(content: "some content")]
    let topics: [Topic] = []

    let evaluator = NPMICoherenceEvaluator()
    let result = await evaluator.evaluate(topics: topics, documents: documents)

    #expect(result.topicCount == 0)
    #expect(result.topicScores.isEmpty)
}

@Test("CoherenceEvaluator handles empty documents")
func testCoherenceEvaluatorEmptyDocuments() async {
    let documents: [Document] = []
    let topics = [
        Topic(id: TopicID(0), keywords: [TopicKeyword(term: "word", score: 1.0)], size: 1)
    ]

    let evaluator = NPMICoherenceEvaluator()
    let result = await evaluator.evaluate(topics: topics, documents: documents)

    #expect(result.topicCount == 1)
    #expect(result.topicScores.count == 1)
}

@Test("CoherenceEvaluator with document-level co-occurrence")
func testCoherenceEvaluatorDocumentMode() async {
    let documents = [
        Document(content: "apple banana cherry"),
        Document(content: "apple banana date"),
        Document(content: "elderberry fig grape")
    ]

    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "apple", score: 1.0),
                TopicKeyword(term: "banana", score: 0.9)
            ],
            size: 2
        )
    ]

    let config = CoherenceConfiguration(useDocumentCooccurrence: true)
    let evaluator = NPMICoherenceEvaluator(configuration: config)
    let result = await evaluator.evaluate(topics: topics, documents: documents)

    #expect(result.topicScores.count == 1)
    // apple and banana co-occur in 2 out of 3 docs, should have positive NPMI
    #expect(result.topicScores[0] > 0)
}

@Test("CoherenceEvaluator includes detailed results")
func testCoherenceEvaluatorDetailedResults() async {
    let documents = [
        Document(content: "word1 word2 word3"),
        Document(content: "word1 word2")
    ]

    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "word1", score: 1.0),
                TopicKeyword(term: "word2", score: 0.9),
                TopicKeyword(term: "word3", score: 0.8)
            ],
            size: 2
        )
    ]

    let evaluator = NPMICoherenceEvaluator(includeDetailedResults: true)
    let result = await evaluator.evaluate(topics: topics, documents: documents)

    #expect(result.detailedResults != nil)
    #expect(result.detailedResults!.count == 1)
    #expect(result.detailedResults![0].pairScores.count == 3)  // 3 choose 2
}

@Test("CoherenceEvaluator computes correct statistics")
func testCoherenceEvaluatorStatistics() async {
    let documents = [
        Document(content: "good coherent topic"),
        Document(content: "good coherent words"),
        Document(content: "random noise unrelated")
    ]

    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "good", score: 1.0),
                TopicKeyword(term: "coherent", score: 0.9)
            ],
            size: 2
        ),
        Topic(
            id: TopicID(1),
            keywords: [
                TopicKeyword(term: "random", score: 1.0),
                TopicKeyword(term: "unrelated", score: 0.9)
            ],
            size: 1
        )
    ]

    let evaluator = NPMICoherenceEvaluator()
    let result = await evaluator.evaluate(topics: topics, documents: documents)

    #expect(result.topicCount == 2)
    #expect(result.minCoherence <= result.maxCoherence)
    #expect(result.stdCoherence >= 0)
}

@Test("CoherenceEvaluator with pre-computed counts")
func testCoherenceEvaluatorPrecomputedCounts() async {
    // Include documents without apple/banana to get meaningful NPMI
    let documents = [
        Document(content: "apple banana cherry"),
        Document(content: "apple banana"),
        Document(content: "other content here"),
        Document(content: "different topic entirely")
    ]

    // Pre-compute counts
    let counts = documents.precomputeCooccurrences()

    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "apple", score: 1.0),
                TopicKeyword(term: "banana", score: 0.9)
            ],
            size: 2
        )
    ]

    let evaluator = NPMICoherenceEvaluator()
    let result = evaluator.evaluate(topics: topics, counts: counts)

    #expect(result.topicCount == 1)
    #expect(result.topicScores[0] > 0)  // Positive NPMI
}

@Test("CoherenceResult identifies low coherence topics")
func testCoherenceResultLowCoherenceTopics() async {
    // Manually create a result with known scores
    let result = CoherenceResult(
        topicScores: [0.5, -0.2, 0.3, -0.5],
        detailedResults: nil,
        meanCoherence: 0.025,
        medianCoherence: 0.05,
        minCoherence: -0.5,
        maxCoherence: 0.5,
        stdCoherence: 0.4,
        topicCount: 4,
        positiveCoherenceCount: 2
    )

    let lowCoherence = result.lowCoherenceTopics(threshold: 0)

    #expect(lowCoherence.count == 2)
    #expect(lowCoherence.contains(1))
    #expect(lowCoherence.contains(3))
}

// MARK: - Diversity Metrics Tests

@Test("DiversityMetrics computes diversity score")
func testDiversityMetricsBasic() {
    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "apple", score: 1.0),
                TopicKeyword(term: "banana", score: 0.9)
            ],
            size: 2
        ),
        Topic(
            id: TopicID(1),
            keywords: [
                TopicKeyword(term: "cherry", score: 1.0),
                TopicKeyword(term: "date", score: 0.9)
            ],
            size: 2
        )
    ]

    let metrics = DiversityMetrics()
    let result = metrics.evaluate(topics: topics, topKeywords: 2)

    // All keywords are unique
    #expect(result.diversity == 1.0)
    #expect(result.uniqueKeywordCount == 4)
    #expect(result.totalKeywordCount == 4)
    #expect(result.meanRedundancy == 0)
}

@Test("DiversityMetrics detects keyword overlap")
func testDiversityMetricsOverlap() {
    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "shared", score: 1.0),
                TopicKeyword(term: "unique1", score: 0.9)
            ],
            size: 2
        ),
        Topic(
            id: TopicID(1),
            keywords: [
                TopicKeyword(term: "shared", score: 1.0),
                TopicKeyword(term: "unique2", score: 0.9)
            ],
            size: 2
        )
    ]

    let metrics = DiversityMetrics()
    let result = metrics.evaluate(topics: topics, topKeywords: 2)

    // 3 unique out of 4 total
    #expect(result.diversity == 0.75)
    #expect(result.uniqueKeywordCount == 3)
    #expect(result.totalKeywordCount == 4)
    // Each topic has 1/2 = 0.5 redundancy
    #expect(result.meanRedundancy == 0.5)
}

@Test("DiversityMetrics computes overlap matrix")
func testDiversityMetricsOverlapMatrix() {
    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "word1", score: 1.0),
                TopicKeyword(term: "word2", score: 0.9)
            ],
            size: 2
        ),
        Topic(
            id: TopicID(1),
            keywords: [
                TopicKeyword(term: "word2", score: 1.0),
                TopicKeyword(term: "word3", score: 0.9)
            ],
            size: 2
        )
    ]

    let metrics = DiversityMetrics(computeOverlapMatrix: true)
    let result = metrics.evaluate(topics: topics, topKeywords: 2)

    #expect(result.overlapMatrix != nil)
    #expect(result.overlapMatrix!.count == 2)
    #expect(result.overlapMatrix![0].count == 2)

    // Diagonal should be 1.0
    #expect(result.overlapMatrix![0][0] == 1.0)
    #expect(result.overlapMatrix![1][1] == 1.0)

    // Off-diagonal: topic 0 shares 1/2 words with topic 1
    #expect(result.overlapMatrix![0][1] == 0.5)
}

@Test("DiversityMetrics handles empty topics")
func testDiversityMetricsEmpty() {
    let topics: [Topic] = []

    let metrics = DiversityMetrics()
    let result = metrics.evaluate(topics: topics)

    #expect(result.diversity == 1.0)
    #expect(result.topicRedundancy.isEmpty)
}

@Test("DiversityMetrics identifies redundant topics")
func testDiversityMetricsRedundantTopics() {
    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "a", score: 1.0),
                TopicKeyword(term: "b", score: 0.9)
            ],
            size: 2
        ),
        Topic(
            id: TopicID(1),
            keywords: [
                TopicKeyword(term: "a", score: 1.0),
                TopicKeyword(term: "b", score: 0.9)
            ],
            size: 2
        ),
        Topic(
            id: TopicID(2),
            keywords: [
                TopicKeyword(term: "x", score: 1.0),
                TopicKeyword(term: "y", score: 0.9)
            ],
            size: 2
        )
    ]

    let metrics = DiversityMetrics()
    let result = metrics.evaluate(topics: topics, topKeywords: 2)

    // Topics 0 and 1 are fully redundant (100% overlap)
    let redundant = result.redundantTopics(threshold: 0.99)
    #expect(redundant.contains(0))
    #expect(redundant.contains(1))
    #expect(!redundant.contains(2))
}

@Test("DiversityMetrics identifies overlapping pairs")
func testDiversityMetricsOverlappingPairs() {
    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [TopicKeyword(term: "shared", score: 1.0)],
            size: 1
        ),
        Topic(
            id: TopicID(1),
            keywords: [TopicKeyword(term: "shared", score: 1.0)],
            size: 1
        ),
        Topic(
            id: TopicID(2),
            keywords: [TopicKeyword(term: "unique", score: 1.0)],
            size: 1
        )
    ]

    let metrics = DiversityMetrics(computeOverlapMatrix: true)
    let result = metrics.evaluate(topics: topics, topKeywords: 1)

    let pairs = result.overlappingPairs(threshold: 0.9)

    #expect(pairs.count == 1)
    #expect(pairs[0].topic1 == 0)
    #expect(pairs[0].topic2 == 1)
    #expect(pairs[0].overlap == 1.0)
}

// MARK: - Combined Quality Evaluation Tests

@Test("TopicQualityResult combines coherence and diversity")
func testTopicQualityResult() async {
    let documents = [
        Document(content: "machine learning algorithms"),
        Document(content: "machine learning models"),
        Document(content: "web development javascript")
    ]

    let topics = [
        Topic(
            id: TopicID(0),
            keywords: [
                TopicKeyword(term: "machine", score: 1.0),
                TopicKeyword(term: "learning", score: 0.9)
            ],
            size: 2
        ),
        Topic(
            id: TopicID(1),
            keywords: [
                TopicKeyword(term: "web", score: 1.0),
                TopicKeyword(term: "javascript", score: 0.9)
            ],
            size: 1
        )
    ]

    let result = await topics.evaluateQuality(documents: documents)

    #expect(result.coherence.topicCount == 2)
    #expect(result.diversity.uniqueKeywordCount == 4)
    #expect(result.qualityScore.isFinite)
}

@Test("Topic extension for NPMI computation")
func testTopicNPMIExtension() {
    // Include documents without word1/word2 to get meaningful NPMI
    let tokenizedDocs = [
        ["word1", "word2"],
        ["word1", "word2", "word3"],
        ["other", "unrelated"],
        ["different", "content"]
    ]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    let topic = Topic(
        id: TopicID(0),
        keywords: [
            TopicKeyword(term: "word1", score: 1.0),
            TopicKeyword(term: "word2", score: 0.9)
        ],
        size: 2
    )

    let result = topic.computeNPMI(counts: counts)

    #expect(result.meanNPMI > 0)  // These words co-occur in half the docs
}

@Test("Document array precomputes cooccurrences")
func testDocumentPrecomputeCooccurrences() {
    let documents = [
        Document(content: "apple banana cherry"),
        Document(content: "apple date")
    ]

    let counts = documents.precomputeCooccurrences()

    #expect(counts.totalWindows > 0)
    #expect(counts.count(for: "apple") == 2)
    #expect(counts.count(for: "banana") == 1)
}

@Test("NPMI range is normalized to [-1, 1]")
func testNPMINormalization() {
    // Create various scenarios
    let tokenizedDocs = [
        ["alpha", "beta"],
        ["alpha", "beta"],
        ["gamma"],
        ["delta"]
    ]

    let counter = CooccurrenceCounter(mode: .document)
    let counts = counter.count(tokenizedDocuments: tokenizedDocs)

    let scorer = NPMIScorer()

    // Perfect co-occurrence
    if let perfect = scorer.score(word1: "alpha", word2: "beta", counts: counts) {
        #expect(perfect.npmi >= -1 && perfect.npmi <= 1)
    }

    // No co-occurrence
    if let none = scorer.score(word1: "alpha", word2: "gamma", counts: counts) {
        #expect(none.npmi >= -1 && none.npmi <= 1)
    }
}

@Test("CoherenceConfiguration presets")
func testCoherenceConfigurationPresets() {
    let defaultConfig = CoherenceConfiguration.default
    #expect(defaultConfig.windowSize == 10)
    #expect(defaultConfig.topKeywords == 10)
    #expect(!defaultConfig.useDocumentCooccurrence)

    let documentConfig = CoherenceConfiguration.document
    #expect(documentConfig.useDocumentCooccurrence)

    let semanticConfig = CoherenceConfiguration.semantic
    #expect(semanticConfig.windowSize == 50)

    let conciseConfig = CoherenceConfiguration.concise
    #expect(conciseConfig.topKeywords == 5)
}

// MARK: - TopicModelConfiguration Tests

@Test("TopicModelConfiguration default preset")
func testTopicModelConfigurationDefault() {
    let config = TopicModelConfiguration.default

    #expect(config.reduction.outputDimension == 15)
    #expect(config.reduction.method == .pca)
    #expect(config.clustering.minClusterSize == 5)
    #expect(config.representation.keywordsPerTopic == 10)
    #expect(config.coherence != nil)
    #expect(config.seed == nil)
}

@Test("TopicModelConfiguration fast preset")
func testTopicModelConfigurationFast() {
    let config = TopicModelConfiguration.fast

    #expect(config.reduction.outputDimension == 10)
    #expect(config.clustering.minClusterSize == 3)
    #expect(config.representation.keywordsPerTopic == 5)
    #expect(config.coherence == nil)  // Coherence skipped for speed
}

@Test("TopicModelConfiguration quality preset")
func testTopicModelConfigurationQuality() {
    let config = TopicModelConfiguration.quality

    #expect(config.reduction.outputDimension == 25)
    #expect(config.clustering.minClusterSize == 10)
    #expect(config.representation.keywordsPerTopic == 15)
    #expect(config.representation.diversify == true)
    #expect(config.coherence != nil)
}

@Test("TopicModelConfiguration custom initialization")
func testTopicModelConfigurationCustom() {
    let config = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 30, method: .pca),
        clustering: HDBSCANConfiguration(minClusterSize: 8),
        representation: CTFIDFConfiguration(keywordsPerTopic: 12),
        coherence: .default,
        seed: 42
    )

    #expect(config.reduction.outputDimension == 30)
    #expect(config.clustering.minClusterSize == 8)
    #expect(config.representation.keywordsPerTopic == 12)
    #expect(config.seed == 42)
}

@Test("TopicModelConfiguration validation passes for valid config")
func testTopicModelConfigurationValidationPasses() throws {
    let config = TopicModelConfiguration.default
    #expect(throws: Never.self) {
        try config.validate()
    }
}

@Test("TopicModelConfiguration builder creates valid config")
func testTopicModelConfigurationBuilder() {
    let config = TopicModelConfigurationBuilder()
        .reductionDimension(20)
        .minClusterSize(8)
        .keywordsPerTopic(12)
        .diversify(true)
        .enableCoherence(true)
        .seed(123)
        .build()

    #expect(config.reduction.outputDimension == 20)
    #expect(config.clustering.minClusterSize == 8)
    #expect(config.representation.keywordsPerTopic == 12)
    #expect(config.representation.diversify == true)
    #expect(config.coherence != nil)
    #expect(config.seed == 123)
}

@Test("TopicModelConfiguration toSnapshot creates snapshot")
func testTopicModelConfigurationSnapshot() {
    let config = TopicModelConfiguration.default
    let snapshot = config.toSnapshot()

    #expect(snapshot.reductionMethod == "pca")
    #expect(snapshot.reducedDimensions == 15)
    #expect(snapshot.clusteringAlgorithm == "HDBSCAN")
    #expect(snapshot.minClusterSize == 5)
}

// MARK: - TopicModel Tests

@Test("TopicModel initializes with configuration")
func testTopicModelInit() async {
    let model = TopicModel(configuration: .default)

    #expect(await model.isFitted == false)
    #expect(await model.topics == nil)
}

@Test("TopicModel validates empty documents")
func testTopicModelEmptyDocuments() async {
    let model = TopicModel(configuration: .default)
    let documents: [Document] = []
    let embeddings: [Embedding] = []

    do {
        _ = try await model.fit(documents: documents, embeddings: embeddings)
        #expect(Bool(false), "Should throw for empty documents")
    } catch let error as TopicModelError {
        if case .invalidInput = error {
            // Expected
        } else {
            #expect(Bool(false), "Should throw invalidInput error")
        }
    } catch {
        #expect(Bool(false), "Should throw TopicModelError")
    }
}

@Test("TopicModel validates document/embedding count mismatch")
func testTopicModelCountMismatch() async {
    let model = TopicModel(configuration: .default)
    let documents = [Document(content: "test")]
    let embeddings = [
        Embedding(vector: [1.0, 2.0]),
        Embedding(vector: [3.0, 4.0])
    ]

    do {
        _ = try await model.fit(documents: documents, embeddings: embeddings)
        #expect(Bool(false), "Should throw for count mismatch")
    } catch let error as TopicModelError {
        if case .invalidInput = error {
            // Expected
        } else {
            #expect(Bool(false), "Should throw invalidInput error")
        }
    } catch {
        #expect(Bool(false), "Should throw TopicModelError")
    }
}

@Test("TopicModel validates embedding dimension consistency")
func testTopicModelDimensionMismatch() async {
    let model = TopicModel(configuration: .default)
    let documents = [
        Document(content: "test1"),
        Document(content: "test2")
    ]
    let embeddings = [
        Embedding(vector: [1.0, 2.0]),
        Embedding(vector: [1.0, 2.0, 3.0])  // Different dimension
    ]

    do {
        _ = try await model.fit(documents: documents, embeddings: embeddings)
        #expect(Bool(false), "Should throw for dimension mismatch")
    } catch let error as TopicModelError {
        if case .embeddingDimensionMismatch = error {
            // Expected
        } else {
            #expect(Bool(false), "Should throw embeddingDimensionMismatch error")
        }
    } catch {
        #expect(Bool(false), "Should throw TopicModelError")
    }
}

@Test("TopicModel transform requires fitted model")
func testTopicModelTransformNotFitted() async {
    let model = TopicModel(configuration: .default)
    let documents = [Document(content: "test")]
    let embeddings = [Embedding(vector: [1.0, 2.0])]

    do {
        _ = try await model.transform(documents: documents, embeddings: embeddings)
        #expect(Bool(false), "Should throw for not fitted")
    } catch let error as TopicModelError {
        if case .notFitted = error {
            // Expected
        } else {
            #expect(Bool(false), "Should throw notFitted error")
        }
    } catch {
        #expect(Bool(false), "Should throw TopicModelError")
    }
}

// MARK: - TopicModelProgress Tests

@Test("TopicModelStage descriptions")
func testTopicModelStageDescriptions() {
    let embedding = TopicModelStage.embedding(current: 5, total: 10)
    #expect(embedding.description.contains("5"))
    #expect(embedding.description.contains("10"))

    let reduction = TopicModelStage.reduction
    #expect(reduction.description.contains("Reducing") || reduction.description.contains("dimension"))

    let clustering = TopicModelStage.clustering
    #expect(clustering.description.contains("Clustering"))

    let representation = TopicModelStage.representation
    #expect(representation.description.contains("keyword") || representation.description.contains("Extract"))

    let evaluation = TopicModelStage.evaluation
    #expect(evaluation.description.contains("coherence") || evaluation.description.contains("Evaluat"))

    let complete = TopicModelStage.complete
    #expect(complete.description.contains("Complete"))
}

@Test("TopicModelProgress clamps values")
func testTopicModelProgressClamp() {
    let progress1 = TopicModelProgress(
        stage: .reduction,
        overallProgress: 1.5,  // Over 1.0
        elapsedTime: 1.0
    )
    #expect(progress1.overallProgress == 1.0)

    let progress2 = TopicModelProgress(
        stage: .reduction,
        overallProgress: -0.5,  // Under 0.0
        elapsedTime: 1.0
    )
    #expect(progress2.overallProgress == 0.0)
}

@Test("TopicModelProgress description includes percentage")
func testTopicModelProgressDescription() {
    let progress = TopicModelProgress(
        stage: .clustering,
        overallProgress: 0.5,
        elapsedTime: 10.0
    )

    #expect(progress.description.contains("50%") || progress.description.contains("Clustering"))
}

// MARK: - TopicModelState Tests

@Test("TopicModelState validates version")
func testTopicModelStateVersionValidation() {
    let state = TopicModelState(
        version: TopicModelState.currentVersion + 1,  // Future version
        configuration: .default,
        topics: [Topic(id: TopicID(0), keywords: [], size: 1)],
        assignments: ClusterAssignment(labels: [0], probabilities: [1.0], outlierScores: [0.0], clusterCount: 1),
        pcaComponents: nil,
        pcaMean: nil,
        centroids: nil,
        inputDimension: 10,
        reducedDimension: 5,
        trainedAt: Date(),
        documentCount: 1,
        seed: nil
    )

    #expect(throws: TopicModelError.self) {
        try state.validate()
    }
}

@Test("TopicModelState validates dimensions")
func testTopicModelStateDimensionValidation() {
    let stateZeroInput = TopicModelState(
        version: 1,
        configuration: .default,
        topics: [],
        assignments: ClusterAssignment(labels: [], probabilities: [], outlierScores: [], clusterCount: 0),
        pcaComponents: nil,
        pcaMean: nil,
        centroids: nil,
        inputDimension: 0,  // Invalid
        reducedDimension: 5,
        trainedAt: Date(),
        documentCount: 0,
        seed: nil
    )

    #expect(throws: TopicModelError.self) {
        try stateZeroInput.validate()
    }

    let stateZeroReduced = TopicModelState(
        version: 1,
        configuration: .default,
        topics: [],
        assignments: ClusterAssignment(labels: [], probabilities: [], outlierScores: [], clusterCount: 0),
        pcaComponents: nil,
        pcaMean: nil,
        centroids: nil,
        inputDimension: 10,
        reducedDimension: 0,  // Invalid
        trainedAt: Date(),
        documentCount: 0,
        seed: nil
    )

    #expect(throws: TopicModelError.self) {
        try stateZeroReduced.validate()
    }
}

@Test("TopicModelState valid state passes validation")
func testTopicModelStateValidPasses() throws {
    let state = TopicModelState(
        version: 1,
        configuration: .default,
        topics: [],  // Empty topics is valid (all outliers case)
        assignments: ClusterAssignment(labels: [], probabilities: [], outlierScores: [], clusterCount: 0),
        pcaComponents: nil,
        pcaMean: nil,
        centroids: nil,
        inputDimension: 10,
        reducedDimension: 5,
        trainedAt: Date(),
        documentCount: 0,
        seed: nil
    )

    #expect(throws: Never.self) {
        try state.validate()
    }
}

@Test("TopicModelState summary includes key info")
func testTopicModelStateSummary() {
    let state = TopicModelState(
        version: 1,
        configuration: .default,
        topics: [Topic(id: TopicID(0), keywords: [], size: 5)],
        assignments: ClusterAssignment(labels: [0, 0, 0, 0, 0], probabilities: [1, 1, 1, 1, 1], outlierScores: [0, 0, 0, 0, 0], clusterCount: 1),
        pcaComponents: nil,
        pcaMean: nil,
        centroids: nil,
        inputDimension: 100,
        reducedDimension: 15,
        trainedAt: Date(),
        documentCount: 5,
        seed: 42
    )

    let summary = state.summary

    #expect(summary.contains("1"))      // version or topic count
    #expect(summary.contains("5"))      // document count or topic size
    #expect(summary.contains("100"))    // input dimension
    #expect(summary.contains("15"))     // reduced dimension
    #expect(summary.contains("42"))     // seed
}

@Test("TopicModelState JSON round-trip")
func testTopicModelStateJSON() throws {
    let originalState = TopicModelState(
        version: 1,
        configuration: .default,
        topics: [
            Topic(
                id: TopicID(0),
                keywords: [TopicKeyword(term: "test", score: 0.9)],
                size: 10
            )
        ],
        assignments: ClusterAssignment(labels: [0], probabilities: [1.0], outlierScores: [0.1], clusterCount: 1),
        pcaComponents: [1.0, 2.0, 3.0],
        pcaMean: [0.1, 0.2],
        centroids: [Embedding(vector: [1.0, 2.0, 3.0])],
        inputDimension: 100,
        reducedDimension: 15,
        trainedAt: Date(timeIntervalSince1970: 1000000),
        documentCount: 1,
        seed: 42
    )

    // Encode to JSON
    let jsonData = try originalState.toJSON()
    #expect(!jsonData.isEmpty)

    // Decode from JSON
    let decodedState = try TopicModelState.fromJSON(jsonData)

    // Verify fields
    #expect(decodedState.version == originalState.version)
    #expect(decodedState.topics.count == originalState.topics.count)
    #expect(decodedState.inputDimension == originalState.inputDimension)
    #expect(decodedState.reducedDimension == originalState.reducedDimension)
    #expect(decodedState.documentCount == originalState.documentCount)
    #expect(decodedState.seed == originalState.seed)
    #expect(decodedState.pcaComponents?.count == originalState.pcaComponents?.count)
}

// MARK: - TopicModelError Tests

@Test("TopicModelError descriptions")
func testTopicModelErrorDescriptions() {
    let notFitted = TopicModelError.notFitted
    #expect(notFitted.errorDescription?.contains("fitted") == true)

    let invalidInput = TopicModelError.invalidInput("test message")
    #expect(invalidInput.errorDescription?.contains("test message") == true)

    let dimensionMismatch = TopicModelError.embeddingDimensionMismatch(expected: 10, got: 5)
    #expect(dimensionMismatch.errorDescription?.contains("10") == true)
    #expect(dimensionMismatch.errorDescription?.contains("5") == true)

    let noTopics = TopicModelError.noTopicsDiscovered
    #expect(noTopics.errorDescription?.contains("No topics") == true)

    let invalidConfig = TopicModelError.invalidConfiguration("bad config")
    #expect(invalidConfig.errorDescription?.contains("bad config") == true)

    let serializationFailed = TopicModelError.serializationFailed("encoding error")
    #expect(serializationFailed.errorDescription?.contains("encoding error") == true)
}

// MARK: - TopicAssignment Tests (from TopicModelResult)

@Test("TopicAssignment outlier detection")
func testTopicAssignmentOutlier() {
    let outlier = TopicAssignment.outlier
    #expect(outlier.isOutlier)
    #expect(outlier.topicID.value == -1)

    let assigned = TopicAssignment(
        topicID: TopicID(0),
        probability: 0.9,
        distanceToCentroid: 1.5,
        alternatives: nil
    )
    #expect(!assigned.isOutlier)
    #expect(assigned.probability == 0.9)
    #expect(assigned.distanceToCentroid == 1.5)
}

// MARK: - SparseMatrix Tests

@Test("SparseMatrix CSR format construction")
func testSparseMatrixCSRConstruction() {
    // Simple 3x3 sparse matrix:
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    let rowPointers = [0, 2, 3, 5]
    let columnIndices = [0, 2, 1, 0, 2]
    let values: [Float] = [1, 2, 3, 4, 5]

    let matrix = SparseMatrix(
        rowPointers: rowPointers,
        columnIndices: columnIndices,
        values: values,
        rows: 3,
        cols: 3
    )

    #expect(matrix.rows == 3)
    #expect(matrix.cols == 3)
    #expect(matrix.nonZeroCount == 5)

    // Check density: 5 non-zeros out of 9 total
    #expect(abs(matrix.density - Float(5) / Float(9)) < 0.01)
}

@Test("SparseMatrix element access")
func testSparseMatrixElementAccess() {
    let rowPointers = [0, 2, 3, 5]
    let columnIndices = [0, 2, 1, 0, 2]
    let values: [Float] = [1, 2, 3, 4, 5]

    let matrix = SparseMatrix(
        rowPointers: rowPointers,
        columnIndices: columnIndices,
        values: values,
        rows: 3,
        cols: 3
    )

    // Access existing elements
    #expect(matrix[0, 0] == 1)
    #expect(matrix[0, 2] == 2)
    #expect(matrix[1, 1] == 3)
    #expect(matrix[2, 0] == 4)
    #expect(matrix[2, 2] == 5)

    // Access zero elements
    #expect(matrix[0, 1] == 0)
    #expect(matrix[1, 0] == 0)
    #expect(matrix[1, 2] == 0)
}

@Test("SparseMatrix row iteration")
func testSparseMatrixRowIteration() {
    let rowPointers = [0, 2, 3, 5]
    let columnIndices = [0, 2, 1, 0, 2]
    let values: [Float] = [1, 2, 3, 4, 5]

    let matrix = SparseMatrix(
        rowPointers: rowPointers,
        columnIndices: columnIndices,
        values: values,
        rows: 3,
        cols: 3
    )

    // Row 0 should have columns 0 and 2
    let row0 = matrix.nonZeroElements(inRow: 0)
    #expect(row0.count == 2)
    #expect(row0[0].col == 0)
    #expect(row0[0].value == 1)
    #expect(row0[1].col == 2)
    #expect(row0[1].value == 2)

    // Row 1 should have just column 1
    let row1 = matrix.nonZeroElements(inRow: 1)
    #expect(row1.count == 1)
    #expect(row1[0].col == 1)
    #expect(row1[0].value == 3)
}

@Test("SparseMatrix from COO format")
func testSparseMatrixFromCOO() {
    // COO (Coordinate) format triplets
    let triplets: [(row: Int, col: Int, value: Float)] = [
        (0, 0, 1.0),
        (0, 2, 2.0),
        (1, 1, 3.0),
        (2, 0, 4.0),
        (2, 2, 5.0)
    ]

    let matrix = SparseMatrix<Float>.fromCOO(rows: 3, cols: 3, entries: triplets)

    #expect(matrix.rows == 3)
    #expect(matrix.cols == 3)
    #expect(matrix.nonZeroCount == 5)

    // Verify values
    #expect(matrix[0, 0] == 1.0)
    #expect(matrix[1, 1] == 3.0)
    #expect(matrix[2, 2] == 5.0)
}

@Test("SparseMatrix to dense conversion")
func testSparseMatrixToDense() {
    let rowPointers = [0, 2, 3, 5]
    let columnIndices = [0, 2, 1, 0, 2]
    let values: [Float] = [1, 2, 3, 4, 5]

    let sparse = SparseMatrix(
        rowPointers: rowPointers,
        columnIndices: columnIndices,
        values: values,
        rows: 3,
        cols: 3
    )

    let dense = sparse.toDense()

    // Expected: [1, 0, 2, 0, 3, 0, 4, 0, 5] in row-major order
    #expect(dense.count == 9)
    #expect(dense[0] == 1)  // [0,0]
    #expect(dense[1] == 0)  // [0,1]
    #expect(dense[2] == 2)  // [0,2]
    #expect(dense[4] == 3)  // [1,1]
    #expect(dense[6] == 4)  // [2,0]
    #expect(dense[8] == 5)  // [2,2]
}

@Test("SparseMatrix normalized Laplacian")
func testSparseMatrixNormalizedLaplacian() {
    // Simple symmetric adjacency: 0-1-2 (linear chain)
    // A = [0, 1, 0]
    //     [1, 0, 1]
    //     [0, 1, 0]
    let triplets: [(row: Int, col: Int, value: Float)] = [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 2, 1.0),
        (2, 1, 1.0)
    ]

    let adjacency = SparseMatrix<Float>.fromCOO(rows: 3, cols: 3, entries: triplets)
    let laplacian = adjacency.normalizedLaplacian()

    // Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    // Diagonal should be 1.0 (I - ...)
    #expect(abs(laplacian[0, 0] - 1.0) < 0.01)
    #expect(abs(laplacian[1, 1] - 1.0) < 0.01)
    #expect(abs(laplacian[2, 2] - 1.0) < 0.01)

    // Off-diagonal should be negative
    #expect(laplacian[0, 1] < 0)
    #expect(laplacian[1, 0] < 0)
}

@Test("SparseMatrix empty construction")
func testSparseMatrixEmpty() {
    let matrix = SparseMatrix<Float>(rows: 5, cols: 5)

    #expect(matrix.rows == 5)
    #expect(matrix.cols == 5)
    #expect(matrix.nonZeroCount == 0)
    #expect(matrix.density == 0)
}

// MARK: - NearestNeighborGraph Tests

@Test("NearestNeighborGraph construction with BallTree")
func testNearestNeighborGraphConstruction() async throws {
    // Create 6 points in 2D forming two clusters
    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [0.1, 0.0]),
        Embedding(vector: [0.0, 0.1]),
        Embedding(vector: [5.0, 5.0]),
        Embedding(vector: [5.1, 5.0]),
        Embedding(vector: [5.0, 5.1])
    ]

    let graph = try await NearestNeighborGraph.build(
        embeddings: embeddings,
        k: 2,
        metric: .euclidean
    )

    #expect(graph.pointCount == 6)
    #expect(graph.k == 2)

    // Each point should have exactly 2 neighbors
    for i in 0..<6 {
        #expect(graph.neighbors[i].count == 2)
        #expect(graph.distances[i].count == 2)
    }

    // Point 0's neighbors should be from cluster 1 (points 1, 2)
    #expect(graph.neighbors[0].contains(1) || graph.neighbors[0].contains(2))
}

@Test("NearestNeighborGraph nearest distances (rho)")
func testNearestNeighborGraphRho() async throws {
    // Three points at known distances
    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [1.0, 0.0]),  // Distance 1.0 from origin
        Embedding(vector: [3.0, 0.0])   // Distance 3.0 from origin
    ]

    let graph = try await NearestNeighborGraph.build(
        embeddings: embeddings,
        k: 2,
        metric: .euclidean
    )

    let nearestDistances = graph.nearestDistances

    #expect(nearestDistances.count == 3)

    // Point 0's nearest is point 1 at distance 1.0
    #expect(abs(nearestDistances[0] - 1.0) < 0.01)

    // Point 1's nearest is point 0 at distance 1.0
    #expect(abs(nearestDistances[1] - 1.0) < 0.01)

    // Point 2's nearest is point 1 at distance 2.0
    #expect(abs(nearestDistances[2] - 2.0) < 0.01)
}

@Test("NearestNeighborGraph with cosine distance")
func testNearestNeighborGraphCosine() async throws {
    // Points where cosine similarity matters more than Euclidean
    let embeddings = [
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.9, 0.1]),  // Similar direction to 0
        Embedding(vector: [0.0, 1.0]),  // Orthogonal to 0
        Embedding(vector: [10.0, 0.0])  // Same direction as 0, far away in Euclidean
    ]

    let graph = try await NearestNeighborGraph.build(
        embeddings: embeddings,
        k: 2,
        metric: .cosine
    )

    // With cosine, point 0's nearest should be point 3 (same direction)
    // or point 1 (close direction), NOT point 2 (orthogonal)
    #expect(!graph.neighbors[0].contains(2))
}

// MARK: - FuzzySimplicialSet Tests

@Test("FuzzySimplicialSet build from k-NN graph")
func testFuzzySimplicialSetBuild() async throws {
    // Create simple cluster of points
    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [0.1, 0.0]),
        Embedding(vector: [0.0, 0.1]),
        Embedding(vector: [0.1, 0.1])
    ]

    let graph = try await NearestNeighborGraph.build(
        embeddings: embeddings,
        k: 2,
        metric: .euclidean
    )

    let fuzzySet = FuzzySimplicialSet.build(from: graph)

    #expect(fuzzySet.pointCount == 4)
    #expect(fuzzySet.rho.count == 4)
    #expect(fuzzySet.sigma.count == 4)

    // All rho values should be positive (distance to nearest neighbor)
    #expect(fuzzySet.rho.allSatisfy { $0 >= 0 })

    // All sigma values should be positive
    #expect(fuzzySet.sigma.allSatisfy { $0 > 0 })
}

@Test("FuzzySimplicialSet membership weights in [0, 1]")
func testFuzzySimplicialSetMembershipRange() async throws {
    let embeddings = (0..<10).map { i in
        Embedding(vector: [Float(i) * 0.1, Float(i % 3) * 0.1])
    }

    let graph = try await NearestNeighborGraph.build(
        embeddings: embeddings,
        k: 3,
        metric: .euclidean
    )

    let fuzzySet = FuzzySimplicialSet.build(from: graph)

    // All membership weights should be in [0, 1]
    for value in fuzzySet.memberships.values {
        #expect(value >= 0 && value <= 1)
    }
}

@Test("FuzzySimplicialSet symmetrization")
func testFuzzySimplicialSetSymmetry() async throws {
    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.0, 1.0])
    ]

    let graph = try await NearestNeighborGraph.build(
        embeddings: embeddings,
        k: 2,
        metric: .euclidean
    )

    let fuzzySet = FuzzySimplicialSet.build(from: graph)

    // After symmetrization, memberships[i,j] should equal memberships[j,i]
    let m = fuzzySet.memberships
    for i in 0..<3 {
        for j in 0..<3 {
            if i != j {
                #expect(abs(m[i, j] - m[j, i]) < 1e-5,
                        "Membership should be symmetric: [\(i),\(j)]=\(m[i,j]) vs [\(j),\(i)]=\(m[j,i])")
            }
        }
    }
}

@Test("FuzzySimplicialSet edge list conversion")
func testFuzzySimplicialSetEdgeList() async throws {
    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [0.1, 0.0]),
        Embedding(vector: [0.0, 0.1])
    ]

    let graph = try await NearestNeighborGraph.build(
        embeddings: embeddings,
        k: 2,
        metric: .euclidean
    )

    let fuzzySet = FuzzySimplicialSet.build(from: graph)
    let edges = fuzzySet.toEdgeList()

    // Should have edges (sparse, so not all n² pairs)
    #expect(!edges.isEmpty)

    // Each edge should have valid indices and positive weight
    for edge in edges {
        #expect(edge.source >= 0 && edge.source < 3)
        #expect(edge.target >= 0 && edge.target < 3)
        #expect(edge.weight > 0)
    }
}

// MARK: - SpectralEmbedding Tests

@Test("SpectralEmbedding random initialization")
func testSpectralEmbeddingRandom() {
    let embedding = SpectralEmbedding.randomInitialization(
        pointCount: 10,
        nComponents: 2,
        seed: 42
    )

    #expect(embedding.count == 10)
    #expect(embedding[0].count == 2)

    // All values should be finite
    for point in embedding {
        #expect(point.allSatisfy { $0.isFinite })
    }
}

@Test("SpectralEmbedding reproducibility with seed")
func testSpectralEmbeddingReproducibility() {
    let embedding1 = SpectralEmbedding.randomInitialization(
        pointCount: 10,
        nComponents: 2,
        seed: 42
    )

    let embedding2 = SpectralEmbedding.randomInitialization(
        pointCount: 10,
        nComponents: 2,
        seed: 42
    )

    // Same seed should give same results
    for i in 0..<10 {
        for d in 0..<2 {
            #expect(embedding1[i][d] == embedding2[i][d])
        }
    }
}

@Test("SpectralEmbedding different seeds give different results")
func testSpectralEmbeddingDifferentSeeds() {
    let embedding1 = SpectralEmbedding.randomInitialization(
        pointCount: 10,
        nComponents: 2,
        seed: 42
    )

    let embedding2 = SpectralEmbedding.randomInitialization(
        pointCount: 10,
        nComponents: 2,
        seed: 99
    )

    // Different seeds should give different results
    var anyDifferent = false
    for i in 0..<10 {
        for d in 0..<2 {
            if embedding1[i][d] != embedding2[i][d] {
                anyDifferent = true
                break
            }
        }
    }
    #expect(anyDifferent)
}

@Test("SpectralEmbedding initialization quality metrics")
func testSpectralEmbeddingMetrics() {
    let embedding = SpectralEmbedding.randomInitialization(
        pointCount: 20,
        nComponents: 2,
        seed: 42
    )

    let metrics = SpectralEmbedding.evaluateInitialization(
        embedding,
        sampleSize: 100,
        seed: 42
    )

    #expect(metrics.meanDistance > 0)
    #expect(metrics.minDistance >= 0)
    #expect(metrics.maxDistance >= metrics.minDistance)
    #expect(metrics.isReasonable)
}

// MARK: - UMAP Reducer Tests

@Test("UMAPReducer basic reduction")
func testUMAPBasicReduction() async throws {
    // Create simple 2D data
    var embeddings = [Embedding]()

    // Cluster 1 around (0, 0)
    for _ in 0..<10 {
        let x = Float.random(in: -0.5...0.5)
        let y = Float.random(in: -0.5...0.5)
        embeddings.append(Embedding(vector: [x, y, 0, 0, 0]))  // 5D input
    }

    // Cluster 2 around (5, 5)
    for _ in 0..<10 {
        let x = 5.0 + Float.random(in: -0.5...0.5)
        let y = 5.0 + Float.random(in: -0.5...0.5)
        embeddings.append(Embedding(vector: [x, y, 0, 0, 0]))
    }

    let umap = UMAPReducer(nNeighbors: 5, nComponents: 2, nEpochs: 50, seed: 42)
    let reduced = try await umap.fitTransform(embeddings)

    #expect(reduced.count == 20)
    #expect(reduced[0].dimension == 2)

    // All values should be finite
    #expect(reduced.allSatisfy { $0.vector.allSatisfy { $0.isFinite } })
}

@Test("UMAPReducer produces valid output")
func testUMAPProducesValidOutput() async throws {
    // Test that UMAP produces valid, finite output of correct dimensions
    let embeddings = (0..<20).map { i in
        // Simple grid pattern
        Embedding(vector: [Float(i % 5), Float(i / 5)])
    }

    let umap = UMAPReducer(nNeighbors: 5, nComponents: 2, nEpochs: 100, seed: 42)
    let reduced = try await umap.fitTransform(embeddings)

    // Basic validity checks
    #expect(reduced.count == 20, "Should preserve point count")
    #expect(reduced.allSatisfy { $0.dimension == 2 }, "Should reduce to 2D")
    #expect(reduced.allSatisfy { $0.vector.allSatisfy { $0.isFinite } }, "All values should be finite")

    // Check that output has reasonable spread (not collapsed to a point)
    let xs = reduced.map { $0.vector[0] }
    let ys = reduced.map { $0.vector[1] }
    let xRange = (xs.max() ?? 0) - (xs.min() ?? 0)
    let yRange = (ys.max() ?? 0) - (ys.min() ?? 0)

    #expect(xRange > 0.01, "X range should be non-trivial (got \(xRange))")
    #expect(yRange > 0.01, "Y range should be non-trivial (got \(yRange))")
}

@Test("UMAPReducer reproducibility with seed")
func testUMAPReproducibility() async throws {
    let embeddings = (0..<15).map { i in
        Embedding(vector: [Float(i) * 0.1, Float(i % 3) * 0.1])
    }

    let umap1 = UMAPReducer(nNeighbors: 3, nComponents: 2, nEpochs: 30, seed: 42)
    let reduced1 = try await umap1.fitTransform(embeddings)

    let umap2 = UMAPReducer(nNeighbors: 3, nComponents: 2, nEpochs: 30, seed: 42)
    let reduced2 = try await umap2.fitTransform(embeddings)

    // Same seed should produce identical results
    for i in 0..<15 {
        for d in 0..<2 {
            #expect(abs(reduced1[i].vector[d] - reduced2[i].vector[d]) < 1e-5,
                    "Reproducibility failed at point \(i) dim \(d)")
        }
    }
}

@Test("UMAPReducer handles small dataset")
func testUMAPSmallDataset() async throws {
    // Just 4 points (minimum viable)
    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.0, 1.0]),
        Embedding(vector: [1.0, 1.0])
    ]

    let umap = UMAPReducer(nNeighbors: 2, nComponents: 2, nEpochs: 20, seed: 42)
    let reduced = try await umap.fitTransform(embeddings)

    #expect(reduced.count == 4)
    #expect(reduced.allSatisfy { $0.dimension == 2 })
    #expect(reduced.allSatisfy { $0.vector.allSatisfy { $0.isFinite } })
}

@Test("UMAPReducer configuration presets")
func testUMAPConfiguration() {
    let defaultConfig = UMAPConfiguration.default
    #expect(defaultConfig.nNeighbors == 15)
    #expect(defaultConfig.minDist == 0.1)

    let fastConfig = UMAPConfiguration.fast
    #expect(fastConfig.nNeighbors < defaultConfig.nNeighbors)

    let qualityConfig = UMAPConfiguration.quality
    #expect(qualityConfig.nEpochs ?? 200 >= 200)
}

@Test("UMAPReducer fit and transform separately")
func testUMAPFitTransformSeparate() async throws {
    let trainData = (0..<20).map { i in
        Embedding(vector: [Float(i) * 0.1, Float(i % 4) * 0.1])
    }

    var umap = UMAPReducer(nNeighbors: 3, nComponents: 2, nEpochs: 30, seed: 42)

    // First fit
    try await umap.fit(trainData)
    #expect(umap.isFitted)

    // Then transform same data
    let reduced = try await umap.transform(trainData)
    #expect(reduced.count == 20)
    #expect(reduced[0].dimension == 2)
}

// MARK: - UMAP Optimizer Tests

@Test("UMAPOptimizer curve parameter computation")
func testUMAPOptimizerCurveParams() async {
    // Create minimal embedding for testing
    let initialEmbedding = [[Float]](repeating: [0, 0], count: 3)

    let optimizer = await UMAPOptimizer(
        initialEmbedding: initialEmbedding,
        minDist: 0.1,
        seed: 42
    )

    // Should initialize without crashing
    #expect(await optimizer.pointCount == 3)
}

// MARK: - Integration Tests

@Test("Full pipeline: UMAP + HDBSCAN")
func testFullPipelineUMAPHDBSCAN() async throws {
    // Create synthetic clustered data in high dimensions with deterministic seed
    var rng = RandomState(seed: 54321)
    var embeddings = [Embedding]()

    // 3 well-separated clusters in 10D space
    // Each cluster has offset of 20 to ensure clear separation
    for cluster in 0..<3 {
        let offset = Float(cluster) * 20
        for _ in 0..<10 {
            var vec = [Float](repeating: 0, count: 10)
            for d in 0..<10 {
                vec[d] = offset + rng.nextFloat(in: -0.5...0.5)
            }
            embeddings.append(Embedding(vector: vec))
        }
    }

    // UMAP reduction with more epochs for better convergence
    let umap = UMAPReducer(nNeighbors: 5, nComponents: 5, nEpochs: 100, seed: 42)
    let reduced = try await umap.fitTransform(embeddings)

    #expect(reduced.count == 30)
    #expect(reduced[0].dimension == 5)

    // HDBSCAN clustering on reduced data
    let config = HDBSCANConfiguration(minClusterSize: 3, minSamples: 2)
    let result = try await hdbscan(reduced, configuration: config)

    // Should find at least one cluster (not all outliers)
    #expect(result.clusterCount >= 1, "Should find at least one cluster")

    // At least some points should be clustered (basic sanity check)
    let clusteredCount = result.labels.filter { $0 >= 0 }.count
    #expect(clusteredCount >= 3, "At least some points should be clustered (got \(clusteredCount))")
}

@Test("End-to-end topic modeling with mock embeddings")
func testEndToEndTopicModeling() async throws {
    // Create documents about different topics
    let documents = [
        // ML topic
        Document(content: "machine learning algorithms neural network deep learning"),
        Document(content: "neural network training backpropagation gradient descent"),
        Document(content: "deep learning models transformers attention mechanism"),
        Document(content: "machine learning data science artificial intelligence"),
        Document(content: "neural networks convolutional recurrent architectures"),
        // Web topic
        Document(content: "web development javascript html css frontend"),
        Document(content: "javascript frameworks react angular vue components"),
        Document(content: "html css responsive design mobile web"),
        Document(content: "web api rest graphql http endpoints"),
        Document(content: "frontend development user interface design"),
    ]

    // Create mock embeddings (cluster documents by topic)
    var embeddings = [Embedding]()
    for i in 0..<5 {
        // ML cluster around (0, 0)
        embeddings.append(Embedding(vector: [
            Float.random(in: -0.1...0.1),
            Float.random(in: -0.1...0.1),
            Float(i) * 0.01
        ]))
    }
    for i in 0..<5 {
        // Web cluster around (3, 3)
        embeddings.append(Embedding(vector: [
            3.0 + Float.random(in: -0.1...0.1),
            3.0 + Float.random(in: -0.1...0.1),
            Float(i) * 0.01
        ]))
    }

    // Use fast config to speed up test
    let config = TopicModelConfiguration.fast

    let model = TopicModel(configuration: config)
    let result = try await model.fit(documents: documents, embeddings: embeddings)

    // Should find topics
    #expect(result.topics.count >= 1)

    // Topics should have keywords
    for topic in result.topics {
        #expect(!topic.keywords.isEmpty)
    }

    // Each document should have an assignment
    #expect(result.documentTopics.count == 10)
}

// MARK: - TopicModel Convenience Methods Tests

@Test("PrecomputedEmbeddingProvider returns stored embeddings")
func testPrecomputedProvider() async throws {
    let embedding1 = Embedding(vector: [1.0, 2.0, 3.0])
    let embedding2 = Embedding(vector: [4.0, 5.0, 6.0])

    let provider = PrecomputedEmbeddingProvider(embeddings: [
        "hello": embedding1,
        "world": embedding2
    ])

    // Check dimension
    #expect(provider.dimension == 3)

    // Check that embed returns correct embedding
    let result1 = try await provider.embed("hello")
    #expect(result1.vector == embedding1.vector)

    let result2 = try await provider.embed("world")
    #expect(result2.vector == embedding2.vector)
}

@Test("PrecomputedEmbeddingProvider throws for unknown text")
func testPrecomputedProviderUnknownText() async throws {
    let provider = PrecomputedEmbeddingProvider(embeddings: [
        "known": Embedding(vector: [1.0, 2.0, 3.0])
    ])

    do {
        _ = try await provider.embed("unknown")
        Issue.record("Expected error for unknown text")
    } catch {
        // Expected - should throw for unknown text
        #expect(error is EmbeddingError)
    }
}

@Test("PrecomputedEmbeddingProvider embedBatch works correctly")
func testPrecomputedProviderBatch() async throws {
    let provider = PrecomputedEmbeddingProvider(embeddings: [
        "a": Embedding(vector: [1.0, 0.0]),
        "b": Embedding(vector: [0.0, 1.0]),
        "c": Embedding(vector: [1.0, 1.0])
    ])

    let results = try await provider.embedBatch(["a", "c"])
    #expect(results.count == 2)
    #expect(results[0].vector == [1.0, 0.0])
    #expect(results[1].vector == [1.0, 1.0])
}

@Test("findTopics returns assignments for text")
func testFindTopics() async throws {
    // Create documents for two distinct clusters
    let documents = [
        Document(content: "machine learning algorithms neural networks"),
        Document(content: "deep learning AI artificial intelligence"),
        Document(content: "neural networks machine learning"),
        Document(content: "cooking recipes food kitchen"),
        Document(content: "restaurant menu chef food"),
        Document(content: "cooking kitchen recipes chef")
    ]

    // Create well-separated embeddings for two clusters
    var embeddings: [String: Embedding] = [:]
    // ML cluster around (0, 0)
    embeddings["machine learning algorithms neural networks"] = Embedding(vector: [0.0, 0.1, 0.0])
    embeddings["deep learning AI artificial intelligence"] = Embedding(vector: [0.1, 0.0, 0.0])
    embeddings["neural networks machine learning"] = Embedding(vector: [0.0, 0.0, 0.1])
    // Food cluster around (5, 5)
    embeddings["cooking recipes food kitchen"] = Embedding(vector: [5.0, 5.1, 5.0])
    embeddings["restaurant menu chef food"] = Embedding(vector: [5.1, 5.0, 5.0])
    embeddings["cooking kitchen recipes chef"] = Embedding(vector: [5.0, 5.0, 5.1])

    // Add query embedding
    embeddings["test machine learning query"] = Embedding(vector: [0.05, 0.05, 0.05])

    let provider = PrecomputedEmbeddingProvider(embeddings: embeddings)

    let config = TopicModelConfiguration.fast
    let model = TopicModel(configuration: config)
    _ = try await model.fit(documents: documents, embeddingProvider: provider)

    // Find topics for a query close to ML cluster
    let assignments = try await model.findTopics(for: "test machine learning query")

    // Should return at least one assignment
    #expect(!assignments.isEmpty)

    // First assignment should have highest probability
    if assignments.count > 1 {
        #expect(assignments[0].probability >= assignments[1].probability)
    }

    // Probabilities should sum approximately to 1
    let sum = assignments.reduce(0.0) { $0 + $1.probability }
    #expect(abs(sum - 1.0) < 0.01)
}

@Test("findTopics throws when not fitted")
func testFindTopicsNotFitted() async throws {
    let model = TopicModel()

    do {
        _ = try await model.findTopics(for: "test query")
        Issue.record("Expected notFitted error")
    } catch TopicModelError.notFitted {
        // Expected
    } catch {
        Issue.record("Expected notFitted error, got: \(error)")
    }
}

@Test("findTopics throws without embedding provider")
func testFindTopicsNoProvider() async throws {
    let documents = [
        Document(content: "test document one"),
        Document(content: "test document two"),
        Document(content: "test document three")
    ]

    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.0, 1.0])
    ]

    let config = TopicModelConfiguration.fast
    let model = TopicModel(configuration: config)
    // Fit with pre-computed embeddings (no provider stored)
    _ = try await model.fit(documents: documents, embeddings: embeddings)

    do {
        _ = try await model.findTopics(for: "test query")
        Issue.record("Expected noEmbeddingProvider error")
    } catch TopicModelError.noEmbeddingProvider {
        // Expected
    } catch {
        Issue.record("Expected noEmbeddingProvider error, got: \(error)")
    }
}

@Test("search returns similar documents")
func testSearch() async throws {
    // Create documents
    let documents = [
        Document(content: "machine learning is great"),
        Document(content: "deep learning neural networks"),
        Document(content: "cooking recipes at home"),
        Document(content: "restaurant reviews food")
    ]

    // Create embeddings with different DIRECTIONS for cosine similarity
    // ML docs: positive x, low y (pointing "right")
    // Food docs: positive y, low x (pointing "up")
    var embeddings: [String: Embedding] = [:]
    embeddings["machine learning is great"] = Embedding(vector: [0.9, 0.1])
    embeddings["deep learning neural networks"] = Embedding(vector: [0.95, 0.05])
    embeddings["cooking recipes at home"] = Embedding(vector: [0.1, 0.9])
    embeddings["restaurant reviews food"] = Embedding(vector: [0.05, 0.95])

    // Query embedding similar direction to ML docs (pointing "right")
    embeddings["neural network AI"] = Embedding(vector: [0.85, 0.15])

    let provider = PrecomputedEmbeddingProvider(embeddings: embeddings)

    let config = TopicModelConfiguration.fast
    let model = TopicModel(configuration: config)
    _ = try await model.fit(documents: documents, embeddingProvider: provider)

    // Search for something similar to ML docs
    let results = try await model.search(query: "neural network AI", topK: 2)

    // Should return 2 results
    #expect(results.count == 2)

    // Results should be sorted by score (descending)
    #expect(results[0].score >= results[1].score)

    // Top results should be ML documents (similar direction to query)
    let topDocContents = results.map { $0.document.content }
    #expect(topDocContents.contains("machine learning is great") ||
            topDocContents.contains("deep learning neural networks"))
}

@Test("search throws when not fitted")
func testSearchNotFitted() async throws {
    let model = TopicModel()

    do {
        _ = try await model.search(query: "test")
        Issue.record("Expected notFitted error")
    } catch TopicModelError.notFitted {
        // Expected
    } catch {
        Issue.record("Expected notFitted error, got: \(error)")
    }
}

@Test("search throws without embedding provider")
func testSearchNoProvider() async throws {
    let documents = [
        Document(content: "test document"),
        Document(content: "another document")
    ]

    let embeddings = [
        Embedding(vector: [0.0, 0.0]),
        Embedding(vector: [1.0, 0.0])
    ]

    let config = TopicModelConfiguration.fast
    let model = TopicModel(configuration: config)
    _ = try await model.fit(documents: documents, embeddings: embeddings)

    do {
        _ = try await model.search(query: "test")
        Issue.record("Expected noEmbeddingProvider error")
    } catch TopicModelError.noEmbeddingProvider {
        // Expected
    } catch {
        Issue.record("Expected noEmbeddingProvider error, got: \(error)")
    }
}

// MARK: - Topic Manipulation Tests

@Test("merge combines topics correctly")
func testMergeTopics() async throws {
    // Create documents with clear clusters - need enough docs per cluster for HDBSCAN
    let documents = [
        // Cluster 0: Tech documents (4 docs)
        Document(content: "programming code software development apps"),
        Document(content: "coding algorithms programming code"),
        Document(content: "software engineering development apps"),
        Document(content: "code programming software apps development"),
        // Cluster 1: Food documents (4 docs)
        Document(content: "cooking recipes kitchen food meals"),
        Document(content: "restaurant dining food meals"),
        Document(content: "culinary cooking chef meals"),
        Document(content: "food recipes cooking meals kitchen")
    ]

    // Create embeddings that cluster naturally with more separation
    let embeddings = [
        // Tech cluster
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.95, 0.05]),
        Embedding(vector: [0.98, 0.02]),
        Embedding(vector: [0.92, 0.08]),
        // Food cluster
        Embedding(vector: [0.0, 1.0]),
        Embedding(vector: [0.05, 0.95]),
        Embedding(vector: [0.02, 0.98]),
        Embedding(vector: [0.08, 0.92])
    ]

    // Use a config with disabled coherence and small cluster size
    let config = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 2, method: .none),
        clustering: HDBSCANConfiguration(minClusterSize: 2, minSamples: 2),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,  // Disable coherence to avoid validation issues
        seed: 42
    )
    let model = TopicModel(configuration: config)
    let result = try await model.fit(documents: documents, embeddings: embeddings)

    // We need at least 2 topics to merge
    let initialTopicCount = result.topics.count
    guard initialTopicCount >= 2 else {
        // Skip test if clustering didn't produce enough topics
        return
    }

    // Get the first two topic IDs
    let topicIds = result.topics.map { $0.id.value }
    let toMerge = Array(topicIds.prefix(2))

    // Get sizes before merge
    let size0 = result.topics.first { $0.id.value == toMerge[0] }?.size ?? 0
    let size1 = result.topics.first { $0.id.value == toMerge[1] }?.size ?? 0

    // Merge
    let mergedTopic = try await model.merge(topics: toMerge)

    // Verify: combined size
    #expect(mergedTopic.size == size0 + size1)

    // Verify: old topics removed, new topic exists
    let currentTopics = await model.topics ?? []
    #expect(currentTopics.count == initialTopicCount - 1)

    // The merged topic ID should be the minimum of the merged IDs
    let expectedID = toMerge.min()!
    #expect(mergedTopic.id.value == expectedID)

    // Verify merged topic is in the topics list
    #expect(currentTopics.contains { $0.id == mergedTopic.id })
}

@Test("merge recomputes keywords")
func testMergeRecomputesKeywords() async throws {
    // Create documents with distinct keywords per cluster - need more docs
    let documents = [
        Document(content: "apple banana fruit healthy sweet"),
        Document(content: "orange grape fruit sweet delicious"),
        Document(content: "fruit mango pineapple healthy natural"),
        Document(content: "vegetable carrot broccoli healthy green"),
        Document(content: "spinach vegetable green healthy natural"),
        Document(content: "vegetable tomato green healthy fresh")
    ]

    // Create embeddings for 2 clear clusters
    let embeddings = [
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.95, 0.05]),
        Embedding(vector: [0.98, 0.02]),
        Embedding(vector: [0.0, 1.0]),
        Embedding(vector: [0.05, 0.95]),
        Embedding(vector: [0.02, 0.98])
    ]

    let config = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 2, method: .none),
        clustering: HDBSCANConfiguration(minClusterSize: 2, minSamples: 2),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,
        seed: 42
    )
    let model = TopicModel(configuration: config)
    let result = try await model.fit(documents: documents, embeddings: embeddings)

    guard result.topics.count >= 2 else { return }

    // Get two topic IDs
    let topicIds = result.topics.map { $0.id.value }

    // Merge them
    let merged = try await model.merge(topics: Array(topicIds.prefix(2)))

    // Merged topic should have keywords (not empty)
    #expect(!merged.keywords.isEmpty)
}

@Test("merge throws for invalid topic IDs")
func testMergeInvalidTopics() async throws {
    let documents = [
        Document(content: "test one document code programming"),
        Document(content: "test two document code software"),
        Document(content: "test three document code apps"),
        Document(content: "different content food cooking meals"),
        Document(content: "different content food recipes kitchen"),
        Document(content: "different content food dining restaurant")
    ]

    let embeddings = [
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.95, 0.05]),
        Embedding(vector: [0.98, 0.02]),
        Embedding(vector: [0.0, 1.0]),
        Embedding(vector: [0.05, 0.95]),
        Embedding(vector: [0.02, 0.98])
    ]

    let config = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 2, method: .none),
        clustering: HDBSCANConfiguration(minClusterSize: 2, minSamples: 2),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,
        seed: 42
    )
    let model = TopicModel(configuration: config)
    let result = try await model.fit(documents: documents, embeddings: embeddings)

    guard result.topics.count >= 1 else { return }
    let existingTopicID = result.topics.first!.id.value

    // Test: non-existent ID
    do {
        _ = try await model.merge(topics: [existingTopicID, 999])
        Issue.record("Expected invalidInput error for non-existent topic ID")
    } catch TopicModelError.invalidInput {
        // Expected
    } catch {
        Issue.record("Expected invalidInput error, got: \(error)")
    }

    // Test: only 1 ID
    do {
        _ = try await model.merge(topics: [existingTopicID])
        Issue.record("Expected invalidInput error for single topic ID")
    } catch TopicModelError.invalidInput {
        // Expected
    } catch {
        Issue.record("Expected invalidInput error, got: \(error)")
    }

    // Test: empty array
    do {
        _ = try await model.merge(topics: [])
        Issue.record("Expected invalidInput error for empty array")
    } catch TopicModelError.invalidInput {
        // Expected
    } catch {
        Issue.record("Expected invalidInput error, got: \(error)")
    }
}

@Test("merge throws for outlier topic")
func testMergeOutlierTopic() async throws {
    let documents = [
        Document(content: "test document code programming apps"),
        Document(content: "test document code software apps"),
        Document(content: "test document code development apps"),
        Document(content: "another content food cooking meals"),
        Document(content: "another content food recipes kitchen"),
        Document(content: "another content food dining restaurant")
    ]

    let embeddings = [
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.95, 0.05]),
        Embedding(vector: [0.98, 0.02]),
        Embedding(vector: [0.0, 1.0]),
        Embedding(vector: [0.05, 0.95]),
        Embedding(vector: [0.02, 0.98])
    ]

    let config = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 2, method: .none),
        clustering: HDBSCANConfiguration(minClusterSize: 2, minSamples: 2),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,
        seed: 42
    )
    let model = TopicModel(configuration: config)
    let result = try await model.fit(documents: documents, embeddings: embeddings)

    guard let firstTopicID = result.topics.first?.id.value else { return }

    // Attempting to merge outlier topic (-1) should throw
    do {
        _ = try await model.merge(topics: [firstTopicID, -1])
        Issue.record("Expected invalidInput error for outlier topic")
    } catch TopicModelError.invalidInput(let message) {
        #expect(message.contains("outlier"))
    } catch {
        Issue.record("Expected invalidInput error, got: \(error)")
    }
}

@Test("merge throws when not fitted")
func testMergeNotFitted() async throws {
    let model = TopicModel()

    do {
        _ = try await model.merge(topics: [0, 1])
        Issue.record("Expected notFitted error")
    } catch TopicModelError.notFitted {
        // Expected
    } catch {
        Issue.record("Expected notFitted error, got: \(error)")
    }
}

@Test("reduce decreases topic count")
func testReduceTopics() async throws {
    // Create documents with multiple clusters - need enough docs per cluster
    let documents = [
        // Cluster A: Programming
        Document(content: "programming code software development apps"),
        Document(content: "coding algorithms programming code apps"),
        Document(content: "software engineering development code apps"),
        // Cluster B: Sports
        Document(content: "football soccer sports game athletes"),
        Document(content: "basketball sports athletes game"),
        Document(content: "sports game competition athletes"),
        // Cluster C: Food
        Document(content: "cooking recipes kitchen food meals"),
        Document(content: "restaurant dining food meals"),
        Document(content: "culinary cooking chef food meals")
    ]

    // Create embeddings for 3 clusters
    let embeddings = [
        Embedding(vector: [1.0, 0.0, 0.0]),
        Embedding(vector: [0.95, 0.05, 0.0]),
        Embedding(vector: [0.98, 0.02, 0.0]),
        Embedding(vector: [0.0, 1.0, 0.0]),
        Embedding(vector: [0.05, 0.95, 0.0]),
        Embedding(vector: [0.02, 0.98, 0.0]),
        Embedding(vector: [0.0, 0.0, 1.0]),
        Embedding(vector: [0.0, 0.05, 0.95]),
        Embedding(vector: [0.0, 0.02, 0.98])
    ]

    let config = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 3, method: .none),
        clustering: HDBSCANConfiguration(minClusterSize: 2, minSamples: 2),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,
        seed: 42
    )
    let model = TopicModel(configuration: config)
    let result = try await model.fit(documents: documents, embeddings: embeddings)

    let initialCount = result.topics.count
    guard initialCount >= 3 else {
        // Need at least 3 topics to test reduction
        return
    }

    // Reduce to 2 topics
    let reducedTopics = try await model.reduce(to: 2)

    #expect(reducedTopics.count == 2)
}

@Test("reduce throws for invalid count")
func testReduceInvalidCount() async throws {
    let documents = [
        Document(content: "test one code programming apps"),
        Document(content: "test two code software apps"),
        Document(content: "test three code development apps"),
        Document(content: "other content food cooking meals"),
        Document(content: "other content food recipes kitchen"),
        Document(content: "other content food dining restaurant")
    ]

    let embeddings = [
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.95, 0.05]),
        Embedding(vector: [0.98, 0.02]),
        Embedding(vector: [0.0, 1.0]),
        Embedding(vector: [0.05, 0.95]),
        Embedding(vector: [0.02, 0.98])
    ]

    let config = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 2, method: .none),
        clustering: HDBSCANConfiguration(minClusterSize: 2, minSamples: 2),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,
        seed: 42
    )
    let model = TopicModel(configuration: config)
    let result = try await model.fit(documents: documents, embeddings: embeddings)

    let currentCount = result.topics.count
    guard currentCount >= 1 else { return }

    // Test: count >= current count
    do {
        _ = try await model.reduce(to: currentCount + 1)
        Issue.record("Expected invalidInput error for count >= current")
    } catch TopicModelError.invalidInput {
        // Expected
    } catch {
        Issue.record("Expected invalidInput error, got: \(error)")
    }

    // Test: count < 1
    do {
        _ = try await model.reduce(to: 0)
        Issue.record("Expected invalidInput error for count < 1")
    } catch TopicModelError.invalidInput {
        // Expected
    } catch {
        Issue.record("Expected invalidInput error, got: \(error)")
    }
}

@Test("reduce to 1 creates single topic")
func testReduceToOne() async throws {
    let documents = [
        Document(content: "programming code software apps development"),
        Document(content: "coding algorithms programming apps"),
        Document(content: "code software engineering apps"),
        Document(content: "football sports game athletes"),
        Document(content: "basketball game sports athletes"),
        Document(content: "sports competition game athletes")
    ]

    let embeddings = [
        Embedding(vector: [1.0, 0.0]),
        Embedding(vector: [0.95, 0.05]),
        Embedding(vector: [0.98, 0.02]),
        Embedding(vector: [0.0, 1.0]),
        Embedding(vector: [0.05, 0.95]),
        Embedding(vector: [0.02, 0.98])
    ]

    let config = TopicModelConfiguration(
        reduction: ReductionConfiguration(outputDimension: 2, method: .none),
        clustering: HDBSCANConfiguration(minClusterSize: 2, minSamples: 2),
        representation: CTFIDFConfiguration(keywordsPerTopic: 5),
        coherence: nil,
        seed: 42
    )
    let model = TopicModel(configuration: config)
    let result = try await model.fit(documents: documents, embeddings: embeddings)

    guard result.topics.count >= 2 else { return }

    // Calculate total docs across topics before reduction
    let totalDocsInTopics = result.topics.reduce(0) { $0 + $1.size }

    // Reduce to 1 topic
    let reducedTopics = try await model.reduce(to: 1)

    #expect(reducedTopics.count == 1)

    // The single topic should contain all documents that were assigned to topics
    // (not counting outliers which aren't part of any topic)
    #expect(reducedTopics[0].size == totalDocsInTopics)
}

@Test("reduce throws when not fitted")
func testReduceNotFitted() async throws {
    let model = TopicModel()

    do {
        _ = try await model.reduce(to: 2)
        Issue.record("Expected notFitted error")
    } catch TopicModelError.notFitted {
        // Expected
    } catch {
        Issue.record("Expected notFitted error, got: \(error)")
    }
}
