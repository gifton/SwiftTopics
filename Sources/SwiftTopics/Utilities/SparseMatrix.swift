// SparseMatrix.swift
// SwiftTopics
//
// Compressed Sparse Row (CSR) matrix for efficient sparse operations

import Foundation

// MARK: - Sparse Matrix

/// A sparse matrix in Compressed Sparse Row (CSR) format.
///
/// CSR is efficient for row-wise operations and sparse matrix-vector multiplication.
/// It stores only non-zero values, making it ideal for large sparse matrices like
/// the fuzzy simplicial set in UMAP where each row has only ~k non-zero entries.
///
/// ## Memory Layout
/// - `rowPointers`: Index into `columnIndices` and `values` for each row start
/// - `columnIndices`: Column index for each non-zero value
/// - `values`: The non-zero values
///
/// For an m×n matrix with nnz non-zeros:
/// - rowPointers has m+1 elements
/// - columnIndices and values each have nnz elements
///
/// ## Example
/// ```
/// Matrix:          CSR representation:
/// [1 0 2]          rowPointers:   [0, 2, 3, 5]
/// [0 3 0]          columnIndices: [0, 2, 1, 0, 2]
/// [4 0 5]          values:        [1, 2, 3, 4, 5]
/// ```
///
/// ## Performance Characteristics
/// - Element access: O(k) where k = nnz in that row (binary search possible)
/// - Row iteration: O(nnz_row)
/// - Space: O(nnz + rows)
/// - Construction from COO: O(nnz log nnz)
///
/// ## Thread Safety
/// `SparseMatrix` is immutable and `Sendable`. Safe for concurrent reads.
public struct SparseMatrix<T: Numeric & Sendable>: Sendable {

    // MARK: - Properties

    /// Row pointers (CSR format). Length = rows + 1.
    /// rowPointers[i] is the index into columnIndices/values where row i starts.
    /// rowPointers[rows] = total number of non-zeros.
    public let rowPointers: [Int]

    /// Column indices for each non-zero value. Length = nnz.
    public let columnIndices: [Int]

    /// Non-zero values. Length = nnz.
    public let values: [T]

    /// Number of rows.
    public let rows: Int

    /// Number of columns.
    public let cols: Int

    /// Number of non-zero elements.
    @inlinable
    public var nonZeroCount: Int {
        values.count
    }

    /// Density of the matrix (nnz / total elements).
    @inlinable
    public var density: Float {
        let total = Float(rows * cols)
        guard total > 0 else { return 0 }
        return Float(nonZeroCount) / total
    }

    // MARK: - Initialization

    /// Creates a sparse matrix from CSR components.
    ///
    /// - Parameters:
    ///   - rowPointers: Row pointer array (length = rows + 1).
    ///   - columnIndices: Column indices for non-zeros.
    ///   - values: Non-zero values.
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    public init(
        rowPointers: [Int],
        columnIndices: [Int],
        values: [T],
        rows: Int,
        cols: Int
    ) {
        precondition(rowPointers.count == rows + 1, "rowPointers must have rows+1 elements")
        precondition(columnIndices.count == values.count, "columnIndices and values must have same length")
        precondition(rowPointers.last == values.count, "Last rowPointer must equal nnz")

        self.rowPointers = rowPointers
        self.columnIndices = columnIndices
        self.values = values
        self.rows = rows
        self.cols = cols
    }

    /// Creates an empty sparse matrix.
    ///
    /// - Parameters:
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    public init(rows: Int, cols: Int) {
        self.rowPointers = [Int](repeating: 0, count: rows + 1)
        self.columnIndices = []
        self.values = []
        self.rows = rows
        self.cols = cols
    }

    // MARK: - Element Access

    /// Gets the value at (row, col), returning zero if not present.
    ///
    /// - Complexity: O(k) where k = number of non-zeros in row.
    public subscript(row: Int, col: Int) -> T {
        precondition(row >= 0 && row < rows, "Row index out of bounds")
        precondition(col >= 0 && col < cols, "Column index out of bounds")

        let start = rowPointers[row]
        let end = rowPointers[row + 1]

        // Linear search within row (could use binary search for sorted columns)
        for i in start..<end {
            if columnIndices[i] == col {
                return values[i]
            }
        }

        return 0  // Not found - return zero
    }

    /// Gets the value at (row, col), returning nil if not present.
    ///
    /// - Complexity: O(k) where k = number of non-zeros in row.
    public func get(row: Int, col: Int) -> T? {
        guard row >= 0 && row < rows && col >= 0 && col < cols else {
            return nil
        }

        let start = rowPointers[row]
        let end = rowPointers[row + 1]

        for i in start..<end {
            if columnIndices[i] == col {
                return values[i]
            }
        }

        return nil
    }

    // MARK: - Row Operations

    /// Returns all non-zero elements in a row.
    ///
    /// - Parameter row: The row index.
    /// - Returns: Array of (column, value) pairs.
    /// - Complexity: O(k) where k = number of non-zeros in row.
    public func nonZeroElements(inRow row: Int) -> [(col: Int, value: T)] {
        precondition(row >= 0 && row < rows, "Row index out of bounds")

        let start = rowPointers[row]
        let end = rowPointers[row + 1]

        return (start..<end).map { i in
            (col: columnIndices[i], value: values[i])
        }
    }

    /// Returns the number of non-zeros in a row.
    ///
    /// - Parameter row: The row index.
    /// - Returns: Number of non-zero elements.
    @inlinable
    public func nonZeroCount(inRow row: Int) -> Int {
        rowPointers[row + 1] - rowPointers[row]
    }

    /// Iterates over all non-zero elements.
    ///
    /// - Parameter body: Closure called for each (row, col, value).
    public func forEachNonZero(_ body: (Int, Int, T) -> Void) {
        for row in 0..<rows {
            let start = rowPointers[row]
            let end = rowPointers[row + 1]
            for i in start..<end {
                body(row, columnIndices[i], values[i])
            }
        }
    }

    // MARK: - Matrix Operations

    /// Computes the sum of all values in a row.
    ///
    /// - Parameter row: The row index.
    /// - Returns: Sum of non-zero values.
    public func rowSum(_ row: Int) -> T {
        let start = rowPointers[row]
        let end = rowPointers[row + 1]

        var sum: T = 0
        for i in start..<end {
            sum = sum + values[i]
        }
        return sum
    }

    /// Computes the transpose of this matrix.
    ///
    /// - Returns: Transposed sparse matrix.
    /// - Complexity: O(nnz)
    public func transposed() -> SparseMatrix<T> {
        // Count non-zeros per column (will be rows in transpose)
        var colCounts = [Int](repeating: 0, count: cols)
        for col in columnIndices {
            colCounts[col] += 1
        }

        // Build row pointers for transpose
        var transposeRowPointers = [Int](repeating: 0, count: cols + 1)
        var cumulative = 0
        for col in 0..<cols {
            transposeRowPointers[col] = cumulative
            cumulative += colCounts[col]
        }
        transposeRowPointers[cols] = cumulative

        // Fill transpose data
        var transposeColIndices = [Int](repeating: 0, count: nonZeroCount)
        var transposeValues = [T](repeating: 0, count: nonZeroCount)
        var currentPos = transposeRowPointers  // Working copy

        for row in 0..<rows {
            let start = rowPointers[row]
            let end = rowPointers[row + 1]
            for i in start..<end {
                let col = columnIndices[i]
                let pos = currentPos[col]
                transposeColIndices[pos] = row
                transposeValues[pos] = values[i]
                currentPos[col] += 1
            }
        }

        return SparseMatrix(
            rowPointers: transposeRowPointers,
            columnIndices: transposeColIndices,
            values: transposeValues,
            rows: cols,
            cols: rows
        )
    }
}

// MARK: - COO Construction

extension SparseMatrix {

    /// A coordinate (COO) format entry.
    public struct COOEntry: Sendable {
        public let row: Int
        public let col: Int
        public let value: T

        public init(row: Int, col: Int, value: T) {
            self.row = row
            self.col = col
            self.value = value
        }
    }

    /// Creates a sparse matrix from COO (coordinate) format.
    ///
    /// COO format is a simple list of (row, col, value) triplets.
    /// This method converts to CSR for efficient operations.
    ///
    /// - Parameters:
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    ///   - entries: List of (row, col, value) entries.
    /// - Returns: Sparse matrix in CSR format.
    /// - Complexity: O(nnz log nnz) for sorting.
    public static func fromCOO(
        rows: Int,
        cols: Int,
        entries: [(row: Int, col: Int, value: T)]
    ) -> SparseMatrix<T> {
        guard !entries.isEmpty else {
            return SparseMatrix(rows: rows, cols: cols)
        }

        // Sort entries by row, then by column for CSR construction
        let sorted = entries.sorted { a, b in
            if a.row != b.row {
                return a.row < b.row
            }
            return a.col < b.col
        }

        // Build CSR arrays
        var rowPointers = [Int](repeating: 0, count: rows + 1)
        var columnIndices = [Int]()
        var values = [T]()

        columnIndices.reserveCapacity(sorted.count)
        values.reserveCapacity(sorted.count)

        var currentRow = 0
        for entry in sorted {
            // Fill row pointers for empty rows
            while currentRow < entry.row {
                currentRow += 1
                rowPointers[currentRow] = columnIndices.count
            }

            columnIndices.append(entry.col)
            values.append(entry.value)
        }

        // Fill remaining row pointers
        while currentRow < rows {
            currentRow += 1
            rowPointers[currentRow] = columnIndices.count
        }

        return SparseMatrix(
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values,
            rows: rows,
            cols: cols
        )
    }

    /// Creates a sparse matrix from COO entries, combining duplicates.
    ///
    /// If multiple entries have the same (row, col), their values are summed.
    ///
    /// - Parameters:
    ///   - rows: Number of rows.
    ///   - cols: Number of columns.
    ///   - entries: List of entries (may contain duplicates).
    ///   - combine: Function to combine duplicate values (default: addition).
    /// - Returns: Sparse matrix with duplicates combined.
    public static func fromCOOCombiningDuplicates(
        rows: Int,
        cols: Int,
        entries: [(row: Int, col: Int, value: T)],
        combine: (T, T) -> T = { $0 + $1 }
    ) -> SparseMatrix<T> {
        guard !entries.isEmpty else {
            return SparseMatrix(rows: rows, cols: cols)
        }

        // Use dictionary to combine duplicates
        var combined: [Int: [Int: T]] = [:]
        for entry in entries {
            if combined[entry.row] == nil {
                combined[entry.row] = [:]
            }
            if let existing = combined[entry.row]![entry.col] {
                combined[entry.row]![entry.col] = combine(existing, entry.value)
            } else {
                combined[entry.row]![entry.col] = entry.value
            }
        }

        // Convert to sorted list
        var sortedEntries: [(row: Int, col: Int, value: T)] = []
        for row in combined.keys.sorted() {
            for col in combined[row]!.keys.sorted() {
                sortedEntries.append((row: row, col: col, value: combined[row]![col]!))
            }
        }

        return fromCOO(rows: rows, cols: cols, entries: sortedEntries)
    }
}

// MARK: - Float-Specific Operations

extension SparseMatrix where T == Float {

    /// Computes sparse matrix-vector product: y = A × x
    ///
    /// - Parameter x: Dense vector (length = cols).
    /// - Returns: Dense result vector (length = rows).
    /// - Complexity: O(nnz)
    public func matVec(_ x: [Float]) -> [Float] {
        precondition(x.count == cols, "Vector length must equal number of columns")

        var result = [Float](repeating: 0, count: rows)

        for row in 0..<rows {
            let start = rowPointers[row]
            let end = rowPointers[row + 1]

            var sum: Float = 0
            for i in start..<end {
                sum += values[i] * x[columnIndices[i]]
            }
            result[row] = sum
        }

        return result
    }

    /// Computes the degree (row sum) for each row.
    ///
    /// - Returns: Array of row sums.
    public func degrees() -> [Float] {
        (0..<rows).map { rowSum($0) }
    }

    /// Normalizes rows so each row sums to 1.
    ///
    /// - Returns: Row-normalized sparse matrix.
    public func rowNormalized() -> SparseMatrix<Float> {
        var normalizedValues = [Float](repeating: 0, count: nonZeroCount)

        for row in 0..<rows {
            let start = rowPointers[row]
            let end = rowPointers[row + 1]
            let sum = rowSum(row)

            if sum > Float.ulpOfOne {
                for i in start..<end {
                    normalizedValues[i] = values[i] / sum
                }
            }
        }

        return SparseMatrix(
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: normalizedValues,
            rows: rows,
            cols: cols
        )
    }

    /// Computes the normalized graph Laplacian: L = I - D^(-1/2) × W × D^(-1/2)
    ///
    /// Used for spectral embedding initialization.
    ///
    /// - Returns: Normalized Laplacian as sparse matrix.
    public func normalizedLaplacian() -> SparseMatrix<Float> {
        // Compute D^(-1/2) where D is diagonal of row sums
        let degrees = self.degrees()
        var invSqrtDegrees = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            if degrees[i] > Float.ulpOfOne {
                invSqrtDegrees[i] = 1.0 / degrees[i].squareRoot()
            }
        }

        // Compute L = I - D^(-1/2) W D^(-1/2)
        // For each entry W[i,j], the corresponding Laplacian entry is:
        // L[i,j] = -invSqrtDegrees[i] * W[i,j] * invSqrtDegrees[j]
        // L[i,i] = 1 (diagonal)

        var entries: [(row: Int, col: Int, value: Float)] = []
        entries.reserveCapacity(nonZeroCount + rows)

        // Track which diagonals we've seen
        var hasDiagonal = [Bool](repeating: false, count: rows)

        for row in 0..<rows {
            let start = rowPointers[row]
            let end = rowPointers[row + 1]

            for i in start..<end {
                let col = columnIndices[i]
                let value = values[i]
                let laplacianValue = -invSqrtDegrees[row] * value * invSqrtDegrees[col]

                if row == col {
                    // Diagonal entry: 1 - normalized value
                    entries.append((row: row, col: col, value: 1.0 + laplacianValue))
                    hasDiagonal[row] = true
                } else {
                    entries.append((row: row, col: col, value: laplacianValue))
                }
            }
        }

        // Add diagonal entries that weren't in the original matrix
        for row in 0..<rows where !hasDiagonal[row] {
            entries.append((row: row, col: row, value: 1.0))
        }

        return SparseMatrix.fromCOO(rows: rows, cols: cols, entries: entries)
    }

    /// Symmetrizes the matrix using the fuzzy set union formula:
    /// sym(A)[i,j] = A[i,j] + A[j,i] - A[i,j] * A[j,i]
    ///
    /// This is used in UMAP to combine directed membership weights.
    ///
    /// - Returns: Symmetrized sparse matrix.
    public func fuzzySetUnion() -> SparseMatrix<Float> {
        // Collect all entries and their transposes
        var combined: [Int: [Int: Float]] = [:]

        // Initialize with original entries
        forEachNonZero { row, col, value in
            if combined[row] == nil {
                combined[row] = [:]
            }
            combined[row]![col] = value
        }

        // Build symmetric version using fuzzy union
        var entries: [(row: Int, col: Int, value: Float)] = []
        entries.reserveCapacity(nonZeroCount * 2)

        // Track processed pairs to avoid duplicates
        var processed = Set<Int>()

        for row in 0..<rows {
            let start = rowPointers[row]
            let end = rowPointers[row + 1]

            for i in start..<end {
                let col = columnIndices[i]

                // Create unique key for this pair (use smaller index first)
                let pairKey = row < col ? row * cols + col : col * cols + row

                if !processed.contains(pairKey) {
                    processed.insert(pairKey)

                    let a = values[i]  // A[row, col]
                    let b = get(row: col, col: row) ?? 0  // A[col, row]

                    // Fuzzy set union: a + b - a*b
                    let symValue = a + b - a * b

                    if symValue > Float.ulpOfOne {
                        entries.append((row: row, col: col, value: symValue))
                        if row != col {
                            entries.append((row: col, col: row, value: symValue))
                        }
                    }
                }
            }
        }

        return SparseMatrix.fromCOO(rows: rows, cols: cols, entries: entries)
    }

    /// Converts to a dense matrix.
    ///
    /// - Warning: Only use for small matrices. Memory = O(rows × cols).
    /// - Returns: Dense matrix in row-major order.
    public func toDense() -> [Float] {
        var dense = [Float](repeating: 0, count: rows * cols)
        forEachNonZero { row, col, value in
            dense[row * cols + col] = value
        }
        return dense
    }
}

// MARK: - Edge List Conversion

extension SparseMatrix where T == Float {

    /// An edge with weight for graph-based operations.
    public struct Edge: Sendable {
        public let source: Int
        public let target: Int
        public let weight: Float

        public init(source: Int, target: Int, weight: Float) {
            self.source = source
            self.target = target
            self.weight = weight
        }
    }

    /// Converts to an edge list (for SGD optimization).
    ///
    /// - Returns: Array of weighted edges.
    public func toEdgeList() -> [Edge] {
        var edges: [Edge] = []
        edges.reserveCapacity(nonZeroCount)

        forEachNonZero { row, col, value in
            edges.append(Edge(source: row, target: col, weight: value))
        }

        return edges
    }

    /// Creates a sparse matrix from an edge list.
    ///
    /// - Parameters:
    ///   - edges: Array of weighted edges.
    ///   - nodeCount: Number of nodes (determines matrix size).
    /// - Returns: Adjacency matrix.
    public static func fromEdgeList(_ edges: [Edge], nodeCount: Int) -> SparseMatrix<Float> {
        let entries = edges.map { edge in
            (row: edge.source, col: edge.target, value: edge.weight)
        }
        return fromCOO(rows: nodeCount, cols: nodeCount, entries: entries)
    }
}

// MARK: - Codable Conformance

extension SparseMatrix: Codable where T: Codable {
    enum CodingKeys: String, CodingKey {
        case rowPointers, columnIndices, values, rows, cols
    }
}

// MARK: - CustomStringConvertible

extension SparseMatrix: CustomStringConvertible {
    public var description: String {
        "SparseMatrix<\(T.self)>(\(rows)×\(cols), nnz=\(nonZeroCount), density=\(String(format: "%.2f%%", density * 100)))"
    }
}
