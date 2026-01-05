// HungarianMatcher.swift
// SwiftTopics
//
// Optimal bipartite matching using the Hungarian algorithm

import Foundation

// MARK: - Hungarian Matcher

/// Optimal bipartite matching using the Hungarian algorithm.
///
/// The Hungarian algorithm (also known as Kuhn-Munkres algorithm) solves
/// the assignment problem in O(n³) time: given an n×m cost matrix, find
/// the assignment of rows to columns that minimizes total cost.
///
/// ## Topic Matching Use Case
///
/// For topic matching, we use this to find the optimal mapping between
/// new topics (rows) and old topics (columns) where:
/// - Cost = 1 - cosineSimilarity(newCentroid, oldCentroid)
/// - Lower cost = higher similarity = better match
///
/// ## Algorithm Overview
///
/// 1. **Reduction**: Subtract row/column minimums to create zeros
/// 2. **Covering**: Find minimum lines to cover all zeros
/// 3. **Augmentation**: If lines < size, adjust and repeat
/// 4. **Assignment**: Extract optimal assignment from final matrix
///
/// ## Thread Safety
///
/// `HungarianMatcher` is `Sendable` and stateless.
public struct HungarianMatcher: Sendable {

    // MARK: - Public API

    /// Finds the optimal assignment for a cost matrix.
    ///
    /// Returns the assignment that minimizes total cost.
    ///
    /// - Parameter costs: An n×m cost matrix where `costs[i][j]` is the cost
    ///   of assigning row i to column j. Use `Float.infinity` for forbidden
    ///   assignments.
    /// - Returns: Array of (row, col) pairs representing the optimal assignment.
    ///   Each row appears at most once, and each column appears at most once.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let costs: [[Float]] = [
    ///     [0.1, 0.5, 0.3],  // Row 0
    ///     [0.4, 0.2, 0.6],  // Row 1
    ///     [0.3, 0.4, 0.1]   // Row 2
    /// ]
    /// let matcher = HungarianMatcher()
    /// let result = matcher.solve(costs: costs)
    /// // result: [(0, 0), (1, 1), (2, 2)] with total cost 0.4
    /// ```
    ///
    /// ## Complexity
    ///
    /// - Time: O(n³) where n = max(rows, cols)
    /// - Space: O(n²)
    public func solve(costs: [[Float]]) -> [(row: Int, col: Int)] {
        guard !costs.isEmpty, !costs[0].isEmpty else {
            return []
        }

        let nRows = costs.count
        let nCols = costs[0].count

        // Make the matrix square by padding with high-cost dummy entries
        let n = max(nRows, nCols)
        var matrix = makeSquareMatrix(costs: costs, size: n)

        // Run Hungarian algorithm
        let assignment = hungarianSolve(&matrix, size: n)

        // Filter to only include valid (non-padded) assignments
        return assignment.filter { $0.row < nRows && $0.col < nCols }
    }

    /// Finds the optimal assignment and returns total cost.
    ///
    /// - Parameter costs: The cost matrix.
    /// - Returns: Tuple of (assignments, totalCost).
    public func solveWithCost(costs: [[Float]]) -> (assignments: [(row: Int, col: Int)], totalCost: Float) {
        let assignments = solve(costs: costs)
        var totalCost: Float = 0
        for (row, col) in assignments {
            totalCost += costs[row][col]
        }
        return (assignments, totalCost)
    }

    // MARK: - Private Implementation

    /// Creates a square matrix by padding with high-cost dummy entries.
    private func makeSquareMatrix(costs: [[Float]], size: Int) -> [[Float]] {
        let nRows = costs.count
        let nCols = costs[0].count
        let padding: Float = 1e9  // High cost for padding (not infinity to avoid arithmetic issues)

        var matrix = [[Float]](repeating: [Float](repeating: padding, count: size), count: size)
        for i in 0..<nRows {
            for j in 0..<nCols {
                matrix[i][j] = costs[i][j]
            }
        }
        return matrix
    }

    /// Core Hungarian algorithm implementation.
    ///
    /// This implements the Kuhn-Munkres algorithm with O(n³) complexity.
    private func hungarianSolve(_ matrix: inout [[Float]], size n: Int) -> [(row: Int, col: Int)] {
        // u[i] = potential for row i
        // v[j] = potential for column j
        var u = [Float](repeating: 0, count: n + 1)
        var v = [Float](repeating: 0, count: n + 1)

        // p[j] = which row is matched to column j (0 = unmatched)
        var p = [Int](repeating: 0, count: n + 1)

        // way[j] = previous column in augmenting path
        var way = [Int](repeating: 0, count: n + 1)

        // Process each row
        for i in 1...n {
            // Start matching row i-1 (0-indexed in matrix)

            // p[0] temporarily holds current row being matched
            p[0] = i

            // Column j0 = 0 is the "virtual" starting column
            var j0 = 0

            // minv[j] = minimum reduced cost to reach column j
            var minv = [Float](repeating: Float.infinity, count: n + 1)

            // used[j] = whether column j is in current alternating tree
            var used = [Bool](repeating: false, count: n + 1)

            // Build alternating tree until we find an augmenting path
            repeat {
                used[j0] = true
                let i0 = p[j0]  // Row matched to column j0 (or current row if j0=0)
                var delta: Float = .infinity
                var j1 = 0

                // Find minimum reduced cost to unvisited columns
                for j in 1...n {
                    if !used[j] {
                        // Reduced cost = matrix cost - potentials
                        let cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j] {
                            minv[j] = cur
                            way[j] = j0
                        }
                        if minv[j] < delta {
                            delta = minv[j]
                            j1 = j
                        }
                    }
                }

                // Update potentials
                for j in 0...n {
                    if used[j] {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }

                j0 = j1
            } while p[j0] != 0  // Until we reach unmatched column

            // Augment along the path
            repeat {
                let j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while j0 != 0
        }

        // Extract assignment from p[]
        var result = [(row: Int, col: Int)]()
        for j in 1...n {
            if p[j] != 0 {
                result.append((row: p[j] - 1, col: j - 1))
            }
        }

        // Sort by row for consistency
        result.sort { $0.row < $1.row }

        return result
    }
}

// MARK: - Hungarian Matcher Extensions

extension HungarianMatcher {

    /// Creates a cost matrix from similarity matrix.
    ///
    /// Converts similarities (higher = better) to costs (lower = better)
    /// using the formula: cost = 1 - similarity
    ///
    /// - Parameter similarities: An n×m similarity matrix with values in [0, 1].
    /// - Returns: The corresponding cost matrix.
    public static func costMatrixFromSimilarities(_ similarities: [[Float]]) -> [[Float]] {
        similarities.map { row in
            row.map { 1.0 - $0 }
        }
    }

    /// Creates a cost matrix from distances.
    ///
    /// Distances are already costs (lower = better), so this just validates
    /// and returns the matrix.
    ///
    /// - Parameter distances: An n×m distance matrix.
    /// - Returns: The cost matrix (same as input).
    public static func costMatrixFromDistances(_ distances: [[Float]]) -> [[Float]] {
        distances
    }
}
