# Bug Fix: Eigendecomposition Convergence Failure

## Problem Summary

The Jacobi eigendecomposition algorithm in `Eigendecomposition.swift` fails to converge for certain input matrices, causing 3 PCA tests to fail with:

```
numericalInstability("Eigendecomposition failed: convergenceFailed")
```

## Failing Tests

```bash
swift test 2>&1 | grep -E "(failed|convergence)"
```

1. `testPCAArrayExtension` - Failed
2. `testPCAVarianceRatio` - Failed
3. `testPCAPreservesSampleCount` - convergenceFailed

## Root Cause Location

**File**: `Sources/SwiftTopics/Utilities/Eigendecomposition.swift`

The Jacobi algorithm was implemented to replace deprecated LAPACK `ssyev_`/`dsyev_` calls. The implementation has convergence issues with:
- Random covariance matrices from PCA
- Matrices with clustered eigenvalues
- Near-singular or ill-conditioned matrices

## Context

### Why Jacobi Was Chosen
- LAPACK's `ssyev_`/`dsyev_` are deprecated on Apple platforms
- Jacobi is numerically stable for symmetric matrices
- Pure Swift implementation (no C dependencies)

### The Algorithm (Current Implementation)
The Jacobi method iteratively applies Givens rotations to zero off-diagonal elements:
1. Find largest off-diagonal element
2. Compute rotation angle to zero it
3. Apply rotation to matrix and eigenvector accumulator
4. Repeat until off-diagonal norm < tolerance

### Suspected Issues
1. **Max iterations too low** - May need more iterations for large matrices
2. **Tolerance too tight** - `1e-10` may be unrealistic for Float32
3. **Rotation angle computation** - Edge cases when diagonal elements are equal
4. **Convergence check** - May be checking wrong condition

## Files to Read

1. **Primary**: `Sources/SwiftTopics/Utilities/Eigendecomposition.swift`
   - Contains the Jacobi implementation
   - Look for `convergenceFailed` error throw

2. **Consumer**: `Sources/SwiftTopics/Reduction/PCA.swift`
   - Calls `Eigendecomposition.symmetric()` in `fit()` method
   - Line ~163-170: eigendecomposition call with regularization

3. **Tests**: `Tests/SwiftTopicsTests/SwiftTopicsTests.swift`
   - Search for `testPCA` to see failing test patterns
   - Lines 445-700: PCA test section

## Expected Fix Approach

### Option A: Fix Jacobi Convergence (Preferred)
1. Increase max iterations (e.g., 100 → 1000)
2. Relax tolerance for Float32 (e.g., `1e-10` → `1e-6`)
3. Add fallback for slow convergence (switch to slower but guaranteed method)
4. Handle edge case: equal diagonal elements in rotation angle calculation

### Option B: Use Accelerate's Modern API
If Jacobi can't be fixed reliably, use `vDSP` or `BLAS` operations that aren't deprecated:
- `vDSP_eigenvectors` (if available)
- Implement power iteration for largest eigenvalues only
- Use QR algorithm via Householder reflections

### Option C: Hybrid Approach
- Use Jacobi for small matrices (< 50×50)
- Fall back to iterative methods for larger matrices

## Validation

After fixing, all tests must pass:

```bash
swift test 2>&1 | grep -E "testPCA"
```

Expected output (all passing):
```
✓ testPCAReducesDimensionality
✓ testPCAPreservesSampleCount
✓ testPCAVarianceRatio
✓ testPCAArrayExtension
... (all 15 PCA tests pass)
```

## Additional Test to Add

After fixing, add a dedicated eigendecomposition test:

```swift
@Test("Eigendecomposition converges for random symmetric matrix")
func testEigendecompositionConvergence() throws {
    // Create a random symmetric positive definite matrix
    let n = 20
    var matrix = [Float](repeating: 0, count: n * n)

    // Generate A = B^T * B (guaranteed positive semi-definite)
    var rng = RandomState(seed: 42)
    var B = (0..<(n*n)).map { _ in rng.nextFloat(in: -1...1) }

    // Compute B^T * B
    for i in 0..<n {
        for j in 0..<n {
            var sum: Float = 0
            for k in 0..<n {
                sum += B[k * n + i] * B[k * n + j]
            }
            matrix[i * n + j] = sum
        }
    }

    // Should not throw
    let result = try Eigendecomposition.symmetric(matrix, dimension: n)

    // Eigenvalues should be non-negative (positive semi-definite)
    #expect(result.eigenvalues.allSatisfy { $0 >= -1e-5 })

    // Should have correct count
    #expect(result.eigenvalues.count == n)
    #expect(result.eigenvectors.count == n * n)
}
```

## Commands

```bash
# Run just PCA tests
swift test --filter "testPCA"

# Run full test suite
swift test

# Build only (faster iteration)
swift build
```

## Exit Criteria

- [ ] All 3 failing PCA tests pass
- [ ] No new test failures introduced
- [ ] `swift build` has no warnings related to eigendecomposition
- [ ] Add at least 1 dedicated eigendecomposition convergence test

---

*Created: January 2025*
*Priority: Critical - Blocking production use*
