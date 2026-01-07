// HardwareInfo.swift
// SwiftTopicsTests
//
// System introspection via sysctl and Metal for benchmark context.
// Part of Day 3: Polish

import Foundation
import Metal

// MARK: - Hardware Info

/// Detailed hardware information for benchmark context.
///
/// Captures system, CPU, memory, and GPU information to provide
/// context for benchmark results. This enables:
/// - Reproducing results on similar hardware
/// - Understanding performance differences across machines
/// - Estimating theoretical performance limits
///
/// ## Capture
/// ```swift
/// let info = HardwareInfo.capture()
/// print(info.summary)
/// ```
///
/// ## Thread Safety
/// `HardwareInfo` is `Sendable` and can be safely shared across concurrency domains.
public struct HardwareInfo: Sendable, Codable {

    // MARK: - System

    /// Operating system version (e.g., "macOS 26.0").
    public let osVersion: String

    /// Swift version (e.g., "6.2").
    public let swiftVersion: String

    /// CPU architecture (e.g., "arm64").
    public let architecture: String

    // MARK: - CPU

    /// CPU model name (e.g., "Apple M3 Max").
    public let cpuModel: String

    /// Total number of CPU cores.
    public let cpuCoreCount: Int

    /// Number of performance cores (estimated).
    public let cpuPerformanceCores: Int

    /// Number of efficiency cores (estimated).
    public let cpuEfficiencyCores: Int

    // MARK: - Memory

    /// Total system RAM in bytes.
    public let totalRAM: UInt64

    /// Available RAM in bytes (at capture time).
    public let availableRAM: UInt64

    // MARK: - GPU

    /// GPU device name (e.g., "Apple M3 Max").
    public let gpuName: String

    /// Number of GPU cores (from Metal).
    public let gpuCoreCount: Int

    /// GPU memory in bytes (unified memory).
    public let gpuMemory: UInt64

    /// Metal feature set version (e.g., "Metal 4").
    public let metalVersion: String

    // MARK: - Theoretical Performance

    /// Estimated GPU memory bandwidth in GB/s (nil if unknown).
    public let gpuMemoryBandwidth: Double?

    /// Estimated GPU peak TFLOPS (nil if unknown).
    public let gpuPeakTFLOPS: Double?

    // MARK: - Initialization

    /// Creates hardware info with all values specified.
    public init(
        osVersion: String,
        swiftVersion: String,
        architecture: String,
        cpuModel: String,
        cpuCoreCount: Int,
        cpuPerformanceCores: Int,
        cpuEfficiencyCores: Int,
        totalRAM: UInt64,
        availableRAM: UInt64,
        gpuName: String,
        gpuCoreCount: Int,
        gpuMemory: UInt64,
        metalVersion: String,
        gpuMemoryBandwidth: Double?,
        gpuPeakTFLOPS: Double?
    ) {
        self.osVersion = osVersion
        self.swiftVersion = swiftVersion
        self.architecture = architecture
        self.cpuModel = cpuModel
        self.cpuCoreCount = cpuCoreCount
        self.cpuPerformanceCores = cpuPerformanceCores
        self.cpuEfficiencyCores = cpuEfficiencyCores
        self.totalRAM = totalRAM
        self.availableRAM = availableRAM
        self.gpuName = gpuName
        self.gpuCoreCount = gpuCoreCount
        self.gpuMemory = gpuMemory
        self.metalVersion = metalVersion
        self.gpuMemoryBandwidth = gpuMemoryBandwidth
        self.gpuPeakTFLOPS = gpuPeakTFLOPS
    }

    // MARK: - Capture

    /// Captures current hardware information.
    ///
    /// Queries system via sysctl and Metal to gather hardware details.
    ///
    /// - Returns: HardwareInfo populated with current system details.
    public static func capture() -> HardwareInfo {
        let cpuModel = sysctlString("machdep.cpu.brand_string") ?? "Unknown CPU"
        let cpuCoreCount = ProcessInfo.processInfo.processorCount
        let (perfCores, effCores) = estimateCoreTypes(total: cpuCoreCount, model: cpuModel)

        let totalRAM = ProcessInfo.processInfo.physicalMemory
        let availableRAM = getAvailableMemory()

        let osVersion = getOSVersion()
        let swiftVersion = getSwiftVersion()
        let architecture = getArchitecture()

        let (gpuName, gpuCoreCount, gpuMemory, metalVersion) = getMetalInfo()
        let (bandwidth, tflops) = lookupPerformanceMetrics(model: cpuModel)

        return HardwareInfo(
            osVersion: osVersion,
            swiftVersion: swiftVersion,
            architecture: architecture,
            cpuModel: cpuModel,
            cpuCoreCount: cpuCoreCount,
            cpuPerformanceCores: perfCores,
            cpuEfficiencyCores: effCores,
            totalRAM: totalRAM,
            availableRAM: availableRAM,
            gpuName: gpuName,
            gpuCoreCount: gpuCoreCount,
            gpuMemory: gpuMemory,
            metalVersion: metalVersion,
            gpuMemoryBandwidth: bandwidth,
            gpuPeakTFLOPS: tflops
        )
    }

    // MARK: - Summary

    /// Formatted summary string for console output.
    public var summary: String {
        let ramGB = totalRAM / 1_073_741_824
        let bandwidthStr = gpuMemoryBandwidth.map { String(format: "%.0f GB/s", $0) } ?? "unknown"

        return """
        Hardware: \(cpuModel), \(ramGB)GB RAM
        GPU: \(gpuName) (\(gpuCoreCount) cores, \(metalVersion))
        \(osVersion), Swift \(swiftVersion)
        Memory Bandwidth: \(bandwidthStr)
        """
    }

    /// Short one-line summary.
    public var shortSummary: String {
        let ramGB = totalRAM / 1_073_741_824
        return "\(cpuModel), \(ramGB)GB RAM, \(gpuCoreCount) GPU cores"
    }
}

// MARK: - Private Helpers

/// Reads a sysctl string value.
private func sysctlString(_ name: String) -> String? {
    var size: Int = 0
    sysctlbyname(name, nil, &size, nil, 0)
    guard size > 0 else { return nil }

    var buffer = [CChar](repeating: 0, count: size)
    sysctlbyname(name, &buffer, &size, nil, 0)
    // Convert to String, truncating at null terminator
    if let nullIndex = buffer.firstIndex(of: 0) {
        return String(decoding: buffer[..<nullIndex].map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
    return String(decoding: buffer.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

/// Reads a sysctl integer value.
private func sysctlInt(_ name: String) -> Int? {
    var value: Int = 0
    var size = MemoryLayout<Int>.size
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
    return value
}

/// Gets available memory using vm_statistics.
private func getAvailableMemory() -> UInt64 {
    var pageSize: vm_size_t = 0
    host_page_size(mach_host_self(), &pageSize)

    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)

    let result = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }

    guard result == KERN_SUCCESS else {
        return ProcessInfo.processInfo.physicalMemory / 2
    }

    let freePages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
    return freePages * UInt64(pageSize)
}

/// Gets the OS version string.
private func getOSVersion() -> String {
    let info = ProcessInfo.processInfo
    let version = info.operatingSystemVersion

    #if os(macOS)
    return "macOS \(version.majorVersion).\(version.minorVersion)"
    #elseif os(iOS)
    return "iOS \(version.majorVersion).\(version.minorVersion)"
    #elseif os(visionOS)
    return "visionOS \(version.majorVersion).\(version.minorVersion)"
    #else
    return "Unknown OS \(version.majorVersion).\(version.minorVersion)"
    #endif
}

/// Gets the Swift version.
private func getSwiftVersion() -> String {
    #if swift(>=6.0)
    return "6.x"
    #elseif swift(>=5.10)
    return "5.10"
    #elseif swift(>=5.9)
    return "5.9"
    #else
    return "5.x"
    #endif
}

/// Gets the CPU architecture.
private func getArchitecture() -> String {
    #if arch(arm64)
    return "arm64"
    #elseif arch(x86_64)
    return "x86_64"
    #else
    return "unknown"
    #endif
}

/// Estimates performance vs efficiency core counts.
private func estimateCoreTypes(total: Int, model: String) -> (performance: Int, efficiency: Int) {
    // Apple Silicon core configurations (approximate)
    let coreConfigs: [(pattern: String, perf: Int, eff: Int)] = [
        ("M4 Max", 14, 4),     // 14P + 4E
        ("M4 Pro", 12, 4),     // 12P + 4E
        ("M4", 4, 6),          // 4P + 6E (base M4)
        ("M3 Max", 12, 4),     // 12P + 4E
        ("M3 Pro", 6, 6),      // 6P + 6E
        ("M3", 4, 4),          // 4P + 4E
        ("M2 Max", 12, 4),     // 12P + 4E
        ("M2 Pro", 8, 4),      // 8P + 4E
        ("M2", 4, 4),          // 4P + 4E
        ("M1 Max", 8, 2),      // 8P + 2E
        ("M1 Pro", 8, 2),      // 8P + 2E
        ("M1", 4, 4),          // 4P + 4E
        ("A18 Pro", 2, 4),     // 2P + 4E
        ("A18", 2, 4),         // 2P + 4E
        ("A17 Pro", 2, 4),     // 2P + 4E
    ]

    for (pattern, perf, eff) in coreConfigs {
        if model.contains(pattern) {
            return (perf, eff)
        }
    }

    // Fallback: assume roughly equal split
    let half = total / 2
    return (half, total - half)
}

/// Gets Metal device information.
private func getMetalInfo() -> (name: String, cores: Int, memory: UInt64, version: String) {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return ("No GPU", 0, 0, "None")
    }

    let name = device.name
    let memory = device.recommendedMaxWorkingSetSize
    let version = getMetalVersion(device)

    // GPU core count estimation based on chip model
    let cores = estimateGPUCores(deviceName: name)

    return (name, cores, memory, version)
}

/// Gets the Metal version string.
private func getMetalVersion(_ device: MTLDevice) -> String {
    if device.supportsFamily(.metal3) {
        return "Metal 3"
    } else if device.supportsFamily(.apple7) {
        return "Metal 3 (Apple 7)"
    } else {
        return "Metal 2"
    }
}

/// Estimates GPU core count based on device name.
private func estimateGPUCores(deviceName: String) -> Int {
    let gpuCoreConfigs: [(pattern: String, cores: Int)] = [
        ("M4 Max", 40),       // Up to 40 GPU cores
        ("M4 Pro", 20),       // Up to 20 GPU cores
        ("M4", 10),           // Up to 10 GPU cores
        ("M3 Max", 40),       // Up to 40 GPU cores
        ("M3 Pro", 18),       // Up to 18 GPU cores
        ("M3", 10),           // Up to 10 GPU cores
        ("M2 Max", 38),       // Up to 38 GPU cores
        ("M2 Pro", 19),       // Up to 19 GPU cores
        ("M2", 10),           // Up to 10 GPU cores
        ("M1 Max", 32),       // Up to 32 GPU cores
        ("M1 Pro", 16),       // Up to 16 GPU cores
        ("M1", 8),            // Up to 8 GPU cores
        ("A18 Pro", 6),       // 6 GPU cores
        ("A18", 5),           // 5 GPU cores
        ("A17 Pro", 6),       // 6 GPU cores
    ]

    for (pattern, cores) in gpuCoreConfigs {
        if deviceName.contains(pattern) {
            return cores
        }
    }

    return 8 // Default assumption
}

/// Looks up memory bandwidth and peak TFLOPS for known chips.
private func lookupPerformanceMetrics(model: String) -> (bandwidth: Double?, tflops: Double?) {
    // Memory bandwidth lookup table (GB/s)
    let bandwidthTable: [String: Double] = [
        // Primary targets
        "Apple M3 Max": 400.0,      // 48GB config
        "Apple A18 Pro": 75.0,      // iPhone 16 Pro Max
        "Apple M4": 120.0,          // Base M4
        "Apple M4 Pro": 273.0,
        "Apple M4 Max": 546.0,

        // Secondary
        "Apple M1": 68.25,
        "Apple M1 Pro": 200.0,
        "Apple M1 Max": 400.0,
        "Apple M2": 100.0,
        "Apple M2 Pro": 200.0,
        "Apple M2 Max": 400.0,
        "Apple M3": 100.0,
        "Apple M3 Pro": 150.0,
    ]

    // Peak GPU TFLOPS lookup (FP32)
    let tflopsTable: [String: Double] = [
        "Apple M4 Max": 17.4,
        "Apple M4 Pro": 8.7,
        "Apple M4": 4.3,
        "Apple M3 Max": 14.2,
        "Apple M3 Pro": 7.0,
        "Apple M3": 4.1,
        "Apple M2 Max": 13.6,
        "Apple M2 Pro": 6.8,
        "Apple M2": 3.6,
        "Apple M1 Max": 10.4,
        "Apple M1 Pro": 5.2,
        "Apple M1": 2.6,
    ]

    var bandwidth: Double?
    var tflops: Double?

    for (pattern, value) in bandwidthTable {
        if model.contains(pattern.replacingOccurrences(of: "Apple ", with: "")) {
            bandwidth = value
            break
        }
    }

    for (pattern, value) in tflopsTable {
        if model.contains(pattern.replacingOccurrences(of: "Apple ", with: "")) {
            tflops = value
            break
        }
    }

    return (bandwidth, tflops)
}
