// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftTopics",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
        .visionOS(.v26)
    ],
    products: [
        .library(
            name: "SwiftTopics",
            targets: ["SwiftTopics"]
        ),
        .library(
            name: "SwiftTopicsApple",
            targets: ["SwiftTopicsApple"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/gifton/VectorAccelerate.git", from: "0.3.2"),
        // EmbedKit for Apple embedding integration
        .package(url: "https://github.com/gifton/EmbedKit.git", from: "0.2.7"),
    ],
    targets: [
        .target(
            name: "SwiftTopics",
            dependencies: [
                .product(name: "VectorAccelerate", package: "VectorAccelerate"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),
        .target(
            name: "SwiftTopicsApple",
            dependencies: [
                "SwiftTopics",
                .product(name: "EmbedKit", package: "EmbedKit"),
            ]
        ),
        .testTarget(
            name: "SwiftTopicsTests",
            dependencies: ["SwiftTopics"]
        ),
    ]
)
