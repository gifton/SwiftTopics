# 6.4 Next Steps

> **Beyond the basics—incremental updates, customization, and production patterns.**

---

## The Challenge

You've built a topic model. Now what?

```
Real-world questions:

  "My users add journal entries daily. How do I update topics?"
  "Can I customize the tokenizer for my domain?"
  "How do I persist and reload my model?"
  "What about visualizing topics?"
```

This guide covers the advanced topics that take you from prototype to production.

---

## Incremental Topic Updates

### The Problem with Batch Processing

```
Batch approach:
  Day 1: Train on 100 entries → 5 topics
  Day 30: Retrain on 130 entries → 6 topics
  Day 60: Retrain on 160 entries → Different topics!

Problems:
  - Topic IDs change each retrain
  - Full retraining is slow (30+ seconds)
  - User sees topics "jump around"
```

### The Incremental Solution

```
Incremental approach:
  Day 1: Initial training → 5 topics
  Day 2: New entry → Assign to existing topic (instant)
  Day 30: Micro-retrain → Refine topics slightly
  Day 60: Full refresh → Only if drift detected

Benefits:
  - Immediate topic assignment for new docs
  - Topics evolve gradually
  - User experience is smooth
```

### Using IncrementalTopicUpdater

```swift
// 📍 See: Sources/SwiftTopics/Incremental/IncrementalTopicUpdater.swift

import SwiftTopics

// 1. Set up storage
let modelDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("TopicModel")

let storage = try FileBasedTopicModelStorage(directory: modelDir)

// 2. Create updater
let updater = try await IncrementalTopicUpdater(
    storage: storage,
    modelConfiguration: .default,
    updateConfiguration: IncrementalUpdateConfiguration(
        coldStartThreshold: 50,        // Min docs for initial training
        microRetrainThreshold: 20,     // Docs before micro-retrain
        fullRefreshGrowthRatio: 2.0    // Double corpus → full refresh
    )
)

// 3. On app launch, resume any interrupted training
try await updater.resumeIfNeeded()

// 4. Process new documents
let assignment = try await updater.processDocument(
    Document(content: "Great run this morning!"),
    embedding: await embedder.embed("Great run this morning!")
)

print("Assigned to topic: \(assignment.topicID.value)")
print("Confidence: \(assignment.confidence)")
print("Keywords: \(assignment.topicKeywords.joined(separator: ", "))")
```

### The Incremental Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Incremental Update Flow                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  NEW DOCUMENT ARRIVES                                                   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐                                                        │
│  │ Model       │─── No ───► Buffer document                            │
│  │ exists?     │            └──► If buffer ≥ threshold → Initial train │
│  └─────────────┘                                                        │
│         │ Yes                                                           │
│         ▼                                                               │
│  ┌─────────────────────────┐                                           │
│  │ Assign via centroid     │ ← Instant! (cosine similarity)           │
│  │ Return topic + keywords │                                           │
│  └─────────────────────────┘                                           │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐                                                        │
│  │ Buffer for  │                                                        │
│  │ micro-retrain│                                                       │
│  └─────────────┘                                                        │
│         │                                                               │
│  ┌──────┴──────┐                                                        │
│  │Buffer full? │─── No ───► Done                                       │
│  └─────────────┘                                                        │
│         │ Yes                                                           │
│         ▼                                                               │
│  ┌─────────────────────────┐                                           │
│  │ Background micro-retrain│ ← Runs asynchronously                     │
│  │ • Train on buffer       │                                           │
│  │ • Match new topics      │                                           │
│  │ • Merge into main model │                                           │
│  └─────────────────────────┘                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration Options

```swift
// 📍 See: Sources/SwiftTopics/Incremental/IncrementalUpdateConfiguration.swift

public struct IncrementalUpdateConfiguration: Sendable, Codable {
    /// Documents needed before initial training.
    public let coldStartThreshold: Int              // Default: 50

    /// Documents buffered before micro-retrain.
    public let microRetrainThreshold: Int           // Default: 20

    /// Growth ratio triggering full refresh.
    public let fullRefreshGrowthRatio: Float        // Default: 2.0

    /// Maximum time before full refresh.
    public let fullRefreshMaxInterval: TimeInterval? // Default: nil

    /// Similarity threshold for transform outliers.
    public let transformOutlierThreshold: Float     // Default: 0.3

    /// Topic matching similarity threshold.
    public let topicMatchingSimilarityThreshold: Float // Default: 0.5

    /// Window size for drift statistics.
    public let driftWindowSize: Int                 // Default: 100

    /// Drift ratio threshold for triggering refresh.
    public let driftRatioThreshold: Float           // Default: 0.3

    /// Outlier rate threshold for triggering refresh.
    public let outlierRateThreshold: Float          // Default: 0.5
}
```

### Lifecycle Management

```swift
// App launch
func applicationDidFinishLaunching() async throws {
    // Resume interrupted training
    if try await updater.resumeIfNeeded() {
        print("Resumed interrupted training")
    }

    // Check if refresh recommended
    if updater.shouldTriggerFullRefresh() {
        // Schedule for background processing
        scheduleBackgroundRefresh()
    }
}

// App termination
func applicationWillTerminate() async throws {
    // Save checkpoint if training in progress
    try await updater.prepareForTermination()
}

// Background processing (iOS)
func handleBackgroundProcessing() async throws {
    try await updater.triggerFullRefresh { progress in
        print("Refresh progress: \(progress.overallProgress)")
    }
}
```

### Cold Start Handling

```swift
// Before model is trained, documents get cold start assignments
let assignment = try await updater.processDocument(doc, embedding: emb)

if assignment.isColdStart {
    // No topic assigned yet
    print("Document buffered for initial training")
    print("Buffer size: \(try await updater.getPendingBufferCount())")
    print("Need \(config.coldStartThreshold) for initial training")
}
```

---

## Model Persistence

### Saving and Loading Results

```swift
// TopicModelResult is Codable
let result = try await model.fit(documents: docs, embeddings: embs)

// Save to file
let encoder = JSONEncoder()
encoder.outputFormatting = .prettyPrinted
let data = try encoder.encode(result)
try data.write(to: modelURL)

// Load from file
let loadedData = try Data(contentsOf: modelURL)
let loadedResult = try JSONDecoder().decode(TopicModelResult.self, from: loadedData)

print("Loaded model with \(loadedResult.topicCount) topics")
```

### Storage Protocol

```swift
// 📍 See: Sources/SwiftTopics/Incremental/Storage/TopicModelStorage.swift

public protocol TopicModelStorage: Sendable {
    func saveModelState(_ state: IncrementalTopicModelState) async throws
    func loadModelState() async throws -> IncrementalTopicModelState?

    func appendToPendingBuffer(_ entries: [BufferedEntry]) async throws
    func drainPendingBuffer() async throws -> [BufferedEntry]
    func pendingBufferCount() async throws -> Int

    func appendEmbeddings(_ embeddings: [(DocumentID, Embedding)]) async throws
    func loadEmbeddings(for ids: [DocumentID]) async throws -> [Embedding]
    func loadAllEmbeddings() async throws -> [(DocumentID, Embedding)]

    func saveCheckpoint(_ checkpoint: TrainingCheckpoint) async throws
    func loadCheckpoint() async throws -> TrainingCheckpoint?
    func clearCheckpoint() async throws
}
```

### File-Based Storage

```swift
// 📍 See: Sources/SwiftTopics/Incremental/Storage/FileBasedTopicModelStorage.swift

let storage = try FileBasedTopicModelStorage(directory: modelDir)

// Files created:
// modelDir/
// ├── model_state.json      ← Topics, centroids, assignments
// ├── embeddings.bin        ← Binary embedding storage
// ├── pending_buffer.json   ← Documents awaiting training
// └── checkpoint.json       ← Interrupted training state
```

---

## Custom Embedding Providers

### The EmbeddingProvider Protocol

```swift
// Create your own embedding source
public protocol EmbeddingProvider: Sendable {
    func embed(_ text: String) async throws -> Embedding
    func embedBatch(_ texts: [String]) async throws -> [Embedding]
}
```

### Example: OpenAI Embeddings

```swift
struct OpenAIEmbeddingProvider: EmbeddingProvider {
    let apiKey: String
    let model: String

    func embed(_ text: String) async throws -> Embedding {
        let embeddings = try await embedBatch([text])
        return embeddings[0]
    }

    func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/embeddings")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "model": model,
            "input": texts
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await URLSession.shared.data(for: request)

        // Parse response
        let response = try JSONDecoder().decode(OpenAIEmbeddingResponse.self, from: data)

        return response.data.map { item in
            Embedding(vector: item.embedding)
        }
    }
}

// Usage
let provider = OpenAIEmbeddingProvider(
    apiKey: "sk-...",
    model: "text-embedding-3-small"
)

let result = try await model.fit(
    documents: documents,
    embeddingProvider: provider
)
```

### Example: Core ML Model

```swift
struct CoreMLEmbeddingProvider: EmbeddingProvider {
    let model: MLModel
    let tokenizer: BERTTokenizer

    func embed(_ text: String) async throws -> Embedding {
        let tokens = tokenizer.tokenize(text)
        let input = try createInput(tokens: tokens)
        let output = try model.prediction(from: input)
        return extractEmbedding(from: output)
    }

    func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        // Process in parallel
        try await withThrowingTaskGroup(of: Embedding.self) { group in
            for text in texts {
                group.addTask {
                    try await self.embed(text)
                }
            }

            var results: [Embedding] = []
            for try await embedding in group {
                results.append(embedding)
            }
            return results
        }
    }
}
```

---

## Visualization Ideas

### Topic Word Clouds

```swift
// Generate data for word cloud visualization
struct WordCloudData {
    let word: String
    let weight: Float
}

func wordCloudData(for topic: Topic) -> [WordCloudData] {
    topic.keywords.map { keyword in
        WordCloudData(
            word: keyword.term,
            weight: keyword.score
        )
    }
}

// In SwiftUI
ForEach(wordCloudData(for: topic), id: \.word) { item in
    Text(item.word)
        .font(.system(size: CGFloat(item.weight * 50 + 12)))
        .foregroundColor(topicColor(topic.id))
}
```

### Topic Timeline (for dated documents)

```swift
// Track topic distribution over time
struct TopicTimeline {
    let topic: Topic
    let dates: [Date]
    let counts: [Int]
}

func buildTimeline(
    for topic: Topic,
    documents: [Document],
    result: TopicModelResult,
    bucketBy: Calendar.Component = .month
) -> TopicTimeline {
    let calendar = Calendar.current

    let topicDocs = result.documents(for: topic.id)
    let datedDocs = documents.filter { topicDocs.contains($0.id) }
        .compactMap { doc -> (Document, Date)? in
            guard let date = doc.metadata?["date"]?.dateValue else { return nil }
            return (doc, date)
        }

    // Group by bucket
    var buckets: [Date: Int] = [:]
    for (_, date) in datedDocs {
        let bucket = calendar.startOfDay(for: date)
        buckets[bucket, default: 0] += 1
    }

    let sorted = buckets.sorted { $0.key < $1.key }
    return TopicTimeline(
        topic: topic,
        dates: sorted.map(\.key),
        counts: sorted.map(\.value)
    )
}
```

### Similarity Heatmap

```swift
// Compute topic similarity matrix for visualization
func topicSimilarityMatrix(topics: [Topic]) -> [[Float]] {
    var matrix: [[Float]] = []

    for topicA in topics {
        var row: [Float] = []
        for topicB in topics {
            if let centroidA = topicA.centroid,
               let centroidB = topicB.centroid {
                let sim = centroidA.cosineSimilarity(centroidB)
                row.append(sim)
            } else {
                row.append(0)
            }
        }
        matrix.append(row)
    }

    return matrix
}

// Render as heatmap in SwiftUI
struct SimilarityHeatmap: View {
    let matrix: [[Float]]
    let topics: [Topic]

    var body: some View {
        Grid {
            ForEach(0..<matrix.count, id: \.self) { row in
                GridRow {
                    ForEach(0..<matrix[row].count, id: \.self) { col in
                        Rectangle()
                            .fill(heatColor(matrix[row][col]))
                            .frame(width: 30, height: 30)
                    }
                }
            }
        }
    }

    func heatColor(_ value: Float) -> Color {
        Color(red: Double(value), green: 0, blue: Double(1 - value))
    }
}
```

---

## Performance Best Practices

### Batch Processing for Large Corpora

```swift
// Process in batches to manage memory
func processLargeCorpus(
    documents: [Document],
    embeddingProvider: EmbeddingProvider,
    batchSize: Int = 1000
) async throws -> TopicModelResult {
    var allEmbeddings: [Embedding] = []
    allEmbeddings.reserveCapacity(documents.count)

    for batch in documents.chunked(into: batchSize) {
        let batchTexts = batch.map(\.content)
        let batchEmbeddings = try await embeddingProvider.embedBatch(batchTexts)
        allEmbeddings.append(contentsOf: batchEmbeddings)

        print("Embedded \(allEmbeddings.count)/\(documents.count)")
    }

    let model = TopicModel(configuration: .largeCorpus)
    return try await model.fit(documents: documents, embeddings: allEmbeddings)
}

// Helper extension
extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
```

### Caching Embeddings

```swift
// Cache embeddings to avoid recomputation
actor EmbeddingCache {
    private var cache: [String: Embedding] = [:]
    private let provider: EmbeddingProvider

    init(provider: EmbeddingProvider) {
        self.provider = provider
    }

    func embed(_ text: String) async throws -> Embedding {
        let key = text.hashValue.description

        if let cached = cache[key] {
            return cached
        }

        let embedding = try await provider.embed(text)
        cache[key] = embedding
        return embedding
    }

    func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        var results: [Embedding] = []
        var uncachedTexts: [String] = []
        var uncachedIndices: [Int] = []

        // Check cache
        for (i, text) in texts.enumerated() {
            let key = text.hashValue.description
            if let cached = cache[key] {
                results.append(cached)
            } else {
                uncachedTexts.append(text)
                uncachedIndices.append(i)
                results.append(Embedding(vector: []))  // Placeholder
            }
        }

        // Fetch uncached
        if !uncachedTexts.isEmpty {
            let fetched = try await provider.embedBatch(uncachedTexts)
            for (i, embedding) in zip(uncachedIndices, fetched) {
                results[i] = embedding
                cache[texts[i].hashValue.description] = embedding
            }
        }

        return results
    }
}
```

### Progress Reporting for UI

```swift
// SwiftUI integration with progress
@MainActor
class TopicModelViewModel: ObservableObject {
    @Published var progress: Float = 0
    @Published var stage: String = ""
    @Published var result: TopicModelResult?

    func train(documents: [Document], embeddings: [Embedding]) async {
        let model = TopicModel(configuration: .default)

        await model.setProgressHandler { [weak self] progress in
            Task { @MainActor in
                self?.progress = progress.overallProgress
                self?.stage = progress.stage.description
            }
        }

        do {
            let result = try await model.fit(documents: documents, embeddings: embeddings)
            self.result = result
        } catch {
            print("Training failed: \(error)")
        }
    }
}

// SwiftUI View
struct TrainingView: View {
    @StateObject var viewModel = TopicModelViewModel()

    var body: some View {
        VStack {
            if viewModel.result == nil {
                ProgressView(value: viewModel.progress)
                Text(viewModel.stage)
            } else {
                TopicListView(result: viewModel.result!)
            }
        }
    }
}
```

---

## Integration Patterns

### With SwiftData

```swift
import SwiftData

@Model
class JournalEntry {
    var id: UUID
    var content: String
    var date: Date
    var topicID: Int?

    init(content: String, date: Date = .now) {
        self.id = UUID()
        self.content = content
        self.date = date
    }
}

// After training, update entries with topic assignments
func updateEntryTopics(
    entries: [JournalEntry],
    result: TopicModelResult,
    context: ModelContext
) {
    for entry in entries {
        let docID = DocumentID(uuid: entry.id)
        if let assignment = result.topicAssignment(for: docID) {
            entry.topicID = assignment.topicID.value
        }
    }
    try? context.save()
}
```

### With CloudKit

```swift
// Sync topic model across devices
actor CloudTopicSync {
    let container: CKContainer

    func uploadTopics(_ topics: [Topic]) async throws {
        let records = topics.map { topic in
            let record = CKRecord(recordType: "Topic")
            record["id"] = topic.id.value
            record["keywords"] = topic.keywords.map(\.term)
            record["size"] = topic.size
            return record
        }

        let database = container.privateCloudDatabase
        let (_, _) = try await database.modifyRecords(saving: records, deleting: [])
    }

    func downloadTopics() async throws -> [CKRecord] {
        let database = container.privateCloudDatabase
        let query = CKQuery(recordType: "Topic", predicate: NSPredicate(value: true))
        let (results, _) = try await database.records(matching: query)
        return results.compactMap { try? $0.1.get() }
    }
}
```

---

## What's Next?

### Explore the Source Code

```
Key files to study:

Sources/SwiftTopics/
├── Model/TopicModel.swift           ← Start here
├── Clustering/HDBSCAN/HDBSCAN.swift ← Core clustering
├── Representation/cTFIDF.swift      ← Keyword extraction
├── Evaluation/NPMIScorer.swift      ← Quality metrics
└── Incremental/                     ← Production patterns
```

### Contribute

SwiftTopics is open source. Consider contributing:
- Bug fixes
- Performance improvements
- New embedding provider integrations
- Documentation improvements

### Build Something

Now you have all the tools:

```
Ideas to build:

📔 Journal Analyzer
   - Daily entries → topics over time
   - Mood correlation with topics
   - Topic trends visualization

📧 Email Organizer
   - Inbox → automatic categorization
   - Topic-based search
   - Priority by topic

📚 Research Assistant
   - Papers → key themes
   - Citation clustering
   - Literature review automation

💬 Chat Analyzer
   - Conversations → topic extraction
   - Sentiment per topic
   - Conversation summaries
```

---

## Key Takeaways

1. **Incremental updates** provide instant assignments while evolving topics gradually.

2. **Persistence** through Codable and the Storage protocol enables production use.

3. **Custom embedding providers** let you use any embedding source.

4. **Visualization** brings topics to life—word clouds, timelines, heatmaps.

5. **Performance patterns**—batching, caching, progress reporting—enable large-scale use.

6. **Integration** with SwiftData, CloudKit, and other frameworks is straightforward.

---

## 💡 Key Insight

```
Topic modeling is a tool, not a destination.

The real value comes from:
  - Surfacing insights users couldn't find manually
  - Organizing information at scale
  - Enabling new kinds of search and discovery
  - Tracking how themes evolve over time

SwiftTopics provides the engine.
You provide the application.

The best topic modeling system is one that:
  - Users don't have to think about
  - Improves their experience invisibly
  - Scales with their data
  - Adapts to their changing needs

Go build something great.
```

---

## Congratulations!

You've completed the SwiftTopics Learning Guide.

You now understand:
- How embeddings capture semantic meaning
- Why dimensionality reduction enables clustering
- How HDBSCAN discovers natural groupings
- How c-TF-IDF extracts representative keywords
- How NPMI measures topic quality
- How to build production topic modeling systems

**Happy topic modeling!**

---

*Guide 6.4 of 6.4 • Chapter 6: Capstone*

---

*SwiftTopics Learning Guide • Complete*
