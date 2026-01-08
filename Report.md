# LLM-Evaluator

## Implementation Choices

### Architectural Efficiency
- **Batched LLM evaluation** (single prompt with multiple field comparisons) leverages full context window to minimize API calls
- Reduces latency and cost compared to per-field evaluation

### Semantic Normalization Strategy
- **Key normalization only** (not values) via `FIELD_MAPPINGS` dictionary
- Preserves raw data integrity while solving schema misalignment between JSON and CSV

### Stateful Agent Management
- Single `AgentCode` instance reused across evaluations with built-in retry logic
- Maintains LLM session persistence and handles hallucinations through structured response validation

### Observability First
- OpenTelemetry instrumentation provides granular tracing of LLM inference latency/token usage

## Going Beyond

### Intelligent Field Matching
- Dual-phase matching (exact → normalized → case-insensitive) with CSV column discovery
- Handles imperfect data schemas without preprocessing

### Extensible Validation Framework
- `EXCLUDED_FIELDS` set and configurable `FIELD_MAPPINGS` allow domain adaptation without code changes

### Deterministic Caching
- Hash-based judgment cache prevents redundant LLM calls for identical field pairs
- Maintains evaluation reproducibility

## Future Improvements

### Dynamic Batching Algorithm
- Implement adaptive context window utilization with fallback to chunked evaluation when field count exceeds token limits

### Rule Engine Hybridization
- Create decision tree where trivial matches (exact/empty) bypass LLM
- Reserve semantic evaluation for complex cases only

### Performance Optimization
1. Implement async evaluation pipeline for concurrent document processing
2. Add Redis-backed distributed caching for multi-instance deployments

### AgentCode Enhancement Fork
- Fork/modify GroundX SDK to expose temperature/token controls

### Schema Evolution System
- Machine learning-based field mapping suggestion engine

### Observability
- Adding Arize Phoenix support for production
