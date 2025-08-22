# Evaluation Metrics

This document describes the evaluation metrics used in ORCNorm for assessing OCR text quality and correction effectiveness.

## Word Error Rate (WER)

**Formula**: `WER = (S + D + I) / N`

Where:

- **S** = Substitutions (words replaced)
- **D** = Deletions (words missing)
- **I** = Insertions (extra words)
- **N** = Total words in ground truth

**Implementation**: [`src/evaluation.py:74-90`](https://github.com/your-repo/ORCNorm/blob/main/src/evaluation.py#L74-L90)

**Preprocessing**:

- **With normalization** (`normalize=True`): Text is lowercased, punctuation removed, whitespace collapsed
- **Without normalization** (`normalize=False`): Only whitespace normalization applied

**Example**:

```
Ground Truth: "The quick brown fox"
OCR Text:     "The quick brown fox"
WER: 0.0 (perfect match)

Ground Truth: "The quick brown fox"
OCR Text:     "The quick brown fox jumps"
WER: 0.25 (1 insertion / 4 words)
```

## Character Error Rate (CER)

**Formula**: `CER = (S + D + I) / N`

Where:

- **S** = Character substitutions
- **D** = Character deletions
- **I** = Character insertions
- **N** = Total characters in ground truth

**Implementation**: [`src/evaluation.py:93-109`](https://github.com/your-repo/ORCNorm/blob/main/src/evaluation.py#L93-L109)

**Preprocessing**: Same as WER - configurable normalization via `normalize` parameter

**Example**:

```
Ground Truth: "Hello"
OCR Text:     "Hallo"
CER: 0.2 (1 substitution / 5 characters)
```

## Semantic Similarity (SS)

**Formula**: `SS = cos(θ) = (A · B) / (||A|| × ||B||)`

Where:

- **A, B** = Sentence embeddings for ground truth and OCR text
- **θ** = Angle between embedding vectors
- **Range**: 0.0 (completely different) to 1.0 (identical meaning)

**Implementation**: [`src/evaluation.py:115-130`](https://github.com/your-repo/ORCNorm/blob/main/src/evaluation.py#L115-L130)

**Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` for embedding generation

**Preprocessing**: No text normalization applied - raw text is embedded as-is

**Example**:

```
Ground Truth: "The cat is on the mat"
OCR Text:     "A cat sits on the mat"
SS: 0.85 (high semantic similarity despite word differences)
```

## Text Preprocessing Details

### Normalization Pipeline

The preprocessing applied before WER/CER calculation is controlled by the `normalize_before_eval` configuration flag:

**When `normalize_before_eval: true`** (default):

```python
# src/evaluation.py:6-18
def preprocess_text(s: str) -> str:
    s = s.lower()                    # Convert to lowercase
    s = re.sub(r"[^\w\s]", "", s)    # Remove punctuation
    s = re.sub(r"\s+", " ", s)       # Collapse whitespace
    s = s.strip()                     # Trim leading/trailing spaces
    return s
```

**When `normalize_before_eval: false`**:

```python
# src/evaluation.py:21-30, 33-42
def tokenize_words(s: str, normalize: bool = True) -> list[str]:
    if normalize:
        s = preprocess_text(s)
    else:
        # Only whitespace normalization
        s = re.sub(r"\s+", " ", s).strip()
    return s.split()
```

### Impact on Metrics

**Normalized evaluation** (`normalize=True`):

- Focuses on word/character content rather than formatting
- More lenient to case differences and punctuation variations
- Standard approach for most OCR evaluation scenarios

**Raw evaluation** (`normalize=False`):

- Preserves case sensitivity and punctuation
- Useful when formatting accuracy is important
- More strict evaluation criteria

## Configuration

Set the normalization behavior in your experiment configuration:

```yaml
experiment:
  normalize_before_eval: true # Enable text normalization
  # ... other settings
```

## Performance Considerations

- **WER/CER**: O(n×m) time complexity using optimized Levenshtein distance
- **SS**: O(1) after initial model loading (embeddings cached)
- **Large texts**: Consider `normalize=True` for faster processing
- **Memory**: Character-level CER can be memory-intensive for very long texts

## Best Practices

1. **For OCR quality assessment**: Use `normalize_before_eval: true` (default)
2. **For formatting-sensitive tasks**: Use `normalize_before_eval: false`
3. **For semantic evaluation**: SS is always computed on raw text
4. **For reproducible results**: Set `llm.seed` in configuration
5. **For long runs**: Enable `experiment.checkpoint_every` for progress saving
