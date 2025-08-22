# src/evaluation.py
import re
import logging
from sentence_transformers import SentenceTransformer, util

# Optional faster distance backends
try:
    from rapidfuzz.distance import Levenshtein as RFLevenshtein  # type: ignore

    _rf_available = True
except Exception:
    _rf_available = False

# One-time warning flag for long inputs when no optimized backend is available
_warned_levenshtein_performance = False


def preprocess_text(s: str) -> str:
    """
    Normalize text: lowercase, remove non-word punctuation, collapse whitespace, strip.
    Args:
        s (str): The input text to normalize.
    Returns:
        str: The normalized text.
    """
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def tokenize_words(s: str) -> list[str]:
    """
    Tokenize into words separated by spaces after preprocessing.
    Args:
        s (str): The input text to tokenize.
    Returns:
        list[str]: The list of words.
    """
    p = preprocess_text(s)
    return p.split()


def tokenize_chars(s: str) -> list[str]:
    """
    Tokenize into characters after preprocessing.
    Args:
        s (str): The input text to tokenize.
    Returns:
        list[str]: The list of characters.
    """
    p = preprocess_text(s)
    return list(p)


def levenshtein_distance(seq1: list[str], seq2: list[str]) -> int:
    """
    Levenshtein edit distance.

    Complexity notes:
    - Time: O(n*m)
    - Space: O(min(n, m)) using a rolling-row DP

    For long sequences, installing a faster backend is recommended:
    - rapidfuzz (pip install rapidfuzz) for optimized C++ implementation
    """
    global _warned_levenshtein_performance

    n = len(seq1)
    m = len(seq2)

    if n == 0:
        return m
    if m == 0:
        return n

    # Use RapidFuzz when available (supports generic sequences)
    if _rf_available:
        try:
            return int(RFLevenshtein.distance(seq1, seq2))
        except Exception:
            # Fallback to Python implementation
            pass

    # Warn once if inputs are large and no optimized backend is available
    if not _warned_levenshtein_performance and (n * m) > 5_000_000:
        logging.warning(
            "Large Levenshtein inputs detected and no optimized backend available. "
            "Install 'rapidfuzz' for significant speedups."
        )

        _warned_levenshtein_performance = True

    # Space-optimized DP (two-row), O(min(n,m)) memory
    # Ensure seq2 is the longer sequence to minimize row size
    if m < n:
        seq1, seq2 = seq2, seq1
        n, m = m, n

    previous_row = list(range(m + 1))
    for i in range(1, n + 1):
        current_row = [i] + [0] * m
        a = seq1[i - 1]
        for j in range(1, m + 1):
            b = seq2[j - 1]
            cost = 0 if a == b else 1
            deletion = previous_row[j] + 1
            insertion = current_row[j - 1] + 1
            substitution = previous_row[j - 1] + cost
            current_row[j] = min(deletion, insertion, substitution)
        previous_row = current_row
    return previous_row[m]


def compute_wer(ground_truth: str, recognized: str) -> float:
    """
    Word Error Rate based on word-level Levenshtein distance.
    Args:
        ground_truth (str): The ground truth text.
        recognized (str): The recognized text.
    Returns:
        float: The Word Error Rate.
    """
    gt_tokens = tokenize_words(ground_truth)
    rec_tokens = tokenize_words(recognized)

    if len(gt_tokens) == 0:
        return 0.0

    dist = levenshtein_distance(gt_tokens, rec_tokens)
    return dist / len(gt_tokens)


def compute_cer(ground_truth: str, recognized: str) -> float:
    """
    Character Error Rate based on character-level Levenshtein distance.
    Args:
        ground_truth (str): The ground truth text.
        recognized (str): The recognized text.
    Returns:
        float: The Character Error Rate.
    """
    gt_chars = tokenize_chars(ground_truth)
    rec_chars = tokenize_chars(recognized)

    if len(gt_chars) == 0:
        return 0.0

    dist = levenshtein_distance(gt_chars, rec_chars)
    return dist / len(gt_chars)


_model = None


def compute_semantic_similarity(ground_truth: str, recognized: str) -> float:
    """
    Cosine similarity between sentence embeddings (0..1).
    Args:
        ground_truth (str): The ground truth text.
        recognized (str): The recognized text.
    Returns:
        float: The cosine similarity between the two sentences.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = _model.encode([ground_truth, recognized], convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings[0], embeddings[1])
    return float(cos_sim[0][0])
