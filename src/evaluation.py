# src/evaluation.py

import re
from sentence_transformers import SentenceTransformer, util


def preprocess_text(s: str) -> str:
    """
    Lowercase, remove punctuation, collapse multiple spaces, and strip.
    """
    s = s.lower()
    # Remove all characters except letters, digits, spaces, underscores
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def tokenize_words(s: str) -> list[str]:
    """
    Convert a string into a list of "words" split by spaces.
    """
    p = preprocess_text(s)
    return p.split()


def tokenize_chars(s: str) -> list[str]:
    """
    Convert a string into a list of characters (after preprocessing).
    """
    p = preprocess_text(s)
    return list(p)


def levenshtein_distance(seq1: list[str], seq2: list[str]) -> int:
    """
    Classic Levenshtein distance algorithm (insert/delete/substitute) for
    lists of words or characters.
    """
    n = len(seq1)
    m = len(seq2)
    # dp[i][j] = distance between first i elements of seq1 and first j elements of seq2
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i  # need i deletions
    for j in range(m + 1):
        dp[0][j] = j  # need j insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution (cost=0 if equal)
            )
    return dp[n][m]


def compute_wer(ground_truth: str, recognized: str) -> float:
    """
    Word Error Rate:
     - Tokenize texts into words
     - Compute Levenshtein distance between lists of words
     - Divide by the number of words in ground_truth
    """
    gt_tokens = tokenize_words(ground_truth)
    rec_tokens = tokenize_words(recognized)

    if len(gt_tokens) == 0:
        # If ground truth is empty: return 0.0 for simplicity
        return 0.0

    dist = levenshtein_distance(gt_tokens, rec_tokens)
    return dist / len(gt_tokens)


def compute_cer(ground_truth: str, recognized: str) -> float:
    """
    Character Error Rate:
     - Tokenize texts into characters (after preprocessing)
     - Compute Levenshtein distance between lists of characters
     - Divide by the number of characters in ground_truth
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
    Cosine similarity between embeddings of two strings (0..1).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = _model.encode([ground_truth, recognized], convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings[0], embeddings[1])
    return float(cos_sim[0][0])
