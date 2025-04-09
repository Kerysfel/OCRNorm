# src/evaluation.py

import re
from typing import List
from sentence_transformers import SentenceTransformer, util


# -----------------------------
# 1) Предобработка текста
# -----------------------------
def preprocess_text(s: str) -> str:
    """
    Приводим к нижнему регистру, убираем знаки препинания,
    объединяем множественные пробелы в один, тримим.
    """
    s = s.lower()
    # Удаляем все символы, кроме букв, цифр, пробелов, подчёркиваний
    s = re.sub(r"[^\w\s]", "", s)
    # Убираем множественные пробелы
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def tokenize_words(s: str) -> List[str]:
    """
    Превращаем строку в список «слов» (разделённых пробелами).
    """
    p = preprocess_text(s)
    return p.split()


def tokenize_chars(s: str) -> List[str]:
    """
    Превращаем строку в список «символов» (после очистки).
    """
    p = preprocess_text(s)
    return list(p)


# -----------------------------
# 2) Левенштейн (edit distance)
# -----------------------------
def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Классический алгоритм вычисления расстояния Левенштейна
    (кол-во правок: вставка/удаление/замена),
    применимый к спискам «слов» или «символов».
    """
    n = len(seq1)
    m = len(seq2)
    # dp[i][j] = дистанция между первыми i эл-тами seq1 и первыми j эл-тами seq2
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i  # нужно i удалений
    for j in range(m + 1):
        dp[0][j] = j  # нужно j вставок

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # удаление
                dp[i][j - 1] + 1,  # вставка
                dp[i - 1][j - 1] + cost,  # замена (cost=0, если символы совпадают)
            )
    return dp[n][m]


# -----------------------------
# 3) Подсчёт WER и CER
# -----------------------------
def compute_wer(ground_truth: str, recognized: str) -> float:
    """
    Word Error Rate:
     - Токенизируем тексты по словам.
     - Считаем расстояние Левенштейна между двумя списками слов.
     - Делим на кол-во слов в ground_truth.
    """
    gt_tokens = tokenize_words(ground_truth)
    rec_tokens = tokenize_words(recognized)

    if len(gt_tokens) == 0:
        # Если реальный текст пуст – можно вернуть 0.0 (если оба пусты)
        # или 1.0 (если recognized не пуст). Тут упрощённо вернём 0.0
        return 0.0

    dist = levenshtein_distance(gt_tokens, rec_tokens)
    return dist / len(gt_tokens)


def compute_cer(ground_truth: str, recognized: str) -> float:
    """
    Character Error Rate:
     - Токенизируем тексты по символам (после очистки).
     - Считаем расстояние Левенштейна между двумя списками символов.
     - Делим на кол-во символов в ground_truth.
    """
    gt_chars = tokenize_chars(ground_truth)
    rec_chars = tokenize_chars(recognized)

    if len(gt_chars) == 0:
        return 0.0

    dist = levenshtein_distance(gt_chars, rec_chars)
    return dist / len(gt_chars)


# -----------------------------
# 4) Семантическая близость
# -----------------------------
_model = None  # ленивое сохранение модели


def compute_semantic_similarity(ground_truth: str, recognized: str) -> float:
    """
    Косинусное сходство между эмбеддингами двух строк (0..1).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = _model.encode([ground_truth, recognized], convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings[0], embeddings[1])
    return float(cos_sim[0][0])
