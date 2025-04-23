from pathlib import Path
import difflib
import re
from typing import List, Tuple

# === Настройки ===
GT_DIR = Path("TMP/RETAS/Ground Truth")
OCR_DIR = Path("TMP/RETAS/OCR Text")
OUTPUT_GT_DIR = GT_DIR / "chunks"
OUTPUT_OCR_DIR = OCR_DIR / "chunks"
CHUNK_TOKEN_TARGET = 350  # Средний размер чанка в словах
MIN_CHUNK_TOKENS = 100  # Минимальный размер чанка

# Создание выходных директорий
OUTPUT_GT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_OCR_DIR.mkdir(parents=True, exist_ok=True)


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Возвращает список (токен, start, end), где start/end — индексы
    начала и конца токена в исходном тексте.
    """
    tokens = []
    for m in re.finditer(r"\b\w+\b", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)


def find_anchor_matches(
    gt_words: List[str], ocr_words: List[str]
) -> List[Tuple[int, int, int]]:
    matcher = difflib.SequenceMatcher(None, gt_words, ocr_words, autojunk=False)
    # Берем только значимые блоки
    matches = [
        (gt_i, ocr_i, n) for gt_i, ocr_i, n in matcher.get_matching_blocks() if n > 5
    ]
    # Исключаем финальный пустой матч
    if matches and matches[-1][2] == 0:
        matches.pop()
    return matches


def split_into_chunks(gt_text: str, ocr_text: str) -> List[Tuple[str, str]]:
    """
    Разбивает gt_text и ocr_text на синхронные чанки примерно по CHUNK_TOKEN_TARGET слов,
    возвращает список пар (gt_chunk, ocr_chunk), сохраняя всю пунктуацию.
    """
    gt_toks = tokenize_with_spans(gt_text)
    ocr_toks = tokenize_with_spans(ocr_text)

    chunks: List[Tuple[str, str]] = []
    idx = 0
    # идём по токенам, делим порциями
    while idx < len(gt_toks) and idx < len(ocr_toks):
        # граница по количеству слов
        end = min(idx + CHUNK_TOKEN_TARGET, len(gt_toks), len(ocr_toks))

        # получаем символьные границы для GT
        gt_start_char = gt_toks[idx][1]
        gt_end_char = gt_toks[end - 1][2]
        gt_chunk = gt_text[gt_start_char:gt_end_char]

        # и для OCR
        ocr_start_char = ocr_toks[idx][1]
        ocr_end_char = ocr_toks[end - 1][2]
        ocr_chunk = ocr_text[ocr_start_char:ocr_end_char]

        # проверяем на минимальный размер
        if end - idx >= MIN_CHUNK_TOKENS:
            chunks.append((gt_chunk, ocr_chunk))

        idx = end

    return chunks


def save_chunks(gt_chunks: List[Tuple[str, str]], base_name: str):
    for i, (gt_chunk, ocr_chunk) in enumerate(gt_chunks):
        gt_path = OUTPUT_GT_DIR / f"{base_name}_chunk_{i + 1}.txt"
        ocr_path = OUTPUT_OCR_DIR / f"{base_name}_chunk_{i + 1}.txt"
        gt_path.write_text(gt_chunk, encoding="utf-8")
        ocr_path.write_text(ocr_chunk, encoding="utf-8")


def process_all():
    for gt_file in sorted(GT_DIR.glob("*.txt")):
        ocr_file = OCR_DIR / gt_file.name.replace("GT", "OCR")
        if not ocr_file.exists():
            print(f"Пропущен: {gt_file.name} (нет OCR файла)")
            continue

        gt_text = gt_file.read_text(encoding="utf-8")
        ocr_text = ocr_file.read_text(encoding="utf-8")

        chunks = split_into_chunks(gt_text, ocr_text)
        save_chunks(chunks, gt_file.stem)
        print(f"✔ Обработан: {gt_file.name} → {len(chunks)} чанков")


if __name__ == "__main__":
    process_all()
