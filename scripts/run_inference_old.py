#!/usr/bin/env python
# scripts/run_inference.py

import argparse
import yaml
import os
import glob
import re

import pytesseract
import easyocr

# Метрики
from src.evaluation import compute_wer, compute_cer, compute_semantic_similarity

# Пайплайн с LLM
from src.normalization_pipeline import NormalizationPipeline

# ---------- Промпты (два разных) ----------
CLEANUP_PROMPT = """You are an assistant specialized in cleaning raw OCR text.
Goals:
1) Remove strange characters or obviously broken fragments.
2) Keep normal words/punctuation if possible.
3) Do not add new text.

OCR text to clean:
"{input_text}"
"""

CORRECTION_PROMPT = """You are an expert text corrector specialized in fixing OCR errors.
Please:
1) Fix spelling/grammar mistakes.
2) Keep the original wording if correct.
3) Do NOT add new sentences or words.

Text to correct:
"{input_text}"
"""


def load_iam_data(iam_base_path: str):
    """
    Пример загрузки датасета IAM.
    Рекурсивно ищем *.png в подпапках 000/, 001/, ...
    Для каждого file.png пытаемся найти file.txt.
    """
    image_paths = sorted(
        glob.glob(os.path.join(iam_base_path, "**", "*.png"), recursive=True)
    )
    image_list = []
    gt_list = []
    for img_path in image_paths:
        base, _ = os.path.splitext(img_path)
        txt_path = base + ".txt"
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
        else:
            line = ""
        image_list.append(img_path)
        gt_list.append(line)
    return image_list, gt_list


def load_mlt19_data(mlt_base_path: str):
    """
    Пример загрузки MLT19:
     - mlt_base_path/images/*.jpg | *.png
     - mlt_base_path/texts/*.txt
    В .txt берем последнее поле после split(',') в качестве gt.
    """
    images_dir = os.path.join(mlt_base_path, "images")
    texts_dir = os.path.join(mlt_base_path, "texts")
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    image_list = []
    gt_list = []
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(texts_dir, base_name + ".txt")
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            parts = line.split(",")
            if len(parts) > 0:
                gt = parts[-1]
            else:
                gt = ""
        else:
            gt = ""
        image_list.append(img_path)
        gt_list.append(gt)
    return image_list, gt_list


def recognize_tesseract(image_path: str, tess_config: str = "") -> str:
    """
    OCR через pytesseract
    """
    return pytesseract.image_to_string(image_path, config=tess_config)


def recognize_easyocr(
    image_path: str, lang_list=None, gpu=False, paragraph=False
) -> str:
    """
    OCR через EasyOCR
    """
    if lang_list is None:
        lang_list = ["en"]
    reader = easyocr.Reader(lang_list, gpu=gpu)
    result = reader.readtext(image_path, detail=0, paragraph=paragraph)
    if isinstance(result, list):
        return "\n\n".join(result)
    else:
        return str(result)


def remove_service_text(full_text: str) -> str:
    """
    Удаляем служебные строки:
      - "Sentence Database"
      - Линии, начинающиеся с A\d\d-
      - Всё после "Name:"
    """
    lines = full_text.splitlines()
    new_lines = []
    skip_rest = False

    for line in lines:
        lstrip = line.strip()
        # Если встретили "Name:", выходим (отрезаем всё)
        if lstrip.startswith("Name:"):
            skip_rest = True
        if skip_rest:
            break

        # Пропускаем строку, если содержит "Sentence Database"
        if "Sentence Database" in lstrip:
            continue

        # Пропускаем строку, если подходит под шаблон A\d\d-
        if re.match(r"^A\d\d-\S+", lstrip):
            continue

        new_lines.append(line)

    cleaned = "\n".join(new_lines)
    return cleaned.strip()


def split_paragraphs(cleaned_text: str):
    """
    Делим на 2 части:
      - 1) Первый абзац => typed_text (идеальный)
      - 2) Остальное => handwritten_text (рукопись)
    """
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        typed_text = cleaned_text
        handwritten_text = ""
    else:
        typed_text = paragraphs[0]
        handwritten_text = "\n\n".join(paragraphs[1:])
    return typed_text, handwritten_text


def main(config_path: str):
    # 1) Считываем конфиг
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]  # IAM / MLT19
    base_path = config["dataset"]["path"]  # Путь к данным
    ocr_engine_name = config["ocr"]["engine"]  # Tesseract / EasyOCR
    ocr_params = config["ocr"].get("params", {})

    debug_mode = config.get("experiment", {}).get("debug", False)
    output_path = config["experiment"].get(
        "output_path", "results/recognition_results.yaml"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Инициализация LLM (normalization_pipeline) - будем вызывать 2 промпта
    # Но внутри pipeline у нас обычно один и тот же model_name, temperature, etc.
    # Мы просто будем process(..., custom_prompt) вызывать.
    pipeline = NormalizationPipeline(config["llm"])

    # 2) Загрузка датасета
    if dataset_name == "IAM":
        images, gt_texts = load_iam_data(base_path)
        print(f"[INFO] Loaded IAM dataset. Found {len(images)} images.")
    elif dataset_name == "MLT19":
        images, gt_texts = load_mlt19_data(base_path)
        print(f"[INFO] Loaded MLT19 dataset. Found {len(images)} images.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not images:
        print("[WARNING] No images found. Exiting.")
        return

    results = []

    # 3) Цикл по изображениям
    for idx, img_path in enumerate(images):
        # --- OCR ---
        if ocr_engine_name.lower() == "tesseract":
            tess_config = ocr_params.get("config", "")
            recognized_text = recognize_tesseract(img_path, tess_config)
        elif ocr_engine_name.lower() == "easyocr":
            lang_list = ocr_params.get("lang_list", ["en"])
            gpu_flag = ocr_params.get("gpu", False)
            paragraph_flag = ocr_params.get("paragraph", False)
            recognized_text = recognize_easyocr(
                img_path, lang_list, gpu_flag, paragraph_flag
            )
        else:
            raise ValueError(f"Unknown OCR engine: {ocr_engine_name}")

        if debug_mode:
            print("\n" + "=" * 70)
            print(f"[DEBUG] Image #{idx} => {os.path.basename(img_path)}")
            print("[DEBUG] Raw OCR text:")
            print(recognized_text)

        # --- 4) Удаляем служебное ---
        cleaned_text = remove_service_text(recognized_text)
        if debug_mode:
            print("[DEBUG] After remove_service_text:")
            print(cleaned_text)

        # --- 5) Делим на "печатный" (typed_text) и "рукописный" (handwritten_text)
        typed_text, handwritten_text = split_paragraphs(cleaned_text)

        if debug_mode:
            print("[DEBUG] typed_text (first paragraph):")
            print(typed_text)
            print("[DEBUG] handwritten_text (rest paragraphs):")
            print(handwritten_text)

        # --- 6) Два вызова LLM на рукопись ---
        # 6.1) Сначала "cleanup" (CLEANUP_PROMPT)
        cleanup_filled = CLEANUP_PROMPT.format(input_text=handwritten_text)
        cleaned_handwritten = pipeline.process(
            handwritten_text, custom_prompt=cleanup_filled
        )

        # 6.2) Затем "correction" (CORRECTION_PROMPT) на результат очистки
        correction_filled = CORRECTION_PROMPT.format(input_text=cleaned_handwritten)
        corrected_text = pipeline.process(
            cleaned_handwritten, custom_prompt=correction_filled
        )

        if debug_mode:
            print("[DEBUG] After cleanup:")
            print(cleaned_handwritten)
            print("[DEBUG] Final corrected text:")
            print(corrected_text)

        # --- 7) Метрики ---
        # Считаем с typed_text как "идеал" (или ground_truth, если хотите)
        noise_text = handwritten_text  # до всякой LLM-правки
        wer_noise = compute_wer(typed_text, noise_text)
        cer_noise = compute_cer(typed_text, noise_text)
        ss_noise = compute_semantic_similarity(typed_text, noise_text)

        wer_corr = compute_wer(typed_text, corrected_text)
        cer_corr = compute_cer(typed_text, corrected_text)
        ss_corr = compute_semantic_similarity(typed_text, corrected_text)

        item = {
            "image_id": idx,
            "image_path": img_path,
            # Про всякий случай сохраняем все этапы
            "typed_text": typed_text,  # Печатный (идеал)
            "handwritten_text_raw": handwritten_text,  # Рукопись (до LLM)
            "handwritten_text_cleaned": cleaned_handwritten,  # После очистки
            "handwritten_text_corrected": corrected_text,  # Итог
            "metrics": {
                "noise": {"WER": wer_noise, "CER": cer_noise, "SS": ss_noise},
                "corrected": {"WER": wer_corr, "CER": cer_corr, "SS": ss_corr},
            },
        }
        results.append(item)

        # Перезапись результатов (чтобы файл оставался валидным, даже если прервётся)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(results, f, allow_unicode=True)

    print(f"\n[INFO] Done. Processed {len(images)} images. Results => {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
