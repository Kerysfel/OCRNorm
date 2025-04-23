import argparse
import yaml
import os
from src.evaluation import compute_wer, compute_cer, compute_semantic_similarity
from src.normalization_pipeline import NormalizationPipeline
from typing import List, Tuple
from pathlib import Path


def load_text_pairs(
    base_path: str, ocr_subdir: str = "OCR Text", gt_subdir: str = "Ground Truth"
) -> List[Tuple[str, str, str]]:
    """
    Сканирует base_path/ocr_subdir и base_path/gt_subdir,
    находит пары .txt файлов по общему имени (stem) и
    возвращает список кортежей (doc_id, ocr_text, gt_text).
    """
    base = Path(base_path)
    ocr_dir = base / ocr_subdir
    gt_dir = base / gt_subdir

    samples: List[Tuple[str, str, str]] = []
    for ocr_file in sorted(ocr_dir.glob("*.txt")):
        doc_id = ocr_file.stem
        gt_file = gt_dir / f"{doc_id}.txt"

        if not gt_file.exists():
            print(f'Пропущен: {doc_id} (нет файла в "{gt_subdir}")')
            continue

        ocr_text = ocr_file.read_text(encoding="utf-8").strip()
        gt_text = gt_file.read_text(encoding="utf-8").strip()
        samples.append((doc_id, ocr_text, gt_text))

    return samples


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_path = config["dataset"]["path"]
    output_path = config["experiment"].get(
        "output_path", f"results/{config['dataset']['name']}_results.yaml"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    normalization_pipeline = NormalizationPipeline(config["llm"])

    samples = load_text_pairs(dataset_path)
    print(
        f"[INFO] Loaded {config['dataset']['name']} dataset. Found {len(samples)} samples."
    )

    results = []

    for idx, (doc_id, ocr_text, gt_text) in enumerate(samples):
        print(f"[INFO] Processing #{idx}: {doc_id}")

        # Apply normalization
        corrected_text = normalization_pipeline.process(
            ocr_text, custom_prompt="correction"
        )

        # Metrics for raw OCR
        noise_wer = compute_wer(gt_text, ocr_text)
        noise_cer = compute_cer(gt_text, ocr_text)
        noise_ss = compute_semantic_similarity(gt_text, ocr_text)

        # Metrics for corrected text
        corr_wer = compute_wer(gt_text, corrected_text)
        corr_cer = compute_cer(gt_text, corrected_text)
        corr_ss = compute_semantic_similarity(gt_text, corrected_text)

        result_item = {
            "image_id": doc_id,
            "typed_text": gt_text,
            "handwritten_text_raw": ocr_text,
            "handwritten_text_corrected": corrected_text,
            "metrics": {
                "noise": {"WER": noise_wer, "CER": noise_cer, "SS": noise_ss},
                "corrected": {"WER": corr_wer, "CER": corr_cer, "SS": corr_ss},
            },
        }

        results.append(result_item)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(results, f, allow_unicode=True)

    print(f"[INFO] Done. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
