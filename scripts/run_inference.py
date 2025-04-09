import argparse
import yaml
import os
import glob
import re
from src.evaluation import compute_wer, compute_cer, compute_semantic_similarity
from src.normalization_pipeline import NormalizationPipeline


def load_bln600_data(bln_base_path: str):
    image_dir = os.path.join(bln_base_path, "Images")
    ocr_dir = os.path.join(bln_base_path, "OCR Text")
    gt_dir = os.path.join(bln_base_path, "Ground Truth")

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

    samples = []
    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]  # e.g., 3200797029
        ocr_path = os.path.join(ocr_dir, base + ".txt")
        gt_path = os.path.join(gt_dir, base + ".txt")

        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr_text = f.read().strip()

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()

        samples.append((base, ocr_text, gt_text))

    return samples


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_path = config["dataset"]["path"]
    output_path = config["experiment"].get("output_path", "results/bln600_results.yaml")
    debug = config["experiment"].get("debug", False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    normalization_pipeline = NormalizationPipeline(config["llm"])

    samples = load_bln600_data(dataset_path)
    print(f"[INFO] Loaded BLN600 dataset. Found {len(samples)} samples.")

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
