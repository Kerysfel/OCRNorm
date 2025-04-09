#!/usr/bin/env python
# scripts/evaluate_metrics.py
import argparse
import yaml
import os

from src.evaluation import compute_wer, compute_cer, compute_semantic_similarity


def main(results_path: str, out_path: str):
    # 1. Считываем результаты распознавания
    with open(results_path, "r", encoding="utf-8") as f:
        recognition_results = yaml.safe_load(f)

    metrics_data = []

    # 2. Для каждого элемента считаем метрики
    for item in recognition_results:
        gt = item.get("ground_truth", "")
        recognized = item.get(
            "corrected_text", ""
        )  # или 'recognized_text' — зависит от того, что сравниваем

        wer = compute_wer(gt, recognized)
        cer = compute_cer(gt, recognized)
        sem_sim = compute_semantic_similarity(gt, recognized)

        item_metrics = {
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "wer": wer,
            "cer": cer,
            "semantic_similarity": sem_sim,
        }
        metrics_data.append(item_metrics)

    # 3. Сохраняем
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics_data, f, allow_unicode=True)

    print(f"[INFO] Metrics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=str, required=True, help="Path to recognition results YAML"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/metrics.yaml",
        help="Path to output metrics YAML",
    )
    args = parser.parse_args()

    main(args.results, args.out)
