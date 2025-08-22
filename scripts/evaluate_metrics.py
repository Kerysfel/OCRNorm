# pyright: reportMissingTypeStubs=false
# scripts/evaluate_metrics.py
"""Flatten and export per-item metrics from run_inference results.

Input format is the list of dicts produced by scripts/run_inference.py, e.g.:
{
  "image_id": str,
  "typed_text": str,
  "handwritten_text_raw": str,
  "handwritten_text_corrected": str,
  "metrics": {
    "noise": {"WER": float, "CER": float, "SS": float},
    "corrected": {"WER": float, "CER": float, "SS": float}
  }
}

This script writes a flat list with raw/corrected metrics and deltas per item.
"""

import argparse
import os
import yaml  # type: ignore
import logging

from src.evaluation import compute_wer, compute_cer, compute_semantic_similarity  # noqa: F401


def main(results_path: str, out_path: str):
    with open(results_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("Input results must be a list of result items")

    flattened: list[dict[str, float | str]] = []

    for item in data:
        # Backward compatibility: accept either doc_id (new) or image_id (legacy)
        image_id = item.get("doc_id", item.get("image_id", ""))
        metrics = item.get("metrics", {}) or {}
        noise = metrics.get("noise", {}) or {}
        corrected = metrics.get("corrected", {}) or {}

        wer_noise = float(noise.get("WER", 0.0))
        cer_noise = float(noise.get("CER", 0.0))
        ss_noise = float(noise.get("SS", 0.0))

        wer_corr = float(corrected.get("WER", 0.0))
        cer_corr = float(corrected.get("CER", 0.0))
        ss_corr = float(corrected.get("SS", 0.0))

        flattened.append(
            {
                "doc_id": image_id,
                "wer_raw": wer_noise,
                "cer_raw": cer_noise,
                "ss_raw": ss_noise,
                "wer_corrected": wer_corr,
                "cer_corrected": cer_corr,
                "ss_corrected": ss_corr,
                "delta_wer": wer_noise - wer_corr,
                "delta_cer": cer_noise - cer_corr,
                "delta_ss": ss_corr - ss_noise,
            }
        )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(flattened, f, allow_unicode=True, sort_keys=False)

    logging.info("Per-item metrics saved to %s", out_path)


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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    main(args.results, args.out)
