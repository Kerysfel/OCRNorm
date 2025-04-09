#!/usr/bin/env python
# scripts/full_pipeline.py

import subprocess
import argparse
import yaml
import os

def main(config_path):
    # 1) Запускаем run_inference
    cmd_inference = [
        "python",
        "scripts/run_inference.py",
        "--config",
        config_path
    ]
    subprocess.run(cmd_inference, check=True)

    # 2) Считываем config, берём путь к results
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    results_path = cfg['experiment'].get('output_path', 'results/recognition_results.yaml')

    # Формируем имя для метрик
    base, ext = os.path.splitext(results_path)
    metrics_path = f"{base}_metrics{ext}"

    # 3) Запускаем evaluate_metrics
    cmd_eval = [
        "python",
        "scripts/evaluate_metrics.py",
        "--results", results_path,
        "--out", metrics_path
    ]
    subprocess.run(cmd_eval, check=True)

    print(f"[INFO] Full pipeline done. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
