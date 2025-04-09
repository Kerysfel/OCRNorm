#!/usr/bin/env python
# scripts/evaluate_aggregates.py

import argparse
import yaml
import statistics


def main(results_path: str):
    # 1) Считываем YAML
    with open(results_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Если data не список, значит формат неправильный
    if not isinstance(data, list):
        print(f"[ERROR] The file {results_path} does not contain a list.")
        return

    # 2) Будем хранить все метрики в списках
    noise_wer = []
    noise_cer = []
    noise_ss = []

    corr_wer = []
    corr_cer = []
    corr_ss = []

    # 3) Пробегаем по записям в файле
    for item in data:
        metrics = item.get("metrics")
        if not metrics:
            # Если нет метрик, пропускаем
            continue

        noise = metrics.get("noise")
        corrected = metrics.get("corrected")

        if noise:
            if "WER" in noise:
                noise_wer.append(noise["WER"])
            if "CER" in noise:
                noise_cer.append(noise["CER"])
            if "SS" in noise:
                noise_ss.append(noise["SS"])

        if corrected:
            if "WER" in corrected:
                corr_wer.append(corrected["WER"])
            if "CER" in corrected:
                corr_cer.append(corrected["CER"])
            if "SS" in corrected:
                corr_ss.append(corrected["SS"])

    # 4) Средние (mean) метрики
    def safe_mean(values):
        return statistics.mean(values) if values else 0.0

    avg_noise_wer = safe_mean(noise_wer)
    avg_noise_cer = safe_mean(noise_cer)
    avg_noise_ss = safe_mean(noise_ss)

    avg_corr_wer = safe_mean(corr_wer)
    avg_corr_cer = safe_mean(corr_cer)
    avg_corr_ss = safe_mean(corr_ss)

    # 5) Выводим
    print("==== AVERAGE METRICS ====")
    print(
        f"Noise (raw)   => WER={avg_noise_wer:.4f}, CER={avg_noise_cer:.4f}, SS={avg_noise_ss:.4f}"
    )
    print(
        f"Corrected     => WER={avg_corr_wer:.4f}, CER={avg_corr_cer:.4f}, SS={avg_corr_ss:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", required=True, help="Path to the results YAML file"
    )
    args = parser.parse_args()
    main(args.results)
