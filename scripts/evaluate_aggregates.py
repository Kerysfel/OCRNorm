# scripts/evaluate_aggregates.py

import argparse
import yaml  # type: ignore
import statistics
import logging


def main(results_path: str):
    with open(results_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        logging.error("The file %s does not contain a list.", results_path)
        return

    noise_wer = []
    noise_cer = []
    noise_ss = []

    corr_wer = []
    corr_cer = []
    corr_ss = []

    for item in data:
        metrics = item.get("metrics")
        if not metrics:
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

    def safe_mean(values):
        return statistics.mean(values) if values else 0.0

    avg_noise_wer = safe_mean(noise_wer)
    avg_noise_cer = safe_mean(noise_cer)
    avg_noise_ss = safe_mean(noise_ss)

    avg_corr_wer = safe_mean(corr_wer)
    avg_corr_cer = safe_mean(corr_cer)
    avg_corr_ss = safe_mean(corr_ss)

    logging.info("==== AVERAGE METRICS ====")
    logging.info(
        "Noise (raw)   => WER=%.4f, CER=%.4f, SS=%.4f",
        avg_noise_wer,
        avg_noise_cer,
        avg_noise_ss,
    )
    logging.info(
        "Corrected     => WER=%.4f, CER=%.4f, SS=%.4f",
        avg_corr_wer,
        avg_corr_cer,
        avg_corr_ss,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", required=True, help="Path to the results YAML file"
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
    main(args.results)
