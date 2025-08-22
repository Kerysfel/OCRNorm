# pyright: reportMissingTypeStubs=false
# scripts/run_inference.py
import argparse
import os
import yaml  # type: ignore
from src.evaluation import compute_wer, compute_cer, compute_semantic_similarity
from src.normalization_pipeline import NormalizationPipeline
from pathlib import Path
import random
import numpy as np
import logging

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    # Fallback no-op if tqdm is not installed
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


def load_text_pairs(
    base_path: str, ocr_subdir: str = "OCR Text", gt_subdir: str = "Ground Truth"
) -> list[tuple[str, str, str]]:
    """
    Find (doc_id, ocr_text, gt_text) pairs by matching *.txt file stems.
    Args:
        base_path (str): The base path to the dataset.
        ocr_subdir (str): The subdirectory containing OCR text files.
        gt_subdir (str): The subdirectory containing ground truth text files.
    Returns:
        list[tuple[str, str, str]]: A list of tuples containing (doc_id, ocr_text, gt_text).
    """
    base: Path = Path(base_path)
    ocr_dir: Path = base / ocr_subdir
    gt_dir: Path = base / gt_subdir

    samples: list[tuple[str, str, str]] = []
    for ocr_file in sorted(ocr_dir.glob("*.txt")):
        doc_id: str = ocr_file.stem
        gt_file: Path = gt_dir / f"{doc_id}.txt"

        if not gt_file.exists():
            logging.warning('Skipped: %s (no file in "%s")', doc_id, gt_subdir)
            continue

        ocr_text: str = ocr_file.read_text(encoding="utf-8").strip()
        gt_text: str = gt_file.read_text(encoding="utf-8").strip()
        samples.append((doc_id, ocr_text, gt_text))

    return samples


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def _type_name(v) -> str:
        return type(v).__name__

    def _validate_config(cfg: dict) -> None:
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a mapping (YAML dict)")

        # dataset
        if "dataset" not in cfg:
            raise ValueError("Missing required section: dataset")
        ds = cfg["dataset"]
        if not isinstance(ds, dict):
            raise ValueError("dataset must be a mapping (dict)")
        if (
            "name" not in ds
            or not isinstance(ds["name"], str)
            or not ds["name"].strip()
        ):
            raise ValueError("dataset.name must be a non-empty string")
        if (
            "path" not in ds
            or not isinstance(ds["path"], str)
            or not ds["path"].strip()
        ):
            raise ValueError("dataset.path must be a non-empty string")
        if "ocr_subdir" in ds and not isinstance(ds["ocr_subdir"], str):
            raise ValueError(
                f"dataset.ocr_subdir must be string, got {_type_name(ds['ocr_subdir'])}"
            )
        if "gt_subdir" in ds and not isinstance(ds["gt_subdir"], str):
            raise ValueError(
                f"dataset.gt_subdir must be string, got {_type_name(ds['gt_subdir'])}"
            )

        # experiment
        if "experiment" not in cfg:
            raise ValueError("Missing required section: experiment")
        exp = cfg["experiment"]
        if not isinstance(exp, dict):
            raise ValueError("experiment must be a mapping (dict)")
        if "output_path" in exp and not isinstance(exp["output_path"], str):
            raise ValueError(
                f"experiment.output_path must be string, got {_type_name(exp['output_path'])}"
            )
        if "checkpoint_every" in exp and not isinstance(
            exp["checkpoint_every"], (int, float)
        ):
            raise ValueError(
                f"experiment.checkpoint_every must be int, got {_type_name(exp['checkpoint_every'])}"
            )

        # llm
        if "llm" not in cfg:
            raise ValueError("Missing required section: llm")
        llm = cfg["llm"]
        if not isinstance(llm, dict):
            raise ValueError("llm must be a mapping (dict)")
        if "use_llm" in llm and not isinstance(llm["use_llm"], (bool, int)):
            raise ValueError(
                f"llm.use_llm must be boolean, got {_type_name(llm['use_llm'])}"
            )
        if llm.get("use_llm", False):
            # if LLM enabled, ensure model_name and strategy exist and are strings
            if (
                "model_name" not in llm
                or not isinstance(llm["model_name"], str)
                or not llm["model_name"].strip()
            ):
                raise ValueError(
                    "llm.model_name must be a non-empty string when use_llm=true"
                )
            if "strategy" in llm and not isinstance(llm["strategy"], str):
                raise ValueError(
                    f"llm.strategy must be string, got {_type_name(llm['strategy'])}"
                )
        if "temperature" in llm and not isinstance(llm["temperature"], (int, float)):
            raise ValueError(
                f"llm.temperature must be number, got {_type_name(llm['temperature'])}"
            )
        for key, typ in (
            ("top_p", float),
            ("top_k", int),
            ("max_tokens", int),
            ("timeout_seconds", float),
            ("seed", int),
        ):
            if key in llm and not isinstance(
                llm[key], (int, float) if typ in (int, float) else typ
            ):
                # Simplify: accept int for floats
                expected = "number" if typ in (int, float) else typ.__name__
                raise ValueError(
                    f"llm.{key} must be {expected}, got {_type_name(llm[key])}"
                )

    try:
        _validate_config(config)
    except Exception as e:
        logging.error("Invalid configuration: %s", e)
        raise

    dataset_path: str = config["dataset"]["path"]
    ocr_subdir: str = config["dataset"].get("ocr_subdir", "OCR Text")
    gt_subdir: str = config["dataset"].get("gt_subdir", "Ground Truth")
    output_path: str = config["experiment"].get(
        "output_path", f"results/{config['dataset']['name']}_results.yaml"
    )
    normalize_before_eval: bool = bool(
        config.get("experiment", {}).get("normalize_before_eval", True)
    )

    out_dir_path = Path(output_path).parent
    if str(out_dir_path) not in ("", "."):
        os.makedirs(out_dir_path, exist_ok=True)

    # Seeding for determinism of Python/NumPy/Torch (LLM server may also use seed)
    seed = int(config["llm"].get("seed", 42))
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)  # type: ignore
        except Exception:
            pass

    normalization_pipeline: NormalizationPipeline = NormalizationPipeline(config["llm"])

    samples = load_text_pairs(dataset_path, ocr_subdir=ocr_subdir, gt_subdir=gt_subdir)
    logging.info(
        "Loaded %s dataset. Found %d samples.", config["dataset"]["name"], len(samples)
    )

    results: list[dict[str, str | dict[str, dict[str, float]]]] = []
    checkpoint_every: int = int(
        config.get("experiment", {}).get("checkpoint_every", 0) or 0
    )

    def _write_yaml_atomic(path: str, data):
        tmp_path = str(Path(path).with_suffix(Path(path).suffix + ".tmp"))
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)
        os.replace(tmp_path, path)

    for idx, (doc_id, ocr_text, gt_text) in enumerate(
        tqdm(samples, desc="Inference", unit="doc")
    ):
        logging.info("Processing #%d: %s", idx, doc_id)
        strategy = config["llm"].get("strategy", "full")
        corrected_text: str = normalization_pipeline.process(
            ocr_text, custom_prompt=strategy
        )
        noise_wer: float = compute_wer(
            gt_text, ocr_text, normalize=normalize_before_eval
        )
        noise_cer: float = compute_cer(
            gt_text, ocr_text, normalize=normalize_before_eval
        )
        noise_ss: float = compute_semantic_similarity(gt_text, ocr_text)
        corr_wer: float = compute_wer(
            gt_text, corrected_text, normalize=normalize_before_eval
        )
        corr_cer: float = compute_cer(
            gt_text, corrected_text, normalize=normalize_before_eval
        )
        corr_ss: float = compute_semantic_similarity(gt_text, corrected_text)

        result_item: dict[str, str | dict[str, dict[str, float]]] = {
            "doc_id": doc_id,
            "typed_text": gt_text,
            "handwritten_text_raw": ocr_text,
            "handwritten_text_corrected": corrected_text,
            "metrics": {
                "noise": {"WER": noise_wer, "CER": noise_cer, "SS": noise_ss},
                "corrected": {"WER": corr_wer, "CER": corr_cer, "SS": corr_ss},
            },
        }

        results.append(result_item)

        # Optional checkpointing for long runs
        if checkpoint_every and ((idx + 1) % checkpoint_every == 0):
            _write_yaml_atomic(output_path, results)

    logging.info("Done. Results saved to %s", output_path)
    # Final write once after processing all samples (atomic)
    _write_yaml_atomic(output_path, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
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
    main(args.config)
