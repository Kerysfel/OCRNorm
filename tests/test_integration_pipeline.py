import os
from pathlib import Path

from scripts.run_inference import load_text_pairs
from src.normalization_pipeline import NormalizationPipeline


def test_pipeline_integration_regex_only(tmp_path):
    base = Path("tests/data/BLNmini")

    # Sanity: dataset layout
    assert (base / "OCR Text").is_dir()
    assert (base / "Ground Truth").is_dir()

    samples = load_text_pairs(str(base))
    assert len(samples) == 2

    # LLM disabled -> regex cleanup only
    pipeline = NormalizationPipeline(
        {
            "use_llm": False,
            "model_name": "noop",
        }
    )

    outputs = []
    for doc_id, ocr_text, gt_text in samples:
        corrected = pipeline.process(ocr_text, custom_prompt="correction")
        assert isinstance(corrected, str)
        assert len(corrected) > 0
        outputs.append((doc_id, corrected))

    # Basic expectations: whitespace collapsed, punctuation retained
    out = dict(outputs)
    assert "  " not in out["a"]
    assert out["b"].lower().startswith("hello")
