# src/normalization_pipeline.py
import re

from src.prompt_strategies import get_prompt_for_llm
from scripts.llm_inference import call_local_llm


class NormalizationPipeline:
    def __init__(self, llm_config: dict[str, bool | str | float]):
        use_llm_value = llm_config.get("use_llm", False)
        self.use_llm: bool = bool(use_llm_value)

        model_name_value = llm_config.get("model_name", "")
        self.model_name: str = (
            model_name_value if isinstance(model_name_value, str) else ""
        )

        temperature_value: float | str | int = llm_config.get("temperature", 0.2)
        self.temperature: float = (
            float(temperature_value)
            if isinstance(temperature_value, (int, float))
            else 0.2
        )

        # Optional sampling and reproducibility controls
        top_p_value = llm_config.get("top_p")
        self.top_p: float | None = (
            float(top_p_value) if isinstance(top_p_value, (int, float)) else None
        )
        top_k_value = llm_config.get("top_k")
        self.top_k: int | None = (
            int(top_k_value) if isinstance(top_k_value, (int, float)) else None
        )
        max_tokens_value = llm_config.get("max_tokens")
        self.max_tokens: int | None = (
            int(max_tokens_value)
            if isinstance(max_tokens_value, (int, float))
            else None
        )
        seed_value = llm_config.get("seed")
        self.seed: int | None = (
            int(seed_value) if isinstance(seed_value, (int, float)) else None
        )

        # Optional endpoint and timeout from config; can be overridden by env
        endpoint_value = llm_config.get("host") or llm_config.get("url")
        self.endpoint_url: str | None = (
            str(endpoint_value) if isinstance(endpoint_value, str) else None
        )
        timeout_value = llm_config.get("timeout_seconds") or llm_config.get("timeout")
        self.timeout_seconds: float | None = (
            float(timeout_value) if isinstance(timeout_value, (int, float)) else None
        )

        # Double-pass is inferred from the provided strategy name at call time.
        # If strategy is exactly "cleanup" or "correction" -> single-pass.
        # Any other strategy name implies a two-stage pipeline: cleanup then correction.

    def process(self, text: str, custom_prompt: str) -> str:
        # Step 1: RegExp normalization
        text_cleaned: str = self._regex_cleanup(text)

        # If LLM usage is disabled, return early with regex-cleaned text
        if not self.use_llm:
            return text_cleaned

        # Determine execution mode by strategy name:
        # - "cleanup" or "correction": run single stage with that strategy
        # - any other value: run two stages (cleanup then correction)
        strategy: str = custom_prompt

        if strategy == "cleanup" or strategy == "correction":
            prompt_template: str = get_prompt_for_llm(strategy)
            user_prompt: str = prompt_template.format(input_text=text_cleaned)
            result_text: str = call_local_llm(
                model_name=self.model_name,
                user_text=user_prompt,
                temperature=self.temperature,
                url=self.endpoint_url,
                timeout_seconds=self.timeout_seconds,
                seed=self.seed,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
            )
            return result_text.strip()

        # Two-stage pipeline: cleanup -> correction
        cleanup_template: str = get_prompt_for_llm("cleanup")
        cleanup_prompt: str = cleanup_template.format(input_text=text_cleaned)
        cleaned_by_llm: str = call_local_llm(
            model_name=self.model_name,
            user_text=cleanup_prompt,
            temperature=self.temperature,
            url=self.endpoint_url,
            timeout_seconds=self.timeout_seconds,
            seed=self.seed,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        ).strip()

        correction_template: str = get_prompt_for_llm("correction")
        correction_prompt: str = correction_template.format(input_text=cleaned_by_llm)
        corrected_text: str = call_local_llm(
            model_name=self.model_name,
            user_text=correction_prompt,
            temperature=self.temperature,
            url=self.endpoint_url,
            timeout_seconds=self.timeout_seconds,
            seed=self.seed,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )
        return corrected_text.strip()

    def _regex_cleanup(self, text: str) -> str:
        text = re.sub(r"[\(\)\[\]\"\']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
