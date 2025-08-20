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

        temperature_value = llm_config.get("temperature", 0.2)
        self.temperature: float = (
            float(temperature_value)
            if isinstance(temperature_value, (int, float))
            else 0.2
        )

    def process(self, text: str, custom_prompt: str) -> str:
        # 1) RegExp cleanup
        text_cleaned: str = self._regex_cleanup(text)

        prompt_template: str = get_prompt_for_llm(custom_prompt)

        user_prompt: str = prompt_template.format(input_text=text_cleaned)

        # 2) LLM correction
        corrected_text: str = call_local_llm(
            model_name=self.model_name,
            user_text=user_prompt,
            temperature=self.temperature,
        )
        return corrected_text.strip()

    def _regex_cleanup(self, text: str) -> str:
        text = re.sub(r"[\(\)\[\]\"\']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
