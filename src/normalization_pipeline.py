# src/normalization_pipeline.py
import re

from src.prompt_strategies import get_prompt_for_llm
from scripts.llm_inference import call_local_llm  # <-- вызов локального эндпоинта


class NormalizationPipeline:
    def __init__(self, llm_config: dict):
        self.use_llm = llm_config.get("use_llm", False)
        self.model_name = llm_config.get("model_name", None)
        self.temperature = llm_config.get("temperature", 0.2)

    def process(self, text: str, custom_prompt: str) -> str:
        # 1. RegExp очистка
        text_cleaned = self._regex_cleanup(text)

        prompt_template = get_prompt_for_llm(custom_prompt)

        user_prompt = prompt_template.format(input_text=text_cleaned)

        # Вызов локальной модели (mistral, llama и т.д.)
        corrected_text = call_local_llm(
            model_name=self.model_name,
            user_text=user_prompt,
            temperature=self.temperature,
        )
        return corrected_text.strip()

    def _regex_cleanup(self, text: str) -> str:
        text = re.sub(r"[\(\)\[\]\"\']", "", text)  # удаляем скобки, кавычки
        text = re.sub(r"\s+", " ", text)  # множественные пробелы
        return text.strip()
