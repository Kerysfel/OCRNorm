# src/prompt_strategies.py


def get_prompt_for_llm(strategy_name: str) -> str:
    cleanup_prompt = """You are an assistant specialized in cleaning raw OCR text.
Your goal:
1) Remove strange characters, random symbols, or broken fragments that clearly do not form real words.
2) Keep all normal words and punctuation if possible.
3) Do not add new text.

Return only the cleaned text, nothing else.

OCR text to clean:
"{input_text}"
"""
    correction_prompt = """You are an expert text corrector, specialized in fixing OCR errors.
Please:
1) Fix any spelling/grammar mistakes.
2) Preserve original wording if it's correct.
3) Do NOT add any new sentences.
4) Remove or repair only words that seem garbled.

Return only the fully corrected text, nothing else.

Text to correct:
"{input_text}"
"""

    if strategy_name == "cleanup":
        return cleanup_prompt
    elif strategy_name == "correction":
        return correction_prompt
    else:
        # fallback
        return correction_prompt
