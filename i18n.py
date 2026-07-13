"""Lightweight i18n: language codes, prompt instructions, and TTS voice selection."""

from typing import Optional

INPUT_LANGUAGES = [
    ("English", "en"),
    ("Hindi", "hi"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Chinese (Simplified)", "zh-Hans"),
    ("Arabic", "ar"),
    ("Italian", "it"),
]

OUTPUT_LANGUAGES = [name for name, _ in INPUT_LANGUAGES]

OUT_LANG_INSTRUCT = {
    "English": "You MUST answer strictly in English. Do not use Chinese or any other language.",
    "Hindi": "उत्तर केवल हिंदी (देवनागरी लिपि) में दें। अंग्रेज़ी शब्दों से बचें।",
    "Spanish": "Responde en español.",
    "French": "Réponds en français.",
    "German": "Antworte auf Deutsch.",
    "Japanese": "日本語で答えてください。",
    "Korean": "한국어로 대답해 주세요.",
    "Portuguese": "Responda em português.",
    "Russian": "Отвечай на русском.",
    "Chinese (Simplified)": "请用简体中文回答。",
    "Arabic": "أجب بالعربية.",
    "Italian": "Rispondi in italiano.",
}

PIPER_VOICES = {
    "English": "en_US-amy-low",
    "Hindi": "hi_IN-pratham-medium",
    "Spanish": "es_ES-davefx-medium",
    "French": "fr_FR-siwis-medium",
    "German": "de_DE-thorsten-low",
    "Japanese": "ja_JP-natsu-medium",
    "Korean": "ko_KO-kss-medium",
    "Portuguese": "pt_PT-tugão-medium",
    "Russian": "ru_RU-dmitri-medium",
    "Chinese (Simplified)": "zh_CN-huayan-medium",
    "Arabic": "ar_JO-kareem-medium",
    "Italian": "it_IT-paola-medium",
}

def lang_instruction(language_name: str) -> str:
    """Returns the prompt-fragment that forces the LLM to answer in `language_name`."""
    return OUT_LANG_INSTRUCT.get(language_name, f"Answer in {language_name}.")

def piper_voice_for(language_name: str) -> str:
    """Returns the piper voice ID for the given display language name."""
    return PIPER_VOICES.get(language_name, "en_US-amy-low")

WHISPER_LANG_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese (Simplified)": "zh",
    "Arabic": "ar",
    "Italian": "it",
}

def whisper_lang_for(language_name: str) -> Optional[str]:
    """Returns the Whisper/ctranslate2 language code for the given display
    language name, or None if unknown (lets Whisper auto-detect instead of
    silently mistranscribing)."""
    return WHISPER_LANG_CODES.get(language_name)