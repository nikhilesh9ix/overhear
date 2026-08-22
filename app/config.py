from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    sarvam_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    hf_token: str = ""

    embed_model: str = "BAAI/bge-small-en-v1.5"
    groq_model: str = "qwen/qwen3.6-27b"
    cerebras_model: str = "gpt-oss-120b"

    speculation_debounce_ms: int = 120
    top_k: int = 5
    request_deadline_ms: int = 8000

    sarvam_ws_url: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
    sarvam_model: str = "saaras:v3-realtime"
    sarvam_language: str = "auto"
    sarvam_stream_type: str = "fast"
    sarvam_mode: str = "translate"  # Indic speech -> English text, matching our English index
    sarvam_sample_rate: int = 16000
    sarvam_silence_duration_ms: int = 500
    sarvam_min_speech_duration_ms: int = 250
    sarvam_vad_threshold: float = 0.3
    groq_url: str = "https://api.groq.com/openai/v1/chat/completions"
    cerebras_url: str = "https://api.cerebras.ai/v1/chat/completions"


settings = Settings()
