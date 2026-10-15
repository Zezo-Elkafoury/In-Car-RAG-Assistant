"""Central configuration. All secrets and tunables come from the environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Provider ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Models ---
    openrouter_chat_model: str = "openai/gpt-4o-mini"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    # Stronger judge for RAGAS (the answer model is too weak to grade reliably).
    eval_llm_model: str = "openai/gpt-4o"
    rerank_model: str = "cohere/rerank-v3.5"

    # Caps keep per-request cost predictable (OpenRouter blocks requests whose
    # worst-case (prompt + max_tokens) cost exceeds the remaining balance).
    answer_max_tokens: int = 512
    judge_max_tokens: int = 512

    # --- Ingestion / retrieval ---
    data_path: str = "data/mg-zs-warning-messages.html"
    chroma_dir: str = "chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 5
    hybrid_bm25_weight: float = 0.5
    rerank_top_n: int = 3

    def require_api_key(self) -> str:
        if not self.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        return self.openrouter_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
