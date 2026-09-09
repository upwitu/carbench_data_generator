"""Configuration module for CarBench SFT Data Generator.
Enhanced with first-class support for NVIDIA NIM API (build.nvidia.com),
Strict 40 RPM Sliding-Window Rate Limiter, Multi-Key Rotation, and Resilient Error Handling.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneratorSettings(BaseSettings):
    """Settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. LLM API Endpoint & Authentication (NVIDIA NIM / OpenAI-compatible)
    OPENAI_API_BASE: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Base URL for NVIDIA NIM or OpenAI-compatible API (https://integrate.api.nvidia.com/v1)",
    )
    OPENAI_API_KEY: str = Field(
        default="nvapi-your-key-here",
        description="NVIDIA NIM API key (e.g. nvapi-...) or comma-separated list of keys for automatic rotation",
    )
    NVIDIA_API_KEY: Optional[str] = Field(
        default=None,
        description="Alternative alias for NVIDIA NIM API key",
    )
    NVIDIA_API_KEYS: Optional[str] = Field(
        default=None,
        description="Comma-separated list of NVIDIA NIM API keys for load-balancing and failover rotation",
    )
    CAR_BENCH_MODEL: str = Field(
        default="deepseek-ai/deepseek-v4-flash-0731",
        description="Model identifier on NVIDIA NIM (e.g. deepseek-ai/deepseek-v4-flash-0731, deepseek-ai/deepseek-r1, deepseek-ai/deepseek-v3)",
    )

    # 2. Strict 40 RPM Rate Limiter & Concurrency Controls
    MAX_RPM: int = Field(
        default=38,
        description="Strict maximum requests per minute per key (Default: 38 RPM with safety buffer for 40 RPM ceiling)",
    )
    CONCURRENCY_LIMIT: int = Field(
        default=80,
        description="Maximum concurrent async workers (Default: 80 workers for DeepSeek high-throughput burst)",
    )
    REQUEST_DELAY_INTERVAL: float = Field(
        default=0.01,
        description="Minimum inter-dispatch delay (seconds)",
    )
    REQUEST_TIMEOUT: float = Field(
        default=180.0,
        description="Timeout per LLM request in seconds",
    )
    MAX_RETRIES: int = Field(
        default=4,
        description="Max retry attempts per failed/rate-limited task generation",
    )
    RETRY_DELAY: float = Field(
        default=2.0,
        description="Initial delay in seconds for exponential backoff",
    )
    MAX_BACKOFF_DELAY: float = Field(
        default=30.0,
        description="Maximum backoff delay ceiling in seconds",
    )

    # 3. Data Generation Controls
    VARIATIONS_PER_TASK: int = Field(
        default=10,
        description="Number of diverse multi-turn variations to synthesize per seed task",
    )
    MAX_TOKENS: int = Field(
        default=3072,
        description="Maximum generation tokens per LLM completion (Default: 3072 tokens)",
    )
    THINKING_MODE: bool = Field(
        default=False,
        description="Enable/disable DeepSeek Thinking Mode (default: False to save tokens & cost)",
    )
    TEMPERATURE: float = Field(
        default=0.7,
        description="Sampling temperature for diverse data generation (bounded between 0.0 and 1.0 for NIM)",
    )
    TOP_P: float = Field(
        default=0.95,
        description="Top-p nucleus sampling",
    )

    # 4. Hugging Face Storage & Dataset Repo
    HF_TOKEN: Optional[str] = Field(
        default=None,
        description="Hugging Face User Access Token (write scope)",
    )
    HF_DATASET_REPO: str = Field(
        default="upwitu/carbench_sft_winner_dataset",
        description="Target Hugging Face Dataset repository",
    )
    HF_SEED_REPO: Optional[str] = Field(
        default="upwitu/carbench_sft_benchmark_data",
        description="Source Hugging Face seed dataset repo if pulling from HF",
    )

    # 5. Local File Paths
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    OUTPUT_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "outputs"
    )
    SEED_DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data"
    )

    def get_api_keys(self) -> List[str]:
        """Returns list of active API keys supporting rotation across multiple keys."""
        raw_keys = (
            os.getenv("DEEPSEEK_API_KEYS")
            or os.getenv("DEEPSEEK_API_KEY")
            or self.NVIDIA_API_KEYS
            or self.NVIDIA_API_KEY
            or self.OPENAI_API_KEY
        )
        if not raw_keys:
            return ["dummy-key"]
        keys = [k.strip() for k in raw_keys.split(",") if k.strip() and k.strip() != "your_api_key_here"]
        return keys if keys else [raw_keys.strip()]


# Singleton settings instance
settings = GeneratorSettings()
