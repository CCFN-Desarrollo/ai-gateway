from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import RuntimeConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    PROJECT_NAME: str = "AI Gateway"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    CORS_ALLOWED_ORIGINS: str = "http://localhost,http://127.0.0.1"
    AI_PROVIDER: str = "anthropic"
    AI_PROVIDER_IDENTITY: str = ""
    AI_PROVIDER_RECEIPT: str = ""
    AI_PROVIDER_CSF: str = ""
    OCR_PROVIDER_IDENTITY: str = ""
    OCR_PROVIDER_RECEIPT: str = ""
    OCR_PROVIDER_CSF: str = ""
    VISION_PROVIDER_IDENTITY: str = ""
    VISION_PROVIDER_RECEIPT: str = ""

    # Claude / Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    ANTHROPIC_TIMEOUT_SECONDS: float = 20.0
    ANTHROPIC_MAX_RETRIES: int = 1

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4.1-mini"
    OPENAI_TIMEOUT_SECONDS: float = 20.0
    OPENAI_MAX_RETRIES: int = 1

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2-vision:11b"
    OLLAMA_TIMEOUT_SECONDS: float = 30.0
    OLLAMA_MAX_RETRIES: int = 1

    # API Key auth (comma-separated list)
    API_KEYS: str = "dev-key-1"

    # Confidence score thresholds (0-100)
    SCORE_AUTO_APPROVE: float = 95.0
    SCORE_HUMAN_REVIEW: float = 70.0

    # Max file size in MB
    MAX_FILE_SIZE_MB: int = 10
    PREPROCESS_DEBUG_DIR: str = ""
    VALIDATION_CASES_DIR: str = "data/validation_cases"
    CRM_ENABLED: bool = False
    CRM_BASE_URL: str = ""
    CRM_API_KEY: str = ""
    CRM_API_KEY_HEADER: str = "X-API-Key"
    CRM_API_KEY_PREFIX: str = ""
    CRM_TIMEOUT_SECONDS: float = 20.0

    @property
    def api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    @property
    def cors_allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def ai_provider(self) -> str:
        return self.AI_PROVIDER.strip().lower()

    @staticmethod
    def _normalize_provider(value: str) -> str:
        return value.strip().lower()

    def provider_for_pipeline(self, pipeline_name: str) -> str:
        provider_by_pipeline = {
            "identity": self.AI_PROVIDER_IDENTITY,
            "receipt": self.AI_PROVIDER_RECEIPT,
            "csf": self.AI_PROVIDER_CSF,
        }
        configured_provider = provider_by_pipeline.get(pipeline_name, "")
        provider = self._normalize_provider(configured_provider or self.AI_PROVIDER)
        if provider not in {"anthropic", "openai", "ollama"}:
            raise RuntimeConfigurationError(
                f"Provider for pipeline '{pipeline_name}' must be 'anthropic', 'openai' or "
                f"'ollama', got '{configured_provider or self.AI_PROVIDER}'."
            )
        return provider

    def provider_for_stage(self, pipeline_name: str, stage_name: str) -> str:
        stage = stage_name.strip().lower()
        stage_provider_by_pipeline = {
            "ocr": {
                "identity": self.OCR_PROVIDER_IDENTITY,
                "receipt": self.OCR_PROVIDER_RECEIPT,
                "csf": self.OCR_PROVIDER_CSF,
            },
            "vision": {
                "identity": self.VISION_PROVIDER_IDENTITY,
                "receipt": self.VISION_PROVIDER_RECEIPT,
            },
        }
        configured_stage_provider = stage_provider_by_pipeline.get(stage, {}).get(pipeline_name, "")
        if configured_stage_provider.strip():
            provider = self._normalize_provider(configured_stage_provider)
        else:
            provider = self.provider_for_pipeline(pipeline_name)

        allowed_providers = {"anthropic", "openai", "ollama"}
        if provider not in allowed_providers:
            allowed_text = "', '".join(sorted(allowed_providers))
            raise RuntimeConfigurationError(
                f"Provider for stage '{stage}' in pipeline '{pipeline_name}' must be "
                f"'{allowed_text}', got '{configured_stage_provider or provider}'."
            )
        return provider

    def validate_runtime(self) -> None:
        """Fail fast for unsafe production-like configurations."""
        for provider_name, raw_value in (
            ("AI_PROVIDER", self.AI_PROVIDER),
            ("AI_PROVIDER_IDENTITY", self.AI_PROVIDER_IDENTITY),
            ("AI_PROVIDER_RECEIPT", self.AI_PROVIDER_RECEIPT),
            ("AI_PROVIDER_CSF", self.AI_PROVIDER_CSF),
            ("OCR_PROVIDER_IDENTITY", self.OCR_PROVIDER_IDENTITY),
            ("OCR_PROVIDER_RECEIPT", self.OCR_PROVIDER_RECEIPT),
            ("OCR_PROVIDER_CSF", self.OCR_PROVIDER_CSF),
            ("VISION_PROVIDER_IDENTITY", self.VISION_PROVIDER_IDENTITY),
            ("VISION_PROVIDER_RECEIPT", self.VISION_PROVIDER_RECEIPT),
        ):
            if raw_value.strip() and self._normalize_provider(raw_value) not in {
                "anthropic",
                "openai",
                "ollama",
            }:
                raise RuntimeConfigurationError(
                    f"{provider_name} must be 'anthropic', 'openai' or 'ollama', got '{raw_value}'."
                )

        if self.ENVIRONMENT.lower() == "development":
            return

        if not self.api_keys_list or self.API_KEYS.strip() == "dev-key-1":
            raise RuntimeConfigurationError("API_KEYS must be configured outside development.")

        if not self.cors_allowed_origins:
            raise RuntimeConfigurationError("CORS_ALLOWED_ORIGINS must define at least one origin.")

        configured_providers = {
            self.provider_for_stage("identity", "ocr"),
            self.provider_for_stage("identity", "vision"),
            self.provider_for_stage("receipt", "ocr"),
            self.provider_for_stage("receipt", "vision"),
        }

        if "anthropic" in configured_providers and not self.ANTHROPIC_API_KEY.strip():
            raise RuntimeConfigurationError(
                "ANTHROPIC_API_KEY must be configured when an Anthropic pipeline is enabled."
            )

        if "openai" in configured_providers and not self.OPENAI_API_KEY.strip():
            raise RuntimeConfigurationError(
                "OPENAI_API_KEY must be configured when an OpenAI pipeline is enabled."
            )


settings = Settings()
