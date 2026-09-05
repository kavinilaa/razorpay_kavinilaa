from pydantic_settings import BaseSettings, SettingsConfigDict

from risk_engine import thresholds as _thresholds  # src/ already on sys.path - see backend/__init__.py


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_API_")

    app_name: str = "AI Fraud-Spike & Risk Detection API"
    app_version: str = "0.1.0"  # backend service version - independent of the model version reported by /model-info
    log_level: str = "INFO"
    default_operating_mode: str = _thresholds.DEFAULT_MODE  # "BALANCED" - sourced from thresholds.py, not retyped

    # Phase 6/7: comma-separated list of allowed browser origins, e.g.
    # "http://localhost:5173,https://my-deployed-frontend.example.com".
    # Deliberately a plain string (not a Settings-parsed list) so the env var
    # can be a simple comma-separated value rather than requiring JSON-array
    # syntax. Use `cors_allow_origins_list` below to get the parsed form.
    # Still NOT a wildcard by default, and still NOT a substitute for the
    # api_key auth below - see reports/phase7_summary.md.
    cors_allow_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:3000"

    # Phase 7: single shared-secret API key. See backend/core/auth.py for how
    # this is enforced and reports/phase7_summary.md for exactly what this
    # scheme does and does not protect against.
    #
    # IMPORTANT: this default value is committed to version control and is
    # therefore PUBLIC. It exists purely so a fresh local checkout works
    # out of the box for development and so this repo's own test suite has
    # a known key to authenticate with. Any shared or hosted deployment
    # MUST override RISK_API_API_KEY with a real secret - the app does not
    # refuse to start on the default value, so this is a deployment-process
    # responsibility, not something this code enforces for you.
    api_key: str = "local-dev-key-CHANGE-ME"

    # The header name callers must supply the key in. Configurable so a
    # deployment can rename it (e.g. to match an existing API-gateway
    # convention) without a code change.
    api_key_header_name: str = "X-API-Key"

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


settings = Settings()
