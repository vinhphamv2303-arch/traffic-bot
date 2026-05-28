from __future__ import annotations

import os
from pathlib import Path


API_KEY_ENV_NAMES = ("OPEN_ROUTER_API", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        env_path = _find_env_file(env_path)
    if env_path is None or not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _find_env_file(path: Path) -> Path | None:
    if path.is_absolute():
        return None

    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / path
        if candidate.exists():
            return candidate

    module_dir = Path(__file__).resolve().parent
    for base in [module_dir, *module_dir.parents]:
        candidate = base / path
        if candidate.exists():
            return candidate

    return None
