from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Settings:
    repo_root: Path = REPO_ROOT
    cache_dir: Path = REPO_ROOT / "data" / "cache" / "basic5000"
    ckpt_path: Path = REPO_ROOT / "runs" / "basic5000_full" / "best.pt"
    upload_dir: Path = REPO_ROOT / "data" / "uploads"
    sample_rate: int = 16000
    n_mels: int = 80

    db_url: str = os.environ.get("OTOKOE_DB_URL", f"sqlite:///{(REPO_ROOT / 'data' / 'app.db').as_posix()}")
    secret_key: str = os.environ.get("OTOKOE_SECRET", "dev-secret-change-me")
    jwt_algorithm: str = "HS256"
    token_ttl_seconds: int = 60 * 60 * 24

    score_green: float = 0.75
    score_yellow: float = 0.45


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
