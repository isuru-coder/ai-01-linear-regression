import json
from pathlib import Path

import joblib


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_metrics(metrics: dict, out_path: str | Path) -> None:
    out = Path(out_path)
    ensure_parent(out)
    out.write_text(json.dumps(metrics, indent=2))


def save_model(model, out_path: str | Path) -> None:
    out = Path(out_path)
    ensure_parent(out)
    joblib.dump(model, out)
