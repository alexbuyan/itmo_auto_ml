import json
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
EDA_INSIGHTS_PATH = CONFIG_DIR / "eda_insights.json"


def load_eda_insights(path: Path = EDA_INSIGHTS_PATH) -> Dict[str, Any]:
    """Load EDA insights from JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"EDA insights not found at {path}. Run notebooks/01_eda.ipynb first."
        )
    with open(path) as f:
        return json.load(f)


def save_eda_insights(insights: Dict[str, Any], path: Path = EDA_INSIGHTS_PATH) -> None:
    """Save EDA insights to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(insights, f, indent=2)
