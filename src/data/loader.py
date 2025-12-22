import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class DataLoader:
    """DataLoader for Allstate Claims Severity dataset."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None

    def load_train(self) -> pd.DataFrame:
        if self._train_df is None:
            train_path = self.data_dir / "train.csv"
            if not train_path.exists():
                raise FileNotFoundError(
                    f"Training file not found at {train_path}. "
                    "Download with: kaggle competitions download -c allstate-claims-severity"
                )
            self._train_df = pd.read_csv(train_path)
            logger.info(f"Loaded {len(self._train_df):,} training samples")
        return self._train_df

    def load_test(self) -> pd.DataFrame:
        if self._test_df is None:
            test_path = self.data_dir / "test.csv"
            if not test_path.exists():
                raise FileNotFoundError(
                    f"Test file not found at {test_path}. "
                    "Download with: kaggle competitions download -c allstate-claims-severity"
                )
            self._test_df = pd.read_csv(test_path)
            logger.info(f"Loaded {len(self._test_df):,} test samples")
        return self._test_df

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> list:
        return [col for col in df.columns if col.startswith("cat")]

    @staticmethod
    def get_continuous_columns(df: pd.DataFrame) -> list:
        return [col for col in df.columns if col.startswith("cont")]

