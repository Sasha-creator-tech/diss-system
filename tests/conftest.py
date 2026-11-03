import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "tests" / "golden" / "ihd"
IHD_CONFIG = ROOT / "examples" / "ihd" / "config.yaml"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: повний прогін ГА у побітово-точному режимі (~1 хв)")


@pytest.fixture(scope="session")
def golden():
    return {
        "t3": pd.read_csv(GOLDEN / "Table_3_Clinical_Data.csv"),
        "t4": pd.read_csv(GOLDEN / "Table_4_Normalized_Data.csv", dtype=str),
        "t5": pd.read_csv(GOLDEN / "Table_5_Weights.csv"),
        "t6": pd.read_csv(GOLDEN / "Table_6_Comparison.csv", dtype=str),
    }


@pytest.fixture(scope="session")
def ihd_cfg():
    from core.config import load_config
    return load_config(IHD_CONFIG)
