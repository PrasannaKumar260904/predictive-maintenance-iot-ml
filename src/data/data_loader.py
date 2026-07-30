"""NASA CMAPSS & IoT Sensor Data Loader."""

import urllib.request
from pathlib import Path

import pandas as pd

from src.data.generator import generate_iot_sensor_data
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


CMAPSS_COLUMNS = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def download_file_if_not_exists(url: str, dest_path: Path) -> bool:
    """Downloads a dataset file if it does not already exist locally."""
    if dest_path.exists():
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading dataset from {url} -> {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Successfully downloaded {dest_path.name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download from {url}: {e}")
        return False


def calculate_rul(df: pd.DataFrame, max_rul_clip: int = 125) -> pd.DataFrame:
    """Calculates Remaining Useful Life (RUL) for each engine cycle.

    Args:
        df (pd.DataFrame): Telemetry data containing 'engine_id' and 'cycle'.
        max_rul_clip (int): Maximum RUL threshold clipping value (standard is 125).

    Returns:
        pd.DataFrame: Dataframe with added 'RUL' and clipped 'RUL_clipped' columns.
    """
    # Max cycle per engine
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]
    df["RUL_clipped"] = df["RUL"].clip(upper=max_rul_clip)

    # Risk categories
    df["failure_risk"] = "Normal"
    df.loc[df["RUL"] <= 30, "failure_risk"] = "Warning"
    df.loc[df["RUL"] <= 15, "failure_risk"] = "Failure"

    # Binary target
    df["is_failure"] = (df["RUL"] <= 15).astype(int)

    return df


def load_cmapss_fd001(config_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Loads NASA CMAPSS FD001 Turbofan Engine Degradation Dataset.

    Falls back to generating synthetic sensor data if offline or download fails.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: (train_df, test_df, test_rul_series)
    """
    config = load_config(config_path)
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "train_FD001.txt"
    test_path = raw_dir / "test_FD001.txt"
    rul_path = raw_dir / "RUL_FD001.txt"

    train_url = config["data"]["cmapss_url"]
    test_url = config["data"]["cmapss_test_url"]
    rul_url = config["data"]["cmapss_rul_url"]

    s1 = download_file_if_not_exists(train_url, train_path)
    s2 = download_file_if_not_exists(test_url, test_path)
    s3 = download_file_if_not_exists(rul_url, rul_path)

    max_rul_clip = config["data"].get("max_rul_clip", 125)

    if s1 and s2 and s3 and train_path.exists() and test_path.exists() and rul_path.exists():
        logger.info("Loading official NASA CMAPSS FD001 dataset from raw directory...")
        train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
        test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
        test_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL_truth"])["RUL_truth"]

        train_df = calculate_rul(train_df, max_rul_clip=max_rul_clip)
        return train_df, test_df, test_rul

    logger.info("Using synthetic Industrial IoT Telemetry Data Generator as fallback...")
    full_synth = generate_iot_sensor_data(
        num_engines=config["data"].get("synthetic_num_engines", 100),
        min_cycles=120,
        max_cycles=250,
        seed=config["data"].get("random_seed", 42),
    )

    # Split 80/20 train/test engines
    engine_ids = full_synth["engine_id"].unique()
    split_idx = int(len(engine_ids) * 0.8)
    train_engines = engine_ids[:split_idx]
    test_engines = engine_ids[split_idx:]

    train_df = full_synth[full_synth["engine_id"].isin(train_engines)].copy()
    test_df = full_synth[full_synth["engine_id"].isin(test_engines)].copy()

    train_df = calculate_rul(train_df, max_rul_clip=max_rul_clip)
    test_df = calculate_rul(test_df, max_rul_clip=max_rul_clip)

    test_rul_truth = test_df.groupby("engine_id")["RUL"].last()

    return train_df, test_df, test_rul_truth
