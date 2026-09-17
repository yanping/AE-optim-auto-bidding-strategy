"""
Unit tests for data download and preparation pipeline.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data.download_data import generate_sample_dataset, check_existing_data
from src.data.prepare_data import run_preparation_pipeline


def test_generate_sample_dataset(tmp_path):
    campaign_id = "test_9999"
    out_file = generate_sample_dataset(
        campaign_id=campaign_id,
        num_records=100,
        output_dir=str(tmp_path / campaign_id),
    )
    assert out_file.exists()
    
    df = pd.read_parquet(out_file)
    assert len(df) == 100
    assert "market_price" in df.columns
    assert "click" in df.columns
    assert "day" in df.columns


def test_prepare_data_pipeline_with_sample(tmp_path):
    # Test prepare_data with sample mode on synthetic data
    campaign_id = "test_sample_campaign"
    sample_file = generate_sample_dataset(
        campaign_id=campaign_id,
        num_records=500,
        output_dir=str(tmp_path / campaign_id),
    )
    
    metrics = run_preparation_pipeline(
        archive_path=str(tmp_path / "nonexistent.7z"),
        raw_dir=str(tmp_path / "season2"),
        campaign_id=campaign_id,
        output_dir=str(tmp_path / campaign_id),
        plot_output=str(tmp_path / "calib.png"),
        skip_model_fit=True,
    )
    
    assert metrics["campaign_id"] == campaign_id
    assert metrics["total_records"] == 500
    assert (tmp_path / campaign_id / "train_advertiser.parquet").exists()
    assert (tmp_path / campaign_id / "val_advertiser.parquet").exists()
    assert (tmp_path / campaign_id / "test_oracle.parquet").exists()

