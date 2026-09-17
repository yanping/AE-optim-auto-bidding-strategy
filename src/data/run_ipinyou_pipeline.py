"""
End-to-End Pipeline for iPinYou Real Campaign 1458 (Wrapper for prepare_data).
Extracts Season 2 logs, processes campaign 1458, performs chronological splitting,
masks advertiser feedback, trains CTR and Kaplan-Meier models, and verifies calibration.
"""

from pathlib import Path
from typing import Dict, Optional
from .prepare_data import run_preparation_pipeline


def run_pipeline(
    archive_path: str = "data/raw/ipinyou.contest.dataset.7z",
    campaign_id: str = "1458",
    plot_output: str = "data/calibration_curve_ipinyou_1458.png",
) -> Dict:
    """Wrapper function preserving legacy run_pipeline interface."""
    return run_preparation_pipeline(
        archive_path=archive_path,
        campaign_id=campaign_id,
        plot_output=plot_output,
    )


if __name__ == "__main__":
    run_pipeline()
