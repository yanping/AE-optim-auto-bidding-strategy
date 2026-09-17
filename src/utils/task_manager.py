# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Task Management & Artifact Isolation Architecture.

Provides functionality for:
1. Automatic unique task ID generation based on execution timestamp: `task_YYYYMMDD_HHMMSS`.
2. Isolated directory resolution for task artifacts under `artifacts/<task_id>/`.
3. Maintenance of `artifacts/latest` symlink and `artifacts/latest_task.txt` pointer file.
"""

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("alpha_evolve.task_manager")


def generate_task_id(dt: Optional[datetime] = None) -> str:
    """
    Generates a unique task identifier based on timestamp.
    
    Format: task_YYYYMMDD_HHMMSS (e.g. task_20260916_192239)
    """
    if dt is None:
        dt = datetime.now()
    return f"task_{dt.strftime('%Y%m%d_%H%M%S')}"


def get_latest_task_id(artifacts_base: Path = Path("./artifacts")) -> Optional[str]:
    """
    Retrieves the most recent task ID from artifacts/latest_task.txt or artifacts/latest.
    """
    pointer_file = artifacts_base / "latest_task.txt"
    if pointer_file.exists():
        try:
            with open(pointer_file, "r", encoding="utf-8") as f:
                task_id = f.read().strip()
                if task_id:
                    return task_id
        except Exception as e:
            logger.warning("Failed to read %s: %s", pointer_file, e)

    latest_symlink = artifacts_base / "latest"
    if latest_symlink.is_symlink():
        try:
            target = os.readlink(latest_symlink)
            return Path(target).name
        except Exception as e:
            logger.warning("Failed to resolve symlink %s: %s", latest_symlink, e)

    return None


def resolve_task_artifacts_dir(
    task_id: Optional[str] = None,
    artifacts_base: Path = Path("./artifacts"),
    create: bool = True,
) -> Tuple[str, Path]:
    """
    Resolves the isolated artifacts directory for a given task ID or latest pointer.

    Args:
        task_id: Explicit task identifier. If None:
                 - When create=True: generates a new timestamped task ID.
                 - When create=False: resolves the latest existing task ID or 'latest'.
        artifacts_base: Base artifacts root directory (default: ./artifacts).
        create: Whether to create the resolved task directory if it doesn't exist.

    Returns:
        Tuple of (effective_task_id, task_artifacts_dir_path).
    """
    artifacts_base = Path(artifacts_base)

    if task_id:
        effective_task_id = task_id.strip()
    else:
        if create:
            effective_task_id = generate_task_id()
        else:
            latest_id = get_latest_task_id(artifacts_base)
            if latest_id and (artifacts_base / latest_id).is_dir():
                effective_task_id = latest_id
            elif (artifacts_base / "latest").exists():
                effective_task_id = "latest"
            else:
                effective_task_id = generate_task_id()

    task_dir = artifacts_base / effective_task_id
    if create:
        task_dir.mkdir(parents=True, exist_ok=True)

    return effective_task_id, task_dir


def update_latest_pointer(task_id: str, artifacts_base: Path = Path("./artifacts")) -> None:
    """
    Updates `artifacts/latest_task.txt` and `artifacts/latest` symlink to point to the specified task.

    Args:
        task_id: The completed task identifier.
        artifacts_base: Root directory of artifacts.
    """
    artifacts_base = Path(artifacts_base)
    artifacts_base.mkdir(parents=True, exist_ok=True)

    # 1. Write pointer text file
    pointer_file = artifacts_base / "latest_task.txt"
    try:
        with open(pointer_file, "w", encoding="utf-8") as f:
            f.write(f"{task_id}\n")
    except Exception as e:
        logger.warning("Could not write %s: %s", pointer_file, e)

    # 2. Update relative symlink
    latest_symlink = artifacts_base / "latest"
    try:
        if latest_symlink.is_symlink() or latest_symlink.exists():
            latest_symlink.unlink()
        # Use relative target (task_id) so artifacts/ can be relocated safely
        latest_symlink.symlink_to(task_id, target_is_directory=True)
        logger.info("Updated latest symlink: %s -> %s", latest_symlink, task_id)
    except Exception as e:
        logger.warning("Could not update symlink %s: %s", latest_symlink, e)
