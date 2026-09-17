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
Unit tests for Task Manager and Artifact Isolation Architecture.
"""

from datetime import datetime
from pathlib import Path
import pytest

from src.utils.task_manager import (
    generate_task_id,
    get_latest_task_id,
    resolve_task_artifacts_dir,
    update_latest_pointer,
)


def test_generate_task_id_format():
    dt = datetime(2026, 9, 16, 21, 30, 45)
    task_id = generate_task_id(dt)
    assert task_id == "task_20260916_213045"
    assert task_id.startswith("task_")
    assert len(task_id) == len("task_YYYYMMDD_HHMMSS")


def test_task_dir_resolution_and_isolation(tmp_path):
    artifacts_base = tmp_path / "artifacts"

    # 1. Resolve with explicit task ID and creation
    task_id_1 = "task_20260916_100000"
    resolved_id, task_dir_1 = resolve_task_artifacts_dir(
        task_id=task_id_1,
        artifacts_base=artifacts_base,
        create=True,
    )
    assert resolved_id == task_id_1
    assert task_dir_1 == artifacts_base / task_id_1
    assert task_dir_1.is_dir()

    # 2. Update pointer to task 1
    update_latest_pointer(task_id_1, artifacts_base=artifacts_base)
    assert (artifacts_base / "latest_task.txt").read_text(encoding="utf-8").strip() == task_id_1
    assert (artifacts_base / "latest").is_symlink()
    assert (artifacts_base / "latest").resolve() == task_dir_1.resolve()
    assert get_latest_task_id(artifacts_base) == task_id_1

    # 3. Create a second task
    task_id_2 = "task_20260916_120000"
    _, task_dir_2 = resolve_task_artifacts_dir(
        task_id=task_id_2,
        artifacts_base=artifacts_base,
        create=True,
    )
    update_latest_pointer(task_id_2, artifacts_base=artifacts_base)
    assert get_latest_task_id(artifacts_base) == task_id_2
    assert (artifacts_base / "latest").resolve() == task_dir_2.resolve()

    # 4. Resolve without task_id in read mode (create=False) should point to latest
    resolved_latest_id, resolved_latest_dir = resolve_task_artifacts_dir(
        task_id=None,
        artifacts_base=artifacts_base,
        create=False,
    )
    assert resolved_latest_id == task_id_2
    assert resolved_latest_dir.resolve() == task_dir_2.resolve()
