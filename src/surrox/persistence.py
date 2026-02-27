from __future__ import annotations

from pathlib import Path

from surrox.result import SurroxResult


def save_result(result: SurroxResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))


def load_result(path: Path) -> SurroxResult:
    return SurroxResult.model_validate_json(path.read_text())
