from __future__ import annotations

import logging
from pathlib import Path

from surrox._logging import log_duration
from surrox.result import SurroxResult

_logger = logging.getLogger(__name__)


def save_result(result: SurroxResult, path: Path) -> None:
    with log_duration(_logger, "save_result", path=str(path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2))


def load_result(path: Path) -> SurroxResult:
    with log_duration(_logger, "load_result", path=str(path)):
        return SurroxResult.model_validate_json(path.read_text())
