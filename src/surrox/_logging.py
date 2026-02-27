from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def log_duration(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **context: Any,
) -> Generator[None]:
    logger.log(level, "%s started", event, extra=context)
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        duration_s = round(time.perf_counter() - start, 3)
        logger.log(
            logging.ERROR, "%s failed", event,
            extra={**context, "duration_s": duration_s},
            exc_info=True,
        )
        raise
    duration_s = round(time.perf_counter() - start, 3)
    logger.log(
        level, "%s complete", event,
        extra={**context, "duration_s": duration_s},
    )
