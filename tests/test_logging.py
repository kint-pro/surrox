from __future__ import annotations

import logging
import time

import pytest

from surrox._logging import log_duration


class TestLogDuration:
    def test_emits_start_and_complete(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = logging.getLogger("test.duration")
        with (
            caplog.at_level(logging.INFO, logger="test.duration"),
            log_duration(logger, "test_event"),
        ):
            pass

        messages = [r.message for r in caplog.records]
        assert "test_event started" in messages
        assert "test_event complete" in messages

    def test_complete_has_duration_extra(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        logger = logging.getLogger("test.duration")
        with (
            caplog.at_level(logging.INFO, logger="test.duration"),
            log_duration(logger, "timed_op"),
        ):
            time.sleep(0.01)

        complete_record = next(
            r for r in caplog.records if "complete" in r.message
        )
        assert hasattr(complete_record, "duration_s")
        assert complete_record.duration_s >= 0.01  # type: ignore[attr-defined]

    def test_passes_context_as_extra(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        logger = logging.getLogger("test.duration")
        with (
            caplog.at_level(logging.INFO, logger="test.duration"),
            log_duration(logger, "ctx_op", column="cost", n_trials=10),
        ):
            pass

        start_record = next(
            r for r in caplog.records if "started" in r.message
        )
        assert start_record.column == "cost"  # type: ignore[attr-defined]
        assert start_record.n_trials == 10  # type: ignore[attr-defined]

    def test_respects_log_level(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        logger = logging.getLogger("test.duration")
        with (
            caplog.at_level(logging.DEBUG, logger="test.duration"),
            log_duration(logger, "debug_op", level=logging.DEBUG),
        ):
            pass

        assert all(r.levelno == logging.DEBUG for r in caplog.records)

    def test_propagates_exception(self) -> None:
        logger = logging.getLogger("test.duration")
        with (
            pytest.raises(ValueError, match="boom"),
            log_duration(logger, "failing_op"),
        ):
            raise ValueError("boom")

    def test_logs_failure_on_exception(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        logger = logging.getLogger("test.duration")
        with (
            caplog.at_level(logging.DEBUG, logger="test.duration"),
            pytest.raises(ValueError, match="boom"),
            log_duration(logger, "failing_op"),
        ):
            raise ValueError("boom")

        failed_record = next(
            r for r in caplog.records if "failed" in r.message
        )
        assert failed_record.levelno == logging.ERROR
        assert hasattr(failed_record, "duration_s")


class TestLibraryLoggingHygiene:
    def test_no_handlers_on_surrox_logger(self) -> None:
        logger = logging.getLogger("surrox")
        assert len(logger.handlers) == 0

    def test_no_handlers_on_child_loggers(self) -> None:
        child_names = [
            "surrox.surrogate.manager",
            "surrox.surrogate.pipeline",
            "surrox.optimizer.runner",
            "surrox.analysis.analyzer",
            "surrox.analysis.summary",
            "surrox.persistence",
        ]
        for name in child_names:
            logger = logging.getLogger(name)
            assert len(logger.handlers) == 0, (
                f"logger '{name}' has handlers"
            )
