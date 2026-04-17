import io
import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(name)s: %(message)s")

    # Build a UTF-8 stream for the console handler.
    # On Windows, sys.stderr defaults to cp1252 which can't encode emoji or
    # box-drawing characters used in log messages.  We create a fresh
    # TextIOWrapper on the raw buffer so encoding errors are replaced with '?'
    # rather than crashing the process.
    try:
        _stream = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        # sys.stderr has no .buffer (e.g. IDLE / certain test runners) —
        # fall back to the stream as-is; emoji may break but the process won't.
        _stream = sys.stderr

    sh = logging.StreamHandler(stream=_stream)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / "goldregimex.log", mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log_regime_transition(logger, timestamp, old_state, new_state, state_names):
    logger.info(
        "REGIME CHANGE at %s: %s -> %s",
        timestamp,
        state_names.get(old_state, str(old_state)),
        state_names.get(new_state, str(new_state)),
    )


def log_trade_signal(logger, timestamp, direction, probability, hmm_state):
    logger.info(
        "SIGNAL at %s: %s (prob=%.3f, hmm_state=%d)",
        timestamp, direction, probability, hmm_state,
    )
