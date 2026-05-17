from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def configure_logging(
    output_dir: Union[str, Path],
    *,
    name: str = "legal_parser",
    log_filename: str = "parse.log",
    verbose: bool = True,
) -> logging.Logger:
    """Configure a module logger with console + file handlers."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(out / log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    if hasattr(console_handler.stream, "reconfigure"):
        console_handler.stream.reconfigure(encoding="utf-8", errors="replace")
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.debug("Logging configured: %s", out / log_filename)
    return logger


def get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    return logger or logging.getLogger("legal_parser")
