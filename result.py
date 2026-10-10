"""The five fields of ``eval_framework.result_processors.base.Result`` that the
aggregation actually reads, as a stand-in so the extracted code runs on its own.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    metric_name: str
    subject: str
    key: str | None
    value: float | None
    error: Any | None = None
