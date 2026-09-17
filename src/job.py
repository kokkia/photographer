from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PrintJob:
    """カメラスレッドからInstaxスレッドへ渡す不変の印刷要求。"""

    image_path: Path
    created_at: datetime
    person_ids: tuple[int, ...] = ()
