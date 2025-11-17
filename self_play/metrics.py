"""
Metrics tracking for self-play system.
"""
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class Metrics:
    """Track system metrics."""
    games_generated: int = 0
    games_saved: int = 0
    db_errors: int = 0
    game_errors: int = 0
    start_time: float = field(default_factory=time.time)

    def log_summary(self):
        """Log metrics summary."""
        elapsed = time.time() - self.start_time
        logger.info(
            f"Metrics Summary - Games Generated: {self.games_generated}, "
            f"Games Saved: {self.games_saved}, "
            f"DB Errors: {self.db_errors}, "
            f"Game Errors: {self.game_errors}, "
            f"Elapsed: {elapsed:.1f}s, "
            f"Rate: {self.games_generated / elapsed if elapsed > 0 else 0:.2f} games/s"
        )
