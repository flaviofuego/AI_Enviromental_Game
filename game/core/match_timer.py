"""
Match timer: tracks elapsed time with pause/resume/freeze support.

Single Responsibility: only tracks time, no rendering.
"""
import time


class MatchTimer:
    """Precise match timer with pause and freeze semantics.

    Usage::

        timer = MatchTimer()
        timer.start()          # begin counting
        timer.pause()          # pauses accumulation
        timer.resume()         # resumes from where it left off
        timer.freeze()         # locks elapsed at current value (game over)
        elapsed = timer.elapsed  # seconds since start minus paused time
    """

    __slots__ = (
        "_start_time",
        "_pause_start",
        "_total_paused",
        "_frozen_elapsed",
        "_running",
    )

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._pause_start: float = 0.0
        self._total_paused: float = 0.0
        self._frozen_elapsed: float | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start (or restart) the timer from zero."""
        self._start_time = time.time()
        self._pause_start = 0.0
        self._total_paused = 0.0
        self._frozen_elapsed = None
        self._running = True

    def pause(self) -> None:
        """Pause the timer.  Idempotent — calling twice has no extra effect."""
        if self._running and self._pause_start == 0.0:
            self._pause_start = time.time()

    def resume(self) -> None:
        """Resume after a pause.  Idempotent."""
        if self._pause_start > 0.0:
            self._total_paused += time.time() - self._pause_start
            self._pause_start = 0.0

    def freeze(self) -> None:
        """Lock elapsed at its current value (e.g. when game over)."""
        if self._frozen_elapsed is None:
            self._frozen_elapsed = self._compute_elapsed()

    def reset(self) -> None:
        """Fully reset to initial state — equivalent to creating a new timer."""
        self.__init__()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since *start*, excluding paused intervals.

        After *freeze()*, this returns the frozen value forever.
        """
        if self._frozen_elapsed is not None:
            return self._frozen_elapsed
        return self._compute_elapsed()

    @property
    def is_running(self) -> bool:
        """True when actively counting (not paused, not frozen)."""
        return self._running and self._pause_start == 0.0 and self._frozen_elapsed is None

    @property
    def is_paused(self) -> bool:
        return self._pause_start > 0.0

    @property
    def is_frozen(self) -> bool:
        return self._frozen_elapsed is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_elapsed(self) -> float:
        if self._start_time == 0.0:
            return 0.0
        now = time.time() if self._pause_start == 0.0 else self._pause_start
        return max(0.0, now - self._start_time - self._total_paused)
