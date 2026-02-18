"""Human-controlled mallet (mouse input)."""
from shared.config import COLORS, GameConfig
from shared.entities.mallet import Mallet


class HumanMallet(Mallet):
    """Player-controlled mallet that follows mouse position, restricted to the left half."""

    def __init__(self, config: GameConfig = None, color=COLORS.NEON_RED, custom_image=None):
        cfg = config or GameConfig()
        super().__init__(cfg.width // 4, cfg.height // 2, color, cfg, custom_image)

    def update(self, mouse_pos=None):
        if mouse_pos is None:
            return
        # Paralysis: ignore mouse movement, keep current position
        if self.paralyzed:
            self.velocity = [0.0, 0.0]
            return
        cfg = self.config
        self.prev_position = self.position.copy()
        target_x = min(max(mouse_pos[0], self.radius), cfg.half_width - self.radius)
        target_y = min(max(mouse_pos[1], self.radius), cfg.height - self.radius)
        self.position = [float(target_x), float(target_y)]
        self.rect.center = (int(self.position[0]), int(self.position[1]))
        self.velocity = [
            self.position[0] - self.prev_position[0],
            self.position[1] - self.prev_position[1],
        ]
