"""
Improved HUD: animated score, timer, power-up indicators, level info, environmental facts.
"""
import math
import time
import pygame
from typing import Optional

from shared.config import GameConfig, COLORS
from game.core.game_state import GameState
from game.core.match_manager import MatchConfig
from game.components.TimerDisplay import TimerDisplay, TimerMode


# Climate change facts shown between goals
CLIMATE_FACTS = [
    "Los glaciares han perdido 9.625 billones de toneladas de hielo desde 1961.",
    "La temperatura global ha subido 1.1C desde la era preindustrial.",
    "El nivel del mar sube 3.6mm cada ano.",
    "El Artico podria estar sin hielo en verano para 2050.",
    "1 millon de especies estan en peligro de extincion.",
    "Los oceanos absorben el 30% del CO2 que producimos.",
    "El 75% de los arrecifes de coral estan amenazados.",
    "La deforestacion causa el 10% de las emisiones globales.",
    "Los eventos climaticos extremos se han quintuplicado en 50 anos.",
    "Solo queda el 3% del agua dulce en el planeta.",
]


class HUD:
    """Enhanced heads-up display for the game."""

    def __init__(self, config: GameConfig):
        self.config = config
        self._fonts = {}
        # Score animation
        self._score_flash_time = 0.0
        self._score_flash_side = None  # "player" or "ai"
        self._last_score = (0, 0)
        # Climate fact
        self._current_fact_idx = 0
        self._fact_display_time = 0.0
        self._show_fact = False
        # Timer display component (created without mode — configured on first draw)
        self._timer_display: Optional[TimerDisplay] = None
        self._timer_configured = False

    def _get_font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def draw(self, screen: pygame.Surface, state: GameState,
             match_config: MatchConfig, elapsed: float,
             level_config: dict, powerup_manager=None):
        """Draw the complete HUD."""
        W, H = self.config.width, self.config.height
        now = time.time()

        # Detect score change for animation
        current_score = (state.player_score, state.ai_score)
        if current_score != self._last_score:
            if state.player_score > self._last_score[0]:
                self._score_flash_side = "player"
            elif state.ai_score > self._last_score[1]:
                self._score_flash_side = "ai"
            self._score_flash_time = now
            self._last_score = current_score
            # Show climate fact
            self._current_fact_idx = (self._current_fact_idx + 1) % len(CLIMATE_FACTS)
            self._fact_display_time = now
            self._show_fact = True

        # --- Score + Timer (single centred bar) ---
        self._draw_score_and_timer(screen, state, match_config, elapsed, W, H, now)

        # --- Level name ---
        self._draw_level_name(screen, level_config, W)

        # --- Power-up indicators ---
        if powerup_manager:
            self._draw_powerup_indicators(screen, powerup_manager, W, H)

        # --- Climate fact ---
        if self._show_fact and (now - self._fact_display_time) < 4.0:
            self._draw_climate_fact(screen, W, H, now)
        else:
            self._show_fact = False

    def _draw_score_and_timer(self, screen, state, match_config, elapsed, W, H, now):
        """Draw score and timer together in a single centred HUD bar."""
        flash_duration = 0.5
        flash_active = (now - self._score_flash_time) < flash_duration

        # --- Build score surfaces ---
        p_size = 42 if flash_active and self._score_flash_side == "player" else 36
        p_color = (100, 255, 100) if flash_active and self._score_flash_side == "player" else COLORS.WHITE
        p_txt = self._get_font(p_size).render(str(state.player_score), True, p_color)

        sep_txt = self._get_font(36).render(" - ", True, COLORS.WHITE)

        a_size = 42 if flash_active and self._score_flash_side == "ai" else 36
        a_color = (255, 100, 100) if flash_active and self._score_flash_side == "ai" else COLORS.WHITE
        a_txt = self._get_font(a_size).render(str(state.ai_score), True, a_color)

        score_w = p_txt.get_width() + sep_txt.get_width() + a_txt.get_width()

        # --- Build timer text ---
        self._ensure_timer_display(match_config)
        timer_text, timer_color = self._timer_display._resolve_display(elapsed)
        timer_str = self._timer_display._format_time(timer_text)
        timer_font = self._get_font(24)
        timer_txt = timer_font.render(timer_str, True, timer_color)

        # --- Layout:  [ score  |  timer ] centred ---
        divider_gap = 16   # space around the divider bar
        divider_w = 1
        total_w = score_w + divider_gap + divider_w + divider_gap + timer_txt.get_width()

        row_h = max(p_txt.get_height(), a_txt.get_height(), timer_txt.get_height())
        y = 10
        x_start = W // 2 - total_w // 2

        # Background pill
        pad_x, pad_y = 14, 5
        bar_rect = pygame.Rect(
            x_start - pad_x, y - pad_y,
            total_w + pad_x * 2, row_h + pad_y * 2,
        )
        bar_surf = pygame.Surface((bar_rect.width, bar_rect.height), pygame.SRCALPHA)
        bar_surf.fill((0, 0, 0, 120))
        screen.blit(bar_surf, bar_rect.topleft)
        pygame.draw.rect(screen, (255, 255, 255, 60), bar_rect, 1, border_radius=8)

        # Score glyphs
        cx = x_start
        screen.blit(p_txt, (cx, y + (row_h - p_txt.get_height()) // 2))
        cx += p_txt.get_width()
        screen.blit(sep_txt, (cx, y + (row_h - sep_txt.get_height()) // 2))
        cx += sep_txt.get_width()
        screen.blit(a_txt, (cx, y + (row_h - a_txt.get_height()) // 2))
        cx += a_txt.get_width()

        # Vertical divider
        cx += divider_gap
        div_top = y + 3
        div_bot = y + row_h - 3
        pygame.draw.line(screen, (255, 255, 255, 90), (cx, div_top), (cx, div_bot), divider_w)
        cx += divider_w + divider_gap

        # Timer text
        screen.blit(timer_txt, (cx, y + (row_h - timer_txt.get_height()) // 2))

    def _ensure_timer_display(self, match_config):
        """Lazy-init / reconfigure the TimerDisplay component."""
        if self._timer_display is None:
            has_limit = (
                match_config.time_limit_seconds is not None
                and match_config.time_limit_seconds > 0
            )
            mode = TimerMode.COUNTDOWN if has_limit else TimerMode.COUNTUP
            self._timer_display = TimerDisplay(
                font_size=26,
                mode=mode,
                time_limit=match_config.time_limit_seconds if has_limit else None,
                warning_threshold=30.0,
                y_offset=50,
            )
            self._timer_configured = True
        elif not self._timer_configured:
            has_limit = (
                match_config.time_limit_seconds is not None
                and match_config.time_limit_seconds > 0
            )
            mode = TimerMode.COUNTDOWN if has_limit else TimerMode.COUNTUP
            self._timer_display.configure(
                mode=mode,
                time_limit=match_config.time_limit_seconds if has_limit else None,
            )
            self._timer_configured = True

    def _draw_level_name(self, screen, level_config, W):
        """Draw level name in the top-left corner."""
        name = level_config.get("name", "")
        if name:
            font = self._get_font(20)
            txt = font.render(name, True, (180, 180, 180))
            screen.blit(txt, (10, 8))

    def _draw_powerup_indicators(self, screen, pm, W, H):
        """Draw active power-up effect indicators for both players."""
        from game.entities.powerups import POWERUP_CONFIG

        font = self._get_font(16)
        y_offset = H - 30

        # Player 1 effects (left side)
        p1_effects = pm.get_active_effects_for_player(0)
        for i, effect in enumerate(p1_effects):
            cfg = POWERUP_CONFIG[effect.type]
            # Bar background
            bar_w = 80
            bar_h = 16
            x = 10
            y = y_offset - i * 22

            bg = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 120))
            screen.blit(bg, (x, y))

            # Fill bar
            fill_pct = effect.remaining / cfg["duration"]
            fill_w = int(bar_w * fill_pct)
            pygame.draw.rect(screen, cfg["color"], (x, y, fill_w, bar_h))
            pygame.draw.rect(screen, COLORS.WHITE, (x, y, bar_w, bar_h), 1)

            # Label
            label = font.render(cfg["icon_char"], True, COLORS.WHITE)
            screen.blit(label, (x + 2, y))

        # Player 2 effects (right side)
        p2_effects = pm.get_active_effects_for_player(1)
        for i, effect in enumerate(p2_effects):
            cfg = POWERUP_CONFIG[effect.type]
            bar_w = 80
            bar_h = 16
            x = W - bar_w - 10
            y = y_offset - i * 22

            bg = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 120))
            screen.blit(bg, (x, y))

            fill_pct = effect.remaining / cfg["duration"]
            fill_w = int(bar_w * fill_pct)
            pygame.draw.rect(screen, cfg["color"], (x, y, fill_w, bar_h))
            pygame.draw.rect(screen, COLORS.WHITE, (x, y, bar_w, bar_h), 1)

            label = font.render(cfg["icon_char"], True, COLORS.WHITE)
            screen.blit(label, (x + 2, y))

    def _draw_climate_fact(self, screen, W, H, now):
        """Draw a climate fact at the bottom of the screen."""
        fact = CLIMATE_FACTS[self._current_fact_idx]
        age = now - self._fact_display_time

        # Fade in/out
        if age < 0.5:
            alpha = int(255 * (age / 0.5))
        elif age > 3.5:
            alpha = int(255 * (1 - (age - 3.5) / 0.5))
        else:
            alpha = 255

        font = self._get_font(18)
        txt = font.render(fact, True, (200, 220, 255))
        txt.set_alpha(max(0, min(255, alpha)))

        # Background bar
        bar = pygame.Surface((txt.get_width() + 20, txt.get_height() + 10), pygame.SRCALPHA)
        bar.fill((0, 0, 0, min(120, alpha // 2)))
        x = W // 2 - bar.get_width() // 2
        y = H - 50
        screen.blit(bar, (x, y))
        screen.blit(txt, (x + 10, y + 5))
