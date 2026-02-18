"""
Main game engine. Consolidates logic from main_improved.py and themed_game.py.
Handles the core game loop, physics updates, and entity management.
"""
import logging
import math
import random
import time
import numpy as np
import pygame

from shared.config import GameConfig, PhysicsConfig, COLORS

logger = logging.getLogger(__name__)
from shared.entities.puck import Puck
from shared.entities.table import Table
from shared.utils.drawing import draw_glow
from shared.utils.sprite_loader import SpriteLoader
from shared.physics import vector_length, normalize_vector

from game.core.game_state import GameState, GamePhase
from game.core.match_manager import MatchManager, MatchConfig, GameMode
from game.core.renderer import Renderer
from game.entities.human_mallet import HumanMallet
from game.entities.ai_mallet import AIMallet
from game.entities.keyboard_mallet import KeyboardMallet
from game.ai.model_loader import find_best_model, load_optimized_model
from game.ai.observation_builder import create_observation
from game.config.level_config import get_level_config
from game.core.mechanics import create_mechanic
from game.components.AudioManager import audio_manager as _audio_manager
from game.components.SkinSelector import DEFAULT_SKINS
from game.components.PowerUpHelpPanel import PowerUpHelpPanel
from game.entities.powerup_renderer import GoalEffectsRenderer


class GameEngine:
    """Core game engine that runs a match."""

    def __init__(
        self, 
        screen: pygame.Surface, 
        match_config: MatchConfig = MatchConfig(),
        game_config: GameConfig = None,
        save_system=None
    ):
        self.screen = screen
        self.config = game_config or GameConfig(width=screen.get_width(), height=screen.get_height())
        self.match_config = match_config
        self.save_system = save_system
        self.state = GameState()
        self.match_manager = MatchManager(self.match_config)
        self.renderer = Renderer(screen, self.config)
        self.physics = PhysicsConfig()

        # Level theme
        self.level_config = get_level_config(self.match_config.level_id)
        self.assets = {}

        # Entities
        self.table = Table(self.config)
        self.puck = None
        self.player1 = None  # HumanMallet
        self.player2 = None  # AIMallet or KeyboardMallet
        self.all_sprites = pygame.sprite.RenderUpdates()

        # AI system
        self.use_rl = False
        self.model = None
        self.model_type = "original"
        self.last_action = 4
        self.last_prediction_time = 0
        self.prediction_interval = 20
        self.frame_count = 0
        self.frame_skip = 2
        self.steps_since_ai_hit = 0

        # Power-up manager (will be set from outside if enabled)
        self.powerup_manager = None

        # Power-up help panel (H key — lazy init on first open)
        self._help_panel: PowerUpHelpPanel | None = None

        # Shield arc renderer (visualiza barrera en portería)
        self._goal_fx = GoalEffectsRenderer()

        # Level mechanic (UV zones, fog, shrinking field, heat waves)
        self.mechanic = None

        # HUD (will be set from outside)
        self.hud = None

        # Clock
        self.clock = pygame.time.Clock()
        self.fixed_dt = 1 / self.config.fps

        # Cached surfaces (avoid per-frame allocations)
        self._overlay_dim = None       # dark overlay for game_over
        self._overlay_dim_pause = None # dark overlay for pause
        self._last_click_time = 0      # debounce game-over buttons

    def setup(self):
        """Initialize all game objects and load assets."""
        self._load_assets()
        self._create_entities()
        self._setup_table()

        if self.match_config.mode == GameMode.PLAYER_VS_AI:
            self._init_ai()

        # Level mechanic
        self.mechanic = create_mechanic(self.config, self.level_config)

        # Pre-render background
        theme_bg = self.assets.get("background")
        self.renderer.pre_render_background(self.table, theme_bg)

        self.state.reset_match()
        self.match_manager.start_match()
        self.steps_since_ai_hit = 0

    def _load_assets(self):
        """Load level-specific assets using SpriteLoader for consistency."""
        level_id = self.match_config.level_id
        logger.info("Loading assets for level %d (mode=%s)", level_id, self.match_config.mode.name)
        self.assets = SpriteLoader.load_level_sprites(level_id, self.config)

    def _get_player_skin_color(self):
        """Get the player's chosen skin color from their profile."""
        skin_id = "default"
        if self.save_system and self.save_system.current_profile:
            skin_id = self.save_system.current_profile.get("skin", "default")
        for skin in DEFAULT_SKINS:
            if skin["id"] == skin_id:
                return skin["color"]
        return DEFAULT_SKINS[0]["color"]

    def _create_entities(self):
        """Create game entities based on mode."""
        puck_img = self.assets.get("puck")
        if puck_img:
            ps = self.config.puck_radius * 2
            puck_img = pygame.transform.smoothscale(puck_img, (ps, ps))
        self.puck = Puck(self.config, self.physics, puck_img)

        # Player mallet: use skin color from profile
        player_color = self._get_player_skin_color()
        p1_img = self.assets.get("mallet_player")
        if p1_img:
            ms = self.config.mallet_radius * 2
            p1_img = pygame.transform.smoothscale(p1_img, (ms, ms))
            self.player1 = HumanMallet(self.config, color=player_color, custom_image=p1_img)
        else:
            # No player-specific sprite: create colored mallet from skin
            self.player1 = HumanMallet(self.config, color=player_color)

        if self.match_config.mode == GameMode.PLAYER_VS_PLAYER:
            p2_img = self.assets.get("mallet_ai")
            if p2_img:
                ms = self.config.mallet_radius * 2
                p2_img = pygame.transform.smoothscale(p2_img, (ms, ms))
            self.player2 = KeyboardMallet(self.config, custom_image=p2_img)
        else:
            ai_cfg = self.level_config
            p2_img = self.assets.get("mallet_ai")
            if p2_img:
                ms = self.config.mallet_radius * 2
                p2_img = pygame.transform.smoothscale(p2_img, (ms, ms))
            self.player2 = AIMallet(
                self.config, custom_image=p2_img,
                reaction_speed=ai_cfg.get("ai_reaction_speed", 0.1),
                prediction_factor=ai_cfg.get("ai_prediction_factor", 0.4),
            )

        self.all_sprites = pygame.sprite.RenderUpdates()
        self.all_sprites.add(self.player1, self.player2, self.puck)

    def _setup_table(self):
        """Configure table theme and goal sprites."""
        theme = self.level_config.get("theme", {})
        self.table.table_color = theme.get("table_color", COLORS.BLACK)
        gl = self.assets.get("goal_left")
        gr = self.assets.get("goal_right")
        if gl and gr:
            self.table.set_goal_sprites(gl, gr)
            logger.info(
                "Table goals configured for level %d — left=%s right=%s",
                self.match_config.level_id,
                gl.get_size() if gl else None,
                gr.get_size() if gr else None,
            )
        else:
            logger.warning(
                "No goal sprites found for level %d — using fallback rendering",
                self.match_config.level_id,
            )

    def _init_ai(self):
        """Load RL model for AI."""
        try:
            model_path, _ = find_best_model()
            if model_path:
                self.model, self.model_type = load_optimized_model(model_path)
                self.use_rl = True
                # Pre-warm with correct observation dimension
                obs_dims = {
                    "enhanced": 21, "improved": 21,
                    "v2": 13, "v2_powerups": 26,
                    "original": 13,
                }
                obs_dim = obs_dims.get(self.model_type, 13)
                self.model.predict(np.zeros(obs_dim, dtype=np.float32), deterministic=True)
            else:
                self.use_rl = False
        except Exception as e:
            print(f"Error loading RL model: {e}")
            self.use_rl = False

    def run(self) -> str:
        """
        Run the game loop. Returns:
        - 'exit' to quit
        - 'back_to_menu' to return to menu
        - 'retry' to play again
        - 'next_level' to advance
        """
        self.setup()
        running = True

        while running:
            mouse_pos = pygame.mouse.get_pos()

            # --- Event handling ---
            for event in pygame.event.get():
                _audio_manager.process_event(event)

                if event.type == pygame.QUIT:
                    return "exit"

                # Help panel has priority: consumes its own events first
                if self._help_panel and self._help_panel.is_open:
                    if self._help_panel.handle_event(event):
                        continue

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state.phase == GamePhase.PLAYING:
                            self.state.phase = GamePhase.PAUSED
                            self.state.timer.pause()
                        elif self.state.phase == GamePhase.PAUSED:
                            self.state.phase = GamePhase.PLAYING
                            self.state.timer.resume()
                        elif self.state.phase == GamePhase.GAME_OVER:
                            return "back_to_menu"
                    elif event.key == pygame.K_F1:
                        self.state.debug_mode = not self.state.debug_mode
                    elif event.key == pygame.K_f:
                        self.state.show_fps = not self.state.show_fps
                    elif event.key == pygame.K_r and self.state.phase in (GamePhase.GAME_OVER, GamePhase.PAUSED):
                        if self.state.phase == GamePhase.PAUSED:
                            self._restart_match()
                        else:
                            return "retry"
                    elif event.key == pygame.K_m and self.state.phase == GamePhase.PAUSED:
                        return "back_to_menu"
                    elif event.key == pygame.K_h:
                        # Toggle powerup help panel (available in any phase)
                        self._toggle_help_panel()

            # --- Update ---
            if self.state.phase == GamePhase.PLAYING:
                self._update_playing(mouse_pos)
            elif self.state.phase == GamePhase.PAUSED:
                result = self._handle_pause(mouse_pos)
                if result:
                    return result
            elif self.state.phase == GamePhase.GAME_OVER:
                result = self._handle_game_over(mouse_pos)
                if result:
                    return result

            # Help panel update (animation)
            if self._help_panel:
                self._help_panel.update(self.fixed_dt)

            # --- Draw ---
            self._draw(mouse_pos)

            pygame.display.flip()
            self.clock.tick(self.config.fps)

        return "back_to_menu"

    def _update_playing(self, mouse_pos):
        """Update all game entities during active play."""
        # Player 1 (mouse)
        self.player1.update(mouse_pos)

        # Player 2 (AI or keyboard)
        if self.match_config.mode == GameMode.PLAYER_VS_PLAYER:
            keys = pygame.key.get_pressed()
            self.player2.update(keys)
        else:
            self._update_ai()

        # Physics
        self.puck.update()

        # Collisions
        if self.puck.check_mallet_collision(self.player1):
            self.state.total_hits_player += 1
            self._ensure_min_puck_speed(self.player1)
        if self.puck.check_mallet_collision(self.player2):
            self.state.total_hits_ai += 1
            self.steps_since_ai_hit = 0
            self._ensure_min_puck_speed(self.player2)
        else:
            self.steps_since_ai_hit += 1

        # Goal check (must run BEFORE goal-frame collision to avoid
        # bouncing the puck away from a valid goal)
        goal = self.table.is_goal(self.puck)
        if goal:
            # Shield interception: cancel goal if the defending player has a shield
            shield_blocked = False
            if self.powerup_manager and self.match_config.powerups_enabled:
                stacks = self.powerup_manager.stacks
                # goal=="player" → player scored on AI's goal → AI (idx 1) is defending
                # goal=="ai"     → AI scored on player's goal → player1 (idx 0) is defending
                defender_idx = 1 if goal == "player" else 0
                if stacks[defender_idx].has_shield():
                    # Bounce puck back toward center
                    self.puck.velocity[0] *= -1
                    # Small nudge away from goal mouth
                    nudge = -1 if goal == "player" else 1
                    self.puck.position[0] += nudge * (self.player1.radius + 4)
                    shield_blocked = True

            if not shield_blocked:
                # Serve puck toward the team that was scored against
                if goal == "player":
                    self.puck.reset("ai")
                elif goal == "ai":
                    self.puck.reset("player")
                self.state.record_goal(goal, self.match_config.score_limit)
                # Notify level mechanic
                if self.mechanic:
                    self.mechanic.on_goal(goal)
        else:
            self.table.check_goal_collision(self.puck)

        # Obstacle collisions (Phase 6 — Iceberg Flotante)
        self._handle_obstacle_collisions()

        # Time limit check
        if self.match_config.time_limit_seconds:
            elapsed = self.state.timer.elapsed
            self.state.check_time_limit(elapsed, self.match_config.time_limit_seconds,
                                        self.match_config.overtime_on_tie)

        # Power-ups
        if self.powerup_manager and self.match_config.powerups_enabled:
            dt = self.clock.get_time() / 1000.0
            self.powerup_manager.update(dt, [self.player1, self.player2], self.puck, self.state)
            self._goal_fx.update(dt)

        # Level mechanic update
        if self.mechanic:
            dt = self.clock.get_time() / 1000.0
            self.mechanic.update(dt, self.puck, [self.player1, self.player2], self.table)

        self.frame_count = (self.frame_count + 1) % max(1, self.frame_skip)

    def _update_ai(self):
        """Update AI mallet using RL model or simple heuristic."""
        if self.use_rl and self.model is not None and self.frame_count == 0:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_prediction_time > self.prediction_interval:
                obs = create_observation(
                    self.player2, self.puck, self.player1,
                    self.state.player_score, self.state.ai_score,
                    self.model_type,
                    steps_since_last_hit=self.steps_since_ai_hit,
                    powerup_manager=self.powerup_manager,
                )

                # Only re-predict if observation changed significantly
                if self._obs_changed(obs):
                    try:
                        import torch
                        with torch.no_grad():
                            action, _ = self.model.predict(obs, deterministic=True)
                        if isinstance(action, np.ndarray):
                            action = int(action.item()) if action.ndim == 0 else int(action[0])
                        else:
                            action = int(action)

                        self.last_action = action
                        self._last_obs = obs
                    except Exception:
                        pass
                self.last_prediction_time = current_time

            self.player2.apply_rl_action(self.last_action, self.physics.ai_move_amount)
        elif not self.use_rl:
            self.player2.update_simple_ai(self.puck.position)

    def _obs_changed(self, obs) -> bool:
        """Check if observation changed significantly (avoid redundant predictions)."""
        if not hasattr(self, '_last_obs') or self._last_obs is None:
            return True
        diff = np.abs(obs - self._last_obs).sum()
        return diff > 0.01  # Threshold for meaningful state change

    def _ensure_min_puck_speed(self, mallet):
        """Ensure puck has minimum speed after collision."""
        if vector_length(self.puck.velocity) < 2:
            d = normalize_vector([
                self.puck.position[0] - mallet.position[0],
                self.puck.position[1] - mallet.position[1],
            ])
            self.puck.velocity[0] += d[0]
            self.puck.velocity[1] += d[1]

    # ------------------------------------------------------------------
    # Phase-6: Obstacle (Iceberg) physics
    # ------------------------------------------------------------------

    def _handle_obstacle_collisions(self):
        """
        Resolve circle-circle elastic collisions between the puck and
        each obstacle spawned by the Iceberg powerup.

        Anti-stuck: obstacles that haven't moved the puck in >1 s apply
        a soft random push.
        """
        obstacles = getattr(self.state, "obstacles", None)
        if not obstacles:
            return

        dt = self.clock.get_time() / 1000.0 or self.fixed_dt
        to_remove = []

        for obs in obstacles:
            # --- Fade-out expired obstacles ---
            if obs.get("fading"):
                obs["alpha"] = max(0, obs["alpha"] - int(255 * dt / 0.5))
                if obs["alpha"] <= 0:
                    to_remove.append(obs)
                continue

            ox, oy = obs["x"], obs["y"]
            or_ = obs["radius"]
            px, py = self.puck.position
            pr = self.puck.radius

            dx = px - ox
            dy = py - oy
            dist = math.hypot(dx, dy)
            min_dist = or_ + pr

            if dist < min_dist and dist > 0:
                # Normalize collision normal
                nx, ny = dx / dist, dy / dist
                # Separate puck (no overlap)
                overlap = min_dist - dist
                self.puck.position[0] += nx * (overlap + 1)
                self.puck.position[1] += ny * (overlap + 1)
                # Reflect velocity along collision normal (elastic, restitution=1)
                dot = self.puck.velocity[0] * nx + self.puck.velocity[1] * ny
                self.puck.velocity[0] -= 2 * dot * nx
                self.puck.velocity[1] -= 2 * dot * ny
                obs["still_timer"] = 0.0  # reset anti-stuck timer on hit
            else:
                # Anti-stuck: if puck is near and nearly motionless for >1 s
                puck_speed = math.hypot(*self.puck.velocity)
                if dist < min_dist * 2 and puck_speed < 0.5:
                    obs["still_timer"] = obs.get("still_timer", 0.0) + dt
                    if obs["still_timer"] > 1.0:
                        angle = random.uniform(0, math.tau)
                        self.puck.velocity[0] += math.cos(angle) * 3
                        self.puck.velocity[1] += math.sin(angle) * 3
                        obs["still_timer"] = 0.0
                else:
                    obs["still_timer"] = 0.0

        for obs in to_remove:
            try:
                self.state.obstacles.remove(obs)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Powerup field-sphere rendering (keeps PowerUpManager pygame-free)
    # ------------------------------------------------------------------

    def _draw_powerup_spheres(self):
        """
        Draw field spheres and obstacle icebergs directly on screen.
        Called from _draw() instead of powerup_manager.draw().
        """
        font = self.renderer.get_font(14)

        # 1. Field spheres (collectible powerups)
        for sphere in self.powerup_manager.get_field_spheres():
            x, y = int(sphere.position[0]), int(sphere.position[1])
            r = sphere.radius
            color = sphere.definition.color
            alpha = sphere.alpha

            # Outer glow circle
            glow_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surf, (*color, max(0, alpha // 4)), (r * 2, r * 2), r * 2
            )
            self.screen.blit(glow_surf, (x - r * 2, y - r * 2))

            # Main circle
            main_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(main_surf, (*color, alpha), (r, r), r)
            pygame.draw.circle(main_surf, (255, 255, 255, alpha // 2), (r, r), r, 2)
            self.screen.blit(main_surf, (x - r, y - r))

            # Icon text
            icon_surf = font.render(sphere.definition.icon, True, (255, 255, 255))
            icon_surf.set_alpha(alpha)
            self.screen.blit(icon_surf, (
                x - icon_surf.get_width() // 2,
                y - icon_surf.get_height() // 2,
            ))

        # 2. Obstacles (Iceberg Flotante — Phase 6)
        obstacles = getattr(self.state, "obstacles", [])
        for obs in obstacles:
            alpha = obs.get("alpha", 255)
            if alpha <= 0:
                continue
            ox, oy, or_ = int(obs["x"]), int(obs["y"]), obs["radius"]
            ice_color = (180, 230, 255)

            obs_surf = pygame.Surface((or_ * 2, or_ * 2), pygame.SRCALPHA)
            # Filled hexagonal-ish block drawn as circle + inner ring
            pygame.draw.circle(obs_surf, (*ice_color, int(alpha * 0.7)), (or_, or_), or_)
            pygame.draw.circle(obs_surf, (220, 245, 255, int(alpha * 0.9)), (or_, or_), or_, 2)
            inner_r = max(1, or_ - 6)
            pygame.draw.circle(obs_surf, (255, 255, 255, int(alpha * 0.4)), (or_, or_), inner_r, 1)
            self.screen.blit(obs_surf, (ox - or_, oy - or_))

    def _handle_game_over(self, mouse_pos) -> str | None:
        """Handle game over input. Returns action or None."""
        # Check for mouse clicks on game over buttons (debounced)
        now = pygame.time.get_ticks()
        if pygame.mouse.get_pressed()[0] and (now - self._last_click_time) > 300:
            self._last_click_time = now
            W, H = self.config.width, self.config.height
            bw, bh = 180, 45
            retry_rect = pygame.Rect(W // 2 - bw - 10, H * 3 // 4, bw, bh)
            menu_rect = pygame.Rect(W // 2 + 10, H * 3 // 4, bw, bh)

            if retry_rect.collidepoint(mouse_pos):
                self._restart_match()
                return None  # Match restarted, stay in loop
            elif menu_rect.collidepoint(mouse_pos):
                return "back_to_menu"

        return None

    def _draw(self, mouse_pos):
        """Draw all game elements with optimized rendering."""
        # Background (pre-rendered static surface — single blit)
        self.renderer.draw_background()

        # Glow effects (themed per level)
        theme = self.level_config.get("theme", {}).get("glow_colors", {})
        p_color = theme.get("player", COLORS.NEON_RED)
        a_color = theme.get("ai", COLORS.NEON_GREEN)
        pk_color = theme.get("puck", COLORS.NEON_BLUE)
        draw_glow(self.screen, p_color, self.player1.position, self.player1.radius)
        draw_glow(self.screen, a_color, self.player2.position, self.player2.radius)
        draw_glow(self.screen, pk_color, self.puck.position, self.puck.radius)

        # Sprites (using RenderUpdates group for dirty-rect tracking)
        self.all_sprites.draw(self.screen)

        # Power-ups on the field (spheres + obstacles)
        if self.powerup_manager and self.match_config.powerups_enabled:
            self._draw_powerup_spheres()
            # Shield arc sobre portería del jugador con escudo activo
            self._goal_fx.draw(self.screen, self.powerup_manager.stacks, self.config)

        # Level mechanic visuals (UV zones, fog, shrinking walls, heat waves)
        if self.mechanic:
            self.mechanic.draw(self.screen)

        # HUD (animated score, timer, power-up bars, climate facts)
        if self.hud:
            self.hud.draw(
                self.screen, self.state, self.match_config,
                self.state.timer.elapsed, self.level_config,
                self.powerup_manager,
            )
        else:
            # Fallback simple score (uses cached font)
            font = self.renderer.get_font(36)
            txt = font.render(f"{self.state.player_score} - {self.state.ai_score}", True, COLORS.WHITE)
            self.screen.blit(txt, (self.config.half_width - txt.get_width() // 2, 20))

        # Game over overlay with stats and buttons
        if self.state.phase == GamePhase.GAME_OVER:
            self._draw_game_over(mouse_pos)

        # Pause overlay
        if self.state.phase == GamePhase.PAUSED:
            self._draw_pause(mouse_pos)

        # Debug / FPS counter
        if self.state.show_fps:
            fps_txt = self.renderer.get_font(24).render(f"FPS: {int(self.clock.get_fps())}", True, COLORS.WHITE)
            self.screen.blit(fps_txt, (10, 10))

        # Power-up help panel (topmost layer)
        if self._help_panel:
            self._help_panel.draw(self.screen)

    def _draw_game_over(self, mouse_pos):
        """Draw game over overlay with stats and buttons."""
        W, H = self.config.width, self.config.height
        if self._overlay_dim is None or self._overlay_dim.get_size() != (W, H):
            self._overlay_dim = pygame.Surface((W, H), pygame.SRCALPHA)
            self._overlay_dim.fill((0, 0, 0, 160))
        self.screen.blit(self._overlay_dim, (0, 0))

        font_big = self.renderer.get_font(48)
        font_med = self.renderer.get_font(32)
        font_small = self.renderer.get_font(24)

        is_pvp = self.match_config.mode == GameMode.PLAYER_VS_PLAYER

        # Winner text
        if is_pvp:
            if self.state.winner == "player":
                title = "Jugador 1 Gana!"
                color = (100, 255, 100)
            elif self.state.winner == "ai":
                title = "Jugador 2 Gana!"
                color = (100, 180, 255)
            else:
                title = "Empate!"
                color = COLORS.WHITE
        else:
            if self.state.winner == "player":
                title = "Has Ganado!"
                color = (100, 255, 100)
            elif self.state.winner == "ai":
                title = "Has Perdido!"
                color = (255, 100, 100)
            else:
                title = "Empate!"
                color = COLORS.WHITE

        title_surf = font_big.render(title, True, color)
        self.screen.blit(title_surf, (W // 2 - title_surf.get_width() // 2, H // 4))

        # Score
        score_txt = font_med.render(
            f"{self.state.player_score} - {self.state.ai_score}", True, COLORS.WHITE
        )
        self.screen.blit(score_txt, (W // 2 - score_txt.get_width() // 2, H // 4 + 60))

        # Stats — mode-aware labels
        elapsed = self.state.timer.elapsed
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        p2_label = "Golpes J2" if is_pvp else "Golpes IA"
        stats = [
            f"Tiempo: {mins:02d}:{secs:02d}",
            f"Golpes J1: {self.state.total_hits_player}",
            f"{p2_label}: {self.state.total_hits_ai}",
        ]
        for i, s in enumerate(stats):
            st = font_small.render(s, True, (200, 200, 200))
            self.screen.blit(st, (W // 2 - st.get_width() // 2, H // 2 + i * 28))

        # Buttons
        bw, bh = 180, 45
        retry_rect = pygame.Rect(W // 2 - bw - 10, H * 3 // 4, bw, bh)
        menu_rect = pygame.Rect(W // 2 + 10, H * 3 // 4, bw, bh)

        for rect, label in [(retry_rect, "Reintentar (R)"), (menu_rect, "Menu (ESC)")]:
            hover = rect.collidepoint(mouse_pos)
            c = (80, 80, 80) if hover else (50, 50, 50)
            pygame.draw.rect(self.screen, c, rect, border_radius=8)
            pygame.draw.rect(self.screen, COLORS.WHITE, rect, 2, border_radius=8)
            bt = font_small.render(label, True, COLORS.WHITE)
            self.screen.blit(bt, (rect.centerx - bt.get_width() // 2, rect.centery - bt.get_height() // 2))


    # ------------------------------------------------------------------
    # Power-up help panel
    # ------------------------------------------------------------------

    def _toggle_help_panel(self) -> None:
        """
        Toggle the PowerUpHelpPanel overlay.

        Lazy-initialises the panel on first call, so there is zero cost
        when powerups are disabled.  Works in any game phase.
        """
        # Only show when powerups are enabled and a registry exists
        registry = None
        if self.powerup_manager and self.match_config.powerups_enabled:
            registry = getattr(self.powerup_manager, "registry", None)

        if registry is None or len(registry) == 0:
            return   # No powerups registered → nothing to show

        if self._help_panel is None:
            W, H = self.config.width, self.config.height
            self._help_panel = PowerUpHelpPanel(registry, W, H)

        self._help_panel.toggle()

    def _restart_match(self):
        """Reset match state for a fresh restart (used from pause and game over)."""
        self.state.reset_match()
        self.puck.reset(zero_velocity=True)
        self.player1.position = [float(self.config.width // 4), float(self.config.height // 2)]
        self.player1.prev_position = self.player1.position.copy()
        self.player1.rect.center = (int(self.player1.position[0]), int(self.player1.position[1]))
        self.player1.velocity = [0.0, 0.0]
        self.player2.position = [float(self.config.width * 3 // 4), float(self.config.height // 2)]
        self.player2.prev_position = self.player2.position.copy()
        self.player2.rect.center = (int(self.player2.position[0]), int(self.player2.position[1]))
        self.player2.velocity = [0.0, 0.0]
        self.last_action = 4
        self.state.timer.start()
        self.steps_since_ai_hit = 0
        if self.powerup_manager:
            self.powerup_manager.reset()
        # Re-create mechanic for fresh state
        if self.mechanic:
            self.mechanic = create_mechanic(self.config, self.level_config)

    def _handle_pause(self, mouse_pos) -> str | None:
        """Handle pause screen input. Returns action or None."""
        now = pygame.time.get_ticks()
        if pygame.mouse.get_pressed()[0] and (now - self._last_click_time) > 300:
            self._last_click_time = now
            W, H = self.config.width, self.config.height
            bw, bh = 200, 45
            btn_y = H // 2 + 80
            spacing = 15
            total_w = bw * 3 + spacing * 2
            start_x = W // 2 - total_w // 2

            resume_rect = pygame.Rect(start_x, btn_y, bw, bh)
            restart_rect = pygame.Rect(start_x + bw + spacing, btn_y, bw, bh)
            menu_rect = pygame.Rect(start_x + (bw + spacing) * 2, btn_y, bw, bh)

            if resume_rect.collidepoint(mouse_pos):
                self.state.phase = GamePhase.PLAYING
                self.state.timer.resume()
            elif restart_rect.collidepoint(mouse_pos):
                self._restart_match()
            elif menu_rect.collidepoint(mouse_pos):
                return "back_to_menu"
        return None

    def _draw_pause(self, mouse_pos=None):
        """Draw pause overlay with buttons."""
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        W, H = self.config.width, self.config.height
        if self._overlay_dim_pause is None or self._overlay_dim_pause.get_size() != (W, H):
            self._overlay_dim_pause = pygame.Surface((W, H), pygame.SRCALPHA)
            self._overlay_dim_pause.fill((0, 0, 0, 120))
        self.screen.blit(self._overlay_dim_pause, (0, 0))

        # Title
        font = self.renderer.get_font(48)
        txt = font.render("PAUSA", True, COLORS.WHITE)
        self.screen.blit(txt, (W // 2 - txt.get_width() // 2, H // 2 - 60))

        # Hint
        font_hint = self.renderer.get_font(20)
        hint = font_hint.render("Presiona ESC para continuar", True, (180, 180, 180))
        self.screen.blit(hint, (W // 2 - hint.get_width() // 2, H // 2 - 10))

        # Buttons
        font_btn = self.renderer.get_font(22)
        bw, bh = 200, 45
        btn_y = H // 2 + 80
        spacing = 15
        total_w = bw * 3 + spacing * 2
        start_x = W // 2 - total_w // 2

        buttons = [
            (pygame.Rect(start_x, btn_y, bw, bh), "Continuar (ESC)", (60, 140, 60)),
            (pygame.Rect(start_x + bw + spacing, btn_y, bw, bh), "Reiniciar (R)", (140, 120, 40)),
            (pygame.Rect(start_x + (bw + spacing) * 2, btn_y, bw, bh), "Menu (M)", (140, 60, 60)),
        ]

        for rect, label, base_color in buttons:
            hover = rect.collidepoint(mouse_pos)
            c = tuple(min(255, ch + 40) for ch in base_color) if hover else base_color
            pygame.draw.rect(self.screen, c, rect, border_radius=8)
            pygame.draw.rect(self.screen, COLORS.WHITE, rect, 2, border_radius=8)
            bt = font_btn.render(label, True, COLORS.WHITE)
            self.screen.blit(bt, (rect.centerx - bt.get_width() // 2, rect.centery - bt.get_height() // 2))

        # Controls hint at bottom
        controls = font_hint.render("R: Reiniciar  |  M: Volver al Menu  |  ESC: Continuar", True, (150, 150, 150))
        self.screen.blit(controls, (W // 2 - controls.get_width() // 2, btn_y + bh + 20))
