"""
AirHockeyWithPowerUpsEnv — Air Hockey environment with the modular powerup system.

This environment extends AirHockeyEnv and integrates the shared/ powerup engine
(PowerUpManager, EffectStack, PowerUpRegistry) through TrainingPowerUpAdapter.

Architecture (Separation of Concerns):
    AirHockeyEnv          ← base physics, action/obs, rewards (unchanged)
    TrainingState         ← lightweight state proxy for PowerUpManager callbacks
    PowerUpManager        ← generic powerup engine (shared/, pygame-free)
    TrainingPowerUpAdapter ← obs encoding + reward signals for RL
    AirHockeyWithPowerUpsEnv ← orchestrates all of the above

Observation space (total = base_obs_dim + powerup_obs_dim):
    Base (13D / 17D / 21D — selected by obs_builder):
        Standard AirHockeyEnv signals.
    Powerup extension (0–23D — determined by active phases):
        Phases 1–3: 16D  (3 sphere slots × 4 + 4 AI effect flags)
        Phases 4–6: +4D  (slow, obstacle, speed_mult_norm, magnet_force_norm)
        Phases 7–8: +3D  (paralyzed, invisible, nearest_sphere_dist)

Usage:
    env = AirHockeyWithPowerUpsEnv(phases=[1, 2, 3])
    env = AirHockeyWithPowerUpsEnv(phases=list(range(1, 9)))  # all phases
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from typing import List, Optional

from shared.config import GameConfig
from shared.powerups.registry import PowerUpRegistry
from shared.powerups.manager import PowerUpManager
from shared.powerups.definitions import register_phases

from training.envs.base_env import AirHockeyEnv
from training.envs.observation_builder import ObservationBuilder
from training.envs.training_state import TrainingState
from training.envs.powerups_adapter import TrainingPowerUpAdapter


class AirHockeyWithPowerUpsEnv(AirHockeyEnv):
    """
    Air Hockey with the shared modular powerup system.

    Parameters
    ----------
    phases : list[int]
        Powerup phases to activate (1-8).  Defaults to phases 1-3.
        - Phase 1: Speed Boost
        - Phase 2: Shield
        - Phase 3: Magnet
        - Phase 4: Duplication
        - Phase 5: Slow Opponent
        - Phase 6: Obstacle
        - Phase 7: Paralyze
        - Phase 8: Invisibility
    render_mode : str | None
        Gymnasium render mode ("human", "rgb_array", None).
    play_mode : bool
        If True, suppresses the algorithmic opponent (used in live game sessions).
    config : GameConfig | None
        Override default field configuration.
    obs_builder : ObservationBuilder | None
        Base observation builder.  Defaults to ObservationBuilder.standard() (13D).
    ai_player_idx : int
        Which player slot (0 or 1) corresponds to the RL agent.
        Default 1 matches AirHockeyEnv convention (AI on right side).
    """

    AI_PLAYER_IDX: int = 1

    def __init__(
        self,
        phases: Optional[List[int]] = None,
        render_mode: Optional[str] = None,
        play_mode: bool = False,
        config: Optional[GameConfig] = None,
        obs_builder: Optional[ObservationBuilder] = None,
        ai_player_idx: int = 1,
    ) -> None:
        self._active_phases = list(phases or [1, 2, 3])
        self._ai_player_idx = ai_player_idx

        cfg = config or GameConfig()

        # Build shared powerup registry & manager
        registry = PowerUpRegistry()
        register_phases(registry, self._active_phases)
        manager = PowerUpManager(cfg, registry, enabled_phases=self._active_phases)

        # Training state proxy (no pygame)
        self._training_state = TrainingState(width=cfg.width, height=cfg.height)

        # Adapter owns obs encoding + reward signal
        self._pu_adapter = TrainingPowerUpAdapter(
            manager=manager,
            phases=self._active_phases,
            ai_player_idx=self._ai_player_idx,
        )

        # Init base env (calls reset() internally)
        super().__init__(
            render_mode=render_mode,
            play_mode=play_mode,
            config=config,
            obs_builder=obs_builder,
        )

        # Extend observation space
        ext_space = self._pu_adapter.get_obs_space_extension()
        if ext_space is not None:
            base_space = self.observation_space
            combined_low  = np.concatenate([base_space.low,  ext_space.low])
            combined_high = np.concatenate([base_space.high, ext_space.high])
            self.observation_space = spaces.Box(
                combined_low, combined_high, dtype=np.float32
            )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if hasattr(self, "_pu_adapter"):
            self._pu_adapter.reset()
        if hasattr(self, "_training_state"):
            self._training_state.reset()
            self._training_state.width  = self.config.width
            self._training_state.height = self.config.height

        return self._get_observation(), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        # Paralysis: force Stay action if AI is paralyzed (plan T-7.4)
        if hasattr(self, "_pu_adapter") and self._pu_adapter.is_ai_paralyzed():
            action = 4  # Stay

        # Apply speed multiplier from EffectStack
        original_move = self.physics.ai_move_amount
        if hasattr(self, "_pu_adapter"):
            speed_mult = self._pu_adapter.get_ai_speed_multiplier()
            self.physics.ai_move_amount = max(1, int(original_move * speed_mult))

        obs, reward, terminated, truncated, info = super().step(action)

        # Restore move amount
        self.physics.ai_move_amount = original_move

        # Mirror scores to training state
        if hasattr(self, "_training_state"):
            self._training_state.player_score = self.player_score
            self._training_state.ai_score     = self.ai_score

        # Build player proxies for PowerUpManager
        players = self._build_players_proxy()

        # Check if a goal was scored while opponent was paralyzed
        dt = 1.0 / 60.0
        opp_idx = 1 - self._ai_player_idx
        opp_paralyzed_goal = (
            info.get("goal") == "ai"
            and hasattr(self, "_pu_adapter")
            and self._pu_adapter.manager.stacks[opp_idx].is_paralyzed()
        )

        # Advance powerup engine
        obs_ext, reward_delta, pu_info = self._pu_adapter.step(
            dt,
            players,
            self.puck,
            self._training_state,
            ai_pos=(self.ai_mallet_position[0], self.ai_mallet_position[1]),
            opponent_paralyzed_goal=opp_paralyzed_goal,
        )

        reward += reward_delta
        info.update(pu_info)
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self):
        """Build base obs and append powerup extension."""
        base_obs = super()._get_observation()
        if not hasattr(self, "_pu_adapter") or self._pu_adapter.obs_dim == 0:
            return base_obs

        obs_ext = self._pu_adapter.obs_builder.build_with_ai_pos(
            self._pu_adapter.manager,
            self.config.width,
            self.config.height,
            ai_pos=(self.ai_mallet_position[0], self.ai_mallet_position[1]),
        )
        if obs_ext.shape[0] == 0:
            return base_obs
        return np.concatenate([base_obs, obs_ext], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_players_proxy(self) -> list:
        """
        Build a minimal list of player proxies for PowerUpManager.

        Slot 0 = human/opponent, Slot 1 = AI (matches AI_PLAYER_IDX = 1).
        PowerUpManager reads .position, .radius and writes
        .speed_multiplier, .paralyzed.
        """
        return [
            _MalletProxy(
                position=list(self.human_mallet.position),
                radius=float(self.human_mallet.radius),
            ),
            _MalletProxy(
                position=list(self.ai_mallet_position),
                radius=float(self.ai_mallet_radius),
            ),
        ]

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    def get_active_phases(self) -> List[int]:
        """Return the list of enabled powerup phases."""
        return list(self._active_phases)

    def get_powerup_obs_dim(self) -> int:
        """Dimension of the powerup obs extension."""
        return self._pu_adapter.obs_dim

    def describe_obs(self) -> List[str]:
        """Return dimension names for the full observation vector."""
        base_names = self._obs_builder.describe()
        pu_names   = self._pu_adapter.describe_obs()
        return base_names + pu_names


# ---------------------------------------------------------------------------
# _MalletProxy — duck-type compatible with PowerUpManager's player interface
# ---------------------------------------------------------------------------

class _MalletProxy:
    """
    Minimal duck-type proxy for PowerUpManager's player interface.

    Avoids coupling the env to concrete Mallet classes.  The manager reads
    .position and .radius, and writes .speed_multiplier / .paralyzed.
    """

    __slots__ = ("position", "radius", "speed_multiplier", "paralyzed")

    def __init__(self, position: list, radius: float) -> None:
        self.position         = position
        self.radius           = radius
        self.speed_multiplier = 1.0
        self.paralyzed        = False
