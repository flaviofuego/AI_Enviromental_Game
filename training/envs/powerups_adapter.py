"""
TrainingPowerUpAdapter — bridges PowerUpManager ↔ RL observation/reward.

Responsibilities (Single Responsibility per class):
    PowerUpObsBuilder   — encodes PowerUpManager state into a flat numpy vector
                          with configurable phases; integrates transparently
                          alongside ObservationBuilder.
    PowerUpRewardSignal — translates manager events → scalar reward deltas.
    TrainingPowerUpAdapter — orchestrates both, exposes a clean API to the env.

Design decisions:
    * Zero pygame imports — compatible with headless training workers.
    * Phases are additive; activating phase N never breaks models trained on
      phases 1..N-1 because the obs vector is concatenated at the end of the
      base observation (standard SB3 practice for incremental obs expansion).
    * The adapter is stateless between steps except for the reward accumulator
      (reset at episode start).  All game state lives in PowerUpManager.
    * MAX_FIELD_SPHERES_OBS controls how many field spheres are encoded;
      extra spheres beyond this count are silently ignored (bounded obs space).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from shared.powerups.manager import PowerUpManager
    from shared.powerups.registry import PowerUpRegistry


# ──────────────────────────────────────────────────────────────────────────────
# Observable slot count per phase group (docs §7.2)
# ──────────────────────────────────────────────────────────────────────────────

# Phases 1–3: 2 field-sphere slots (exists, x, y, type_idx = 4 dims each)
#              + 4 AI active-effect flags (speed, shield, magnet, paralyzed)
# → 12 dims total for phases 1–3
_SPHERE_SLOT_DIMS = 4   # (exists, x_norm, y_norm, type_idx_norm) per sphere
_MAX_OBS_SPHERES  = 3   # matches MAX_POWERUPS_ON_FIELD in shared/powerups/config.py

#  Phases 1–3  → 12 dims:  3×4 (spheres)
#  + AI effects basic: 4 dims (speed_active, shield_active, magnet_active, duplication_active)
#  Phases 4–6  → +4 dims:  slow_active, obstacle_active, speed_mult, magnet_force
#  Phases 7–8  → +3 dims:  is_paralyzed, is_invisible, dist_nearest_sphere
#  Total max   → 23 dims

_PHASE_GROUP_DIMS = {
    "base":    _MAX_OBS_SPHERES * _SPHERE_SLOT_DIMS + 4,  # 16 dims
    "mid":     4,   # additional for phases 4–6
    "advanced": 3,  # additional for phases 7–8
}

# Reward constants (docs §7.3)
REWARD_COLLECT_POSITIVE   = +0.15
REWARD_COLLECT_NEGATIVE   = +0.10   # powerup that harms opponent = positive
REWARD_EVADE_OBSTACLE     = +0.05
REWARD_OPPONENT_PARALYZED_GOAL = +0.25
REWARD_POWERUP_EXPIRE_PENALTY = -0.02  # small: losing an effect hurts slightly


# ──────────────────────────────────────────────────────────────────────────────
# PowerUpObsBuilder
# ──────────────────────────────────────────────────────────────────────────────

class PowerUpObsBuilder:
    """
    Encodes PowerUpManager state into a bounded flat observation vector.

    The observation vector is divided into phase groups so that activating
    a higher phase group only appends new dimensions at the end.

    Parameters
    ----------
    phases : list[int]
        Which powerup phases contribute to the observation.
        - phases 1–3  → enables base group (field spheres + 4 AI effect flags)
        - phases 4–6  → enables mid group (slow, obstacle, speed_mult, magnet_force)
        - phases 7–8  → enables advanced group (paralyzed, invisible, nearest dist)
    ai_player_idx : int
        Which player slot in the manager corresponds to the AI agent (default 1,
        matching AirHockeyEnv convention: slot 0 = human, slot 1 = AI).
    """

    def __init__(
        self,
        phases: List[int],
        ai_player_idx: int = 1,
    ) -> None:
        self._phases = set(phases)
        self._ai_idx = ai_player_idx
        self._opp_idx = 1 - ai_player_idx

        self._use_base     = bool(self._phases & {1, 2, 3})
        self._use_mid      = bool(self._phases & {4, 5, 6})
        self._use_advanced = bool(self._phases & {7, 8})

        self._dim = (
            (_PHASE_GROUP_DIMS["base"]     if self._use_base     else 0)
            + (_PHASE_GROUP_DIMS["mid"]    if self._use_mid      else 0)
            + (_PHASE_GROUP_DIMS["advanced"] if self._use_advanced else 0)
        )

    # --- Public API -------------------------------------------------------

    @property
    def dim(self) -> int:
        """Total number of dimensions this builder appends to the base obs."""
        return self._dim

    def get_obs_space_extension(self) -> Optional[spaces.Box]:
        """Return a Box space representing this builder's obs slice."""
        if self._dim == 0:
            return None
        low  = np.zeros(self._dim, dtype=np.float32)
        high = np.ones(self._dim, dtype=np.float32)
        # velocity-like fields can go negative — override selectively below
        # For simplicity, all values are normalized [0,1] except speed_mult [0,5]
        # which we still clamp to [0,1] in the builder for stability.
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def build(
        self,
        manager: "PowerUpManager",
        width: float,
        height: float,
    ) -> np.ndarray:
        """
        Encode manager state → flat np.float32 vector of length ``self.dim``.

        All values are normalized to [0, 1] for stable RL training.
        """
        if self._dim == 0:
            return np.empty(0, dtype=np.float32)

        parts: List[float] = []
        diag = math.hypot(width, height)

        # ── Base group (phases 1–3) ──────────────────────────────────────
        if self._use_base:
            spheres = manager.get_field_spheres()
            all_ids = sorted({d.id for d in manager.registry.get_all()})
            id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
            n_types   = max(len(all_ids), 1)

            for slot in range(_MAX_OBS_SPHERES):
                if slot < len(spheres):
                    s = spheres[slot]
                    type_norm = id_to_idx.get(s.definition.id, 0) / (n_types - 1 or 1)
                    parts.extend([
                        1.0,
                        s.position[0] / width,
                        s.position[1] / height,
                        float(type_norm),
                    ])
                else:
                    parts.extend([0.0, 0.0, 0.0, 0.0])

            # 4 AI effect flags: speed, shield, magnet, duplication
            ai_stack = manager.stacks[self._ai_idx]
            parts.extend([
                1.0 if ai_stack.has("speed_boost")   else 0.0,
                1.0 if ai_stack.has_shield()          else 0.0,
                1.0 if ai_stack.has_magnet()          else 0.0,
                1.0 if ai_stack.has("duplication")    else 0.0,
            ])

        # ── Mid group (phases 4–6) ───────────────────────────────────────
        if self._use_mid:
            ai_stack  = manager.stacks[self._ai_idx]
            opp_stack = manager.stacks[self._opp_idx]
            # slow_opponent affects opp (check opp stack for it)
            parts.extend([
                1.0 if opp_stack.has("slow_opponent") else 0.0,
                1.0 if ai_stack.has("obstacle")        else 0.0,
                min(ai_stack.get_speed_multiplier() / 5.0, 1.0),  # norm to [0,1]
                min(ai_stack.get_magnet_force() / 2.0,  1.0),
            ])

        # ── Advanced group (phases 7–8) ─────────────────────────────────
        if self._use_advanced:
            ai_stack = manager.stacks[self._ai_idx]
            spheres  = manager.get_field_spheres()

            # Distance to nearest sphere (normalized)
            if spheres:
                # We need ai position — not directly available here, so we
                # encode the min sphere distance normalised; the env passes
                # ai_pos via a separate call to build_with_ai_pos()
                nearest = 1.0   # placeholder if ai_pos not provided
            else:
                nearest = 1.0

            parts.extend([
                1.0 if ai_stack.is_paralyzed()  else 0.0,
                1.0 if ai_stack.is_invisible()  else 0.0,
                nearest,
            ])

        return np.array(parts, dtype=np.float32)

    def build_with_ai_pos(
        self,
        manager: "PowerUpManager",
        width: float,
        height: float,
        ai_pos: tuple,
    ) -> np.ndarray:
        """
        Variant that also computes nearest-sphere distance for the advanced group.
        Prefer this over ``build()`` when ``ai_pos`` is available.
        """
        obs = self.build(manager, width, height)
        if self._use_advanced and self._dim > 0:
            spheres = manager.get_field_spheres()
            diag = math.hypot(width, height)
            if spheres:
                nearest = min(
                    math.hypot(s.position[0] - ai_pos[0], s.position[1] - ai_pos[1])
                    for s in spheres
                )
                nearest_norm = min(nearest / diag, 1.0)
            else:
                nearest_norm = 1.0
            # Overwrite the last 3 dims: [paralyzed, invisible, nearest_dist]
            obs[-1] = nearest_norm
        return obs

    def describe(self) -> List[str]:
        """Human-readable dimension names — useful for logging."""
        names: List[str] = []
        if self._use_base:
            for i in range(_MAX_OBS_SPHERES):
                names += [
                    f"sphere{i}_exists",
                    f"sphere{i}_x",
                    f"sphere{i}_y",
                    f"sphere{i}_type",
                ]
            names += [
                "ai_speed_active",
                "ai_shield_active",
                "ai_magnet_active",
                "ai_duplication_active",
            ]
        if self._use_mid:
            names += [
                "opp_slow_active",
                "ai_obstacle_active",
                "ai_speed_mult_norm",
                "ai_magnet_force_norm",
            ]
        if self._use_advanced:
            names += [
                "ai_is_paralyzed",
                "ai_is_invisible",
                "nearest_sphere_dist",
            ]
        return names

    def __repr__(self) -> str:
        return (
            f"PowerUpObsBuilder(phases={sorted(self._phases)}, "
            f"dim={self._dim}, ai_idx={self._ai_idx})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# PowerUpRewardSignal
# ──────────────────────────────────────────────────────────────────────────────

class PowerUpRewardSignal:
    """
    Translates PowerUpManager events into scalar reward deltas.

    Stateless between calls except for the per-episode accumulator used
    to cap total powerup reward and prevent exploiting spawn luck.

    Parameters
    ----------
    ai_player_idx : int
        Which player slot is the RL agent.
    max_episode_bonus : float
        Upper bound on cumulative powerup reward per episode (anti-exploit).
    """

    MAX_EPISODE_BONUS: float = 3.0   # caps total powerup reward per episode

    def __init__(self, ai_player_idx: int = 1) -> None:
        self._ai_idx = ai_player_idx
        self._episode_bonus: float = 0.0

    def reset(self) -> None:
        """Call at episode start."""
        self._episode_bonus = 0.0

    def process_events(
        self,
        events: List[Dict],
        manager: "PowerUpManager",
        opponent_paralyzed_goal: bool = False,
    ) -> float:
        """
        Compute reward delta from a list of manager events.

        Parameters
        ----------
        events                  : returned by PowerUpManager.update()
        manager                 : reference for current stack state
        opponent_paralyzed_goal : True when a goal was scored while the
                                  opponent was still paralyzed this step.
        """
        delta = 0.0
        remaining_budget = self.MAX_EPISODE_BONUS - self._episode_bonus

        for ev in events:
            ev_type = ev.get("type")

            if ev_type == "collected":
                powerup_id  = ev.get("powerup_id", "")
                player_idx  = ev.get("player_idx", -1)
                target_idx  = ev.get("target_idx", -1)

                if player_idx == self._ai_idx:
                    # AI collected it
                    if target_idx == self._ai_idx:
                        # Positive self-effect (speed, shield, magnet, etc.)
                        bonus = REWARD_COLLECT_POSITIVE
                    else:
                        # Negative effect on opponent (slow, paralyze, etc.)
                        bonus = REWARD_COLLECT_NEGATIVE
                    delta += min(bonus, remaining_budget)
                    remaining_budget -= bonus

            elif ev_type == "expired":
                # Very small penalty for expiry — teaches urgency
                target_idx = ev.get("target_idx", -1)
                if target_idx == self._ai_idx:
                    delta -= REWARD_POWERUP_EXPIRE_PENALTY

        # Extra bonus: goal while opponent is paralyzed
        if opponent_paralyzed_goal:
            bonus = REWARD_OPPONENT_PARALYZED_GOAL
            delta += min(bonus, remaining_budget)
            remaining_budget -= bonus

        self._episode_bonus += max(0.0, delta)
        return float(delta)

    @property
    def episode_bonus(self) -> float:
        """Cumulative bonus granted this episode (for logging)."""
        return self._episode_bonus


# ──────────────────────────────────────────────────────────────────────────────
# TrainingPowerUpAdapter  (main public API)
# ──────────────────────────────────────────────────────────────────────────────

class TrainingPowerUpAdapter:
    """
    Orchestrates PowerUpManager integration into a Gymnasium training environment.

    Owns:
        - PowerUpManager  (shared/ engine, pygame-free)
        - PowerUpObsBuilder  (obs encoding)
        - PowerUpRewardSignal  (event → reward)

    The environment only needs to call:
        adapter.reset()                        → episode boundary
        adapter.step(dt, players, puck, state) → (obs_ext, reward_delta, info)
        adapter.get_obs_space_extension()      → Box space appended to base obs

    Parameters
    ----------
    manager         : pre-built PowerUpManager with registry + phases configured.
    phases          : list of phases whose obs features are included.
    ai_player_idx   : player slot for the RL agent (0 = left, 1 = right).
    """

    def __init__(
        self,
        manager: "PowerUpManager",
        phases: List[int],
        ai_player_idx: int = 1,
    ) -> None:
        self.manager = manager
        self.obs_builder    = PowerUpObsBuilder(phases=phases, ai_player_idx=ai_player_idx)
        self.reward_signal  = PowerUpRewardSignal(ai_player_idx=ai_player_idx)
        self._ai_idx        = ai_player_idx
        self._last_events: List[Dict] = []

    # --- Episode lifecycle ------------------------------------------------

    def reset(self) -> None:
        """Reset engine + reward accumulator at episode start."""
        self.manager.reset()
        self.reward_signal.reset()
        self._last_events = []

    # --- Per-step update --------------------------------------------------

    def step(
        self,
        dt: float,
        players: list,
        puck,
        state,
        ai_pos: Optional[tuple] = None,
        opponent_paralyzed_goal: bool = False,
    ) -> tuple[np.ndarray, float, dict]:
        """
        Advance powerup simulation and return (obs_extension, reward_delta, info).

        Parameters
        ----------
        dt                      : frame delta time in seconds
        players                 : list of 2 mallet-like objects (index = player slot)
        puck                    : puck entity
        state                   : TrainingState (shared between env and adapter)
        ai_pos                  : (x, y) of the AI mallet for nearest-sphere distance
        opponent_paralyzed_goal : True if a goal was scored while opp was paralyzed
        """
        events = self.manager.update(dt, players, puck, state)
        self._last_events = events

        W = getattr(state, "width",  self.manager.config.width)
        H = getattr(state, "height", self.manager.config.height)

        if ai_pos is not None:
            obs_ext = self.obs_builder.build_with_ai_pos(self.manager, W, H, ai_pos)
        else:
            obs_ext = self.obs_builder.build(self.manager, W, H)

        reward_delta = self.reward_signal.process_events(
            events,
            self.manager,
            opponent_paralyzed_goal=opponent_paralyzed_goal,
        )

        info = {
            "powerup_events":      len(events),
            "powerup_episode_bonus": self.reward_signal.episode_bonus,
            "active_effects_ai":   len(self.manager.get_active_effects(self._ai_idx)),
            "field_spheres":       len(self.manager.get_field_spheres()),
        }
        return obs_ext, reward_delta, info

    # --- Observation helpers ---------------------------------------------

    def get_obs_space_extension(self) -> Optional[spaces.Box]:
        """Box space for the powerup obs slice; None if dim == 0."""
        return self.obs_builder.get_obs_space_extension()

    @property
    def obs_dim(self) -> int:
        """Number of dimensions this adapter appends to the base obs."""
        return self.obs_builder.dim

    # --- State queries (for the env and renderer) -----------------------

    def is_ai_paralyzed(self) -> bool:
        return self.manager.stacks[self._ai_idx].is_paralyzed()

    def is_ai_invisible(self) -> bool:
        return self.manager.stacks[self._ai_idx].is_invisible()

    def get_ai_speed_multiplier(self) -> float:
        return self.manager.stacks[self._ai_idx].get_speed_multiplier()

    def describe_obs(self) -> List[str]:
        """Return dimension names for the powerup obs slice."""
        return self.obs_builder.describe()

    def __repr__(self) -> str:
        return (
            f"TrainingPowerUpAdapter("
            f"obs_dim={self.obs_dim}, "
            f"manager={self.manager!r})"
        )
