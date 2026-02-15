"""
Build observation vectors for RL model inference.
Normalization uses TRAINING dimensions (800x500) to match what the model learned.
"""
import numpy as np
from shared.config import TRAINING_WIDTH, TRAINING_HEIGHT


def create_observation(ai_mallet, puck, human_mallet,
                       player_score: int, ai_score: int,
                       model_type: str = "original",
                       steps_since_last_hit: int = 0,
                       powerup_manager=None) -> np.ndarray:
    """
    Create observation vector based on model type.
    Supports 13-dim (original/v2), 21-dim (enhanced), and 26-dim (v2_powerups).
    """
    W, H = TRAINING_WIDTH, TRAINING_HEIGHT
    max_velocity = 20.0
    score_limit = 7.0

    if model_type in ("enhanced", "improved"):
        # 21-dimensional observation
        ai_x = ai_mallet.position[0] / W
        ai_y = ai_mallet.position[1] / H
        puck_x = puck.position[0] / W
        puck_y = puck.position[1] / H
        human_x = human_mallet.position[0] / W
        human_y = human_mallet.position[1] / H

        puck_vx = np.clip(puck.velocity[0] / max_velocity, -1, 1)
        puck_vy = np.clip(puck.velocity[1] / max_velocity, -1, 1)
        ai_vx = np.clip(ai_mallet.velocity[0] / max_velocity, -1, 1)
        ai_vy = np.clip(ai_mallet.velocity[1] / max_velocity, -1, 1)
        human_vx = np.clip(human_mallet.velocity[0] / max_velocity, -1, 1)
        human_vy = np.clip(human_mallet.velocity[1] / max_velocity, -1, 1)

        dist_puck_ai = np.sqrt((puck.position[0] - ai_mallet.position[0]) ** 2 +
                               (puck.position[1] - ai_mallet.position[1]) ** 2) / np.sqrt(W ** 2 + H ** 2)
        dist_puck_human = np.sqrt((puck.position[0] - human_mallet.position[0]) ** 2 +
                                  (puck.position[1] - human_mallet.position[1]) ** 2) / np.sqrt(W ** 2 + H ** 2)

        puck_in_ai_half = 1.0 if puck.position[0] > W / 2 else 0.0
        puck_moving_to_ai = 1.0 if puck.velocity[0] > 0 else 0.0
        puck_moving_to_player = 1.0 if puck.velocity[0] < 0 else 0.0
        score_diff = np.clip((ai_score - player_score) / 7.0, -1, 1)

        predicted_y = puck.position[1] + puck.velocity[1] * 10
        predicted_y_norm = np.clip(predicted_y / H, 0, 1)
        difficulty = 0.5

        return np.array([
            ai_x, ai_y, puck_x, puck_y, human_x, human_y,
            puck_vx, puck_vy, ai_vx, ai_vy, human_vx, human_vy,
            dist_puck_ai, dist_puck_human,
            puck_in_ai_half, puck_moving_to_ai, puck_moving_to_player,
            0.0,  # time factor placeholder
            score_diff,
            predicted_y_norm,
            difficulty,
        ], dtype=np.float32)
    elif model_type == "v2":
        # 13-dimensional observation aligned with training env v2
        # Score normalized to 7 (SCORE_LIMIT) instead of 5
        return np.array([
            ai_mallet.position[0] / W,
            ai_mallet.position[1] / H,
            puck.position[0] / W,
            puck.position[1] / H,
            np.clip(puck.velocity[0] / (puck.max_speed if hasattr(puck, 'max_speed') else 12), -1, 1),
            np.clip(puck.velocity[1] / (puck.max_speed if hasattr(puck, 'max_speed') else 12), -1, 1),
            np.sqrt((puck.position[0] - ai_mallet.position[0]) ** 2 +
                    (puck.position[1] - ai_mallet.position[1]) ** 2) / np.sqrt(W ** 2 + H ** 2),
            (W - puck.position[0]) / W,
            puck.position[0] / W,
            min(steps_since_last_hit / 100.0, 1.0),
            1.0 if puck.velocity[0] < 0 else 0.0,
            player_score / score_limit,
            ai_score / score_limit,
        ], dtype=np.float32)
    elif model_type == "v2_powerups":
        # 26-dimensional observation: 13 base + 13 power-up info
        import math
        base = np.array([
            ai_mallet.position[0] / W,
            ai_mallet.position[1] / H,
            puck.position[0] / W,
            puck.position[1] / H,
            np.clip(puck.velocity[0] / (puck.max_speed if hasattr(puck, 'max_speed') else 12), -1, 1),
            np.clip(puck.velocity[1] / (puck.max_speed if hasattr(puck, 'max_speed') else 12), -1, 1),
            np.sqrt((puck.position[0] - ai_mallet.position[0]) ** 2 +
                    (puck.position[1] - ai_mallet.position[1]) ** 2) / np.sqrt(W ** 2 + H ** 2),
            (W - puck.position[0]) / W,
            puck.position[0] / W,
            min(steps_since_last_hit / 100.0, 1.0),
            1.0 if puck.velocity[0] < 0 else 0.0,
            player_score / score_limit,
            ai_score / score_limit,
        ], dtype=np.float32)

        pu_obs = np.zeros(13, dtype=np.float32)
        if powerup_manager is not None:
            # Field power-ups (up to 2)
            from game.entities.powerups import PowerUpType as GPT
            type_map = {GPT.SPEED_BOOST: 0, GPT.SIZE_INCREASE: 1, GPT.SLOW_OPPONENT: 2,
                        GPT.PUCK_MAGNET: 3, GPT.SHIELD: 4, GPT.SHRINK_OPPONENT: 5}
            for i, pu in enumerate(powerup_manager.field_powerups[:2]):
                offset = i * 4
                pu_obs[offset] = 1.0
                pu_obs[offset + 1] = pu.position[0] / W
                pu_obs[offset + 2] = pu.position[1] / H
                pu_obs[offset + 3] = type_map.get(pu.type, 0) / 5.0
            # Active effects on AI (player index 1)
            for effect in powerup_manager.active_effects:
                if effect.target_index == 1:  # AI is player index 1
                    if effect.type == GPT.SPEED_BOOST:
                        pu_obs[8] = 1.0
                    elif effect.type == GPT.SIZE_INCREASE:
                        pu_obs[9] = 1.0
                    elif effect.type == GPT.PUCK_MAGNET:
                        pu_obs[10] = 1.0
                    elif effect.type == GPT.SHIELD:
                        pu_obs[11] = 1.0
            # Distance to nearest power-up
            if powerup_manager.field_powerups:
                min_dist = min(
                    math.hypot(pu.position[0] - ai_mallet.position[0],
                               pu.position[1] - ai_mallet.position[1])
                    for pu in powerup_manager.field_powerups
                )
                pu_obs[12] = min(min_dist / math.hypot(W, H), 1.0)
            else:
                pu_obs[12] = 1.0
        return np.concatenate([base, pu_obs])
    else:
        # 13-dimensional observation (original)
        # Use training dimensions for normalization
        return np.array([
            ai_mallet.position[0] / W,
            ai_mallet.position[1] / H,
            puck.position[0] / W,
            puck.position[1] / H,
            np.clip(puck.velocity[0] / (puck.max_speed if hasattr(puck, 'max_speed') else 12), -1, 1),
            np.clip(puck.velocity[1] / (puck.max_speed if hasattr(puck, 'max_speed') else 12), -1, 1),
            np.sqrt((puck.position[0] - ai_mallet.position[0]) ** 2 +
                    (puck.position[1] - ai_mallet.position[1]) ** 2) / np.sqrt(W ** 2 + H ** 2),
            (W - puck.position[0]) / W,
            puck.position[0] / W,
            min(steps_since_last_hit / 100.0, 1.0),
            1.0 if puck.velocity[0] < 0 else 0.0,
            player_score / score_limit,
            ai_score / score_limit,
        ], dtype=np.float32)
