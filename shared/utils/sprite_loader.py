"""
Sprite loading utility — loads images with error handling and scaling.
"""
import logging
import os
import pygame

logger = logging.getLogger(__name__)


class SpriteLoader:
    """Loads and manages sprites for different levels."""

    @staticmethod
    def load_sprite(image_path, target_size=None, preserve_aspect=True):
        """
        Load an image and optionally scale it.

        Returns the loaded Surface, or None on failure.
        """
        try:
            if not os.path.exists(image_path):
                logger.warning("Sprite not found: %s", image_path)
                return None
            image = pygame.image.load(image_path).convert_alpha()
            if target_size is None:
                return image
            if preserve_aspect:
                iw, ih = image.get_size()
                tw, th = target_size
                scale = min(tw / iw, th / ih)
                return pygame.transform.smoothscale(
                    image, (int(iw * scale), int(ih * scale))
                )
            return pygame.transform.smoothscale(image, target_size)
        except Exception as exc:
            logger.error("Failed to load sprite '%s': %s", image_path, exc)
            return None

    @staticmethod
    def create_circular_mask(radius):
        mask_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(mask_surf, (255, 255, 255, 255), (radius, radius), radius)
        return pygame.mask.from_surface(mask_surf)

    @staticmethod
    def load_level_sprites(level_id, config=None):
        """Load all sprites for a given level id."""
        from game.config.level_config import get_asset_path
        from shared.config import GameConfig

        cfg = config or GameConfig()
        sprites = {}
        sf = cfg.scale_factor

        logger.info("Loading sprites for level %d (scale=%.2f)", level_id, sf)

        bg_path = get_asset_path(level_id, "background.png")
        if os.path.exists(bg_path):
            s = SpriteLoader.load_sprite(bg_path, (cfg.width, cfg.height), False)
            if s:
                sprites["background"] = s

        puck_path = get_asset_path(level_id, "puck.png")
        if os.path.exists(puck_path):
            ps = int(30 * sf)
            s = SpriteLoader.load_sprite(puck_path, (ps, ps))
            if s:
                sprites["puck"] = s

        mallet_path = get_asset_path(level_id, "mallet_IA.png")
        if os.path.exists(mallet_path):
            ms = int(64 * sf)
            s = SpriteLoader.load_sprite(mallet_path, (ms, ms))
            if s:
                sprites["mallet_ai"] = s
                sprites["mallet_player"] = s.copy()

        for key, fname in [("goal_left", "porteria_izq.png"), ("goal_right", "porteria_der.png")]:
            p = get_asset_path(level_id, fname)
            if os.path.exists(p):
                s = SpriteLoader.load_sprite(p)
                if s:
                    sprites[key] = s
                    logger.info("  Loaded %s: %s (%dx%d)", key, p, *s.get_size())
                else:
                    logger.warning("  Failed to load %s from %s", key, p)
            else:
                logger.warning("  Goal sprite missing: %s", p)

        logger.info("Level %d sprites loaded: %s", level_id, list(sprites.keys()))
        return sprites

    @staticmethod
    def apply_tint(surface, color):
        tinted = surface.copy()
        tinted.fill(color + (128,), special_flags=pygame.BLEND_RGBA_MULT)
        return tinted
