"""
Shared powerup system — framework-agnostic core.

Usage:
    from shared.powerups.registry import PowerUpRegistry, PowerUpDefinition
    from shared.powerups.manager import PowerUpManager
    from shared.powerups.effect_stack import EffectStack
"""
from shared.powerups.registry import PowerUpDefinition, PowerUpRegistry
from shared.powerups.effect_stack import ActiveEffect, EffectStack
from shared.powerups.manager import PowerUpManager

__all__ = [
    "PowerUpDefinition",
    "PowerUpRegistry",
    "ActiveEffect",
    "EffectStack",
    "PowerUpManager",
]
