"""AI核心：NPC感知与决策（兼容入口，实现位于 src/engine.py）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import (  # noqa: E402
    NPC,
    PerceptionSystem,
    MOVE_NOISE_RADIUS,
    MATERIAL_SOUND_FACTOR,
    DEFAULT_MIN_STATE_FRAMES,
    DEFAULT_MEMORY_DURATION,
    DEFAULT_MIN_SEPARATION,
)

__all__ = [
    "NPC",
    "PerceptionSystem",
    "MOVE_NOISE_RADIUS",
    "MATERIAL_SOUND_FACTOR",
    "DEFAULT_MIN_STATE_FRAMES",
    "DEFAULT_MEMORY_DURATION",
    "DEFAULT_MIN_SEPARATION",
]
