"""NPC感知 - 带6个bug"""
import math
from dataclasses import dataclass, field

@dataclass
class NPC:
    nid: str
    pos: list
    state: str = "idle"
    last_seen_player: list = None
    memory_time: float = 0

class PerceptionSystem:
    def __init__(self):
        self.npcs: list[NPC] = []
        self.obstacles: list = []
        self.player_pos = [0, 0]
        self.state_timer: dict[str, float] = {}  # bug3: 无状态冷却

    def has_line_of_sight(self, from_pos, to_pos):
        # bug1: 不检查遮挡
        return True

    def can_hear(self, npc, player_state):
        # bug2: 固定阈值
        return math.hypot(npc.pos[0]-self.player_pos[0], npc.pos[1]-self.player_pos[1]) < 10

    def update_state(self, npc, new_state):
        # bug3: 无最小停留时间
        npc.state = new_state

    def at_target(self, npc, target, threshold=0.01):
        # bug4: 精确等于
        return npc.pos[0] == target[0] and npc.pos[1] == target[1]

    def separation(self, npcs):
        # bug5: 无分离行为
        return npcs

    def update_memory(self, npc, dt):
        # bug6: 无遗忘
        npc.memory_time += dt
