"""NPC感知 - 修复版

修复点：
1. 视野检测：线段与圆形障碍物做射线投射，遮挡则不可见
2. 听觉：可听距离随移动状态（蹲/走/跑）与地面材质动态变化
3. 决策：状态转换有最小停留时间（帧数冷却），冷却期内保持原状态
4. 路径跟随：到达判断使用距离阈值而非精确等于
5. 群体：相邻NPC之间施加分离力，保持最小间距
6. 记忆：记忆随时间衰减，超过记忆时长后遗忘最后看到的位置
"""
import math
from dataclasses import dataclass, field


@dataclass
class NPC:
    nid: str
    pos: list
    state: str = "idle"
    last_seen_player: list = None
    memory_time: float = 0.0
    state_timer: int = 0  # 进入当前状态后已经过的帧数
    state_entered: bool = False  # 是否已正式进入过状态（首次切换无冷却）


# 各移动状态的基础听觉半径（声音越大传得越远）
MOVE_NOISE_RADIUS = {
    "crouch": 4.0,
    "walk": 10.0,
    "run": 18.0,
    "idle": 0.0,
}

# 地面材质对声音传播的系数
MATERIAL_SOUND_FACTOR = {
    "stone": 1.0,
    "concrete": 1.0,
    "wood": 1.3,
    "metal": 1.5,
    "grass": 0.7,
    "carpet": 0.5,
    "water": 0.6,
}

# 默认状态最小停留帧数
DEFAULT_MIN_STATE_FRAMES = 10

# 默认记忆时长（秒）
DEFAULT_MEMORY_DURATION = 5.0

# 默认群体最小间距
DEFAULT_MIN_SEPARATION = 1.5


class PerceptionSystem:
    def __init__(self, min_state_frames=DEFAULT_MIN_STATE_FRAMES,
                 memory_duration=DEFAULT_MEMORY_DURATION):
        self.npcs: list[NPC] = []
        self.obstacles: list = []
        self.player_pos = [0, 0]
        # 状态冷却：记录每个NPC进入当前状态后停留的帧数
        self.state_timer: dict = {}
        self.min_state_frames = min_state_frames
        # 记忆时长（秒）
        self.memory_duration = memory_duration

    # ---------- 1. 视野：射线投射检查遮挡 ----------
    def has_line_of_sight(self, from_pos, to_pos):
        """从 from_pos 到 to_pos 做射线投射，被任何障碍物挡住则返回 False。

        支持的障碍物形式：
        - {"x", "y", "r"}：圆形
        - {"x", "y", "w", "h"}：轴对齐矩形
        - (x, y, r)：圆形元组
        """
        x1, y1 = float(from_pos[0]), float(from_pos[1])
        x2, y2 = float(to_pos[0]), float(to_pos[1])
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy

        for obs in self.obstacles:
            if isinstance(obs, dict):
                if "r" in obs:
                    cx, cy, radius = obs["x"], obs["y"], obs["r"]
                elif "w" in obs and "h" in obs:
                    if self._seg_intersects_rect(x1, y1, x2, y2,
                                                 obs["x"], obs["y"],
                                                 obs["w"], obs["h"]):
                        return False
                    continue
                else:
                    continue
            else:
                cx, cy, radius = obs[0], obs[1], obs[2]

            if self._seg_circle_distance_sq(x1, y1, x2, y2, dx, dy, length2,
                                            cx, cy) <= radius * radius:
                return False
        return True

    @staticmethod
    def _seg_circle_distance_sq(x1, y1, x2, y2, dx, dy, length2, cx, cy):
        """线段到圆心的最短距离平方。"""
        if length2 == 0.0:
            ex, ey = x1 - cx, y1 - cy
            return ex * ex + ey * ey
        t = ((cx - x1) * dx + (cy - y1) * dy) / length2
        t = max(0.0, min(1.0, t))
        qx, qy = x1 + t * dx, y1 + t * dy
        ex, ey = qx - cx, qy - cy
        return ex * ex + ey * ey

    @staticmethod
    def _seg_intersects_rect(x1, y1, x2, y2, rx, ry, rw, rh):
        """线段与轴对齐矩形（含内部）是否相交。"""
        def inside(x, y):
            return rx <= x <= rx + rw and ry <= y <= ry + rh

        if inside(x1, y1) or inside(x2, y2):
            return True
        edges = (
            (rx, ry, rx + rw, ry),
            (rx + rw, ry, rx + rw, ry + rh),
            (rx + rw, ry + rh, rx, ry + rh),
            (rx, ry + rh, rx, ry),
        )
        for ex1, ey1, ex2, ey2 in edges:
            if PerceptionSystem._segments_cross(x1, y1, x2, y2,
                                                ex1, ey1, ex2, ey2):
                return True
        return False

    @staticmethod
    def _segments_cross(x1, y1, x2, y2, x3, y3, x4, y4):
        def orient(ax, ay, bx, by, cx, cy):
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        d1 = orient(x3, y3, x4, y4, x1, y1)
        d2 = orient(x3, y3, x4, y4, x2, y2)
        d3 = orient(x1, y1, x2, y2, x3, y3)
        d4 = orient(x1, y1, x2, y2, x4, y4)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    # ---------- 2. 听觉：移动状态 + 地面材质动态阈值 ----------
    def hearing_radius(self, player_state="walk", material="stone"):
        """根据移动状态和地面材质计算当前可听半径。"""
        if isinstance(player_state, dict):
            material = player_state.get("material", material)
            player_state = player_state.get("move",
                                player_state.get("state", "walk"))
        base = MOVE_NOISE_RADIUS.get(player_state, 10.0)
        factor = MATERIAL_SOUND_FACTOR.get(material, 1.0)
        return base * factor

    def can_hear(self, npc, player_state="walk", material="stone"):
        radius = self.hearing_radius(player_state, material)
        dist = math.hypot(npc.pos[0] - self.player_pos[0],
                          npc.pos[1] - self.player_pos[1])
        return dist < radius

    # ---------- 3. 决策：状态最小停留时间 ----------
    def update_state(self, npc, new_state, min_frames=None):
        """尝试切换状态；未达到最小停留帧数时保持原状态。

        返回最终所处的状态。
        """
        if min_frames is None:
            min_frames = self.min_state_frames
        unlocked = getattr(npc, "state_entered", False) is False
        if unlocked or new_state == npc.state or npc.state_timer >= min_frames:
            if new_state != npc.state:
                npc.state = new_state
                npc.state_timer = 0
            npc.state_entered = True
            self.state_timer[npc.nid] = npc.state_timer
            return npc.state
        # 冷却中：保持原状态
        return npc.state

    def tick_state(self, npc, frames=1):
        """推进状态停留计时，每帧调用一次。"""
        npc.state_timer += frames
        self.state_timer[npc.nid] = npc.state_timer
        return npc.state_timer

    # ---------- 4. 路径跟随：距离阈值到达判断 ----------
    def at_target(self, npc, target, threshold=0.1):
        dist = math.hypot(npc.pos[0] - target[0], npc.pos[1] - target[1])
        return dist <= threshold

    # ---------- 5. 群体：分离力 ----------
    def separation(self, npcs, min_distance=DEFAULT_MIN_SEPARATION,
                   strength=0.5):
        """对相邻过近的NPC施加分离力，原地更新坐标并返回同一列表。"""
        count = len(npcs)
        for i in range(count):
            force_x, force_y = 0.0, 0.0
            pos_i = npcs[i].pos if hasattr(npcs[i], "pos") else npcs[i]
            for j in range(count):
                if i == j:
                    continue
                pos_j = npcs[j].pos if hasattr(npcs[j], "pos") else npcs[j]
                dx, dy = pos_i[0] - pos_j[0], pos_i[1] - pos_j[1]
                dist = math.hypot(dx, dy)
                if dist < min_distance:
                    if dist < 1e-9:
                        # 完全重叠：沿确定性方向推开
                        angle = i * 2.399963229728653  # 黄金角
                        dx, dy, dist = math.cos(angle), math.sin(angle), 1.0
                    overlap = (min_distance - dist) / min_distance
                    force_x += (dx / dist) * overlap
                    force_y += (dy / dist) * overlap
            pos_i[0] += force_x * strength
            pos_i[1] += force_y * strength
        return npcs

    # ---------- 6. 记忆：随时间衰减并遗忘 ----------
    def record_sighting(self, npc, pos=None):
        """NPC看到玩家时记录位置并刷新记忆计时。"""
        seen = pos if pos is not None else self.player_pos
        npc.last_seen_player = [seen[0], seen[1]]
        npc.memory_time = 0.0

    def update_memory(self, npc, dt):
        """记忆随时间衰减；超过记忆时长则遗忘最后看到的位置。"""
        if npc.last_seen_player is None:
            npc.memory_time = 0.0
            return
        npc.memory_time += dt
        if npc.memory_time >= self.memory_duration:
            npc.last_seen_player = None
            npc.memory_time = 0.0

    def has_memory(self, npc):
        return npc.last_seen_player is not None
