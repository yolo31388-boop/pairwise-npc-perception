"""NPC感知测试"""
import pytest, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import PerceptionSystem, NPC

class TestLOS:
    def test_line_of_sight_blocked(self):
        ps = PerceptionSystem()
        ps.obstacles = [{"x": 5, "y": 0, "r": 2}]
        assert not ps.has_line_of_sight([0,0], [10,0]), "视线遮挡未检查"

class TestAtTarget:
    def test_at_target_threshold(self):
        ps = PerceptionSystem()
        npc = NPC("n1", [0.001, 0.001])
        assert ps.at_target(npc, [0, 0], threshold=0.1), "到达判断用精确等于"
