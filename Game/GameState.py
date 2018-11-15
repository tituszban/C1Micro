import numpy as np
from Game import LANES
from Game.PlayerCore import PlayerCore

class GameState():
    def __init__(self, me, enemy, t):
        self.my_info = me
        self.enemy_info = enemy
        self.t = t

    @staticmethod
    def get_empty():
        return GameState(PlayerCore(), PlayerCore(), 0)
