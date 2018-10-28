import numpy as np
from Game import LANES

class GameState():

    def __init__(self, me, enemy, t):

        self.my_info = me
        self.enemy_info = enemy
        self.t = t