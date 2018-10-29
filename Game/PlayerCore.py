import numpy as np
from Game import LANES, HEALTH


class PlayerCore:
    def __init__(self):
        self.bits = np.ceil(LANES / 2)
        self.cores = LANES * 2
        self.health = HEALTH
        self.attack = np.array([0 for _ in range(LANES)])
        self.defense = np.array([0 for _ in range(LANES)])

    def reset_atk(self):
        self.attack = np.array([0 for _ in range(LANES)])
