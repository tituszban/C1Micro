from Bot import Bot
import numpy as np
import random


class RandomBot(Bot):

    def __init__(self):
        self.start_health = 0
        self.lanes = 0

    def on_start(self, config):
        self.start_health = config['START_HEALTH']
        self.lanes = config['LANES']

    def on_turn(self, gs):
        attack = np.zeros(self.lanes, np.int32)
        defense = np.zeros(self.lanes, np.int32)

        for i in range(int(np.floor(gs.my_info.cores))):
            defense[random.randrange(self.lanes)] += 1
        for i in range(int(np.floor(gs.my_info.bits))):
            attack[random.randrange(self.lanes)] += 1
        print(attack, defense)
        return attack, defense

    def on_game_over(self, is_winner):
        pass
