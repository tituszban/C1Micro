from Bot import Bot
from Bot.PaulBot.PaulBrain import gen_model
import numpy as np

class PaulBot(Bot):

    def __init__(self):
        self.start_health = 0
        self.lanes = 0
        self.model = None

    def on_start(self, config):
        self.start_health = config['START_HEALTH']
        self.lanes = config['LANES']
        self.model = gen_model(self.lanes)

    def on_turn(self, gs):

        pass

    def on_game_over(self, is_winner):

        pass # do yey or meh