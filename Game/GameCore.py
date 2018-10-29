import numpy as np
from Game import LANES, D_BITS, D_CORES, HEALTH, HEALTH_STEP, BIT_HEALTH_MULTIPLIER, CORE_HEALTH_MULTIPLIER
from Game.PlayerCore import PlayerCore
from Game.GameState import GameState


class GameCore:
    def __init__(self, p1bot, p2bot):

        self.t = 0
        self.p1 = PlayerCore()
        self.p2 = PlayerCore()
        self.bot1 = p1bot
        self.bot2 = p2bot

    def game_loop(self):

        config = {"LANES": LANES, "START_HEALTH": HEALTH}
        self.bot1.on_start(config)
        self.bot2.on_start(config)
        while self.p1.health > 0 and self.p2.health > 0:
            self.new_turn()
        self.bot1.on_game_over(self.p1.health > 0)
        self.bot2.on_game_over(self.p2.health > 0)
        if not (self.p1.health <= 0 and self.p2.health <= 0):
            print("Game Over: The winner is Bot {}".format(1 if self.p1.health > 0 else 2))
        else:
            print("Game Over: Tie")

    def new_turn(self):

        p1_state = GameState(self.p1, self.p2, self.t)
        p2_state = GameState(self.p2, self.p1, self.t)

        p1_new_atk, p1_new_def = self.bot1.on_turn(p1_state)
        p2_new_atk, p2_new_def = self.bot2.on_turn(p2_state)

        if not self.turn_valid(self.p1, p1_new_atk, p1_new_def) or \
                not self.turn_valid(self.p2, p2_new_atk, p2_new_def):
            raise ArithmeticError("stupid bot ... {}".format(
                1 if self.turn_valid(self.p2, p2_new_atk, p2_new_def) else 2
            ))

        self.update_player(self.p1, p1_new_atk, p1_new_def)
        self.update_player(self.p2, p2_new_atk, p2_new_def)

        self.resolve_combat(self.p1, self.p2)
        self.resolve_combat(self.p2, self.p1)

        self.gen_res(self.p1, self.p2)
        self.gen_res(self.p2, self.p1)

        self.t += 1

    def turn_valid(self, p, attack, defense):

        return not (np.sum(attack) > p.bits or np.sum(defense) > p.cores)

    def update_player(self, p, attack, defense):

        p.attack += attack
        p.defense += defense
        p.bits -= np.sum(attack)
        p.cores -= np.sum(defense)

    def resolve_combat(self, attacker, defender):

        delta = defender.defense - attacker.attack
        def_leftover = np.clip(delta, 0, np.inf)
        damage = np.sum(np.clip(delta, -np.inf, 0))
        attacker.reset_atk()
        defender.defense = def_leftover
        defender.health += damage

    def gen_res(self, one, other):

        one.bits += D_BITS + ((HEALTH - other.health) // HEALTH_STEP) * BIT_HEALTH_MULTIPLIER
        one.cores += D_CORES + (HEALTH - one.health) // HEALTH_STEP * CORE_HEALTH_MULTIPLIER
