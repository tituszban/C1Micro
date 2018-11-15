from Bot import Bot
from Bot.PaulBot.PaulBrain import gen_model
import numpy as np
import collections
from Game.GameState import GameState
from os.path import isfile

GAMMA = 0.99  # decay rate of past observations
INITIAL_EPSILON = 0.1  # starting value of epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
MEMORY_SIZE = 50000  # number of previous transitions to remember
NUM_EPOCHS_OBSERVE = 100
NUM_EPOCHS_TRAIN = 2000
BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

SAVE_PATH = "./Bot/PaulBot/model.h5"

INVALID_PENALTY = -1

DONE = 0
ATTACK = 1
DEFENSE = 2

def normalize(vector):
    return vector / np.sum(vector)

class PaulBot(Bot):
    def __init__(self, start_epoch, loss):
        self.start_health = 0
        self.lanes = 0
        self.model = None
        self.experience = collections.deque(maxlen=MEMORY_SIZE)
        self.epoch = start_epoch
        self.epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS * self.epoch)
        self.loss = loss

        self.prev_gs = GameState.get_empty()
        self.prev_states = [self.gs_to_state(GameState.get_empty())]
        self.prev_actions = []

    def on_start(self, config):
        self.start_health = config['START_HEALTH']
        self.lanes = config['LANES']
        self.model = gen_model(self.lanes)
        if self.epoch > NUM_EPOCHS_OBSERVE and isfile(SAVE_PATH):
            self.model.load_weights(SAVE_PATH)

    def get_next_batch(self):
        batch_indices = np.random.randint(low=0, high=len(self.experience), size=BATCH_SIZE)
        batch = [self.experience[i] for i in batch_indices]
        X = [np.zeros((BATCH_SIZE, 7)), np.zeros((BATCH_SIZE, self.lanes, 3))]
        Y = [np.zeros((BATCH_SIZE, 3)), np.zeros((BATCH_SIZE, self.lanes))]
        for i in range(len(batch)):
            s_t, a_t, s_tp1, r_t, game_over = batch[i]
            info, gamestate = s_t
            X[0][i] = info
            X[1][i] = gamestate.T
            action, placement = self.predict(s_t)
            Y[0][i] = action
            Y[1][i] = placement
            Q_pred = self.predict(s_tp1)
            Q_sa = (np.max(Q_pred[0]) + np.max(Q_pred[1])) / 2

            if game_over:
                Y[0][i, a_t[0]] = r_t
                Y[1][i, a_t[1]] = r_t
            else:
                Y[0][i, a_t[0]] = r_t + GAMMA * Q_sa
                Y[1][i, a_t[1]] = r_t + GAMMA * Q_sa

        return X, Y

    def gs_to_state(self, gs):
        info = np.array([
            gs.t,
            gs.my_info.health,
            gs.my_info.cores,
            gs.my_info.bits,
            gs.enemy_info.health,
            gs.enemy_info.cores,
            gs.enemy_info.bits])
        gamestate = np.stack((gs.my_info.attack, gs.my_info.defense, gs.enemy_info.defense))
        return [info, gamestate]

    def store_experience(self, gs, game_over):
        delta_my_healt = gs.my_info.health - self.prev_gs.my_info.health
        delta_enemy_health = gs.enemy_info.health - self.prev_gs.enemy_info.health

        # TODO: reward
        reward = 0

        assert len(self.prev_actions) + 1 == len(self.prev_states)

        for i in range(len(self.prev_actions)):
            action, placement, is_valid = self.prev_actions[i]
            self.experience.append(
                (self.prev_states[i], (action, placement), self.prev_states[i + 1],
                 reward - (INVALID_PENALTY if is_valid else 0), game_over))


        if self.epoch > NUM_EPOCHS_OBSERVE:
            X, Y = self.get_next_batch()
            #self.loss += self.model.train_on_batch(X, Y)  # TODO how does this work?
            self.model.train_on_batch(X, Y)

        self.prev_states = [self.prev_states[-1]]
        self.prev_actions = []

    def get_random_action(self):
        out_action = np.array([0, 20, 40])
        out_placement = np.zeros((self.lanes,))
        out_placement[np.random.choice(list(range(self.lanes)))] = 1
        return out_action, out_placement

    def predict(self, state):
        out_action, out_placement = self.model.predict(
            [np.reshape(state[0], (1, *state[0].shape)),
             np.reshape(state[1].T, (1, *state[1].T.shape))])
        out_action = out_action[0]
        out_placement = out_placement[0]
        return [out_action, out_placement]

    def on_turn(self, gs):
        if len(self.prev_actions) > 0 and len(self.prev_states) > 0:
            self.store_experience(gs, False)

        attack, defense = np.zeros((self.lanes,), np.int32), np.zeros((self.lanes,), np.int32)
        state = self.gs_to_state(gs)

        # gs -> s_t
        done = False
        while not done:
            if self.epoch <= NUM_EPOCHS_OBSERVE:
                out_action, out_placement = self.get_random_action()
            else:
                if np.random.rand() <= self.epsilon:
                    out_action, out_placement = self.get_random_action()
                else:
                    out_action, out_placement = self.predict(state)
                    #print(out_action, out_placement)

            # Update state, attack, defense
            is_valid = False
            action = 0
            placement = np.argmax(out_placement)
            while not is_valid:
                action = np.random.choice(3, 1, p=normalize(1 + out_action - np.min(out_action)))[0]
                if action == DONE:
                    done = True
                    is_valid = True
                elif action == ATTACK:
                    if state[0][3] > 0:
                        is_valid = True
                        state[0][3] -= 1  # bits - 1
                        state[1][0][placement] += 1  # attack[placement] + 1
                        attack[placement] += 1
                elif action == DEFENSE:
                    if state[0][2] > 0:
                        is_valid = True
                        state[0][2] -= 1  # cores -1
                        state[1][1][placement] += 1  # defense[placement] + 1
                        defense[placement] += 1
                lst_valid_found = False
                duplicate_found = False
                i = 0
                if len(self.prev_actions) > 0 and not is_valid:
                    while not lst_valid_found and not duplicate_found and i <= len(self.prev_actions):
                        if self.prev_actions[-i][2]:
                            lst_valid_found = True
                        elif self.prev_actions[-i][0] == action:
                            duplicate_found = True
                        i += 1
                if not duplicate_found:
                    self.prev_actions.append((action, placement, is_valid))
                    self.prev_states.append(state)
                # if not is_valid:
                #     print("Invalid")

        self.prev_gs = gs
        return attack, defense

    def on_game_over(self, gs, is_winner):
        self.store_experience(gs, True)
        if self.epoch > NUM_EPOCHS_OBSERVE:
            self.model.save_weights(SAVE_PATH)
