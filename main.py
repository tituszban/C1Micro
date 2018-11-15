from Game.GameCore import GameCore

from Bot.RandomBot import RandomBot
from Bot.PaulBot import PaulBot, NUM_EPOCHS


def main():
    epoch = 0
    loss = 0
    while epoch < NUM_EPOCHS:
        print("Game {}".format(epoch))
        bot1 = PaulBot(epoch, loss)
        bot2 = RandomBot()

        game = GameCore(bot1, bot2)

        game.game_loop()

        epoch += 1
        loss += bot1.loss


if __name__ == "__main__":
    main()
