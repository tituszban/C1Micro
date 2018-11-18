from Game.GameCore import GameCore
from Game import LANES

from Bot.RandomBot import RandomBot
from Bot.PaulBot import PaulBot, NUM_EPOCHS, SAVE_PATH


def main():
    epoch = 0
    loss = 0
    model = PaulBot.get_model(LANES)
    print(model.summary())
    experience = PaulBot.get_experience()
    while epoch < NUM_EPOCHS:
        print("Game {}".format(epoch))
        bot1 = PaulBot(epoch, loss, experience, model)
        bot2 = RandomBot()

        game = GameCore(bot1, bot2)

        game.game_loop()

        epoch += 1
        loss += bot1.loss
        model = bot1.model
        experience = bot1.experience
    model.save_weights(SAVE_PATH)


if __name__ == "__main__":
    main()
