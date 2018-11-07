from Game.GameCore import GameCore

from Bot.RandomBot import RandomBot

def main():
    bot1 = RandomBot()
    bot2 = RandomBot()

    game = GameCore(bot1, bot2)

    game.game_loop()




if __name__ == "__main__":
    main()
