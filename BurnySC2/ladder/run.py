import sys

import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer

from __init__ import run_ladder_game
# Load bot
from ladder_agent import LadderBot

bot = Bot(Race.Protoss, LadderBot())

# Start game
if __name__ == "__main__":
    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        print("Starting local_data game...")
        sc2.run_game(sc2.maps.get("ThunderbirdLE"), [bot, Computer(Race.Protoss, Difficulty.VeryHard)], realtime=True)
