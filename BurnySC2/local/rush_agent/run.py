import random

import sc2
from sc2.player import Bot, Computer

import protoss_agent

if __name__ == '__main__':
    enemy_race = random.choice([sc2.Race.Protoss, sc2.Race.Terran, sc2.Race.Zerg, sc2.Race.Random])
    sc2.run_game(sc2.maps.get("Simple128"),
                 [Bot(sc2.Race.Protoss, protoss_agent.ProtossRushBot()),
                  Computer(enemy_race, sc2.Difficulty.Easy)],
                 realtime=False)
