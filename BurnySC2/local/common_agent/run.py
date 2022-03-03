import random

import sc2
from sc2.player import Bot, Computer

import game_config
import protoss_agent

if __name__ == '__main__':
    map_list = random.choice([  # choose from: yourGamePath\StarCraft II\Maps
        "AbyssalReefLE", "BelShirVestigeLE", "CactusValleyLE", "HonorgroundsLE", "NewkirkPrecinctTE",
        "PaladinoTerminalLE", "ProximaStationLE", "AscensiontoAiurLE", "BloodBoilLE", "DefendersLandingLE", "OdysseyLE",
        "ProximaStationLE", "SequencerLE", "AcolyteLE", "FrostLE", "InterloperLE", "MechDepotLE",
        "BattleontheBoardwalkLE", "BlackpinkLE", "NeonVioletSquareLE", "AbiogenesisLE", "AcidPlantLE",
        "BackwaterLE", "EastwatchLE", "(4)DarknessSanctuaryLE", "(2)RedshiftLE", "(2)16-BitLE", "LostandFoundLE",
        "BlueshiftLE", "DreamcatcherLE", "ParaSiteLE", "CeruleanFallLE", "AutomatonLE", "StasisLE", "PortAleksanderLE",
        "KairosJunctionLE", "CyberForestLE", "KingsCoveLE", "NewRepugnancyLE", "YearZeroLE", "AcropolisLE",
        "ThunderbirdLE", "TurboCruise'84LE", "WintersGateLE", "WorldofSleepersLE", "EphemeronLE", "DiscoBloodbathLE"])

    sc2.run_game(sc2.maps.get(map_list),
                 [Bot(sc2.Race.Protoss,
                      protoss_agent.ProtossOperationBot(use_model=game_config.is_use_model, title="Protoss")),
                  Computer(game_config.enemy_race, game_config.enemy_difficulty)],
                 realtime=game_config.is_real_time)
