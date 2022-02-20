import sc2


class ProtossBot(sc2.BotAI):
    async def on_start(self):
        print("Game started")
        # Do things here before the game starts

    async def on_step(self, iteration):
        # Populate this order with whatever your bot should do!
        pass

    def on_end(self, result):
        print("Game ended.")
        # Do things here after the game ends
