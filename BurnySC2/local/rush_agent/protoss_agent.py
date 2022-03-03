from sc2.bot_ai import BotAI

from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId


class ProtossRushBot(BotAI):
    def __init__(self):
        super().__init__()
        self.proxy = None
        self.proxy_pylon = None
        self.proxy_built = False
        self.proxy_location = None
        self.gate_way_num = 0

    async def on_step(self, iteration):
        nexus = self.townhalls.ready.random
        self.auto_train_workers(nexus)
        await self.auto_fix_supply(nexus)
        await self.distribute_workers()
        await self.auto_chrono_boost(nexus)
        await self.build_assimilator()
        await self.build_gate_way()
        await self.build_cybernetic_core()
        await self.research_warp()
        await self.set_proxy_location()
        await self.build_proxy_structures()
        if self.proxy_built:  # warp stalker
            await self.warp_stalker(self.proxy)
            self.micro()

    def auto_train_workers(self, nexus):
        current_state_workers = self.supply_workers + self.already_pending(UnitTypeId.PROBE)
        if current_state_workers < self.townhalls.amount * 22 and nexus.is_idle:
            nexus.train(UnitTypeId.PROBE)

    async def auto_fix_supply(self, nexus):
        # auto fix supply
        if self.supply_left < 6 and self.already_pending(UnitTypeId.PYLON) == 0:
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=nexus)
            return

    async def auto_chrono_boost(self, nexus):
        # auto chrono nexus if cybernetic core is not ready, then chrono cybernetic core
        if not self.structures(UnitTypeId.CYBERNETICSCORE).ready:
            if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not nexus.is_idle:
                if nexus.energy >= 50:
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
        else:
            ccore = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
            if not ccore.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not ccore.is_idle:
                if nexus.energy >= 50:
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, ccore)

    async def build_assimilator(self):
        # Build gas
        for nexus in self.townhalls.ready:
            vespenes = self.vespene_geyser.closer_than(15, nexus)
            for vespene in vespenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.gas_buildings or not self.gas_buildings.closer_than(1, vespene):
                    worker.build(UnitTypeId.ASSIMILATOR, vespene)
                    worker.stop(queue=True)

    async def set_proxy_location(self):
        if self.structures(UnitTypeId.PYLON).ready:
            self.proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

    async def build_pylon(self, nexus):
        if self.structures(UnitTypeId.PYLON).amount < 5 and self.already_pending(UnitTypeId.PYLON) == 0:
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=nexus.position.towards(self.game_info.map_center, 5))

    async def build_gate_way(self):
        if self.structures(UnitTypeId.PYLON).ready:
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            self.gate_way_num = self.structures(UnitTypeId.WARPGATE).amount + self.structures(UnitTypeId.GATEWAY).amount
            if self.can_afford(UnitTypeId.GATEWAY) and self.gate_way_num < 3:
                await self.build(UnitTypeId.GATEWAY, near=pylon)

    async def build_cybernetic_core(self):
        if self.structures(UnitTypeId.PYLON).ready.exists:
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            if self.structures(UnitTypeId.GATEWAY).ready.exists:
                if not self.structures(UnitTypeId.CYBERNETICSCORE):
                    if (self.can_afford(UnitTypeId.CYBERNETICSCORE)
                            and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0):
                        await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)

    async def build_proxy_structures(self):
        if (self.structures(UnitTypeId.CYBERNETICSCORE).amount >= 1
                and not self.proxy_built
                and self.can_afford(UnitTypeId.PYLON)):
            self.proxy_location = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)
            await self.build(UnitTypeId.PYLON, near=self.proxy_location)
            self.proxy_pylon = self.structures(UnitTypeId.PYLON).ready.closest_to(self.proxy_location)
            # we can warp units
            self.proxy_built = True

        if self.proxy_pylon and self.gate_way_num < 4:
            await self.build(UnitTypeId.GATEWAY, near=self.proxy_location)

    async def research_warp(self):
        if (self.structures(UnitTypeId.CYBERNETICSCORE).ready
                and self.can_afford(AbilityId.RESEARCH_WARPGATE)
                and self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0):
            self.structures(UnitTypeId.CYBERNETICSCORE).ready.first.research(UpgradeId.WARPGATERESEARCH)

    async def warp_stalker(self, proxy):
        for warpgate in self.structures(UnitTypeId.WARPGATE).ready:
            abilities = await self.get_available_abilities(warpgate)
            if AbilityId.WARPGATETRAIN_STALKER in abilities:
                pos = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                if placement is None:
                    print("can't warp stalker")
                    return
                warpgate.warp_in(UnitTypeId.STALKER, placement)

    def micro(self):
        if self.units(UnitTypeId.STALKER).ready.amount > 8:
            for stalker in self.units(UnitTypeId.STALKER).ready.idle:
                targets = (self.enemy_units | self.enemy_structures).filter(lambda unit: unit.can_be_attacked)
                if targets:
                    target = targets.closest_to(stalker)
                    stalker.attack(target)
                else:
                    stalker.attack(self.enemy_start_locations[0])
