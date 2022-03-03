import itertools
import math
import random
import time

import numpy as np
from keras.models import load_model
from sc2 import Result
from sc2 import position
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.upgrade_id import UpgradeId
from sc2.units import UnitTypeId

import game_config
import game_data

HEADLESS = False


class ProtossOperationBot(BotAI):
    def __init__(self, use_model=False, title="title"):
        super().__init__()
        # Global
        self.title = title
        # Model
        self.flipped = None
        self.use_model = use_model
        self.train_data = []
        if self.use_model:
            print("--- Use Model To Play ---")
            self.model = load_model(game_config.model_path)

        # Game
        self.can_warp = False
        self.observers_and_spots = {}
        self.enemy_expand_dis_dir = {}  # base expand_position list
        self.ordered_expansions = None
        self.ordered_exp_distances = None
        self.clear_map = None  # is map clear
        self.army_target = None  # unit selects which targets to attack
        # Structures groups
        self.all_nexus = None
        self.all_pylon = None
        self.all_power_pylon = None
        self.power_pylon_closet_enemy = None
        self.all_forge = None
        self.all_cybernetics_core = None
        self.all_robotics_facility = None
        self.all_star_gate = None
        # Unit groups
        self.all_observer = None  # Units group
        self.all_oracle = None
        self.all_warp_prism = None
        self.all_disruptor = None
        self.tactical_army = None
        # Basic build order
        self.current_build_order_step = 0
        self.build_order = [UnitTypeId.PROBE,
                            UnitTypeId.PYLON,
                            UnitTypeId.PROBE,
                            UnitTypeId.PROBE,
                            UnitTypeId.GATEWAY,
                            UnitTypeId.PROBE,
                            UnitTypeId.ASSIMILATOR,
                            UnitTypeId.ASSIMILATOR,
                            UnitTypeId.PROBE,
                            UnitTypeId.PROBE,
                            UnitTypeId.CYBERNETICSCORE,
                            "END"]
        # Total order
        self.choices = {0: self.build_nexus,
                        1: self.build_pylon,
                        2: self.build_assimilator,
                        3: self.build_gateway,
                        4: self.build_cybernetic_core,
                        5: self.build_shield_battery,
                        6: self.build_forge,
                        7: self.build_photon_cannon,
                        8: self.build_twilight_council,
                        9: self.build_templar_archive,
                        10: self.build_dark_shrine,
                        11: self.build_robotics_facility,
                        12: self.build_robotics_bay,
                        13: self.build_stargate,
                        14: self.build_fleet_beacon,
                        15: self.upgrade_warp_gate,
                        16: self.upgrade_stalker,
                        17: self.train_bg_zealot,
                        18: self.train_bg_stalker,
                        19: self.train_bg_sentry,
                        20: self.train_bg_adept,
                        21: self.train_vr_observer,
                        22: self.train_vr_warp_prism,
                        23: self.train_vr_immortal,
                        24: self.train_vr_disruptor,
                        25: self.train_vr_colossus,
                        26: self.train_vs_phoenix,
                        27: self.train_vs_oracle,
                        28: self.train_vs_void_ray,
                        29: self.train_vs_tempest,
                        30: self.train_vs_carrier,
                        31: self.micro}

    async def on_step(self, iteration):
        await self.create_unit_groups()
        await self.distribute_workers()  # auto split workers
        await self.ai_pre_build_order()
        # when build order done, let AI chose by self
        if self.build_order[self.current_build_order_step] == "END":
            self.auto_train_worker(self.all_nexus.random)
            await self.auto_chrono_boost(self.all_nexus.random)
            await self.ai_do_something()
            await self.observer_scout()
            self.set_army_target()
            self.micro()

    async def create_unit_groups(self):
        # structure_info will show the structure info included current queue and building progress
        if game_config.is_show_structure_info:
            game_data.structure_info(self)
        # intel_report will show the game brief included current minerals, gases and supply, also the enemies brief
        if game_config.is_show_intel_report:
            game_data.simple_intel_report(self, HEADLESS)
        # Expand location list
        self.ordered_expansions = sorted(self.expansion_locations.keys(),
                                         key=lambda expansion: expansion.distance_to(self.start_location))

        # Create groups, included structures and units
        self.all_nexus = self.structures(UnitTypeId.NEXUS).ready  # Structures
        self.all_pylon = self.structures(UnitTypeId.PYLON).ready
        if self.all_pylon:  # Add all powered pylon (reduce warp time and provide power field)
            self.all_power_pylon = self.all_pylon.ready.filter(lambda pylon: pylon.is_powered)
            self.power_pylon_closet_enemy = self.structures(UnitTypeId.PYLON).ready.filter(
                lambda pylon: self.all_pylon.closest_to(self.enemy_start_locations[0]))
        self.all_forge = self.structures(UnitTypeId.FORGE).ready
        self.all_cybernetics_core = self.structures(UnitTypeId.CYBERNETICSCORE).ready
        self.all_robotics_facility = self.structures(UnitTypeId.ROBOTICSFACILITY).ready
        self.all_star_gate = self.structures(UnitTypeId.STARGATE).ready
        self.all_observer = self.units(UnitTypeId.OBSERVER).ready  # Army
        self.all_oracle = self.units(UnitTypeId.ORACLE).ready
        self.all_warp_prism = self.units(UnitTypeId.WARPPRISM).ready
        self.all_disruptor = self.units(UnitTypeId.DISRUPTOR).ready
        self.tactical_army = self.units.filter(  # Tactical army
            lambda unit: unit.type_id in {UnitTypeId.SENTRY, UnitTypeId.HIGHTEMPLAR}).ready

    # each game, AI should follow the pre_build_order at once, then choose by AI-self
    async def ai_pre_build_order(self):
        self._client.game_step = 2
        # no enough minerals
        if self.minerals < 50:
            return
        # track current_step
        current_step = self.build_order[self.current_build_order_step]
        if current_step == "END" or not self.can_afford(current_step):
            return
        if current_step == UnitTypeId.PROBE:  # Probe
            self.structures(UnitTypeId.NEXUS).ready.first.train(current_step)
        if current_step == UnitTypeId.PYLON:  # BE
            await self.build_pylon()
            await self.auto_chrono_boost(self.structures(UnitTypeId.NEXUS).ready.random)
        if current_step == UnitTypeId.ASSIMILATOR:  # BA
            await self.build_assimilator()
        if current_step == UnitTypeId.GATEWAY:  # BG
            if self.all_pylon.exists:
                await self.build_gateway()
        if current_step == UnitTypeId.CYBERNETICSCORE:  # BY
            if self.structures(UnitTypeId.GATEWAY).ready.exists:
                await self.build_cybernetic_core()
        # system will auto distribute workers into resources field
        await self.distribute_workers()
        # when each order is in progress, move to next build order step
        print(f"{self.time_formatted} STEP {self.current_build_order_step:2} {current_step.name}")
        self.current_build_order_step += 1

    async def ai_do_something(self):
        if self.use_model:
            # let AI choose next order
            prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
            choice = np.argmax(prediction[0])
        else:
            # Build order
            build_nexus_weight = 1  # BN
            build_pylons_weight = 3  # BE
            build_gas_collector_weight = 1  # BA
            build_gateway_weight = 3  # BG
            build_cybernetic_core_weight = 1  # BY
            build_shield_battery_weight = 1  # BB
            build_forge_weight = 1  # BF
            build_photon_cannon_weight = 1  # BC
            build_twilight_council_weight = 3  # VC
            build_templar_archive_weight = 1  # VT
            build_dark_archive_weight = 1  # VD
            build_robotics_facility_weight = 3  # VR
            build_robotics_bay_weight = 1  # VB
            build_stargate_weight = 3  # VS
            build_fleet_beacon_weight = 1  # VF
            # Upgrade order
            upgrade_warp_gate_weight = 3
            upgrade_blink_weight = 1
            # Train order
            train_bg_zealot_weight = 3
            train_bg_stalker_weight = 3
            train_bg_sentry_weight = 1
            train_bg_adept_weight = 1
            train_vr_observer_weight = 1
            train_vr_warp_prism_weight = 1
            train_vr_immortal_weight = 3
            train_vr_disruptor_weight = 1
            train_vr_colossus_weight = 3
            train_vs_phoenix_weight = 1
            train_vs_oracle_weight = 1
            train_vs_void_ray_weight = 3
            train_vs_tempest_weight = 3
            train_vs_carrier_weight = 3
            # Warp order
            warp_bg_zealot_weight = 1
            warp_bg_stalker_weight = 1
            warp_bg_sentry_weight = 1
            warp_bg_high_templar_weight = 1
            warp_bg_dark_templar_weight = 1
            # Unit order
            micro_control_weight = 6
            choice_weights = (build_nexus_weight * [0]
                              + build_pylons_weight * [1]  # BE
                              + build_gas_collector_weight * [2]  # BA
                              + build_gateway_weight * [3]  # BG
                              + build_cybernetic_core_weight * [4]  # BY
                              + build_shield_battery_weight * [5]  # BB
                              + build_forge_weight * [6]  # BF
                              + build_photon_cannon_weight * [7]  # BC
                              + build_twilight_council_weight * [8]  # VC
                              + build_templar_archive_weight * [9]  # VT
                              + build_dark_archive_weight * [10]  # VD
                              + build_robotics_facility_weight * [11]  # VR
                              + build_robotics_bay_weight * [12]  # VB
                              + build_stargate_weight * [13]  # VS
                              + build_fleet_beacon_weight * [14]  # VF
                              + upgrade_warp_gate_weight * [15]
                              + upgrade_blink_weight * [16]
                              + train_bg_zealot_weight * [17]
                              + train_bg_stalker_weight * [18]
                              + train_bg_sentry_weight * [19]
                              + train_bg_adept_weight * [20]
                              + train_vr_observer_weight * [21]
                              + train_vr_warp_prism_weight * [22]
                              + train_vr_immortal_weight * [23]
                              + train_vr_disruptor_weight * [24]
                              + train_vr_colossus_weight * [25]
                              + train_vs_phoenix_weight * [26]
                              + train_vs_oracle_weight * [27]
                              + train_vs_void_ray_weight * [28]
                              + train_vs_tempest_weight * [29]
                              + train_vs_carrier_weight * [30]
                              # + warp_bg_zealot_weight * [34]
                              # + warp_bg_stalker_weight * [35]
                              # + warp_bg_sentry_weight * [36]
                              # + warp_bg_high_templar_weight * [37]
                              # + warp_bg_dark_templar_weight * [38]
                              + micro_control_weight * [31])

            game_config.total_order_num = len(self.choices)
            choice = random.choice(choice_weights)
        try:
            await self.choices[choice]()
        except Exception as exception:
            print("EXCEPTION | Order ID: {} - ".format(choice) + str(exception))

        # Model
        y = np.zeros(len(self.choices))  # collect the information of AI order list
        y[choice] = 1
        self.train_data.append([y, self.flipped])

    # -----------------------------------------------------------------------------------------------------------------
    # Basic order
    # -----------------------------------------------------------------------------------------------------------------

    async def auto_chrono_boost(self, nexus):
        abilities = await self.get_available_abilities(self.all_nexus)
        if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
            for nexus_speed_up, abilities_nexus in zip(self.all_nexus, abilities):
                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                    nexus_speed_up(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                    break

    def auto_train_worker(self, nexus):
        current_state_workers = self.supply_workers + self.already_pending(UnitTypeId.PROBE)
        if current_state_workers < self.townhalls.amount * 22 and nexus.is_idle:
            nexus.train(UnitTypeId.PROBE)

    # TODO: no in used
    def auto_fix_no_enough_supply(self, nexus):
        # Build pylon when on low supply
        if self.supply_left < 2 and self.already_pending(UnitTypeId.PYLON) == 0:
            # Always check if you can afford something before you build it
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=nexus)
            return

    # -----------------------------------------------------------------------------------------------------------------
    # Build order (Structure)
    # -----------------------------------------------------------------------------------------------------------------

    # BN (also used to base expanded)
    async def build_nexus(self):
        try:
            if self.can_afford(UnitTypeId.NEXUS):
                await self.expand_now()
        except Exception as exception:
            print("EXCEPTION | expand_now: ", str(exception))

    # BE
    async def build_pylon(self):
        # don't block the way to the mineral filed
        pylon_build_position = self.structures(UnitTypeId.NEXUS).random.position.towards(self.game_info.map_center, 5)
        if self.supply_left < 4:
            if self.structures(UnitTypeId.NEXUS).ready:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=pylon_build_position)

    # BA
    async def build_assimilator(self):
        for nexus in self.all_nexus:
            vaspenes = self.vespene_geyser.closer_than(16, nexus)  # closest vaspenes gas location
            for vaspene in vaspenes:
                worker = self.select_build_worker(vaspene.position)
                if not self.can_afford(UnitTypeId.ASSIMILATOR):  # we don't have enough minerals
                    break
                if worker is None:  # there is no worker between the way to vaspenes
                    break
                if not self.structures(UnitTypeId.ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    worker.build(UnitTypeId.ASSIMILATOR, vaspene)

    # BG
    async def build_gateway(self):
        if self.structures(UnitTypeId.GATEWAY).amount < 8:
            if self.can_afford(UnitTypeId.GATEWAY):
                await self.build(UnitTypeId.GATEWAY, near=self.all_pylon.random)

    # BY
    async def build_cybernetic_core(self):
        if self.all_gate_way.exists:
            if not self.all_cybernetics_core.exists or self.all_cybernetics_core.amount < 2:
                if (self.can_afford(UnitTypeId.CYBERNETICSCORE)
                        and not self.already_pending(UnitTypeId.CYBERNETICSCORE)):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=self.all_pylon.random)

    # BB
    async def build_shield_battery(self):
        photo_canon = self.structures(UnitTypeId.PHOTONCANNON).exists.close_to(self.all_nexus.random, 12)
        shield_battery_build_position = [self.all_pylon.random, photo_canon]
        if self.all_cybernetics_core.exists:
            if self.can_afford(UnitTypeId.SHIELDBATTERY):
                await self.build(UnitTypeId.SHIELDBATTERY, near=random.choice(shield_battery_build_position))

    # BF
    async def build_forge(self):
        if not self.all_forge.exists or self.all_forge.amount <= 2:
            if self.can_afford(UnitTypeId.FORGE) and not self.already_pending(UnitTypeId.FORGE):
                await self.build(UnitTypeId.FORGE, near=self.all_pylon.random)

    # BC
    async def build_photon_cannon(self):
        pylon = self.structures(UnitTypeId.PYLON).ready.closest_to(UnitTypeId.NEXUS, 12)
        shield_battery = self.structures(UnitTypeId.SHIELDBATTERY).exists.closest_to(self.all_nexus.random, 12)
        canon_build_position = [pylon, shield_battery]
        if self.all_forge.exists:
            if self.can_afford(UnitTypeId.PHOTONCANNON):
                await self.build(UnitTypeId.PHOTONCANNON, near=random.choice(canon_build_position))

    # VC
    async def build_twilight_council(self):
        if self.all_cybernetics_core.exists:
            if (not self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists
                    and not self.already_pending(UnitTypeId.TWILIGHTCOUNCIL)):
                if self.can_afford(UnitTypeId.TWILIGHTCOUNCIL):
                    await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=self.all_pylon.random)

    # VT
    async def build_templar_archive(self):
        templar_archive = self.structures(UnitTypeId.TEMPLARARCHIVE)
        if self.all_cybernetics_core.exists:
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists:
                if not templar_archive.ready.exists and not self.already_pending(UnitTypeId.TEMPLARARCHIVE):
                    if self.can_afford(UnitTypeId.TEMPLARARCHIVE):
                        await self.build(UnitTypeId.TEMPLARARCHIVE, near=self.all_pylon.random)

    # VD
    async def build_dark_shrine(self):
        dark_shrine = self.structures(UnitTypeId.DARKSHRINE)
        if self.all_cybernetics_core.exists:
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists:
                if not dark_shrine.ready.exists and not self.already_pending(UnitTypeId.DARKSHRINE):
                    if self.can_afford(UnitTypeId.DARKSHRINE):
                        await self.build(UnitTypeId.DARKSHRINE, near=self.all_pylon.random)

    # VR
    async def build_robotics_facility(self):
        if not self.all_cybernetics_core.exists:
            if (not self.all_robotics_facility.exists
                    and not self.already_pending(UnitTypeId.ROBOTICSFACILITY)
                    and self.all_robotics_facility.amount < game_config.vr_num):
                if self.can_afford(UnitTypeId.ROBOTICSFACILITY):
                    await self.build(UnitTypeId.ROBOTICSFACILITY, near=self.all_pylon.random)

    # VB
    async def build_robotics_bay(self):
        robotics_bay = self.structures(UnitTypeId.ROBOTICSBAY)
        if self.all_cybernetics_core.exists:
            if self.all_robotics_facility.exists:
                if not robotics_bay.ready.exists and not self.already_pending(UnitTypeId.ROBOTICSBAY):
                    if self.can_afford(UnitTypeId.ROBOTICSBAY):
                        await self.build(UnitTypeId.ROBOTICSBAY, near=self.all_pylon.random)

    # VS
    async def build_stargate(self):
        if self.all_cybernetics_core.exists:
            if (not self.all_star_gate.exists
                    and not self.already_pending(UnitTypeId.STARGATE)
                    and self.all_star_gate.amount < game_config.vs_num):
                if self.can_afford(UnitTypeId.STARGATE):
                    await self.build(UnitTypeId.STARGATE, near=self.all_pylon.random)

    # VF
    async def build_fleet_beacon(self):
        fleet_beacon = self.structures(UnitTypeId.FLEETBEACON)
        if self.all_cybernetics_core.exists:
            if self.all_star_gate.exists:
                if not fleet_beacon.ready.exists and not self.already_pending(UnitTypeId.FLEETBEACON):
                    if self.can_afford(UnitTypeId.FLEETBEACON):
                        await self.build(UnitTypeId.FLEETBEACON, near=self.all_pylon.random)

    # -----------------------------------------------------------------------------------------------------------------
    # Upgrade order
    # -----------------------------------------------------------------------------------------------------------------

    # BY: Warp
    async def upgrade_warp_gate(self):
        if self.all_cybernetics_core.exists:
            if (self.can_afford(AbilityId.RESEARCH_WARPGATE)
                    and self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0):
                self.all_cybernetics_core.first.research(UpgradeId.WARPGATERESEARCH)

    # VC: Zealot
    async def upgrade_zealot(self):
        if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists:
            if self.can_afford(AbilityId.RESEARCH_CHARGE) and self.already_pending_upgrade(UpgradeId.CHARGE) == 0:
                self.structures(UnitTypeId.TWILIGHTCOUNCIL).researsh(UpgradeId.CHARGE)

    # VC: Stalker
    async def upgrade_stalker(self):
        if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists:
            if self.can_afford(AbilityId.RESEARCH_BLINK) and self.already_pending_upgrade(UpgradeId.BLINKTECH) == 0:
                self.structures(UnitTypeId.TWILIGHTCOUNCIL).researsh(UpgradeId.BLINKTECH)

    # VC: Adept
    async def upgrade_adept(self):
        if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.exists:
            if (self.can_afford(AbilityId.RESEARCH_ADEPTRESONATINGGLAIVES)
                    and self.already_pending_upgrade(UpgradeId.ADEPTPIERCINGATTACK) == 0):
                self.structures(UnitTypeId.TWILIGHTCOUNCIL).researsh(UpgradeId.ADEPTPIERCINGATTACK)

    async def upgrade_ground_weapon_armor(self):
        if self.all_forge.exists:
            if (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1)  # Weapon
                    and self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) == 0):
                self.all_forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1)

            elif (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) == 0):
                self.all_forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2)

            elif (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) == 0):
                self.all_forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3)

            elif (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1)  # Armor
                  and self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) == 0):
                self.all_forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1)

            elif (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL2)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) == 0):
                self.all_forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2)

            elif (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) == 0):
                self.all_forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3)

    async def upgrade_air_weapon_armor(self):
        if self.all_cybernetics_core.exists:
            if (self.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1)  # Weapon
                    and self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) == 0):
                self.all_cybernetics_core.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL1)

            elif (self.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL2) == 0):
                self.all_cybernetics_core.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL2)

            elif (self.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL3) == 0):
                self.all_cybernetics_core.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL3)

            elif (self.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1)  # Armor
                  and self.already_pending_upgrade(UpgradeId.PROTOSSAIRARMORSLEVEL1) == 0):
                self.all_cybernetics_core.research(UpgradeId.PROTOSSAIRARMORSLEVEL1)

            elif (self.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSAIRARMORSLEVEL2) == 0):
                self.all_cybernetics_core.research(UpgradeId.PROTOSSAIRARMORSLEVEL2)

            elif (self.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSAIRARMORSLEVEL3) == 0):
                self.all_cybernetics_core.research(UpgradeId.PROTOSSAIRARMORSLEVEL3)

    async def upgrade_all_shield(self):
        if self.all_forge.exists:
            if (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1)  # Weapon
                    and self.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL1) == 0):
                self.all_forge.research(UpgradeId.PROTOSSSHIELDSLEVEL1)

            elif (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL2) == 0):
                self.all_forge.research(UpgradeId.PROTOSSSHIELDSLEVEL2)

            elif (self.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3)
                  and self.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL3) == 0):
                self.all_forge.research(UpgradeId.PROTOSSSHIELDSLEVEL3)

    # -----------------------------------------------------------------------------------------------------------------
    # Train order
    # -----------------------------------------------------------------------------------------------------------------
    # Unit - BG (only be used for early game, then will be replaced by the warp order)
    # -----------------------------------------------------------------------------------------------------------------

    # Zealot
    async def train_bg_zealot(self):
        if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.can_warp:
            if self.can_afford(UnitTypeId.ZEALOT) and self.supply_left > 2:
                random.choice(self.structures(UnitTypeId.GATEWAY).ready.idle).train(UnitTypeId.ZEALOT)

    # Stalker
    async def train_bg_stalker(self):
        if self.structures(UnitTypeId.GATEWAY).ready.exists and self.all_cybernetics_core.exists:
            if self.can_afford(UnitTypeId.STALKER) and self.supply_left > 2:
                random.choice(self.structures(UnitTypeId.GATEWAY).ready.idle).train(UnitTypeId.STALKER)

    # Sentry
    async def train_bg_sentry(self):
        if self.structures(UnitTypeId.GATEWAY).ready.exists and self.all_cybernetics_core.exists:
            if self.can_afford(UnitTypeId.SENTRY) and self.supply_left > 2:
                random.choice(self.structures(UnitTypeId.GATEWAY).ready.idle).train(UnitTypeId.SENTRY)

    # Adept
    async def train_bg_adept(self):
        if self.structures(UnitTypeId.GATEWAY).ready.exists and self.all_cybernetics_core.exists:
            if self.can_afford(UnitTypeId.ADEPT) and self.supply_left > 2:
                random.choice(self.structures(UnitTypeId.GATEWAY).ready.idle).train(UnitTypeId.ADEPT)

    # -----------------------------------------------------------------------------------------------------------------
    # Unit - VR
    # -----------------------------------------------------------------------------------------------------------------

    # Observer
    async def train_vr_observer(self):
        observer = self.all_observer
        if self.all_cybernetics_core.exists:
            if self.all_robotics_facility.exists or observer.amount < math.floor(self.time / 3):
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 1:
                    random.choice(self.all_robotics_facility.idle).train(UnitTypeId.OBSERVER)

    # Warp prism
    async def train_vr_warp_prism(self):
        if self.all_cybernetics_core.exists:
            if self.all_robotics_facility.exists:
                if self.can_afford(UnitTypeId.WARPPRISM) and self.supply_left > 2:
                    random.choice(self.all_robotics_facility.idle).train(UnitTypeId.WARPPRISM)

    # Immortal
    async def train_vr_immortal(self):
        if self.all_cybernetics_core.exists:
            if self.all_robotics_facility.exists:
                if self.can_afford(UnitTypeId.IMMORTAL) and self.supply_left > 3:
                    random.choice(self.all_robotics_facility.idle).train(UnitTypeId.IMMORTAL)

    # Disruptor
    async def train_vr_disruptor(self):
        if self.all_cybernetics_core.exists:
            if self.all_robotics_facility.exists:
                if self.can_afford(UnitTypeId.DISRUPTOR) and self.supply_left > 3:
                    random.choice(self.all_robotics_facility.idle).train(UnitTypeId.DISRUPTOR)

    # Colossus
    async def train_vr_colossus(self):
        if self.all_cybernetics_core.exists:
            if self.all_robotics_facility.exists:
                if self.can_afford(UnitTypeId.COLOSSUS) and self.supply_left > 6:
                    random.choice(self.all_robotics_facility.idle).train(UnitTypeId.COLOSSUS)

    # -----------------------------------------------------------------------------------------------------------------
    # Unit - VS
    # -----------------------------------------------------------------------------------------------------------------

    # Phoenix
    async def train_vs_phoenix(self):
        if self.all_cybernetics_core.exists:
            if self.all_star_gate.exists:
                if self.can_afford(UnitTypeId.PHOENIX) and self.supply_left > 2:
                    random.choice(self.all_star_gate.idle).train(UnitTypeId.PHOENIX)

    # Oracle
    async def train_vs_oracle(self):
        oracle = self.units(UnitTypeId.ORACLE)
        if self.all_cybernetics_core.exists:
            if self.all_star_gate.exists:
                if self.can_afford(UnitTypeId.ORACLE) and self.supply_left > 2 or oracle.amount <= 3:
                    random.choice(self.all_star_gate.idle).train(UnitTypeId.ORACLE)

    # Void way
    async def train_vs_void_ray(self):
        if self.all_cybernetics_core.exists:
            if self.all_star_gate.exists:
                if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 3:
                    random.choice(self.all_star_gate.idle).train(UnitTypeId.VOIDRAY)

    # Tempest
    async def train_vs_tempest(self):
        fleet_beacon = self.structures(UnitTypeId.FLEETBEACON)
        if self.all_cybernetics_core.exists:
            if self.all_star_gate.exists and fleet_beacon.ready.exists:
                if self.can_afford(UnitTypeId.TEMPEST) and self.supply_left > 4:
                    random.choice(self.all_star_gate.idle).train(UnitTypeId.TEMPEST)

    # Carrier
    async def train_vs_carrier(self):
        fleet_beacon = self.structures(UnitTypeId.FLEETBEACON)
        if self.all_cybernetics_core.exists:
            if self.all_star_gate.exists and fleet_beacon.ready.exists:
                if self.can_afford(UnitTypeId.CARRIER) and self.supply_left > 6:
                    random.choice(self.all_star_gate.idle).train(UnitTypeId.CARRIER)

    # Mothership
    async def train_nexus_mothership(self):
        fleet_beacon = self.structures(UnitTypeId.FLEETBEACON)
        if self.all_cybernetics_core.exists:
            if self.all_star_gate.exists and fleet_beacon.ready.exists:
                if self.can_afford(UnitTypeId.MOTHERSHIP) and self.supply_left > 8:
                    random.choice(self.all_nexus.idle).train(UnitTypeId.MOTHERSHIP)

    # TODO: need to be fixed
    # -----------------------------------------------------------------------------------------------------------------
    # Warp order
    # -----------------------------------------------------------------------------------------------------------------

    # Zealot
    async def warp_bg_zealot(self, proxy):
        for warp_gates in self.structures(UnitTypeId.WARPGATE).ready:
            warp_abilities = await self.get_available_abilities(warp_gates)
            if AbilityId.WARPGATETRAIN_ZEALOT in warp_abilities:
                warp_position = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_ZEALOT, warp_position, placement_step=1)
                if placement is None:
                    print("Can't warp ZEALOT")
                    return
                warp_gates.warp_in(UnitTypeId.ZEALOT, placement)

    # Stalker
    async def warp_bg_stalker(self, proxy):
        for warp_gates in self.structures(UnitTypeId.WARPGATE).ready:
            warp_abilities = await self.get_available_abilities(warp_gates)
            if AbilityId.WARPGATETRAIN_STALKER in warp_abilities:
                warp_position = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, warp_position, placement_step=1)
                if placement is None:
                    print("Can't warp STALKER")
                    return
                warp_gates.warp_in(UnitTypeId.STALKER, placement)

    # Sentry
    async def warp_bg_sentry(self, proxy):
        for warp_gates in self.structures(UnitTypeId.WARPGATE).ready:
            warp_abilities = await self.get_available_abilities(warp_gates)
            if AbilityId.WARPGATETRAIN_SENTRY in warp_abilities:
                warp_position = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_SENTRY, warp_position, placement_step=1)
                if placement is None:
                    print("Can't warp SENTRY")
                    return
                warp_gates.warp_in(UnitTypeId.SENTRY, placement)

    # High Templar
    async def warp_bg_high_templar(self, proxy):
        for warp_gates in self.structures(UnitTypeId.WARPGATE).ready:
            warp_abilities = await self.get_available_abilities(warp_gates)
            if AbilityId.WARPGATETRAIN_HIGHTEMPLAR in warp_abilities:
                warp_position = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_HIGHTEMPLAR, warp_position,
                                                      placement_step=1)
                if placement is None:
                    print("Can't warp HIGH-TEMPLAR")
                    return
                warp_gates.warp_in(UnitTypeId.HIGHTEMPLAR, placement)

    # Dark Tamplar
    async def warp_bg_dark_templar(self, proxy):
        for warp_gates in self.structures(UnitTypeId.WARPGATE).ready:
            warp_abilities = await self.get_available_abilities(warp_gates)
            if AbilityId.WARPGATETRAIN_DARKTEMPLAR in warp_abilities:
                warp_position = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_DARKTEMPLAR, warp_position,
                                                      placement_step=1)
                if placement is None:
                    print("Can't warp DARK-TEMPLAR")
                    return
                warp_gates.warp_in(UnitTypeId.DARKTEMPLAR, placement)

    # -----------------------------------------------------------------------------------------------------------------
    # Unit order
    # -----------------------------------------------------------------------------------------------------------------

    # Scout
    async def observer_scout(self):
        for expansion_location in self.expansion_locations_list:
            distance_to_enemy_start = expansion_location.distance_to(self.enemy_start_locations[0])
            self.enemy_expand_dis_dir[distance_to_enemy_start] = expansion_location

        self.ordered_exp_distances = sorted(k for k in self.enemy_expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]

        # when the observer was destroyed by the enemies
        to_be_removed = []

        for noted_ob in self.observers_and_spots:
            if noted_ob not in existing_ids:
                to_be_removed.append(noted_ob)

        for ob in to_be_removed:
            del self.observers_and_spots[ob]

        if self.structures(UnitTypeId.ROBOTICSFACILITY).ready.amount == 0:
            unit_type = UnitTypeId.PROBE
            unit_limit = 1
        else:
            unit_type = UnitTypeId.OBSERVER
            unit_limit = game_config.total_ob_num

        assign_ob = True

        # we use probe to scout for early game
        if unit_type == UnitTypeId.PROBE:
            for unit in self.units(UnitTypeId.PROBE):
                if unit.tag in self.observers_and_spots:
                    assign_ob = False

        if assign_ob:
            if self.units(unit_type).idle.amount > 0:
                for observer in self.units(unit_type).idle[:unit_limit]:
                    if observer.tag not in self.observers_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = next(
                                    value for key, value in self.enemy_expand_dis_dir.items() if key == dist)
                                # DICT {UNIT_ID:LOCATION}
                                active_locations = [self.observers_and_spots[k] for k in self.observers_and_spots]

                                if location not in active_locations:
                                    if unit_type == UnitTypeId.PROBE:
                                        for unit in self.units(UnitTypeId.PROBE):
                                            if unit.tag in self.observers_and_spots:
                                                continue

                                    self.do(observer.move(location))
                                    self.observers_and_spots[observer.tag] = location
                                    break
                            except Exception as exception:
                                print("EXCEPTION | Observer scout: ", str(exception))

        for observer in self.units(unit_type):
            if observer.tag in self.observers_and_spots:
                if observer in [probe for probe in self.units(UnitTypeId.PROBE)]:
                    self.do(observer.move(self.random_location_variance(self.observers_and_spots[observer.tag])))

    # Random choose location
    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            print("x below")
            x = 0
        if y < 0:
            print("y below")
            y = 0
        if x > self.game_info.map_size[0]:
            print("x above")
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            print("y above")
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    # Find target to attack
    def set_army_target(self):
        if not self.clear_map:
            self.clear_map = itertools.cycle(reversed(self.ordered_expansions))
            self.army_target = next(self.clear_map)
        if self.units.closer_than(6, self.army_target):
            self.army_target = next(self.clear_map)

    # MICRO
    def micro(self):
        # calculate army units
        army = self.units.ready.filter(
            lambda self_army: self_army.type_id in {UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.SENTRY,
                                                    UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS,
                                                    UnitTypeId.VOIDRAY, UnitTypeId.TEMPEST, UnitTypeId.CARRIER})
        # don't do anything if we don't have any army
        if not army:
            return
        # low level targets
        ground_enemies = self.enemy_units.filter(
            lambda enemy: not enemy.is_flying and enemy.type_id not in {UnitTypeId.LARVA,
                                                                        UnitTypeId.EGG})
        # only attack if we have a certain size army
        if army.amount > 10:
            # we don't see anything so start to clear the map
            if not ground_enemies:
                for unit in army:
                    # clear found structures
                    if self.enemy_structures:
                        # focus down low hp structures first
                        in_range_structures = self.enemy_structures.in_attack_range_of(unit)
                        if in_range_structures:
                            lowest_hp = min(in_range_structures, key=lambda e: (e.health + e.shield, e.tag))
                            if unit.weapon_cooldown == 0:
                                self.do(unit.attack(lowest_hp))
                            else:
                                if unit.ground_range > 1:
                                    self.do(unit.move(lowest_hp.position.towards(unit, 1 + lowest_hp.radius)))
                                else:
                                    self.do(unit.move(lowest_hp.position))
                        else:
                            self.do(unit.move(self.enemy_structures.closest_to(unit)))
                    # check bases to find new structures
                    else:
                        self.do(unit.move(self.army_target))
                return
            # create selection of dangerous enemy units.
            enemy_fighters = ground_enemies.filter(lambda u: u.can_attack) + self.enemy_structures(
                {UnitTypeId.BUNKER, UnitTypeId.SPINECRAWLER, UnitTypeId.PHOTONCANNON})
            for unit in army:
                if enemy_fighters:
                    # select enemies in range
                    in_range_enemies = enemy_fighters.in_attack_range_of(unit)
                    if in_range_enemies:
                        # high level targets we want to vanish
                        workers = in_range_enemies({UnitTypeId.DRONE, UnitTypeId.SCV, UnitTypeId.PROBE})
                        if workers:
                            in_range_enemies = workers
                        # special micro for ranged units
                        if unit.ground_range > 1:
                            # attack if weapon not on cooldown
                            if unit.weapon_cooldown == 0:
                                # attack enemy with the lowest hp of the ones in range
                                lowest_hp = min(in_range_enemies, key=lambda e: (e.health + e.shield, e.tag))
                                self.do(unit.attack(lowest_hp))
                            else:
                                # micro away from the closest unit
                                # move further away if too many enemies are near
                                friends_in_range = army.in_attack_range_of(unit)
                                closest_enemy = in_range_enemies.closest_to(unit)
                                distance = unit.ground_range + unit.radius + closest_enemy.radius
                                if (len(friends_in_range) <= len(in_range_enemies)
                                        and closest_enemy.ground_range <= unit.ground_range):
                                    distance += 1
                                else:
                                    # if more than 5 units friends are close, use distance one shorter than range
                                    # to let other friendly units get close enough as well and not block each other
                                    if len(army.closer_than(7, unit.position)) >= 5:
                                        distance -= -1
                                self.do(unit.move(closest_enemy.position.towards(unit, distance)))
                        else:
                            # target fire with melee units
                            lowest_hp = min(in_range_enemies, key=lambda e: (e.health + e.shield, e.tag))
                            self.do(unit.attack(lowest_hp))
                    else:
                        # no unit in range, go to closest
                        self.do(unit.move(enemy_fighters.closest_to(unit)))
                # no dangerous enemy at all, attack closest anything
                else:
                    self.do(unit.attack(ground_enemies.closest_to(unit)))


async def on_end(self, game_result):
    if game_result == Result.Victory:  # if the bot win the game, save data
        np.save("train_data/local_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    print("--- On END ---")
    with open("log.txt", "a") as file:
        if self.use_model:
            file.write("Use Model {} - {}\n".format(game_result, int(time.time())))
        else:
            file.write("Not Use Model {} - {}\n".format(game_result, int(time.time())))
