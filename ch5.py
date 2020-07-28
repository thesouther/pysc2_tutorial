import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, STALKER, \
    CYBERNETICSCORE, GATEWAY
import random

# Protoss 神族
class SentdeBot(sc2.BotAI):
    async def on_step(self, iteration):
        # what to do every step
        await self.distribute_workers()  # in sc2/bot_ai.py
        await self.build_workers() # 生产探针
        await self.build_pylons() # 水晶塔
        await self.build_assimilators() # 收集瓦斯
        await self.expand() # 扩展资源区域
        await self.offensive_force_buildings() # 建造生产军队的建筑，控制核心和传送门
        await self.build_offensive_force() # 建造军队
        await self.attack() # 军队攻击

    # 生成探针
    async def build_workers(self):
        for nexus in self.units(NEXUS).ready.idle:
            if self.can_afford(PROBE):
                await self.do(nexus.train(PROBE))

    # build pylons 水晶塔，用于提高部队容量
    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    # 瓦斯处理厂
    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))
    
    # 向外扩张
    async def expand(self):
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            # 建造控制核心
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE): # 已有传送门，建造控制核心
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            elif len(self.units(GATEWAY)) < 3:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

    # 建造军队，这里主要是stalker追猎者
    async def build_offensive_force(self):
        for gw in self.units(GATEWAY).ready.idle:
            if self.can_afford(STALKER) and self.supply_left > 0:
                await self.do(gw.train(STALKER))

    # 寻找目标
    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    # 攻击
    async def attack(self):
        """
        The way we do this is, if we know where any units are, we attack those first. 
        If we don't know where any units are, but we know where their buildings are, we go for those. 
        Finally, if we do not know where any enemy units or buildings are, we go to where we know their starting point was
        """
        if self.units(STALKER).amount > 15:
            for s in self.units(STALKER).idle:
                await self.do(s.attack(self.find_target(self.state)))

        if self.units(STALKER).amount > 3:
            if len(self.known_enemy_units) > 0:
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))



run_game(maps.get("KingsCoveLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Easy)
    ], realtime=False)
