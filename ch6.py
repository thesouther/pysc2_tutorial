import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, STALKER, \
    CYBERNETICSCORE, GATEWAY, STARGATE, VOIDRAY
import random
from pythonsc2.examples.terran.proxy_rax import ProxyRaxBot

# Protoss 神族
class SentdeBot(sc2.BotAI):
    def __init__(self):
        self.ITERATION_PER_MINUTE = 165
        self.MAX_WORKERS = 70

    async def on_step(self, iteration):
        # what to do every step
        self.iteration = iteration
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
        if (len(self.units(NEXUS))*16) > len(self.units(PROBE)) and len(self.units(PROBE)) < self.MAX_WORKERS:
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
        # if self.units(NEXUS).amount < (self.iteration/self.ITERATION_PER_MINUTE) and self.can_afford(NEXUS):
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        # print(self.iteration/self.ITERATION_PER_MINUTE)
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            # 建造控制核心
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE): # 已有传送门，建造控制核心
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            elif len(self.units(GATEWAY)) < ((self.iteration/self.ITERATION_PER_MINUTE)/2):
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
            
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < ((self.iteration/self.ITERATION_PER_MINUTE)/2):
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    # 建造军队，这里主要是stalker追猎者, 虚空战舰
    async def build_offensive_force(self):
        for gw in self.units(GATEWAY).ready.idle:
            if not self.units(STALKER).amount > self.units(VOIDRAY).amount:
                if self.can_afford(STALKER) and self.supply_left > 0:
                    await self.do(gw.train(STALKER))

        for sg in self.units(STARGATE).ready.idle:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

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
        aggressive_units = {STALKER: [15, 5],
                            VOIDRAY: [8, 3]}
        for UNIT in aggressive_units:
            if self.units(UNIT).amount < aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(self.find_target(self.state)))
            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))

run_game(maps.get("KingsCoveLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Bot(Race.Terran, ProxyRaxBot())
    ], realtime=False)
