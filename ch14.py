"""
修改侦察策略
1. 使用探针进行侦察，直到我们可以制造实际的observe。
2. 使用多个侦察者
3. 将侦察员沿着扩展位置放置
"""

import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, STALKER, \
    CYBERNETICSCORE, GATEWAY, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY

import random
import cv2
import numpy as np
import time
import keras
import math

HEADLESS = True

# Protoss 神族
class SentdeBot(sc2.BotAI):
    def __init__(self, use_model=False):
        self.ITERATION_PER_MINUTE = 165
        self.MAX_WORKERS = 70
        self.do_something_after = 0
        self.train_data = []

        self.scout_and_spots = {}

        self.use_model = use_model
        if self.use_model:
            self.model = keras.models.load_model("./models/BasicCNN-10-epochs-0.0001-LR-STAGE1")

    # 在训练数据中记录游戏结果
    def on_end(self, game_result):
        with open("log.txt", "a") as f:
            if self.use_model:
                f.write("model {}\n".format(game_result))
            else:
                f.write("random {}\n".format(game_result))
        
    async def on_step(self, iteration):
        # what to do every step
        # self.iteration = iteration
        self.times = self.time/60
        await self.build_scout()
        await self.scout()
        await self.distribute_workers()  # in sc2/bot_ai.py
        await self.build_workers() # 生产探针
        await self.build_pylons() # 水晶塔
        await self.build_assimilators() # 收集瓦斯
        await self.expand() # 扩展资源区域
        await self.offensive_force_buildings() # 建造生产军队的建筑，控制核心和传送门
        await self.build_offensive_force() # 建造军队
        await self.intel()
        await self.attack() # 军队攻击

    def random_location_variance(self, enemy_start_location):
        # 这里用硬编码，移动到位置附近的5的范围内
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to

    # 生成侦察兵
    async def build_scout(self):
        if len(self.units(OBSERVER)) < math.floor(self.times/3):
            for rf in self.units(ROBOTICSFACILITY).ready.idle:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

    # 侦察
    async def scout(self):
        '''
        首先选择侦察位置，按距离敌人starting位置的远近顺序。
        所以第一个侦察兵送到敌人的starting位置。
        然后，下一个侦察兵检查第一个点附近，因为这很可能是他们去的地方。
        '''

        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.expand_dis_dir[distance_to_enemy_start] = el
        self.ordered_expd_dustances = sorted(k for k in self.expand_dis_dir)
        # 侦察兵总是被杀，所以要检查
        existing_ids = [unit.tag for unit in self.units]
        # 移除被击毁的
        to_be_removed = []
        for noted_scout in self.scout_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)
        for scout in to_be_removed:
            del self.scout_and_spots[scout]

        # 只希望一个探机进行侦察， 还要保持它运动，否则就是空闲
        # 希望所有的侦察兵进行侦察
        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True
        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scout_and_spots:
                    assign_scout = False
        if assign_scout:
            if len(self.units(unit_type).idle)> 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scout_and_spots:
                        for dist in self.ordered_expd_dustances:
                            try:
                                location = next(value for key,value in self.expand_dis_dir.items() if key == dist)
                                # DICT {UNIT_ID:LOCATION}
                                active_locations = [self.scout_and_spots[k] for k in self.scout_and_spots]
                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scout_and_spots:
                                                continue
                                    await self.do(obs.move(location))
                                    self.scout_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                pass
        
        # 保持探针移动，防止它采矿
        for obs in self.units(unit_type):
            if obs.tag in self.scout_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scout_and_spots[obs.tag])))

    # 使用cv生成训练数据
    async def intel(self):
        # 获得小地图图片，里边标记一些单元或者建筑的位置信息
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        draw_dict = {
                NEXUS: [15, (0, 255, 0)],
                PYLON: [3, (20, 235, 0)],
                PROBE: [1, (55, 200, 0)],

                ASSIMILATOR: [2, (55, 200, 0)],
                GATEWAY: [3, (200, 100, 0)],
                CYBERNETICSCORE: [3, (150, 150, 0)],
                STARGATE: [5, (255, 0, 0)],
                ROBOTICSFACILITY: [5, (215, 155, 0)],
                VOIDRAY: [3, (255, 100, 0)],
                # OBSERVER: [3, (255, 255, 255)],
            }
        # 画出己方兵力和建筑
        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)
        
        # 画出敌方建筑
        main_base_name = ["nexus", "commandcenter", "orbitalcommand", "planetaryfortress", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_name:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_name:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0,0,255), -1)

        # 画出敌方兵力
        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_name = [
                    "probe",
                    "scv",
                    "drone"
                ]
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_name:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (55, 0, 215), -1)
        # 画出observer
        for obs in self.units(OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        # 可视化资源、供给等信息
        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0
        
        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0
        
        cv2.line(game_data, (0,19), (int(line_max*military_weight), 19), (250, 250, 200), 3) # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        if not HEADLESS:
            if self.use_model:
                cv2.imshow('Model Intel', resized)
                cv2.waitKey(1)
            else:
                cv2.imshow('Random Intel', resized)
                cv2.waitKey(1)
        
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
        try:
            # if self.units(NEXUS).amount < (self.iteration/self.ITERATION_PER_MINUTE) and self.can_afford(NEXUS):
            if self.units(NEXUS).amount < (self.times / 2) and self.can_afford(NEXUS):
               await self.expand_now()
        except Exception as e:
            print(str(e))
        

    async def offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            # 建造控制核心
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE): # 已有传送门，建造控制核心
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            elif len(self.units(GATEWAY)) < 1:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)
            
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < self.times:
                # if len(self.units(STARGATE)) < ((self.iteration/self.ITERATION_PER_MINUTE)):
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    # 建造军队，这里主要是stalker追猎者, 虚空战舰
    async def build_offensive_force(self):
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
        if len(self.units(VOIDRAY).idle) > 5:
            target = False
            if self.times > self.do_something_after:
            # if self.iteration > self.do_something_after:
                if self.use_model:
                    prediction = self.model.predict([self.flipped.reshape([-1, 176, 176, 3])])
                    choice = np.argmax(prediction[0])
                else:
                    choice = random.randrange(0,4)

                if choice == 0:
                    # 不攻击
                    # wait = random.randrange(20, 165)
                    # self.do_something_after = self.iteration + wait
                    wait = random.randrange(7, 100) / 100
                    self.do_something_after = self.times + wait
                elif choice == 1:
                    # 攻击最近的nexus
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
                elif choice == 2:
                    # 攻击敌方建筑
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)
                elif choice == 3:
                    # 攻击大本营（start）
                    target = self.enemy_start_locations[0]
                
                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))

                y = np.zeros(4)
                y[choice] =1
                self.train_data.append([y, self.flipped])

run_game(maps.get("KingsCoveLE"), [
    Bot(Race.Protoss, SentdeBot(use_model=False)),
    Computer(Race.Terran, Difficulty.Medium)
    ], realtime=False)
