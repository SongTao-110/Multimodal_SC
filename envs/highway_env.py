from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    高速公路驾驶环境。

    车辆在一条直的高速公路上行驶，具有多个车道，获得奖励以达到高速度、
    保持在最右侧车道并避免碰撞。
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},  # 观察类型
                "action": {
                    "type": "DiscreteMetaAction",  # 动作类型
                },
                "lanes_count": 4,  # 车道数量
                "vehicles_count": 50,  # 车辆总数
                "controlled_vehicles": 1,  # 受控车辆数量
                "initial_lane_id": None,  # 初始车道ID
                "duration": 40,  # 事件持续时间 [s]
                "ego_spacing": 2,  # 自车间距
                "vehicles_density": 1,  # 车辆密度
                "collision_reward": -1,  # 碰撞时的奖励
                "right_lane_reward": 0.1,  # 在最右侧车道行驶的奖励
                "high_speed_reward": 0.4,  # 高速行驶的奖励
                "lane_change_reward": 0,  # 每次变道的奖励
                "reward_speed_range": [20, 30],  # 奖励速度范围
                "normalize_reward": True,  # 是否归一化奖励
                "offroad_terminal": False,  # 是否允许越界终止
            }
        )
        return config

    def _reset(self) -> None:
        """重置环境状态。"""
        self._create_road()  # 创建道路
        self._create_vehicles()  # 创建车辆

    def _create_road(self) -> None:
        """创建一条由直线相邻车道组成的道路。"""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30  # 创建直线道路网络
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],  # 是否记录历史轨迹
        )

    def _create_vehicles(self) -> None:
        """创建新的随机车辆，并将其添加到道路上。"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])  # 获取其他车辆类型
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]  # 分配车辆数量
        )

        self.controlled_vehicles = []  # 初始化受控车辆列表
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,  # 车辆初始速度
                lane_id=self.config["initial_lane_id"],  # 初始车道ID
                spacing=self.config["ego_spacing"],  # 自车间距
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed  # 创建受控车辆
            )
            self.controlled_vehicles.append(vehicle)  # 添加到受控车辆列表
            self.road.vehicles.append(vehicle)  # 添加到道路上的车辆列表

            for _ in range(others):  # 创建其他随机车辆
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]  # 创建随机车辆
                )
                vehicle.randomize_behavior()  # 随机化车辆行为
                self.road.vehicles.append(vehicle)  # 添加到道路上的车辆列表

    def _reward(self, action: Action) -> float:
        """
        奖励设计以促进高速行驶、在最右侧车道行驶和避免碰撞。
        :param action: 上一次执行的动作
        :return: 相应的奖励
        """
        rewards = self._rewards(action)  # 获取各类奖励
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()  # 计算总奖励
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],  # 归一化奖励
            )
        reward *= rewards["on_road_reward"]  # 仅在道路上时的奖励
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        """计算各类奖励。"""
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)  # 获取所有邻近车道
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)  # 使用前向速度
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]  # 映射速度
        )
        return {
            "collision_reward": float(self.vehicle.crashed),  # 碰撞奖励
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),  # 右侧车道奖励
            "high_speed_reward": np.clip(scaled_speed, 0, 1),  # 高速奖励
            "on_road_reward": float(self.vehicle.on_road),  # 道路内奖励
        }

    def _is_terminated(self) -> bool:
        """如果自车发生碰撞则事件结束。"""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road  # 越界终止
        )

    def _is_truncated(self) -> bool:
        """如果达到时间限制则事件被截断。"""
        return self.time >= self.config["duration"]  # 超过持续时间


class HighwayEnvFast(HighwayEnv):
    """
    highway-v0 的变体，具有更快的执行速度：
        - 较低的仿真频率
        - 场景中车辆数量较少（车道数量和事件持续时间较短）
        - 仅检查受控车辆与其他车辆的碰撞
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,  # 仿真频率
                "lanes_count": 3,  # 车道数量
                "vehicles_count": 20,  # 车辆数量
                "duration": 30,  # 持续时间 [s]
                "ego_spacing": 1.5,  # 自车间距
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        """创建车辆并禁用非受控车辆的碰撞检查。"""
        super()._create_vehicles()  # 调用父类的方法创建车辆
        for vehicle in self.road.vehicles:  # 遍历所有车辆
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False  # 禁用非受控车辆的碰撞检查
