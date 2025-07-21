from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
from gymnasium import spaces

from highway_env import utils
from highway_env.utils import Vector
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

# 定义一个联合类型，表示动作可以是整型或NumPy数组
Action = Union[int, np.ndarray]


class ActionType:
    """动作类型基类，定义了动作的空间以及在环境中执行动作的方式"""

    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env  # 保存当前环境
        self.__controlled_vehicle = None  # 初始化受控车辆

    def space(self) -> spaces.Space:
        """定义动作空间"""
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable:
        """
        定义能够执行该动作的车辆类

        必须返回一个Vehicle的子类。
        """
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """
        在主车辆上执行动作

        大部分动作机制由vehicle.act(action)实现，
        该vehicle是由ActionType.vehicle_class指定的车辆实例，
        动作可以根据ActionType配置进行预处理。

        :param action: 要执行的动作
        """
        raise NotImplementedError

    def get_available_actions(self):
        """对于离散动作空间，返回可用的动作列表"""
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        """
        获取受控车辆。如果未设置，则使用第一个受控车辆作为默认值
        """
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class ContinuousAction(ActionType):
    """
    连续动作空间，用于控制油门和/或方向角

    如果油门和转向都启用，则顺序为：[油门, 转向]
    空间范围始终为[-1, 1]，但根据配置映射到具体的油门/转向范围。
    """

    ACCELERATION_RANGE = (-5, 5.0)  # 加速度范围，以m/s²为单位
    STEERING_RANGE = (-np.pi / 4, np.pi / 4)  # 转向角范围，以弧度为单位

    def __init__(
            self,
            env: AbstractEnv,
            acceleration_range: tuple[float, float] | None = None,
            steering_range: tuple[float, float] | None = None,
            speed_range: tuple[float, float] | None = None,
            longitudinal: bool = True,
            lateral: bool = True,
            dynamical: bool = False,
            clip: bool = True,
            **kwargs,
    ) -> None:
        """
        创建一个连续动作空间

        :param env: 环境实例
        :param acceleration_range: 加速度范围，以m/s²为单位
        :param steering_range: 转向角范围，以弧度为单位
        :param speed_range: 可达到的速度范围，以m/s为单位
        :param longitudinal: 是否启用纵向控制（油门控制）
        :param lateral: 是否启用横向控制（转向控制）
        :param dynamical: 是否模拟动态（例如摩擦），而不是运动学
        :param clip: 是否将动作裁剪到定义范围内
        """
        super().__init__(env)
        # 初始化加速度和转向范围，默认使用定义的常量范围
        self.acceleration_range = acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        self.steering_range = steering_range if steering_range else self.STEERING_RANGE
        self.speed_range = speed_range
        self.lateral = lateral
        self.longitudinal = longitudinal
        # 至少启用纵向或横向控制，否则抛出错误
        if not self.lateral and not self.longitudinal:
            raise ValueError("必须启用纵向或横向控制中的至少一个")
        self.dynamical = dynamical
        self.clip = clip
        self.size = 2 if self.lateral and self.longitudinal else 1  # 动作维度大小
        self.last_action = np.zeros(self.size)  # 初始化上一个动作为零向量

    def space(self) -> spaces.Box:
        """定义连续动作空间"""
        return spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        """返回对应的车辆类，选择动态模型或普通车辆"""
        return Vehicle if not self.dynamical else BicycleVehicle

    def get_action(self, action: np.ndarray):
        """获取并处理动作（加速度和/或转向角）"""
        if self.clip:
            action = np.clip(action, -1, 1)  # 裁剪动作
        if self.speed_range:
            (self.controlled_vehicle.MIN_SPEED, self.controlled_vehicle.MAX_SPEED) = self.speed_range
        if self.longitudinal and self.lateral:
            return {
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
            }
        elif self.longitudinal:
            return {"acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range), "steering": 0}
        elif self.lateral:
            return {"acceleration": 0, "steering": utils.lmap(action[0], [-1, 1], self.steering_range)}

    def act(self, action: np.ndarray) -> None:
        """执行动作并更新上一个动作"""
        self.controlled_vehicle.act(self.get_action(action))
        self.last_action = action


class DiscreteAction(ContinuousAction):
    """
    离散动作空间，继承于ContinuousAction，并将连续动作空间离散化
    """

    def __init__(
            self,
            env: AbstractEnv,
            acceleration_range: tuple[float, float] | None = None,
            steering_range: tuple[float, float] | None = None,
            longitudinal: bool = True,
            lateral: bool = True,
            dynamical: bool = False,
            clip: bool = True,
            actions_per_axis: int = 3,
            **kwargs,
    ) -> None:
        super().__init__(
            env,
            acceleration_range=acceleration_range,
            steering_range=steering_range,
            longitudinal=longitudinal,
            lateral=lateral,
            dynamical=dynamical,
            clip=clip,
        )
        self.actions_per_axis = actions_per_axis  # 每个轴的离散动作数量

    def space(self) -> spaces.Discrete:
        """定义离散动作空间"""
        return spaces.Discrete(self.actions_per_axis ** self.size)

    def act(self, action: int) -> None:
        """执行离散化后的动作"""
        cont_space = super().space()
        axes = np.linspace(cont_space.low, cont_space.high, self.actions_per_axis).T  # 创建轴上的动作值
        all_actions = list(itertools.product(*axes))  # 创建所有组合动作
        super().act(all_actions[action])

class DiscreteMetaAction(ActionType):
    """
    元动作的离散动作空间，包括变道和巡航控制
    """

    ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}  # 全动作集合
    ACTIONS_LONGI = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}  # 纵向动作集合
    ACTIONS_LAT = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}  # 横向动作集合

    def __init__(
        self,
        env: AbstractEnv,
        longitudinal: bool = True,
        lateral: bool = True,
        target_speeds: Vector | None = None,
        **kwargs,
    ) -> None:
        """
        创建元动作的离散动作空间

        :param env: 环境实例
        :param longitudinal: 是否包含纵向动作
        :param lateral: 是否包含横向动作
        :param target_speeds: 可追踪的目标速度列表
        """
        super().__init__(env)
        self.longitudinal = longitudinal
        self.lateral = lateral
        self.target_speeds = np.array(target_speeds) if target_speeds else MDPVehicle.DEFAULT_TARGET_SPEEDS
        # 确定动作集合
        self.actions = (
            self.ACTIONS_ALL if longitudinal and lateral else
            (self.ACTIONS_LONGI if longitudinal else self.ACTIONS_LAT if lateral else None)
        )
        if self.actions is None:
            raise ValueError("必须至少包含纵向或横向动作")
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Discrete:
        """定义离散动作空间"""
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        # 返回一个部分应用的MDPVehicle类，用于创建车辆实例，并指定目标速度
        return functools.partial(MDPVehicle, target_speeds=self.target_speeds)

    def act(self, action: int | np.ndarray) -> None:
        # 执行指定的动作，调用controlled_vehicle的act方法
        self.controlled_vehicle.act(self.actions[int(action)])

    def get_available_actions(self) -> list[int]:
        """
        获取当前可用动作列表。

        在道路边界上不允许换道，达到最大或最小速度时不允许速度变化。

        :return: 可用动作的列表
        """
        actions = [self.actions_indexes["IDLE"]]  # 初始可用动作为“闲置”
        network = self.controlled_vehicle.road.network  # 获取车辆所在道路的网络
        for l_index in network.side_lanes(self.controlled_vehicle.lane_index):
            # 检查是否可以向左换道
            if (
                    l_index[2] < self.controlled_vehicle.lane_index[2]
                    and network.get_lane(l_index).is_reachable_from(
                self.controlled_vehicle.position
            )
                    and self.lateral
            ):
                actions.append(self.actions_indexes["LANE_LEFT"])
            # 检查是否可以向右换道
            if (
                    l_index[2] > self.controlled_vehicle.lane_index[2]
                    and network.get_lane(l_index).is_reachable_from(
                self.controlled_vehicle.position
            )
                    and self.lateral
            ):
                actions.append(self.actions_indexes["LANE_RIGHT"])
        # 检查是否可以加速
        if (
                self.controlled_vehicle.speed_index
                < self.controlled_vehicle.target_speeds.size - 1
                and self.longitudinal
        ):
            actions.append(self.actions_indexes["FASTER"])
        # 检查是否可以减速
        if self.controlled_vehicle.speed_index > 0 and self.longitudinal:
            actions.append(self.actions_indexes["SLOWER"])
        return actions  # 返回所有可用动作

class MultiAgentAction(ActionType):
    def __init__(self, env: AbstractEnv, action_config: dict, **kwargs) -> None:
        super().__init__(env)  # 调用父类构造函数
        self.action_config = action_config  # 存储动作配置
        self.agents_action_types = []  # 代理动作类型列表
        for vehicle in self.env.controlled_vehicles:
            # 为每个控制的车辆创建动作类型
            action_type = action_factory(self.env, self.action_config)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        # 返回所有代理的动作空间的元组
        return spaces.Tuple(
            [action_type.space() for action_type in self.agents_action_types]
        )

    @property
    def vehicle_class(self) -> Callable:
        # 返回一个车辆类，用于所有代理
        return action_factory(self.env, self.action_config).vehicle_class

    def act(self, action: Action) -> None:
        # 执行一组动作，确保动作为元组
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)  # 调用每个代理的act方法

    def get_available_actions(self):
        # 获取所有代理的可用动作的笛卡尔积
        return itertools.product(
            *[
                action_type.get_available_actions()
                for action_type in self.agents_action_types
            ]
        )


def action_factory(env: AbstractEnv, config: dict) -> ActionType:
    # 根据配置创建相应的动作类型
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)  # 连续动作
    if config["type"] == "DiscreteAction":
        return DiscreteAction(env, **config)  # 离散动作
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)  # 离散元动作
    elif config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)  # 多代理动作
    else:
        raise ValueError("Unknown action type")  # 未知动作类型
