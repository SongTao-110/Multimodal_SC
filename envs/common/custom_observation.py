from __future__ import annotations

from collections import OrderedDict
from itertools import product
from typing import TYPE_CHECKING
from highway_env.envs.common.observation import ObservationType
import numpy as np
import pandas as pd
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.lane import AbstractLane
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class CustomObservation(ObservationType):
    def __init__(self, env: AbstractEnv):
        super().__init__(env)

    def observe(self) -> np.ndarray:
        # 获取受控车辆
        controlled_vehicle = self.env.controlled_vehicle

        # 速度
        speed = controlled_vehicle.speed

        # 车道索引
        lane_index = controlled_vehicle.lane_index

        # 获取前车信息
        front_vehicle = controlled_vehicle.road.get_vehicle_ahead(controlled_vehicle)
        if front_vehicle is not None:
            # 车间距
            distance_to_front = front_vehicle.position[0] - controlled_vehicle.position[0]
            front_speed = front_vehicle.speed
        else:
            # 如果没有前车，设定默认值
            distance_to_front = float('inf')  # 或者 0，取决于你的需求
            front_speed = 0

        # 返回观测状态数组
        return np.array([speed, lane_index[2], distance_to_front, front_speed])
