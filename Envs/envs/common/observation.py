from __future__ import annotations

from collections import OrderedDict
from itertools import product
from typing import TYPE_CHECKING

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


class ObservationType:
    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env  # 存储环境对象
        self.__observer_vehicle = None  # 观察车辆，默认为None

    def space(self) -> spaces.Space:
        """获取观察空间。"""
        raise NotImplementedError()  # 抽象方法，子类需实现

    def observe(self):
        """获取环境状态的观察。"""
        raise NotImplementedError()  # 抽象方法，子类需实现

    @property
    def observer_vehicle(self):
        """
        观察场景的车辆。

        如果未设置，默认使用第一个受控车辆。
        """
        return self.__observer_vehicle or self.env.vehicle  # 返回观察车辆

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle  # 设置观察车辆


class GrayscaleObservation(ObservationType):
    """
    一个观察类，直接收集模拟器渲染的内容。

    还将收集的帧堆叠起来，类似于自然DQN。
    观察形状为 C x W x H。

    在传递的配置字典中期望特定的键。
    环境配置中的观察字典示例：
        observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # RGB转换的权重，
        }
    """

    def __init__(
            self,
            env: AbstractEnv,
            observation_shape: tuple[int, int],
            stack_size: int,
            weights: list[float],
            scaling: float | None = None,
            centering_position: list[float] | None = None,
            **kwargs,
    ) -> None:
        super().__init__(env)  # 调用父类初始化
        self.observation_shape = observation_shape  # 观察的形状
        self.shape = (stack_size,) + self.observation_shape  # 堆叠后的观察形状
        self.weights = weights  # RGB转换的权重
        self.obs = np.zeros(self.shape, dtype=np.uint8)  # 初始化观察数组

        # 观察者配置与env.render()的配置可能不同（通常较小）
        viewer_config = env.config.copy()
        viewer_config.update(
            {
                "offscreen_rendering": True,  # 离屏渲染
                "screen_width": self.observation_shape[0],
                "screen_height": self.observation_shape[1],
                "scaling": scaling or viewer_config["scaling"],
                "centering_position": centering_position
                                      or viewer_config["centering_position"],
            }
        )
        self.viewer = EnvViewer(env, config=viewer_config)  # 创建环境查看器

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)  # 返回观察空间

    def observe(self) -> np.ndarray:
        new_obs = self._render_to_grayscale()  # 渲染到灰度图像
        self.obs = np.roll(self.obs, -1, axis=0)  # 向前滚动观察数组
        self.obs[-1, :, :] = new_obs  # 更新最新观察
        return self.obs  # 返回观察数组

    def _render_to_grayscale(self) -> np.ndarray:
        self.viewer.observer_vehicle = self.observer_vehicle  # 设置观察车辆
        self.viewer.display()  # 显示环境
        raw_rgb = self.viewer.get_image()  # 获取原始RGB图像 (H x W x C)
        raw_rgb = np.moveaxis(raw_rgb, 0, 1)  # 转换轴顺序
        return np.dot(raw_rgb[..., :3], self.weights).clip(0, 255).astype(np.uint8)  # 返回灰度图像


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: AbstractEnv, horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)  # 调用父类初始化
        self.horizon = horizon  # 时间范围

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(
                shape=self.observe().shape, low=0, high=1, dtype=np.float32
            )  # 返回观察空间
        except AttributeError:
            return spaces.Space()  # 如果发生属性错误，返回空空间

    def observe(self) -> np.ndarray:
        if not self.env.road:  # 如果没有道路信息
            return np.zeros(
                (3, 3, int(self.horizon * self.env.config["policy_frequency"]))  # 返回全零观察
            )
        grid = compute_ttc_grid(
            self.env,
            vehicle=self.observer_vehicle,
            time_quantization=1 / self.env.config["policy_frequency"],
            horizon=self.horizon,
        )  # 计算碰撞时间网格
        padding = np.ones(np.shape(grid))  # 创建填充
        padded_grid = np.concatenate([padding, grid, padding], axis=1)  # 连接网格和填充
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2  # 左边界
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2  # 右边界
        clamped_grid = padded_grid[:, l0: lf + 1, :]  # 限制网格
        repeats = np.ones(clamped_grid.shape[0])  # 复制次数
        repeats[np.array([0, -1])] += clamped_grid.shape[0]  # 对边界进行加倍
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)  # 重复边界网格
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2  # 速度的左边界
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2  # 速度的右边界
        clamped_grid = padded_grid[v0: vf + 1, :, :]  # 限制速度网格
        return clamped_grid.astype(np.float32)  # 返回观察


class KinematicObservation(ObservationType):
    """观察附近车辆的运动学信息。"""

    FEATURES: list[str] = ["presence", "x", "y", "vx", "vy"]  # 观察的特征列表

    def __init__(
            self,
            env: AbstractEnv,
            features: list[str] = None,
            vehicles_count: int = 5,
            features_range: dict[str, list[float]] = None,
            absolute: bool = False,
            order: str = "sorted",
            normalize: bool = True,
            clip: bool = True,
            see_behind: bool = False,
            observe_intentions: bool = False,
            include_obstacles: bool = True,
            **kwargs: dict,
    ) -> None:
        """
        :param env: 要观察的环境
        :param features: 用于观察的特征名称
        :param vehicles_count: 观察的车辆数量
        :param features_range: 特征值范围的字典映射
        :param absolute: 是否使用绝对坐标
        :param order: 观察车辆的顺序。值：sorted, shuffled
        :param normalize: 是否对观察进行归一化
        :param clip: 是否将值限制在所需范围内
        :param see_behind: 是否观察后方车辆
        :param observe_intentions: 是否观察其他车辆的目的地
        """
        super().__init__(env)  # 调用父类初始化
        self.features = features or self.FEATURES  # 设置观察特征
        self.vehicles_count = vehicles_count  # 设置观察车辆数量
        self.features_range = features_range  # 特征范围
        self.absolute = absolute  # 是否使用绝对坐标
        self.order = order  # 观察顺序
        self.normalize = normalize  # 是否归一化
        self.clip = clip  # 是否裁剪
        self.see_behind = see_behind  # 是否观察后方车辆
        self.observe_intentions = observe_intentions  # 是否观察车辆意图
        self.include_obstacles = include_obstacles  # 是否包含障碍物

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),  # 观察空间形状
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        归一化观察值。

        目前假设道路是沿x轴的直线。
        :param Dataframe df: 观察数据
        """
        if not self.features_range:  # 如果没有特征范围
            side_lanes = self.env.road.network.all_side_lanes(
                self.observer_vehicle.lane_index
            )  # 获取所有侧车道
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],  # x的范围
                "y": [
                    -AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                    AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                ],  # y的范围
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],  # vx的范围
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],  # vy的范围
            }
        for feature, f_range in self.features_range.items():  # 遍历特征范围
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])  # 归一化
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)  # 裁剪
        return df  # 返回归一化后的数据

    def observe(self) -> np.ndarray:
        if not self.env.road:  # 如果没有道路信息
            return np.zeros(self.space().shape)  # 返回全零观察

        # 添加自我车辆
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])  # 自我车辆信息
        # 添加附近交通
        close_vehicles = self.env.road.close_objects_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,  # 除了自我车辆外观察的数量
            see_behind=self.see_behind,
            sort=self.order == "sorted",  # 根据顺序排序
            vehicles_only=not self.include_obstacles,  # 仅观察车辆
        )
        if close_vehicles:  # 如果有近的车辆
            origin = self.observer_vehicle if not self.absolute else None  # 计算原点
            vehicles_df = pd.DataFrame.from_records(
                [
                    v.to_dict(origin, observe_intentions=self.observe_intentions)  # 车辆信息
                    for v in close_vehicles[-self.vehicles_count + 1:]  # 取最近的车辆
                ]
            )
            df = pd.concat([df, vehicles_df], ignore_index=True)  # 合并数据

        df = df[self.features]  # 只保留需要的特征

        # 归一化和裁剪
        if self.normalize:
            df = self.normalize_obs(df)  # 归一化观察数据
        # 填充缺失行
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))  # 创建零行
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True  # 合并
            )
        # 重新排序
        df = df[self.features]
        obs = df.values.copy()  # 复制数据
        if self.order == "shuffled":  # 如果顺序为随机
            self.env.np_random.shuffle(obs[1:])  # 随机打乱
        # 扁平化
        return obs.astype(self.space().dtype)  # 返回观察数据


class OccupancyGridObservation(ObservationType):
    """观察附近车辆的占用网格。"""

    FEATURES: list[str] = ["presence", "vx", "vy", "on_road"]  # 特征列表
    GRID_SIZE: list[list[float]] = [[-5.5 * 5, 5.5 * 5], [-5.5 * 5, 5.5 * 5]]  # 网格的真实世界大小
    GRID_STEP: list[int] = [5, 5]  # 网格每个单元的步长

    def __init__(
            self,
            env: AbstractEnv,
            features: list[str] | None = None,
            grid_size: tuple[tuple[float, float], tuple[float, float]] | None = None,
            grid_step: tuple[float, float] | None = None,
            features_range: dict[str, list[float]] = None,
            absolute: bool = False,
            align_to_vehicle_axes: bool = False,
            clip: bool = True,
            as_image: bool = False,
            **kwargs: dict,
    ) -> None:
        """
        :param env: 观察的环境
        :param features: 用于观察的特征名称
        :param grid_size: 网格的真实世界大小 [[min_x, max_x], [min_y, max_y]]
        :param grid_step: 网格单元之间的步长 [step_x, step_y]
        :param features_range: 特征名称到 [min, max] 值的映射字典
        :param absolute: 使用绝对或相对坐标
        :param align_to_vehicle_axes: 如果为真，网格轴与车辆轴对齐。否则，与世界轴对齐。
        :param clip: 将观察值限制在 [-1, 1] 范围内
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = (
            np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        )
        self.grid_step = (
            np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        )
        grid_shape = np.asarray(
            np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step),
            dtype=np.uint8,
        )
        self.grid = np.zeros((len(self.features), *grid_shape))  # 初始化网格
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image

    def space(self) -> spaces.Space:
        """返回观察空间的形状和数据类型。"""
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(
                shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32
            )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        规范化观察值。

        暂时假设道路沿 x 轴是直的。
        :param Dataframe df: 观察数据
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        """获取当前观察值。"""
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # 初始化空数据
            self.grid.fill(np.nan)

            # 获取附近交通数据
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles]
            )
            # 规范化
            df = self.normalize(df)
            # 填充特征
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # 车辆特征
                    for _, vehicle in df[::-1].iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # 恢复未规范化的坐标以计算单元索引
                        if "x" in self.features_range:
                            x = utils.lmap(
                                x,
                                [-1, 1],
                                [
                                    self.features_range["x"][0],
                                    self.features_range["x"][1],
                                ],
                            )
                        if "y" in self.features_range:
                            y = utils.lmap(
                                y,
                                [-1, 1],
                                [
                                    self.features_range["y"][0],
                                    self.features_range["y"][1],
                                ],
                            )
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        if (
                                0 <= cell[0] < self.grid.shape[-2]
                                and 0 <= cell[1] < self.grid.shape[-1]
                        ):
                            self.grid[layer, cell[0], cell[1]] = vehicle[feature]
                elif feature == "on_road":
                    self.fill_road_layer_by_lanes(layer)

            obs = self.grid

            if self.clip:
                obs = np.clip(obs, -1, 1)

            if self.as_image:
                obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            obs = np.nan_to_num(obs).astype(self.space().dtype)

            return obs

    def pos_to_index(self, position: Vector, relative: bool = False) -> tuple[int, int]:
        """
        将世界位置转换为网格单元索引

        如果 align_to_vehicle_axes 为真，则单元格在车辆的坐标系中；否则，在世界坐标系中。

        :param position: 世界位置
        :param relative: 位置是否已经相对于观察者的位置
        :return: 单元索引对 (i,j)
        """
        if not relative:
            position -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(
                self.observer_vehicle.heading
            )
            position = np.array([[c, s], [-s, c]]) @ position
        return (
            int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),
            int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1])),
        )

    def index_to_pos(self, index: tuple[int, int]) -> np.ndarray:
        """将单元索引转换为世界位置。"""
        position = np.array(
            [
                (index[0] + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
                (index[1] + 0.5) * self.grid_step[1] + self.grid_size[1, 0],
            ]
        )

        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(
                -self.observer_vehicle.heading
            )
            position = np.array([[c, s], [-s, c]]) @ position

        position += self.observer_vehicle.position
        return position

    def fill_road_layer_by_lanes(
            self, layer_index: int, lane_perception_distance: float = 100
    ) -> None:
        """
        填充一层以编码在路上 (1) / 离路 (0) 信息

        在这里，我们迭代车道并在这些车道上定期放置的路点来填充相应的单元格。
        如果网格较大而道路网络较小，这种方法更快。

        :param layer_index: 网格中层的索引
        :param lane_perception_distance: 车辆位置左右的车道渲染距离
        """
        lane_waypoints_spacing = np.amin(self.grid_step)
        road = self.env.road

        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for lane in road.network.graph[_from][_to]:
                    origin, _ = lane.local_coordinates(self.observer_vehicle.position)
                    waypoints = np.arange(
                        origin - lane_perception_distance,
                        origin + lane_perception_distance,
                        lane_waypoints_spacing,
                    ).clip(0, lane.length)
                    for waypoint in waypoints:
                        cell = self.pos_to_index(lane.position(waypoint, 0))
                        if (
                                0 <= cell[0] < self.grid.shape[-2]
                                and 0 <= cell[1] < self.grid.shape[-1]
                        ):
                            self.grid[layer_index, cell[0], cell[1]] = 1

    def fill_road_layer_by_cell(self, layer_index) -> None:
        """
        填充一层以编码在路上 (1) / 离路 (0) 信息

        在此实现中，我们迭代网格单元并检查单元中心的对应世界位置是否在路上/离路。
        如果网格较小而道路网络较大，这种方法更快。
        """
        road = self.env.road
        for i, j in product(range(self.grid.shape[-2]), range(self.grid.shape[-1])):
            for _from in road.network.graph.keys():
                for _to in road.network.graph[_from].keys():
                    for lane in road.network.graph[_from][_to]:
                        if lane.on_lane(self.index_to_pos((i, j))):
                            self.grid[layer_index, i, j] = 1


class KinematicsGoalObservation(KinematicObservation):
    """动力学目标观察类。"""

    def __init__(self, env: AbstractEnv, scales: list[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        """返回观察空间。"""
        try:
            obs = self.observe()
            return spaces.Dict(
                dict(
                    desired_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["desired_goal"].shape,
                        dtype=np.float64,
                    ),
                    achieved_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["achieved_goal"].shape,
                        dtype=np.float64,
                    ),
                    observation=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["observation"].shape,
                        dtype=np.float64,
                    ),
                )
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> dict[str, np.ndarray]:
        """获取观察和目标。"""
        if not self.observer_vehicle:
            return OrderedDict(
                [
                    ("observation", np.zeros((len(self.features),))),
                    ("achieved_goal", np.zeros((len(self.features),))),
                    ("desired_goal", np.zeros((len(self.features),))),
                ]
            )

        obs = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        )
        goal = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.goal.to_dict()])[self.features]
        )
        obs = OrderedDict(
            [
                ("observation", obs / self.scales),
                ("achieved_goal", obs / self.scales),
                ("desired_goal", goal / self.scales),
            ]
        )
        return obs


class AttributesObservation(ObservationType):
    """属性观察类。"""

    def __init__(self, env: AbstractEnv, attributes: list[str], **kwargs: dict) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        """返回观察空间。"""
        try:
            obs = self.observe()
            return spaces.Dict(
                {
                    attribute: spaces.Box(
                        -np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float64
                    )
                    for attribute in self.attributes
                }
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> dict[str, np.ndarray]:
        """获取属性观察值。"""
        return OrderedDict(
            [(attribute, getattr(self.env, attribute)) for attribute in self.attributes]
        )


class MultiAgentObservation(ObservationType):
    """多代理观察类，观察所有受控车辆的状态。"""

    def __init__(self, env: AbstractEnv, observation_config: dict, **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        # 为每个受控车辆创建观察类型实例
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        """返回所有代理观察空间的元组。"""
        return spaces.Tuple(
            [obs_type.space() for obs_type in self.agents_observation_types]
        )

    def observe(self) -> tuple:
        """观察所有代理的状态并返回一个元组。"""
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


class TupleObservation(ObservationType):
    """元组观察类，组合多个观察类型。"""

    def __init__(self, env: AbstractEnv, observation_configs: list[dict], **kwargs) -> None:
        super().__init__(env)
        # 根据配置创建观察类型实例
        self.observation_types = [
            observation_factory(self.env, obs_config)
            for obs_config in observation_configs
        ]

    def space(self) -> spaces.Space:
        """返回组合观察空间的元组。"""
        return spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> tuple:
        """观察所有组合的状态并返回一个元组。"""
        return tuple(obs_type.observe() for obs_type in self.observation_types)


class ExitObservation(KinematicObservation):
    """特定于 exit_env，观察距离下一个出口车道的距离。"""

    def observe(self) -> np.ndarray:
        """观察状态并返回与出口车道的距离。"""
        if not self.env.road:
            return np.zeros(self.space().shape)

        # 添加自我车辆信息
        ego_dict = self.observer_vehicle.to_dict()
        exit_lane = self.env.road.network.get_lane(("1", "2", -1))
        ego_dict["x"] = exit_lane.local_coordinates(self.observer_vehicle.position)[0]
        df = pd.DataFrame.from_records([ego_dict])[self.features]

        # 添加附近车辆信息
        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = pd.concat(
                [
                    df,
                    pd.DataFrame.from_records(
                        [
                            v.to_dict(
                                origin, observe_intentions=self.observe_intentions
                            )
                            for v in close_vehicles[-self.vehicles_count + 1:]
                        ]
                    )[self.features],
                ],
                ignore_index=True,
            )
        # 归一化和截断
        if self.normalize:
            df = self.normalize_obs(df)
        # 填充缺失行
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # 重排
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # 扁平化
        return obs.astype(self.space().dtype)


class LidarObservation(ObservationType):
    """激光雷达观察类，观察周围环境。"""

    DISTANCE = 0  # 距离索引
    SPEED = 1  # 速度索引

    def __init__(
            self,
            env,
            cells: int = 16,  # 雷达细分单元数量
            maximum_range: float = 60,  # 最大范围
            normalize: bool = True,  # 是否归一化
            **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.cells = cells
        self.maximum_range = maximum_range
        self.normalize = normalize
        self.angle = 2 * np.pi / self.cells  # 每个单元的角度
        self.grid = np.ones((self.cells, 1)) * float("inf")  # 初始化网格
        self.origin = None  # 原点

    def space(self) -> spaces.Space:
        """返回观察空间。"""
        high = 1 if self.normalize else self.maximum_range
        return spaces.Box(shape=(self.cells, 2), low=-high, high=high, dtype=np.float32)

    def observe(self) -> np.ndarray:
        """观察状态并返回激光雷达数据。"""
        obs = self.trace(
            self.observer_vehicle.position, self.observer_vehicle.velocity
        ).copy()
        if self.normalize:
            obs /= self.maximum_range
        return obs

    def trace(self, origin: np.ndarray, origin_velocity: np.ndarray) -> np.ndarray:
        """跟踪周围环境，计算距离和速度。"""
        self.origin = origin.copy()
        self.grid = np.ones((self.cells, 2)) * self.maximum_range

        # 遍历所有障碍物
        for obstacle in self.env.road.vehicles + self.env.road.objects:
            if obstacle is self.observer_vehicle or not obstacle.solid:
                continue
            center_distance = np.linalg.norm(obstacle.position - origin)
            if center_distance > self.maximum_range:
                continue
            center_angle = self.position_to_angle(obstacle.position, origin)
            center_index = self.angle_to_index(center_angle)
            distance = center_distance - obstacle.WIDTH / 2
            if distance <= self.grid[center_index, self.DISTANCE]:
                direction = self.index_to_direction(center_index)
                velocity = (obstacle.velocity - origin_velocity).dot(direction)
                self.grid[center_index, :] = [distance, velocity]

            # 计算障碍物的角度范围
            corners = utils.rect_corners(
                obstacle.position, obstacle.LENGTH, obstacle.WIDTH, obstacle.heading
            )
            angles = [self.position_to_angle(corner, origin) for corner in corners]
            min_angle, max_angle = min(angles), max(angles)
            if (
                    min_angle < -np.pi / 2 < np.pi / 2 < max_angle
            ):  # 处理障碍物角度包围 +pi
                min_angle, max_angle = max_angle, min_angle + 2 * np.pi
            start, end = self.angle_to_index(min_angle), self.angle_to_index(max_angle)
            if start < end:
                indexes = np.arange(start, end + 1)
            else:  # 处理障碍物角度包围 0
                indexes = np.hstack(
                    [np.arange(start, self.cells), np.arange(0, end + 1)]
                )

            # 计算这些区域的实际距离
            for index in indexes:
                direction = self.index_to_direction(index)
                ray = [origin, origin + self.maximum_range * direction]
                distance = utils.distance_to_rect(ray, corners)
                if distance <= self.grid[index, self.DISTANCE]:
                    velocity = (obstacle.velocity - origin_velocity).dot(direction)
                    self.grid[index, :] = [distance, velocity]
        return self.grid

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        """将位置转换为角度。"""
        return (
                np.arctan2(position[1] - origin[1], position[0] - origin[0])
                + self.angle / 2
        )

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        """将位置转换为索引。"""
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        """将角度转换为索引。"""
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: int) -> np.ndarray:
        """根据索引计算方向。"""
        return np.array([np.cos(index * self.angle), np.sin(index * self.angle)])

class CustomObservation(ObservationType):
    def __init__(self, env):
        """
        自定义的观测类，用于从环境中提取所需的观测信息。
        :param env: 环境实例
        """
        self.env = env

    def space(self):
        """
        定义观测空间，指定状态变量的范围。
        :return: 观测空间
        """
        # 将形状改为 (5,) 与实际返回的观测值相匹配
        return spaces.Box(low=np.array([0, 0, 0, 0, 0]), high=np.array([100, 3, 1000, 30, 1]),
                          dtype=np.float32)

    def observe(self):
        """
        获取环境状态的观测，包括：
            - 受控车辆的速度
            - 受控车辆所在的车道
            - 与前车的车间距
            - 前车的速度
            - 受控车辆是否发生碰撞（终止标志）
        :return: np.array，包含上述观测状态
        """
        # 检查 controlled_vehicle 是否已经初始化
        if not hasattr(self.env, 'controlled_vehicle') or self.env.controlled_vehicle is None:
            # 返回一个默认状态，避免在未初始化时访问 controlled_vehicle
            return np.array([0.0, 0, np.inf, 0.0, 0.0], dtype=np.float32)

        controlled_vehicle = self.env.controlled_vehicle
        lane_id = controlled_vehicle.lane_index[2]  # 获取所在车道的ID
        speed = controlled_vehicle.speed  # 受控车辆的速度

        # 获取与前车的距离和前车速度
        front_vehicle, _ = controlled_vehicle.road.neighbour_vehicles(controlled_vehicle)
        if front_vehicle:
            distance_to_front = front_vehicle.position[0] - controlled_vehicle.position[0]
            front_speed = front_vehicle.speed
        else:
            distance_to_front = np.inf  # 如果没有前车，设置一个较大的距离
            front_speed = 0.0  # 没有前车则认为前车速度为0

        # 添加受控车辆是否发生碰撞的标志
        crashed = float(controlled_vehicle.crashed)  # 1.0 表示碰撞，0.0 表示没有碰撞

        # 返回观测状态，包括速度、车道ID、与前车距离、前车速度、是否发生碰撞
        return np.array([speed, lane_id, distance_to_front, front_speed, crashed], dtype=np.float32)


def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    """观察类型工厂，根据配置生成不同的观察类型。"""
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "TupleObservation":
        return TupleObservation(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "ExitObservation":
        return ExitObservation(env, **config)
    elif config["type"] == "CustomObservation":
        return CustomObservation(env)
    else:
        raise ValueError("Unknown observation type")
