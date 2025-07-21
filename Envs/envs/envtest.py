from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

import numpy as np
from gym import spaces
from gym.utils import seeding
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils


class RSU:
    def __init__(self, position, vertical_distance, coverage_radius):
        self.position = position
        self.vertical_distance = vertical_distance
        self.coverage_radius = coverage_radius
        self.state = None

    def receive_state(self, vehicle_state):
        self.state = vehicle_state

    def calculate_distance(self, vehicle_position):
        rsu_x, rsu_y = self.position
        vehicle_x, vehicle_y = vehicle_position
        return np.sqrt((rsu_x - vehicle_x) ** 2 + (rsu_y - vehicle_y) ** 2 + self.vertical_distance ** 2)

    def calculate_channel_gain(self, vehicle_position, power_transmit, interference_power, noise_density,
                               bandwidth, fc=5.9):
        distance = self.calculate_distance(vehicle_position)
        path_loss = 28 + 22 * np.log10(distance) + 20 * np.log10(fc)
        channel_gain = 10 ** (-path_loss / 10)
        effective_power = power_transmit * channel_gain
        sinr = effective_power / (noise_density * bandwidth + interference_power)
        channel_capacity = bandwidth * np.log2(1 + sinr)
        return channel_capacity

    def calculate_transmission_delay(self, data_size, channel_capacity):
        if channel_capacity == 0:
            return float('inf')
        return data_size / channel_capacity

    def clear(self):
        self.state = None


class testEnv(AbstractEnv):
    def __init__(self, render_mode=None, config=None):
        self._initialized = False
        super().__init__(config=config)
        rsu_x = self.config["distance"] / 2
        lane_width = 3.5
        rsu_y = (self.config["lanes_count"] * lane_width) / 2
        rsu_position = (rsu_x, rsu_y)
        vertical_distance = 25
        coverage_radius = 5000
        self.rsu = RSU(position=rsu_position, vertical_distance=vertical_distance,
                       coverage_radius=coverage_radius)
        self.controlled_vehicle = None
        self.render_mode = render_mode
        self._initialized = True
        self.reset()

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 14,
                "lanes_count": 2,
                "vehicles_count": 25,
                "distance": 2000,
                "ego_spacing": 2,
                "controlled_vehicles": 1,
                "observation": {"type": "CustomObservation"},
                "action": {"type": "DiscreteMetaAction"},
                "initial_lane_id": None,
                "vehicles_density": 0.8,
                "lane_change_reward": 10,
                "collision_penalty": -15,
                "high_speed_reward": 10,
                "delay_reward_factor": 10,
                "reward_speed_range": [60, 100],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return cfg

    def step(self, action):
        result = super().step(action)
        obs, reward, done = result[:3]
        info = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
        truncated = info.get('TimeLimit.truncated', False)
        if self.controlled_vehicle is None:
            raise RuntimeError("Controlled vehicle is not set.")
        vehicle_position = self.controlled_vehicle.position[:2]
        power_transmit = 1.0
        interference_power = 0
        noise_density = 1e-9
        bandwidth = 1e6
        channel_capacity = self.rsu.calculate_channel_gain(
            vehicle_position=vehicle_position,
            power_transmit=power_transmit,
            interference_power=interference_power,
            noise_density=noise_density,
            bandwidth=bandwidth
        )
        data_size = self.calculate_data_size()
        transmission_delay = self.rsu.calculate_transmission_delay(data_size=data_size,
                                                                   channel_capacity=channel_capacity)
        info['channel_capacity'] = channel_capacity
        info['transmission_delay'] = transmission_delay
        vehicle_state = self.vehicle_state()
        self.rsu.receive_state(vehicle_state)
        reward = self._reward(action, transmission_delay)
        return obs, reward, done, truncated, info

    def calculate_data_size(self):
        vehicle_state = self.vehicle_state()
        state_size = len(vehicle_state) * 4 * 8
        image_data_size = 0
        if self.render_mode == 'rgb_array':
            image = self.observation_type.observe_image()
            if image is not None:
                image_data_size = image.size * 8
        total_data_size = state_size + image_data_size
        return total_data_size

    def vehicle_state(self):
        if self.controlled_vehicle is None:
            raise RuntimeError("controlled_vehicle not set.")
        return np.array([
            self.controlled_vehicle.position[0],
            self.controlled_vehicle.position[1],
            self.controlled_vehicle.speed,
            float(self.controlled_vehicle.crashed)
        ])

    def reset(self, seed=None, **kwargs):
        if not self._initialized:
            return
        self.np_random, seed = seeding.np_random(seed)
        self._reset()
        self.rsu.clear()
        if not hasattr(self, 'controlled_vehicles') or not self.controlled_vehicles:
            raise RuntimeError("controlled_vehicles has not been properly initialized.")
        self.controlled_vehicle = self.controlled_vehicles[0]
        observation = self.observation_type.observe()
        return observation, {}

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False),
        )

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(
            self.config.get("other_vehicles_type", "highway_env.vehicle.kinematics.Vehicle"))
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )
        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=50,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.speed = 20
                vehicle.target_speed = 20
                vehicle.plan_route_to = lambda destination: None
                vehicle.randomize_behavior = lambda: None
                vehicle.act = lambda action=None: None
                vehicle.controller = None
                vehicle.check_collisions = False
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action, transmission_delay: float = 0.0) -> float:
        rewards = self._rewards(action)
        safety_distance = 10
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        distance_to_front = (
            front_vehicle.position[0] - self.controlled_vehicle.position[0]
            if front_vehicle
            else float('inf')
        )
        reward = (
                1.5 * rewards["collision_reward"] +
                1.0 * rewards["lane_change_reward"] +
                0.8 * rewards["high_speed_reward"]
        )
        if action in [0, 2] and distance_to_front > safety_distance:
            reward += 5
        if action == 3 and (front_vehicle is None or distance_to_front > safety_distance):
            reward += 3
        delay_reward = max(0.0, (5.0 - transmission_delay)) * 10
        reward += delay_reward * self.config["delay_reward_factor"]
        reward = np.clip(reward, -50, 50)
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        forward_speed = self.controlled_vehicle.speed
        safety_distance = 20
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        distance_to_front = (
            front_vehicle.position[0] - self.controlled_vehicle.position[0]
            if front_vehicle
            else float('inf')
        )
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        if self.controlled_vehicle.crashed and (front_vehicle is not None and front_vehicle.crashed):
            collision_reward = -10
        elif self.controlled_vehicle.crashed:
            collision_reward = -5
        else:
            collision_reward = 0.0
        lane_change_reward = (
            5 if action in [0, 2] and distance_to_front > safety_distance else 0.0
        )
        high_speed_reward = np.clip(scaled_speed, 0, 1) * 8
        proximity_penalty = -5 if distance_to_front < safety_distance else 0.0
        stationary_penalty = -3 if self.controlled_vehicle.speed < 22 else 0.0
        return {
            "collision_reward": collision_reward,
            "lane_change_reward": lane_change_reward,
            "high_speed_reward": high_speed_reward,
            "proximity_penalty": proximity_penalty,
            "stationary_penalty": stationary_penalty,
        }

    def _is_terminated(self) -> bool:
        return (
                self.controlled_vehicle.crashed or
                self.controlled_vehicle.position[0] >= self.config["distance"] or
                (self.config["offroad_terminal"] and not self.controlled_vehicle.on_road)
        )

    def _is_truncated(self) -> bool:
        return self.controlled_vehicle.position[0] >= self.config["distance"]

    def render(self, mode='human'):
        if hasattr(super(), 'render'):
            try:
                super().render(mode=mode)
            except TypeError:
                super().render()
        else:
            if mode == 'human':
                print("Rendering in human mode... (this is a placeholder)")
            elif mode == 'rgb_array':
                print("Rendering as RGB array... (this is a placeholder)")
            else:
                raise ValueError(f"Unsupported render mode: {mode}")
