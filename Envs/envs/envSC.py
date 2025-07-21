from __future__ import annotations
from UDeepSC.model import UDeepSC_model
from torch import numel
from highway_env.envs.common.action import Action
import os
import random
from PIL import Image
import numpy as np
from gym.utils import seeding
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils
from functools import partial
from UDeepSC.trans_deocer import Decoder
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from transformers import BertModel, BertTokenizer
from torchvision import transforms
BERT_PATH = '.\\bert'
from UDeepSC.model_util import *
from UDeepSC.model_util import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from UDeepSC.base_args import IMGC_NUMCLASS, TEXTC_NUMCLASS, IMGR_LENGTH, TEXTR_NUMCLASS, VQA_NUMCLASS, MSA_NUMCLASS
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class SC:
    def __init__(self, position, vertical_distance, coverage_radius,kappa_model=True,theta_std_db=8,
                 img_size=224, patch_size=32, encoder_in_chans=3, encoder_num_classes=0,
                 img_embed_dim=768, text_embed_dim=768,  img_encoder_depth=6,
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=512,
                 decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, num_classes=0):
        self.position = position
        self.vertical_distance = vertical_distance
        self.coverage_radius = coverage_radius
        self.kappa_model = kappa_model
        self.theta_std_db = theta_std_db
        self.state = None
        self.codebook = nn.ModuleDict()
        self.bit_per_digit = 4
        self.codebook['img'] = VectorQuantizer(num_embeddings=8,
                                               embedding_dim=img_embed_dim,
                                               quan_bits=4)
        self.codebook['text'] = VectorQuantizer(num_embeddings=8,
                                                embedding_dim=text_embed_dim,
                                                quan_bits=4)
        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim,
                               drop_rate=drop_rate)
    def decode_state(self,x_img,x_text,img_encode,text_encode):
        img_decoded = self.decoder(x_img,memory=img_encode)
        text_decoded = self.decoder(x_text,memory=text_encode)
        img_size = img_decoded.numel() * 4
        text_size = text_decoded.numel() * 4
        decode_size=img_size+text_size
        return decode_size
    def receive_state(self, vehicle_state):
        self.state = vehicle_state
    def calculate_distance(self, vehicle_position):
        rsu_x, rsu_y = self.position
        vehicle_x, vehicle_y = vehicle_position
        return np.sqrt((rsu_x - vehicle_x) ** 2 + (rsu_y - vehicle_y) ** 2 + self.vertical_distance ** 2)
    def calculate_channel_gain(self, vehicle_position, power_transmit, interference_power, noise_density,
                               bandwidth):
        distance = self.calculate_distance(vehicle_position)
        distance_km = distance / 1000
        if self.kappa_model:
            path_loss_dB = 128.1 + 37.6 * np.log10(distance_km)
            kappa = 10 ** (-path_loss_dB / 10)
        mu_t = np.random.rayleigh(scale=1)
        theta_t_db = np.random.normal(0, self.theta_std_db)
        theta_t = 10 ** (theta_t_db / 10)
        channel_gain = mu_t * np.sqrt(kappa * theta_t)
        effective_power = power_transmit * channel_gain
        sinr = effective_power / (noise_density)
        channel_capacity = bandwidth * np.log2(1 + sinr)
        return channel_capacity
    def calculate_transmission_delay(self, data_size, channel_capacity):
        if channel_capacity == 0:
            return float('inf')
        return data_size / channel_capacity
    def clear(self):
        self.state = None
vertical_distance = 25
coverage_radius = 2000
lane_width = 3.5
power_transmit = 10 ** (23 / 10) / 1000
interference_power = 0
noise_density = 1e-9
bandwidth = 10e6
C = 100000
tau = 0.01
xi_1 = 0.5
xi_2 = 1
a_t = 5
cost=500000
class testEnvSC(AbstractEnv,nn.Module):
    def __init__(self, render_mode=None, config=None,
                 img_size=224, patch_size=51, encoder_in_chans=3, encoder_num_classes=0,
                 img_embed_dim=768, text_embed_dim=768,  img_encoder_depth=6,
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=512,
                 decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, num_classes=0):
        super().__init__(config=config)
        self._initialized = False
        rsu_x = self.config["distance"] / 2
        rsu_y = (self.config["lanes_count"] * lane_width) / 2
        rsu_position = (rsu_x, rsu_y)
        self.rsu = SC(position=rsu_position, vertical_distance=vertical_distance,
                       coverage_radius=coverage_radius)
        self.controlled_vehicle = None
        self.render_mode = render_mode
        self.img_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=img_embed_dim,
            depth=img_encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.text_encoder = BertModel.from_pretrained(BERT_PATH)
        self.num_symbols_img = 16
        self.num_symbols_text = 16
        self.text_encoder_to_channel = nn.Linear(text_embed_dim, self.num_symbols_text)
        self.img_encoder_to_channel = nn.Linear(img_embed_dim, self.num_symbols_img)
        self.text_channel_to_decoder = nn.Linear(text_embed_dim, decoder_embed_dim)
        self.img_channel_to_decoder = nn.Linear(img_embed_dim, decoder_embed_dim)
        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['imgr'] = nn.Embedding(64, decoder_embed_dim)
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)
        self.head = nn.ModuleDict()
        self.head['imgc'] = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)
        self.head['msa'] = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)
        self.codebook = nn.ModuleDict()
        self.bit_per_digit = 4
        self.codebook['img'] = VectorQuantizer(num_embeddings=8,
                                               embedding_dim=img_embed_dim,
                                               quan_bits=4)
        self.codebook['text'] = VectorQuantizer(num_embeddings=8,
                                                embedding_dim=text_embed_dim,
                                                quan_bits=4)
        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim,
                               drop_rate=drop_rate)
        self._initialized = True
        self.reset()
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                 "simulation_frequency": 5,
                "lanes_count": 2,
                "vehicles_count": 55,
                "distance": 3000,
                "ego_spacing": 2,
                "controlled_vehicles": 1,
                "observation": {"type": "CustomObservation"},
                "action": {"type": "DiscreteMetaAction"},
                "initial_lane_id": None,
                "vehicles_density": 1.2,
                "lane_change_reward": 1.5,
                "right_lane_reward": 0.1,
                "collision_penalty": -1.2,
                "reward_speed_range": [40, 60],
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
        text = self.vehicle_state()
        image_dir = '.\image_data'
        image=self.load_random_image(image_dir)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img = transform(image).unsqueeze(0)
        img_encode = self.img_encoder(img,ta_perform='imgc')
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        state_str = ', '.join(map(str, text))
        input = tokenizer(state_str, return_tensors='pt')
        text_encode = self.text_encoder(**input)
        x_img= self.codebook['img'](img_encode)
        x_text= self.codebook['text'](text_encode)
        image_data_size = img_encode.numel() * 4
        if isinstance(text_encode, BaseModelOutputWithPoolingAndCrossAttentions):
            text_encode = text_encode.last_hidden_state
        else:
            text_encode = text_encode
        text_data_size = text_encode.numel() * 4
        total_encoded_size = text_data_size + image_data_size
        encoded_time=total_encoded_size/cost
        decoded_time=self.rsu.decode_state(x_img,x_text,img_encode,text_encode)/(cost)
        vehicle_position = self.controlled_vehicle.position[:2]
        channel_capacity = self.rsu.calculate_channel_gain(
            vehicle_position=vehicle_position,
            power_transmit=power_transmit,
            interference_power=interference_power,
            noise_density=noise_density,
            bandwidth=bandwidth
        )
        min_len = min(x_img.size(1), x_text.size(1))
        x_img = x_img[:, :min_len, :]
        x_text = x_text[:, :min_len, :]
        encoded_state = torch.cat((x_img, x_text), dim=-1)
        data_size = self.calculate_data_size(encoded_state)
        transmission_delay = self.rsu.calculate_transmission_delay(data_size=data_size,
                                                                   channel_capacity=channel_capacity)
        sum_time = encoded_time + decoded_time + transmission_delay
        info['channel_capacity'] = channel_capacity
        info['sum_time'] = sum_time
        print(encoded_time,decoded_time,transmission_delay)
        reward = self._reward(action, sum_time)
        return obs, reward, done, truncated, info
    def load_random_image(self,image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg'))]
        random_image_file = random.choice(image_files)
        image_path = os.path.join(image_dir, random_image_file)
        image = Image.open(image_path)
        return image
    def calculate_data_size(self,encoded_state):
        state_size = len(encoded_state) * 4*8
        return state_size
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
        self._initialized=True
        self.np_random, seed = seeding.np_random(seed)
        self._reset()
        if hasattr(self, 'rsu'):
            self.rsu.clear()
        else:
            rsu_x = self.config["distance"] / 2
            rsu_y = (self.config["lanes_count"] * lane_width) / 2
            rsu_position = (rsu_x, rsu_y)
            self.rsu = SC(position=rsu_position, vertical_distance=vertical_distance,
                       coverage_radius=coverage_radius)
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
                speed=40,
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
    def _reward(self, action: Action,sum_time:float = 0.0) -> float:
        rewards = self._rewards(action)
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        if front_vehicle:
            v_t = self.controlled_vehicle.velocity[0]
            v_f_t = front_vehicle.velocity[0]
            dynamic_safety_distance = xi_1 * abs(v_t - v_f_t) + xi_2 * a_t
        else:
            dynamic_safety_distance = float('inf')
        safety_distance = dynamic_safety_distance
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        distance_to_front = (
            front_vehicle.position[0] - self.controlled_vehicle.position[0]
            if front_vehicle
            else float('inf')
        )
        Total_reward = C / (sum_time + tau)
        if sum_time == 0:
            return 0
        delay_penalty = 0
        if action in [0, 2] and distance_to_front < safety_distance:
            delay_penalty = 1
        constr_penalty = 0
        total_bandwidth = random.uniform(bandwidth - 500, bandwidth + 10)
        if total_bandwidth > bandwidth:
            constr_penalty = 1
        total_power = random.uniform(power_transmit - 0.1, power_transmit + 0.01)
        if total_power > power_transmit:
            constr_penalty = 1
        reward = (
                1.2 * rewards["collision_reward"] +
                1.0 * rewards["lane_change_reward"] +
                1.2 * rewards["high_speed_reward"] +
                1.0 * rewards["proximity_penalty"] +
                1.0 * rewards["stationary_penalty"]
        )
        if action == 3 and (front_vehicle is None or distance_to_front > safety_distance):
            reward += 0.3
        if action in [0, 2] and distance_to_front > 80:
            reward -= 1.5
        sum=Total_reward+reward-0.8*delay_penalty-0.2*constr_penalty
        sum= np.clip(sum, -50, 50)
        return sum

    def _rewards(self, action: Action) -> dict[str, float]:
    front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
    forward_speed = self.controlled_vehicle.speed

    if front_vehicle:
        v_t = self.controlled_vehicle.velocity[0]
        v_f_t = front_vehicle.velocity[0]
        dynamic_safety_distance = xi_1 * abs(v_t - v_f_t) + xi_2 * a_t
        distance_to_front = front_vehicle.position[0] - self.controlled_vehicle.position[0]
    else:
        dynamic_safety_distance = float('inf')
        distance_to_front = float('inf')

    collision_reward = -1.5 if (self.controlled_vehicle.crashed and (front_vehicle and front_vehicle.crashed)) \
                       else -0.5 if self.controlled_vehicle.crashed else 0.0

    lane_change_reward = (
        2 if action in [0, 2] and distance_to_front > dynamic_safety_distance else 0.0
    )

    scaled_speed = np.clip(
        utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1]),
        0, 1
    )
    high_speed_reward = scaled_speed * 0.5

    proximity_penalty = -0.3 if distance_to_front < dynamic_safety_distance else 0.0
    stationary_penalty = -5 if self.controlled_vehicle.speed < self.config["reward_speed_range"][0] else 0.0

    return {
        "collision_reward": collision_reward,
        "lane_change_reward": lane_change_reward,
        "high_speed_reward": high_speed_reward,
        "proximity_penalty": proximity_penalty,
        "stationary_penalty": stationary_penalty,
    }
