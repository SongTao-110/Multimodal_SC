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
        self.kappa_model = kappa_model  # 是否使用路径损耗模型计算 kappa
        self.theta_std_db = theta_std_db  # 阴影衰落的标准差，单位为 dB
        self.state = None
        # 定义码本（Vector Quantization，用于离散化表示）
        self.codebook = nn.ModuleDict()
        self.bit_per_digit = 4  # 每个数字使用的比特数

        # 图像和文本的量化器（将编码后的表示离散化）
        self.codebook['img'] = VectorQuantizer(num_embeddings=8,
                                               embedding_dim=img_embed_dim,
                                               quan_bits=4)
        self.codebook['text'] = VectorQuantizer(num_embeddings=8,
                                                embedding_dim=text_embed_dim,
                                                quan_bits=4)

        # 初始化解码器
        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim,
                               drop_rate=drop_rate)
    def decode_state(self,x_img,x_text,img_encode,text_encode):
        """
        使用解码器解码车辆传输的状态。
        :param encoded_data: 编码后的语义数据
        :return: 解码后的原始数据
        """
        # img_decoded = self.codebook['img'].decoder(x_img)  # 解码图像
        # text_decoded = self.codebook['text'].decoder(x_text,ta_perform='textr')  # 解码文本
        img_decoded = self.decoder(x_img,memory=img_encode)  # 解码图像
        text_decoded = self.decoder(x_text,memory=text_encode)  # 解码文本
        img_size = img_decoded.numel() * 4
        text_size = text_decoded.numel() * 4
        decode_size=img_size+text_size
        return decode_size
    def receive_state(self, vehicle_state):
        """
        接收来自车辆编码后的状态。
        :param vehicle_state: 车辆的状态，包括位置和速度等
        """
        self.state = vehicle_state

    def calculate_distance(self, vehicle_position):
        """
        计算车辆与RSU之间的距离。
        :param vehicle_position: 车辆的位置 (x, y) 坐标
        :return: 距离值
        """
        rsu_x, rsu_y = self.position
        vehicle_x, vehicle_y = vehicle_position
        return np.sqrt((rsu_x - vehicle_x) ** 2 + (rsu_y - vehicle_y) ** 2 + self.vertical_distance ** 2)

    def calculate_channel_gain(self, vehicle_position, power_transmit, interference_power, noise_density,
                               bandwidth):
        # 计算车辆和RSU之间的距离
        distance = self.calculate_distance(vehicle_position)
        distance_km = distance / 1000
        if self.kappa_model:
            path_loss_dB = 128.1 + 37.6 * np.log10(distance_km)
            # 计算信道增益 h(t) = mu(t) * sqrt(d(t)^-kappa * theta(t))
            kappa = 10 ** (-path_loss_dB / 10)  # 将路径损耗转换为路径损耗因子
        # 计算衰落随机变量 mu(t) 和 theta(t)
        mu_t = np.random.rayleigh(scale=1)  # 使用瑞利衰落分布生成 mu(t)
        theta_t_db = np.random.normal(0, self.theta_std_db)  # 阴影衰落，单位为 dB
        theta_t = 10 ** (theta_t_db / 10)  # 转换为线性值

        # 计算信道增益 h(t) = mu(t) * sqrt(d(t)^-kappa * theta(t))
        channel_gain = mu_t * np.sqrt(kappa * theta_t)

        # 计算信道增益的功率
        effective_power = power_transmit * channel_gain

        # 计算信道容量 R(t)
        sinr = effective_power / (noise_density)
        channel_capacity = bandwidth * np.log2(1 + sinr)

        return channel_capacity

    def calculate_transmission_delay(self, data_size, channel_capacity):
        """
        计算数据从车辆传输到RSU的时延。
        :param data_size: 数据大小（单位：比特）
        :param channel_capacity: 信道容量（单位：比特每秒）
        :return: 传输时延（单位：秒）
        """
        if channel_capacity == 0:
            return float('inf')  # 如果信道容量为 0，时延为无穷大（传输不可能完成）
        return data_size / channel_capacity

    def clear(self):
        self.state = None
vertical_distance = 25  # RSU与道路的垂直距离
coverage_radius = 2000   # RSU的覆盖半径
lane_width = 3.5  # 单车道宽度
power_transmit = 10 ** (23 / 10) / 1000  # 假设车辆的发射功率为 32dbm
interference_power = 0  # 假设其他干扰功率为0 无干扰
noise_density = 1e-9  # 假设噪声密度为 1e-9 W/Hz
bandwidth = 10e6  # 假设带宽为 10 MHz
# 定义常数
C = 100000  # 延迟奖励的常数C
tau = 0.01  # 延迟调整因子tau，用于防止延迟为0时的除零错误
# 设置动态安全距离
xi_1 = 0.5  # 相对速度调整系数
xi_2 = 1  # 加速度调整系数
a_t = 5  # 当前车辆的加速度
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
        # 初始化 RSU
        rsu_x = self.config["distance"] / 2  # 道路中心
        rsu_y = (self.config["lanes_count"] * lane_width) / 2
        rsu_position = (rsu_x, rsu_y)
        self.rsu = SC(position=rsu_position, vertical_distance=vertical_distance,
                       coverage_radius=coverage_radius)
        self.controlled_vehicle = None
        self.render_mode = render_mode
        # 初始化图像编码器（基于Vision Transformer，ViT）
        # 这里使用了一个ViTEncoder类来作为图像的编码器，它会将输入图像转换成特征向量。
        self.img_encoder = ViTEncoder(
            img_size=img_size,  # 图像大小（224x224等）
            patch_size=patch_size,  # 图像分块大小
            in_chans=encoder_in_chans,  # 输入图像的通道数（例如RGB为3）
            num_classes=encoder_num_classes,  # 图像编码器输出类别数
            embed_dim=img_embed_dim,  # 图像的嵌入维度
            depth=img_encoder_depth,  # 编码器层数
            num_heads=encoder_num_heads,  # 注意力头数
            mlp_ratio=mlp_ratio,  # 多层感知机的比率
            qkv_bias=qkv_bias,  # 是否使用QKV偏置
            drop_rate=drop_rate,  # Dropout比率
            drop_path_rate=drop_path_rate,  # DropPath比率
            norm_layer=norm_layer,  # 归一化层类型
            init_values=init_values,  # 初始化权重的值
            use_learnable_pos_emb=use_learnable_pos_emb  # 是否使用可学习的位置编码
        )

        # 初始化文本编码器（使用BERT预训练模型）
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        # 使用预训练的BERT模型
        self.text_encoder = BertModel.from_pretrained(BERT_PATH)

        # 设置符号数量（在本模型中用于映射到解码器）
        self.num_symbols_img = 16
        self.num_symbols_text = 16
        # self.num_symbols_spe = 16

        # 定义图像、文本的通道到解码器的线性映射
        self.text_encoder_to_channel = nn.Linear(text_embed_dim, self.num_symbols_text)
        self.img_encoder_to_channel = nn.Linear(img_embed_dim, self.num_symbols_img)

        # 定义通道到解码器的映射（将编码器输出的特征映射到解码器输入的维度）
        self.text_channel_to_decoder = nn.Linear(text_embed_dim, decoder_embed_dim)
        self.img_channel_to_decoder = nn.Linear(img_embed_dim, decoder_embed_dim)

        # 定义任务字典（用于不同的任务，如分类、问答、回归等）
        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)  # 图像分类
        self.task_dict['imgr'] = nn.Embedding(64, decoder_embed_dim)  # 图像回归
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)  # 文本分类
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)  # 文本回归

        # 定义每个任务对应的头部（用于输出任务特定的预测结果）
        self.head = nn.ModuleDict()
        self.head['imgc'] = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)  # 图像分类头
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)  # 文本分类头
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)  # 文本回归头
        self.head['msa'] = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)  # 多任务学习头

        # 定义码本（Vector Quantization，用于离散化表示）
        self.codebook = nn.ModuleDict()
        self.bit_per_digit = 4  # 每个数字使用的比特数

        # 图像和文本的量化器（将编码后的表示离散化）
        self.codebook['img'] = VectorQuantizer(num_embeddings=8,
                                               embedding_dim=img_embed_dim,
                                               quan_bits=4)
        self.codebook['text'] = VectorQuantizer(num_embeddings=8,
                                                embedding_dim=text_embed_dim,
                                                quan_bits=4)

        # 初始化解码器
        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim,
                               drop_rate=drop_rate)

        # 在所有属性初始化完成后再调用 reset()
        self._initialized = True
        self.reset()

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                 "simulation_frequency": 5,  # 仿真频率
                "lanes_count": 2,  # 车道数量
                "vehicles_count": 55,  # 车辆数量
                "distance": 3000,  # 规定行驶距离 [m]
                "ego_spacing": 2,  # 自车间距
                "controlled_vehicles": 1,  # 受控车辆数量，添加默认值
                "observation": {"type": "CustomObservation"},  # 观察类型
                "action": {"type": "DiscreteMetaAction"},  # 动作类型
                "initial_lane_id": None,  # 初始车道ID
                "vehicles_density": 1.2,  # 车辆密度
                "lane_change_reward": 1.5,  # 每次变道的基础奖励，动态调整
                "right_lane_reward": 0.1,  # 在最右侧车道行驶的奖励
                "collision_penalty": -1.2,  # 增加碰撞惩罚
                "reward_speed_range": [40, 60],  # 奖励速度范围
                "normalize_reward": True,  # 是否归一化奖励
                "offroad_terminal": False,  # 是否允许越界终止
            }
        )
        return cfg

    def step(self, action):
        # 调用父类的 step 方法
        result = super().step(action)
        obs, reward, done = result[:3]
        info = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
        truncated = info.get('TimeLimit.truncated', False)

        # 确保受控车辆已经初始化
        if self.controlled_vehicle is None:
            raise RuntimeError("Controlled vehicle is not set.")
        #定义
        # 获取车辆的观测状态
        text = self.vehicle_state()
        #获取图像数据
        image_dir = '.\image_data'
        image=self.load_random_image(image_dir)
        # 定义图像转换操作，将图像转换为Tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 将图像转换为Tensor
        ])

        # 将图像转换为张量
        img = transform(image).unsqueeze(0)
        # 使用UDeepSC模型的编码器对图像和文本进行编码
        img_encode = self.img_encoder(img,ta_perform='imgc')  # 使用ViT对图像进行编码
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        state_str = ', '.join(map(str, text))  # 假设 text 是一个数值数组
        input = tokenizer(state_str, return_tensors='pt')
        # 使用文本编码器对文本进行编码
        # text_encode = self.text_encoder(input,ta_perform='textc')  # BERT编码
        text_encode = self.text_encoder(**input)  # BERT编码
        x_img= self.codebook['img'](img_encode)  # 对图像进行量化编码
        x_text= self.codebook['text'](text_encode)  # 对文本进行量化编码
        # 假设每个元素是浮点数（4字节）
        image_data_size = img_encode.numel() * 4  # numel()返回张量的总元素数，乘以4表示每个元素4字节

        # 计算文本编码后的数据量大小
        if isinstance(text_encode, BaseModelOutputWithPoolingAndCrossAttentions):
            text_encode = text_encode.last_hidden_state  # 获取实际的 tensor 数据
        else:
            text_encode = text_encode  # 如果已经是 tensor，就直接使用
        text_data_size = text_encode.numel() * 4  # 同样计算文本编码后的数据量，假设每个元素是浮点数

        # 计算总的编码数据大小
        total_encoded_size = text_data_size + image_data_size
        encoded_time=total_encoded_size/cost
        decoded_time=self.rsu.decode_state(x_img,x_text,img_encode,text_encode)/(cost)
        # 模拟通信信道
        vehicle_position = self.controlled_vehicle.position[:2]

        # 计算信道容量
        channel_capacity = self.rsu.calculate_channel_gain(
            vehicle_position=vehicle_position,
            power_transmit=power_transmit,
            interference_power=interference_power,
            noise_density=noise_density,
            bandwidth=bandwidth
        )
        # 调整形状以匹配
        min_len = min(x_img.size(1), x_text.size(1))  # 第二维的最小长度
        x_img = x_img[:, :min_len, :]  # 裁剪到最小长度
        x_text = x_text[:, :min_len, :]
        # 拼接
        encoded_state = torch.cat((x_img, x_text), dim=-1)  # 沿着最后一个维度拼接

        # 计算传输数据大小
        data_size = self.calculate_data_size(encoded_state)

        # 计算传输时延
        transmission_delay = self.rsu.calculate_transmission_delay(data_size=data_size,
                                                                   channel_capacity=channel_capacity)
        # 将信道容量和传输时延添加到 info 字典中
        sum_time = encoded_time + decoded_time + transmission_delay
        info['channel_capacity'] = channel_capacity
        info['sum_time'] = sum_time
        print(encoded_time,decoded_time,transmission_delay)
        # 自定义奖励逻辑
        reward = self._reward(action, sum_time)

        return obs, reward, done, truncated, info
    # def sum_time(self,encoded_time,decoded_time,transmission_delay):
    #     sum_time = encoded_time + decoded_time + transmission_delay
    #     return sum_time
    def load_random_image(self,image_dir):
        # 获取所有图像文件（假设图像文件是以常见的格式存储，例如jpg, png）
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg'))]

        # 随机选择一张图像文件
        random_image_file = random.choice(image_files)

        # 获取图像的完整路径
        image_path = os.path.join(image_dir, random_image_file)

        # 打开图像并返回
        image = Image.open(image_path)
        return image

    def calculate_data_size(self,encoded_state):
        """
        计算车辆状态数据的大小（单位：比特）。
        :return: 数据大小（单位：比特）
        """
        # 数据大小计算（假设每个浮点数占用32位，即4字节）
        # 每个状态量为浮点数，因此数据大小为状态量数量乘以4字节，再乘以8得到比特数
        state_size = len(encoded_state) * 4*8   # 以比特为单位

        return state_size

    def vehicle_state(self):
        # 使用 controlled_vehicle 获取状态
        if self.controlled_vehicle is None:
            raise RuntimeError("controlled_vehicle not set.")
        return np.array([
            self.controlled_vehicle.position[0],
            self.controlled_vehicle.position[1],
            self.controlled_vehicle.speed,
            float(self.controlled_vehicle.crashed)  # 添加观测终止标志（是否发生碰撞）
        ])

    def reset(self, seed=None, **kwargs):
        self._initialized=True
        self.np_random, seed = seeding.np_random(seed)
        self._reset()
        if hasattr(self, 'rsu'):
            self.rsu.clear()  # 确保 rsu 已经初始化
        else:
            # 如果 rsu 没有初始化，可以在这里初始化
            rsu_x = self.config["distance"] / 2  # 道路中心
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
        """重置环境状态，创建道路和车辆。"""
        self._create_road()  # 调用方法创建道路
        self._create_vehicles()  # 调用方法创建车辆

    def _create_road(self) -> None:
        """创建一条由直线相邻车道组成的道路。"""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30  # 创建直线道路网络
            ),
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False),  # 是否记录历史轨迹
        )

    def _create_vehicles(self) -> None:
        """创建新的随机车辆，并将其添加到道路上。"""
        other_vehicles_type = utils.class_from_path(
            self.config.get("other_vehicles_type", "highway_env.vehicle.kinematics.Vehicle"))  # 获取其他车辆类型
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]  # 分配车辆数量
        )

        self.controlled_vehicles = []  # 初始化受控车辆列表

        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=40,  # 初始速度提高到 40
                lane_id=self.config["initial_lane_id"],  # 初始车道ID
                spacing=self.config["ego_spacing"],  # 自车间距
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed  # 创建受控车辆
            )

            self.controlled_vehicles.append(vehicle)  # 添加到受控车辆列表
            self.road.vehicles.append(vehicle)  # 添加到道路上的车辆列表

            # 创建其他随机车辆
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]  # 创建随机车辆
                )

                # 设置恒定速度和禁用换道及其他行为
                vehicle.speed = 20  # 设置恒定速度为 20
                vehicle.target_speed = 20  # 确保目标速度也是 20

                # 禁用换道计划方法
                vehicle.plan_route_to = lambda destination: None
                # 禁用随机化行为
                vehicle.randomize_behavior = lambda: None
                # 禁用动作控制，包括加速、减速和换道
                vehicle.act = lambda action=None: None  # 修改 lambda 函数，接受可选的 `action` 参数

                # 禁用其他的加速逻辑
                vehicle.controller = None  # 去掉控制器，以防止自动控制调整速度
                vehicle.check_collisions = False  # 禁用非受控车辆的碰撞检查

                # 添加到道路上的车辆列表
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action,sum_time:float = 0.0) -> float:
        rewards = self._rewards(action)
        # 获取前方车辆
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)

        if front_vehicle:
            v_t = self.controlled_vehicle.velocity[0]  # 当前车辆的速度
            v_f_t = front_vehicle.velocity[0]  # 前车的速度

            # 计算动态安全距离
            dynamic_safety_distance = xi_1 * abs(v_t - v_f_t) + xi_2 * a_t
        else:
            dynamic_safety_distance = float('inf')  # 如果没有前车，则安全距离为无穷大

        # 安全距离设定
        safety_distance = dynamic_safety_distance
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        distance_to_front = (
            front_vehicle.position[0] - self.controlled_vehicle.position[0]
            if front_vehicle
            else float('inf')
        )
        # 计算延迟奖励项: r_{t}^{\text{late}} = C / (T_{\text{total}} + tau)
        Total_reward = C / (sum_time + tau)
        # 时延约束
        if sum_time == 0:
            return 0
        delay_penalty = 0
        if action in [0, 2] and distance_to_front < safety_distance:
            delay_penalty = 1
        # 计算约束惩罚项: r_{t}^{\text{constr}}
        constr_penalty = 0
        total_bandwidth = random.uniform(bandwidth - 500, bandwidth + 10)
        if total_bandwidth > bandwidth:  # 带宽超出限制
            constr_penalty = 1
        total_power = random.uniform(power_transmit - 0.1, power_transmit + 0.01)
        if total_power > power_transmit:  # 功率超出限制
            constr_penalty = 1

        # 调整奖励权重
        reward = (
                1.2 * rewards["collision_reward"] +  # 增加碰撞惩罚权重
                1.0 * rewards["lane_change_reward"] +  # 调整换道奖励权重
                1.2 * rewards["high_speed_reward"] +
                1.0 * rewards["proximity_penalty"] +
                1.0 * rewards["stationary_penalty"]
        )

        # # 鼓励安全换道
        # if action in [0, 2] and distance_to_front > safety_distance:
        #     reward += 0.5

        # 鼓励加速行为
        if action == 3 and (front_vehicle is None or distance_to_front > safety_distance):
            reward += 0.3
        #无车换道惩罚
        if action in [0, 2] and distance_to_front > 80:
            reward -= 1.5
        # 动态范围裁剪
        sum=Total_reward+reward-0.8*delay_penalty-0.2*constr_penalty
        sum= np.clip(sum, -50, 50)
        return sum

    def _rewards(self, action: Action) -> dict[str, float]:
        """
        计算各类奖励的具体值。

        参数:
        - action (Action): 当前采取的动作。

        返回值:
        - dict[str, float]: 奖励名称及其对应的值。
        """
        # 获取前方车辆
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        forward_speed = self.controlled_vehicle.speed  # 当前车辆速度
        if front_vehicle:
            v_t = self.controlled_vehicle.velocity[0]  # 当前车辆的速度
            v_f_t = front_vehicle.velocity[0]  # 前车的速度

            # 计算动态安全距离
            dynamic_safety_distance = xi_1 * abs(v_t - v_f_t) + xi_2 * a_t
        else:
            dynamic_safety_distance = float('inf')  # 如果没有前车，则安全距离为无穷大

        # 安全距离设定
        safety_distance = dynamic_safety_distance
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        distance_to_front = (
            front_vehicle.position[0] - self.controlled_vehicle.position[0]
            if front_vehicle
            else float('inf')
        )

        # 奖励定义
        if self.controlled_vehicle.crashed and (front_vehicle is not None and front_vehicle.crashed):
            collision_reward = -1.5
        elif self.controlled_vehicle.crashed:
            collision_reward = -0.5
        else:
            collision_reward = 0.0
        lane_change_reward = (
            2  # 基础奖励
            if action in [0, 2] and distance_to_front > safety_distance  # 换道且距离足够安全
            else 0.0
        )
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        high_speed_reward = np.clip(scaled_speed, 0, 1) * 0.5

        # 增加距离前车过近的惩罚
        proximity_penalty = -0.3 if distance_to_front < safety_distance else 0.0

        # 增加长时间静止的惩罚
        stationary_penalty = -5 if self.controlled_vehicle.speed < 21 else 0.0

        # 返回奖励字典
        return {
            "collision_reward": collision_reward,
            "lane_change_reward": lane_change_reward,
            "high_speed_reward": high_speed_reward,
            "proximity_penalty": proximity_penalty,
            "stationary_penalty": stationary_penalty,
        }

    def _is_terminated(self) -> bool:
        return (
                self.controlled_vehicle.crashed or  # 碰撞
                self.controlled_vehicle.position[0] >= self.config["distance"] or  # 达到规定行驶距离
                (self.config["offroad_terminal"] and not self.controlled_vehicle.on_road)  # 偏离道路
        )

    def _is_truncated(self) -> bool:
        """如果达到规定行驶距离则事件被截断。"""
        return self.controlled_vehicle.position[0] >= self.config["distance"]  # 达到规定行驶距离

    def render(self, mode='human'):
        """
        渲染环境。
        :param mode: 渲染模式（可选的，如 'human', 'rgb_array' 等）
        """
        # 如果父类的 render 方法存在且不接受 mode 参数
        if hasattr(super(), 'render'):
            try:
                # 尝试调用父类的 render 方法，看看是否接受 mode 参数
                super().render(mode=mode)
            except TypeError:
                # 如果不接受 mode 参数，就调用不带参数的版本
                super().render()
        else:
            # 如果父类没有 render 方法或需要自定义的渲染逻辑
            if mode == 'human':
                # 如果需要可视化
                print("Rendering in human mode... (this is a placeholder)")
            elif mode == 'rgb_array':
                # 如果需要返回图像
                print("Rendering as RGB array... (this is a placeholder)")
            else:
                raise ValueError(f"Unsupported render mode: {mode}")
