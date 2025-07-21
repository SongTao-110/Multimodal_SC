import os
import re
import sys
import pickle
import torch
import pprint
import numpy as np
import torch.nn as nn

from pathlib import Path
# from transformers import *  # 可能用于自然语言处理的预训练模型，但此处未被使用
from tqdm import tqdm_notebook  # 用于显示进度条
# from mmsdk import mmdatasdk as md  # 可能用于多模态数据处理，但此处未被使用
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from subprocess import check_call, CalledProcessError


# 定义两个用于保存和加载pickle文件的函数
# to_pickle函数：将对象保存为pickle文件
# obj: 要保存的对象
# path: 保存对象的文件路径
def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# load_pickle函数：从pickle文件加载对象
# path: 要加载的pickle文件路径
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# 构建word2id字典，自动为新遇到的单词分配一个新的ID
# word2id是一个defaultdict，当遇到未知的单词时，会自动分配一个新的ID
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']  # 未知单词的默认ID
PAD = word2id['<pad>']  # 填充符号的ID


# 关闭word2id自动分配新ID的功能，防止新单词继续增加
# return_unk函数用于返回UNK值，用于未找到单词时的处理
def return_unk():
    return UNK


# 加载词嵌入函数
# 该函数用于从预训练的词嵌入文件中加载词向量到嵌入矩阵中
# w2i：词到ID的映射
# path_to_embedding：嵌入文件的路径
# embedding_size：嵌入向量的维度，默认为300
# embedding_vocab：嵌入文件中的词汇数，默认为2196017
# init_emb：初始化的嵌入矩阵，默认随机初始化
# 返回一个torch张量，表示词汇的嵌入矩阵
def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)  # 初始化嵌入矩阵为随机数
    else:
        emb_mat = init_emb

    f = open(path_to_embedding, 'r')  # 打开嵌入文件
    found = 0  # 记录找到的词汇数量
    # 遍历嵌入文件，加载嵌入向量
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()  # 将每行拆分为词汇和向量
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))  # 获取向量部分，转换为浮点数
        word = ' '.join(content[:-300])  # 获取词汇部分
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector  # 将词汇对应的向量更新到嵌入矩阵中
            found += 1
    print(f"Found {found} words in the embedding file.")  # 打印找到的词汇数量
    return torch.tensor(emb_mat).float()  # 返回嵌入矩阵，转换为torch张量


# 定义MOSI数据集类，用于处理MOSI数据集
class MOSI():
    def __init__(self, config):
        # 检查SDK路径是否指定，如果未指定则退出程序
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))  # 添加SDK路径到系统路径中，便于后续导入SDK

        data_path = str(config.dataset_dir)  # 获取数据集目录路径
        cache_path = data_path + '/embedding_and_mapping.pt'  # 定义缓存文件路径

        try:
            # 尝试加载训练集、验证集和测试集的数据
            self.train = load_pickle(data_path + '/train.pkl')  # 加载训练集数据
            self.dev = load_pickle(data_path + '/dev.pkl')  # 加载验证集数据
            self.test = load_pickle(data_path + '/test.pkl')  # 加载测试集数据
        except:
            print('error')  # 如果加载失败，打印错误信息
            pass

    # 获取数据的方法
    # is_train: 是否获取训练数据，True表示获取训练数据，False表示获取测试数据
    def get_data(self, is_train):
        if is_train:
            return self.train  # 返回训练数据
        else:
            return self.test  # 返回测试数据


# 定义MSA数据集类，继承自torch的Dataset类
# 该类用于将数据集包装成Dataset对象，便于使用DataLoader加载数据
class MSA(Dataset):
    def __init__(self, config, train=True):
        dataset = MOSI(config)  # 创建MOSI数据集对象，传入配置参数

        self.data = dataset.get_data(train)  # 获取训练或测试数据
        self.len = len(self.data)  # 数据集长度

        # 配置可视化和音频特征的大小
        # 获取第一个样本的数据，分别设置可视化和音频特征的维度
        config.visual_size = self.data[0][0][1].shape[1]  # 设置可视化特征的大小
        config.acoustic_size = self.data[0][0][2].shape[1]  # 设置音频特征的大小

    # 根据索引获取数据的方法
    # index: 数据索引
    def __getitem__(self, index):
        return self.data[index]  # 返回指定索引的数据

    # 返回数据集长度的方法
    def __len__(self):
        return self.len  # 返回数据集长度


# 配置类，用于存储和管理实验配置
# 该类用于定义数据集路径、SDK路径、词嵌入路径等配置参数
class Config_MSA(object):
    def __init__(self, ):
        project_dir = Path(__file__).resolve().parent.parent  # 获取项目的根目录路径
        sdk_dir = project_dir.joinpath('/home/hqyyqh888/SemanRes2/MSA/CMU-MultimodalSDK/')  # SDK目录路径
        data_dir = project_dir.joinpath('data/msadata')  # 数据目录路径
        # 定义数据集的字典，包含MOSI、MOSEI和UR_FUNNY数据集
        data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
            'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
        word_emb_path = '/home/hqyyqh888/SemanRes2/MSA/MISA/glove/glove.840B.300d.txt'  # 词嵌入文件路径
        assert (word_emb_path is not None)  # 确保词嵌入文件路径已指定

        # 设置配置参数
        self.dataset_dir = data_dict['mosei']  # 设置数据集目录
        self.sdk_dir = sdk_dir  # 设置SDK目录
        self.word_emb_path = word_emb_path  # 设置词嵌入文件路径
        self.data_dir = self.dataset_dir  # 设置数据目录

    # 打印配置信息的方法
    def __str__(self):
        """以字典形式按字母顺序打印配置信息"""
        config_str = 'Configurations\n'  # 配置信息的标题
        config_str += pprint.pformat(self.__dict__)  # 格式化配置参数为字符串
        return config_str  # 返回配置字符串
