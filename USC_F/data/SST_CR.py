import os
import torch
import pandas as pd
import numpy as np
import pytreebank
import torch.utils.data as data

from loguru import logger
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer

# 加载SST数据集
sst = pytreebank.load_sst()


# 右侧填充函数
# array: 输入的列表
# n: 填充后的目标长度，默认为70
# 如果输入长度小于目标长度，则用0进行填充
# 如果输入长度大于目标长度，则截断
def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)  # 获取输入的当前长度
    if current_len > n:
        return array[: n - 1]  # 如果输入长度大于目标长度，进行截断
    extra = n - current_len  # 计算需要填充的长度
    return array + ([0] * extra)  # 使用0进行填充


# 将细粒度标签转换为二进制标签
# label: 输入的细粒度标签
# 返回二进制标签，0表示消极，1表示积极
# 标签2为中立标签，无法转换为二进制
def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2:
        return 0  # 标签小于2表示消极
    if label > 2:
        return 1  # 标签大于2表示积极
    raise ValueError("Invalid label")  # 标签2为无效标签，抛出异常


# 自定义SST数据集类，继承自torch的Dataset类
class SST_CR(Dataset):
    # 初始化函数
    # root: 是否只使用根节点的数据
    # train: 是否加载训练集，True为训练集，False为测试集
    # binary: 是否将标签转换为二进制，True为二进制标签，False为细粒度标签
    # if_class: 是否返回图像和标签，True返回图像和标签，False返回图像和图像
    def __init__(self, root=True, train=True, binary=True, if_class: bool = True):
        logger.info("Loading the tokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # 加载BERT分词器
        logger.info("Loading SST")

        # 根据train参数选择加载训练集或测试集
        if train:
            self.sst = sst["train"]
        else:
            self.sst = sst["test"]
        self.if_class = if_class  # 设置是否返回类别标签

        # 根据不同参数组合处理数据
        if root and binary:
            # 使用根节点数据并将标签转换为二进制
            self.data = [(rpad(tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66),
                          get_binary_label(tree.label),)
                         for tree in self.sst if tree.label != 2]  # 过滤掉标签为2的中立数据
        elif root and not binary:
            # 使用根节点数据，不转换标签
            self.data = [(rpad(tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66),
                          tree.label,)
                         for tree in self.sst]
        elif not root and not binary:
            # 使用所有节点数据，不转换标签
            self.data = [(rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                          label)
                         for tree in self.sst
                         for label, line in tree.to_labeled_lines()]
        else:
            # 使用所有节点数据，并将标签转换为二进制
            self.data = [(rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                          get_binary_label(label),)
                         for tree in self.sst
                         for label, line in tree.to_labeled_lines()
                         if label != 2]  # 过滤掉标签为2的中立数据

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)  # 返回数据集的长度

    # 获取单个数据的方法
    # index: 数据的索引
    # 返回句子及其对应的标签（或句子自身）
    def __getitem__(self, index):
        sentence, target = self.data[index]  # 获取句子和标签
        sentence = torch.tensor(sentence)  # 将句子转换为torch张量
        if self.if_class:
            return sentence, target  # 返回句子和标签
        else:
            return sentence, sentence  # 返回句子和句子本身（无标签）
