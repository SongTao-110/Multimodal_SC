import math
import numpy as np
from timm.models.registry import register_model
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    将图像转换为 Patch 嵌入 (Image to Patch Embedding)。

    功能:
    ----------
    1. 将输入的图像分割成多个固定大小的 Patch。
    2. 对每个 Patch 进行线性投影，将其映射到指定的嵌入维度。

    参数:
    ----------
    img_size: int, 图像的高度和宽度，默认为 224。
    patch_size: int, 每个 Patch 的高度和宽度，默认为 16。
    in_chans: int, 输入图像的通道数，默认为 3（如 RGB 图像）。
    embed_dim: int, 每个 Patch 的嵌入维度，默认为 768。

    属性:
    ----------
    num_patches: int, 分割后的 Patch 总数。
    patch_shape: tuple, 每个 Patch 的大小。
    proj: nn.Conv2d, 卷积层，用于将每个 Patch 投影到嵌入维度。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # 确保图像尺寸是元组形式 (H, W)
        patch_size = to_2tuple(patch_size)  # 确保 Patch 尺寸是元组形式 (H, W)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 计算总 Patch 数
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 每个 Patch 的形状
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # 卷积层，将输入图像分块并投影到嵌入维度
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        """
        前向传播，将输入图像转换为 Patch 嵌入。

        参数:
        ----------
        x: Tensor, 输入图像，形状为 [B, C, H, W]。

        返回:
        ----------
        Tensor, Patch 嵌入，形状为 [B, num_patches, embed_dim]。
        """
        B, C, H, W = x.shape  # 获取输入图像的 Batch 大小和形状
        # 确保输入图像的尺寸与定义的尺寸一致
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸 ({H}*{W}) 与模型定义尺寸 ({self.img_size[0]}*{self.img_size[1]}) 不匹配。"
        # 通过卷积进行分块和投影，展平并转置为 [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    生成正弦位置编码表，用于为序列添加位置信息。

    参数:
    ----------
    n_position: int, 序列的最大位置数量。
    d_hid: int, 每个位置的嵌入维度。

    返回:
    ----------
    sinusoid_table: Tensor, 位置编码表，形状为 [1, n_position, d_hid]。
    """

    def get_position_angle_vec(position):
        # 计算位置编码向量中每个维度的值
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # 为每个位置计算编码向量

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度使用正弦函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度使用余弦函数

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # 添加 Batch 维度


class PositionalEncoding(nn.Module):
    """
    正弦位置编码模块，为输入添加固定的正弦位置编码。

    参数:
    ----------
    d_model: int, 每个位置的嵌入维度。
    dropout: float, Dropout 比例。
    max_len: int, 最大序列长度。
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # Dropout 层

        # 创建位置编码张量
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # 位置索引，形状为 [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # 缩放因子
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用余弦函数
        pe = pe.unsqueeze(0)  # 添加 Batch 维度
        # 注册为模型的 buffer，不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播，为输入张量添加位置编码。

        参数:
        ----------
        x: Tensor, 输入张量，形状为 [B, T, d_model]。

        返回:
        ----------
        Tensor, 添加位置编码后的张量。
        """
        # 将位置编码加到输入张量中
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)  # 应用 Dropout
        return x


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)。
    功能：
    ----------
    1. 通过多头机制并行处理不同的注意力分布。
    2. 每个头共享相同的 Q、K、V 维度，并独立计算注意力。
    3. 最终将多个头的输出拼接起来，并通过全连接层进行融合。

    参数:
    ----------
    num_heads: int, 注意力头的数量。
    d_model: int, 输入嵌入的维度。
    dropout: float, Dropout 比例，默认值为 0.1。
    """

    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须是 num_heads 的整数倍"
        self.d_k = d_model // num_heads  # 每个头的维度
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)  # 查询向量的线性变换
        self.wk = nn.Linear(d_model, d_model)  # 键向量的线性变换
        self.wv = nn.Linear(d_model, d_model)  # 值向量的线性变换
        self.dense = nn.Linear(d_model, d_model)  # 最终输出的线性变换
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)  # Dropout 层

    def forward(self, query, key, value, policy=None, mask=None):
        """
        前向传播计算注意力输出。

        参数:
        ----------
        query, key, value: Tensor, 注意力机制的输入张量。
        policy: Tensor, 可选，额外的注意力策略。
        mask: Tensor, 可选，注意力掩码。

        返回:
        ----------
        Tensor, 多头注意力的输出张量。
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # 对掩码增加一个维度

        nbatches = query.size(0)  # Batch 大小

        # 线性变换并分多头
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力输出
        x, self.attn = self.attention(query, key, value, policy=policy, mask=mask)

        # 拼接多头输出
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        # 通过全连接层
        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, policy=None, mask=None, eps=1e-6):
        """
        缩放点积注意力 (Scaled Dot Product Attention)。

        参数:
        ----------
        query, key, value: Tensor, 注意力机制的输入张量。
        policy: Tensor, 可选，注意力策略。
        mask: Tensor, 可选，注意力掩码。
        eps: float, 防止除零的小数。

        返回:
        ----------
        Tensor, 注意力输出张量。
        """
        d_k = query.size(-1)  # 查询向量的最后一维
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 注意力得分
        if mask is not None:
            scores += (mask * -1e9)  # 应用掩码

        if policy is None:
            p_attn = F.softmax(scores, dim=-1)  # 使用标准的注意力机制
            return torch.matmul(p_attn, value), p_attn
        else:
            # 如果有注意力策略，则应用策略计算
            B, N1, _ = policy.size()
            B, H, N1, N2 = scores.size()
            attn_policy = policy.reshape(B, 1, 1, N2)
            scores = scores.exp() * attn_policy
            p_attn = (scores + eps / N1) / (scores.sum(dim=-1, keepdim=True) + eps)
            return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    """
    前馈神经网络 (Feed Forward Network, FFN)。
    功能：
    ----------
    1. 实现 Transformer 的位置无关特征提取。
    2. 包含两个线性变换和一个激活函数。

    参数:
    ----------
    d_model: int, 输入张量的维度。
    d_ff: int, FFN 的隐藏层维度。
    dropout: float, Dropout 比例，默认值为 0.1。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一层全连接
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二层全连接
        self.dropout = nn.Dropout(dropout)  # Dropout 层

    def forward(self, x):
        """
        前向传播计算 FFN 输出。

        参数:
        ----------
        x: Tensor, 输入张量。

        返回:
        ----------
        Tensor, FFN 的输出张量。
        """
        x = self.w_1(x)
        x = F.relu(x)  # 激活函数
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    """
    解码器层 (Decoder Layer)，包含自注意力、源-目标注意力和前馈网络。

    参数:
    ----------
    d_model: int, 输入嵌入的维度。
    num_heads: int, 注意力头的数量。
    dff: int, FFN 的隐藏层维度。
    dropout: float, Dropout 比例。
    """

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)  # 自注意力模块
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)  # 源-目标注意力模块
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)  # 前馈网络
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, policy, look_ahead_mask, trg_padding_mask):
        """
        前向传播计算解码器层输出。

        参数:
        ----------
        x: Tensor, 解码器输入。
        memory: Tensor, 编码器输出。
        policy: Tensor, 可选的注意力策略。
        look_ahead_mask: Tensor, 遮挡未来信息的掩码。
        trg_padding_mask: Tensor, 目标序列的填充掩码。

        返回:
        ----------
        Tensor, 解码器层的输出。
        """
        attn_output = self.self_mha(x, x, x, None, look_ahead_mask)  # 自注意力
        x = self.layernorm1(x + attn_output)

        src_output = self.src_mha(x, memory, memory, policy, trg_padding_mask)  # 源-目标注意力
        x = self.layernorm2(x + src_output)

        fnn_output = self.ffn(x)  # 前馈网络
        x = self.layernorm3(x + fnn_output)
        return x


class Decoder(nn.Module):
    """
    解码器 (Decoder)，由多层解码器层组成。

    参数:
    ----------
    depth: int, 解码器层的数量。
    embed_dim: int, 嵌入维度。
    num_heads: int, 注意力头的数量。
    dff: int, FFN 的隐藏层维度。
    drop_rate: float, Dropout 比例。
    """

    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)  # 位置编码
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate)
                                         for _ in range(depth)])  # 解码器层列表

    def forward(self, x, memory, policy=None, look_ahead_mask=None, trg_padding_mask=None):
        """
        前向传播计算解码器输出。

        参数:
        ----------
        x: Tensor, 解码器输入。
        memory: Tensor, 编码器输出。
        policy: Tensor, 可选的注意力策略。
        look_ahead_mask: Tensor, 遮挡未来信息的掩码。
        trg_padding_mask: Tensor, 目标序列的填充掩码。

        返回:
        ----------
        Tensor, 解码器的最终输出。
        """
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, policy, look_ahead_mask, trg_padding_mask)
        return x

