import math
import numpy as np
from timm.models.registry import register_model
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """ 将图像转化为Patch嵌入表示 """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # 确保图像大小是元组格式
        patch_size = to_2tuple(patch_size)  # 确保Patch大小是元组格式
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 计算总Patch数
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # Patch形状
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches  # Patch总数
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)  # 使用卷积操作生成Patch嵌入

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # 确保输入图像大小与预定义的图像大小一致
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像大小 ({H}*{W}) 与模型预定义的大小 ({self.img_size[0]}*{self.img_size[1]}) 不匹配。"
        x = self.proj(x).flatten(2).transpose(1, 2)  # 转换为嵌入表示
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    """ 生成正弦位置编码表 """

    def get_position_angle_vec(position):
        # 根据公式计算位置角度向量
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])  # 计算每个位置的编码
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数位置使用正弦函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数位置使用余弦函数

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # 返回张量


class PositionalEncoding(nn.Module):
    """ 实现位置编码模块 """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # Dropout防止过拟合

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # 每个位置的索引
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # 缩放因子
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度的编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度的编码
        pe = pe.unsqueeze(0)  # 增加Batch维度
        self.register_buffer('pe', pe)  # 注册为模块缓冲区

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # 添加位置编码
        x = self.dropout(x)  # 应用Dropout
        return x


class MultiHeadedAttention(nn.Module):
    """ 实现多头自注意力机制 """

    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0  # 确保特征维度可以被头数整除
        self.d_k = d_model // num_heads  # 每个头的维度
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)  # 查询向量投影
        self.wk = nn.Linear(d_model, d_model)  # 键向量投影
        self.wv = nn.Linear(d_model, d_model)  # 值向量投影
        self.dense = nn.Linear(d_model, d_model)  # 输出层
        self.dropout = nn.Dropout(p=dropout)  # Dropout层

    def forward(self, query, key, value, policy=None, mask=None):
        """ 前向传播 """
        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩展掩码维度，适应多头注意力
        nbatches = query.size(0)

        # 将输入线性投影到多头表示
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        x, self.attn = self.attention(query, key, value, policy=policy, mask=mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)  # 拼接多头输出
        x = self.dense(x)  # 输出层
        x = self.dropout(x)
        return x

    def attention(self, query, key, value, policy=None, mask=None, eps=1e-6):
        """ 计算缩放点积注意力 """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数
        if mask is not None:
            scores += (mask * -1e9)  # 应用掩码
        if policy is None:
            p_attn = F.softmax(scores, dim=-1)  # 标准注意力
            return torch.matmul(p_attn, value), p_attn
        else:
            # 使用策略调整注意力分数
            B, N1, _ = policy.size()
            B, H, N1, N2 = scores.size()
            attn_policy = policy.reshape(B, 1, 1, N2)
            temp = torch.zeros((B, 1, N1, N2), dtype=attn_policy.dtype, device=attn_policy.device)
            attn_policy = attn_policy + temp
            max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
            scores = scores - max_scores
            scores = scores.to(torch.float32).exp_() * attn_policy.to(torch.float32)
            p_attn = (scores + eps / N1) / (scores.sum(dim=-1, keepdim=True) + eps)
            return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    """ 实现前馈网络 """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一层全连接
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二层全连接
        self.dropout = nn.Dropout(dropout)  # Dropout防止过拟合

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)  # 激活函数
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    """ 解码器层由自注意力、源注意力和前馈网络组成 """

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)  # 自注意力
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)  # 源注意力
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)  # 前馈网络
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)  # 规范化
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, policy, look_ahead_mask, trg_padding_mask):
        attn_output = self.self_mha(x, x, x, None, look_ahead_mask)  # 自注意力
        x = self.layernorm1(x + attn_output)
        src_output = self.src_mha(x, memory, memory, policy, trg_padding_mask)  # 源注意力
        x = self.layernorm2(x + src_output)
        fnn_output = self.ffn(x)  # 前馈网络
        x = self.layernorm3(x + fnn_output)
        return x


class Decoder(nn.Module):
    """ 实现Transformer解码器 """

    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)  # 位置编码
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate)
                                         for _ in range(depth)])  # 多层解码器

    def forward(self, x, memory, policy=None, look_ahead_mask=None, trg_padding_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, policy, look_ahead_mask, trg_padding_mask)  # 逐层解码
        return x
