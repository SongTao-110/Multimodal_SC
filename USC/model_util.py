
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from UDeepSC.channel import *
import torch.nn.functional as F


import timm
net = timm.create_model("vit_base_patch16_384", pretrained=True)


# 定义默认配置函数，返回一个包含模型默认参数的字典
def _cfg(url='', **kwargs):
    return {
        'url': url,  # 预训练模型的下载链接
        'num_classes': 1000,  # 分类的类别数，默认为ImageNet的1000类
        'input_size': (3, 224, 224),  # 输入图像的尺寸，通道数为3，大小为224x224
        'pool_size': None,  # 池化层的尺寸，默认为None
        'crop_pct': .9,  # 图像裁剪比例
        'interpolation': 'bicubic',  # 图像插值方式，使用双三次插值
        'mean': (0.5, 0.5, 0.5),  # 图像预处理的均值
        'std': (0.5, 0.5, 0.5),  # 图像预处理的标准差
        **kwargs  # 其他可变参数
    }


# nohup   > log_files/demo_t_14dB.log 2>&1 &
# 定义噪声生成函数，用于模拟信道噪声
def noise_gen(is_train):
    min_snr, max_snr = -6, 18  # 定义最小和最大信噪比（SNR）
    diff_snr = max_snr - min_snr  # 计算信噪比差值

    min_var, max_var = 10 ** (-min_snr / 20), 10 ** (-max_snr / 20)  # 计算最小和最大噪声标准差
    diff_var = max_var - min_var  # 计算噪声标准差差值
    if is_train:
        # 以下为注释掉的代码，用于随机生成训练时的噪声，但目前固定为12dB
        # b = torch.bernoulli(1/5.0*torch.ones(1))
        # if b > 0.5:
        #     channel_snr = torch.FloatTensor([20])
        # else:
        #     channel_snr = torch.rand(1)*diff_snr+min_snr
        # noise_var = 10**(-channel_snr/20)
        # noise_var = torch.rand(1)*diff_var+min_var
        # channel_snr = 10*torch.log10((1/noise_var)**2)
        # channel_snr = torch.rand(1)*diff_snr+min_snr
        # noise_var = 10**(-channel_snr/20)
        channel_snr = torch.FloatTensor([12])  # 固定信噪比为12dB
        noise_var = torch.FloatTensor([1]) * 10 ** (-channel_snr / 20)  # 计算对应的噪声标准差
    else:
        channel_snr = torch.FloatTensor([12])  # 测试时也固定信噪比为12dB
        noise_var = torch.FloatTensor([1]) * 10 ** (-channel_snr / 20)  # 计算对应的噪声标准差
    return channel_snr, noise_var  # 返回信噪比和噪声标准差


# 定义DropPath类，用于实现随机深度（Stochastic Depth）的正则化方法
class DropPath(nn.Module):
    """按照样本随机丢弃路径（在残差块的主路径中应用时）。"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob  # 丢弃概率

    def forward(self, x):
        # 在前向传播中调用drop_path函数
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回模块的额外表示，显示丢弃概率
        return 'p={}'.format(self.drop_prob)


# 定义多层感知机（MLP）模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果未指定输出特征数，则与输入特征数相同
        hidden_features = hidden_features or in_features  # 如果未指定隐藏层特征数，则与输入特征数相同
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层全连接层
        self.act = act_layer()  # 激活函数，默认为GELU
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层全连接层
        self.drop = nn.Dropout(drop)  # Dropout层，防止过拟合

    def forward(self, x):
        x = self.fc1(x)  # 输入经过第一层全连接
        x = self.act(x)  # 应用激活函数
        x = self.fc2(x)  # 经过第二层全连接
        x = self.drop(x)  # 应用Dropout
        return x  # 返回输出


# 定义自注意力（Attention）模块
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads  # 注意力头的数量
        head_dim = dim // num_heads  # 每个头的维度
        if attn_head_dim is not None:
            head_dim = attn_head_dim  # 如果指定了头维度，则使用指定的值
        all_head_dim = head_dim * self.num_heads  # 所有头的总维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子，用于缩放查询向量

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)  # 线性层，生成查询、键、值矩阵
        if qkv_bias:
            # 如果需要偏置，则为查询和值添加可训练的偏置参数
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)  # 注意力Dropout
        self.proj = nn.Linear(all_head_dim, dim)  # 输出线性层，将多头输出合并
        self.proj_drop = nn.Dropout(proj_drop)  # 输出Dropout

    def forward(self, x):
        B, N, C = x.shape  # 获取批次大小、序列长度和通道数
        qkv_bias = None  # 初始化偏置
        if self.q_bias is not None:
            # 如果有偏置，拼接查询偏置、键的零偏置、值偏置
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # 计算查询、键、值矩阵
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)  # 使用F.linear更高效
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 重塑并调整维度
        q, k, v = qkv[0], qkv[1], qkv[2]  # 将查询、键、值分离
        q = q * self.scale  # 缩放查询向量
        attn = (q @ k.transpose(-2, -1))  # 计算注意力权重
        attn = attn.softmax(dim=-1)  # 对最后一个维度进行Softmax
        attn = self.attn_drop(attn)  # 应用注意力Dropout
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)  # 计算注意力输出并调整维度
        x = self.proj(x)  # 通过输出线性层
        x = self.proj_drop(x)  # 应用输出Dropout
        return x  # 返回注意力层的输出


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        # 定义第一个规范化层
        self.norm1 = norm_layer(dim)
        # 定义自注意力模块
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # 定义随机深度模块
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 定义第二个规范化层
        self.norm2 = norm_layer(dim)
        # 定义多层感知机模块
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 初始化可训练的比例参数
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        # 如果没有比例参数，直接加上注意力和MLP模块的输出
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # 如果有比例参数，先乘以比例参数再加上输出
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ 图像到Patch嵌入
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # 将图像尺寸转为元组
        patch_size = to_2tuple(patch_size)  # 将Patch尺寸转为元组
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 计算总的Patch数量
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # 使用卷积实现Patch嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # 确保输入图像尺寸与模型预期一致
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸 ({H}*{W}) 不符合模型预期 ({self.img_size[0]}*{self.img_size[1]})."
        # 使用卷积提取Patch特征并展开为二维
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    """ 正弦位置编码表 """

    # 计算位置编码表的每个位置和隐藏维度的值
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度应用sin函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度应用cos函数

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # 返回张量


class PositionalEncoding(nn.Module):
    """ 位置编码模块 """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # 生成位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # 计算缩放因子
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 将位置编码保存为缓冲区

    def forward(self, x):
        # 将位置编码加到输入上，并应用Dropout
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):
    """ 多头注意力模块 """

    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0  # 确保模型维度可被头数整除
        self.d_k = d_model // num_heads  # 每个头的维度
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)  # 查询矩阵的线性变换
        self.wk = nn.Linear(d_model, d_model)  # 键矩阵的线性变换
        self.wv = nn.Linear(d_model, d_model)  # 值矩阵的线性变换
        self.dense = nn.Linear(d_model, d_model)  # 最后的线性变换
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """ 实现多头注意力机制 """
        if mask is not None:
            mask = mask.unsqueeze(1)  # 将mask扩展到所有头
        nbatches = query.size(0)  # 获取批次大小
        # print(query.shape)  # 打印输入张量的形状
        # print(self.wq.weight.shape)  # 打印权重矩阵的形状
        # 线性变换并调整维度为 [批次, 头数, 序列长度, 头维度]
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 将注意力结果拼接并线性变换
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        """ 计算缩放点积注意力 """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 点积并缩放
        if mask is not None:
            scores += (mask * -1e9)  # 对mask位置赋值极小值
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        return torch.matmul(p_attn, value), p_attn  # 返回注意力结果和权重


# 定义位置前馈网络模块
class PositionwiseFeedForward(nn.Module):
    """ 实现前馈网络（Feed Forward Network，FFN）。 """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一层全连接层，将输入特征维度扩展为隐藏特征维度
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二层全连接层，将隐藏特征维度映射回输入特征维度
        self.dropout = nn.Dropout(dropout)  # Dropout，用于防止过拟合

    def forward(self, x):
        x = self.w_1(x)  # 应用第一层全连接
        x = F.relu(x)  # 激活函数使用ReLU
        x = self.w_2(x)  # 应用第二层全连接
        x = self.dropout(x)  # 应用Dropout
        return x


# 定义解码器层
class DecoderLayer(nn.Module):
    """ 解码器层由自注意力、源注意力和前馈网络组成。 """

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = int(d_model)
        self.dff = int(dff)
        self.self_mha = MultiHeadedAttention(num_heads, self.d_model, dropout=0.1)  # 自注意力
        self.src_mha = MultiHeadedAttention(num_heads, self.d_model, dropout=0.1)  # 源注意力

        self.ffn = PositionwiseFeedForward(self.d_model, self.dff, dropout=0.1)  # 前馈网络

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=1e-6)  # 第一层规范化
        self.layernorm2 = nn.LayerNorm(self.d_model, eps=1e-6)  # 第二层规范化
        self.layernorm3 = nn.LayerNorm(self.d_model, eps=1e-6)  # 第三层规范化

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        """ 解码器层的前向传播 """
        attn_output = self.self_mha(x, x, x, look_ahead_mask)  # 自注意力
        x = self.layernorm1(x + attn_output)  # 加残差并规范化

        src_output = self.src_mha(x, memory, memory, trg_padding_mask)  # 源注意力
        x = self.layernorm2(x + src_output)  # 加残差并规范化

        fnn_output = self.ffn(x)  # 前馈网络
        x = self.layernorm3(x + fnn_output)  # 加残差并规范化
        return x


# 定义解码器
class Decoder(nn.Module):
    """ 解码器由多个解码器层组成。 """

    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)  # 位置编码模块
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate)
                                         for _ in range(depth)])  # 解码器层列表

    def forward(self, x, memory, look_ahead_mask=None, trg_padding_mask=None):
        """ 解码器的前向传播 """
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)  # 依次通过每一层
        return x


# 定义Vision Transformer编码器
class ViTEncoder(nn.Module):
    """ Vision Transformer编码器，支持Patch或混合CNN输入。 """
# embed_dim=768
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # 特征维度
        # 定义不同任务的Patch嵌入模块
        self.patch_embed_imgr = PatchEmbed(img_size=32, patch_size=4, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_imgc = PatchEmbed(img_size=224, patch_size=32, in_chans=in_chans,
                                           embed_dim=embed_dim)
        self.linear_embed_vqa = nn.Linear(2048, self.embed_dim)  # VQA任务的线性嵌入
        self.linear_embed_msa = nn.Linear(35, self.embed_dim)  # MSA任务的线性嵌入

        # 定义分类Token
        self.cls_token = nn.ParameterDict({
            'imgr': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'imgc': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'vqa': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'msa': nn.Parameter(torch.zeros(1, 1, embed_dim))
        })

        # 定义任务嵌入
        self.task_embedd = nn.ParameterDict({
            'imgr': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'imgc': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'vqa': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'msa': nn.Parameter(torch.zeros(1, 1, embed_dim))
        })

        # 定义位置编码
        if use_learnable_pos_emb:
            self.pos_embed_imgc = nn.Parameter(
                torch.zeros(1, self.patch_embed_imgc.num_patches + 1, embed_dim))
            self.pos_embed_imgr = nn.Parameter(
                torch.zeros(1, self.patch_embed_imgr.num_patches + 1, embed_dim))
        else:
            self.pos_embed_imgc = get_sinusoid_encoding_table(self.patch_embed_imgc.num_patches + 1,
                                                              embed_dim)
            self.pos_embed_imgr = get_sinusoid_encoding_table(self.patch_embed_imgr.num_patches + 1,
                                                              embed_dim)

        # 定义Transformer编码器块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)  # 最后一个规范化层

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed_imgc, std=.02)
            trunc_normal_(self.pos_embed_imgr, std=.02)
        for key in self.cls_token.keys():
            trunc_normal_(self.cls_token[key], std=.02)
            trunc_normal_(self.task_embedd[key], std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ 初始化权重 """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, ta_perform):
        """ 前向特征提取，根据任务类型处理输入 """
        if ta_perform.startswith('vqa'):
            x = self.linear_embed_vqa(x)
        elif ta_perform.startswith('msa'):
            x = self.linear_embed_msa(x)
        elif ta_perform.startswith('img'):
            x = self.patch_embed_imgr(x) if ta_perform.startswith('imgr') else self.patch_embed_imgc(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device)
        task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
        x = torch.cat((cls_tokens, x), dim=1)
        if ta_perform.startswith('imgr'):
            x = x + self.pos_embed_imgr.type_as(x).to(x.device).clone().detach()
        elif ta_perform.startswith('imgc'):
            x = x + self.pos_embed_imgc.type_as(x).to(x.device).clone().detach()
        x = torch.cat((x, task_embedd), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x, ta_perform):
        """ 前向传播 """
        x = self.forward_features(x, ta_perform)
        return x

class SPTEncoder(nn.Module):
    """ SPT编码器，支持线性嵌入和Transformer编码器块"""
    def __init__(self, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # 嵌入维度

        self.linear_embed = nn.Linear(74, self.embed_dim)  # 输入数据的线性嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类Token
        self.task_embedd = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 任务嵌入

        # 定义Transformer编码器块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减规则
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)  # 规范化层
        trunc_normal_(self.cls_token, std=.02)  # 初始化分类Token
        trunc_normal_(self.task_embedd, std=.02)  # 初始化任务嵌入
        self.apply(self._init_weights)  # 初始化权重

    def _init_weights(self, m):
        """ 初始化模块权重 """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        """ 获取Transformer编码器的层数 """
        return len(self.blocks)

    def forward_features(self, x, ta_perform):
        """ 特征提取，根据任务类型进行前向传播 """
        if ta_perform.startswith('msa'):
            x = self.linear_embed(x)  # 对输入进行线性嵌入
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(x.device)  # 分类Token扩展
            task_embedd = self.task_embedd.expand(batch_size, -1, -1).to(x.device)  # 任务嵌入扩展
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)  # 拼接分类Token、输入和任务嵌入

        for blk in self.blocks:
            x = blk(x)  # 逐层通过Transformer块
        x = self.norm(x)  # 应用规范化
        return x

    def forward(self, x, ta_perform=None):
        """ 前向传播接口 """
        x = self.forward_features(x, ta_perform)
        return x

class VectorQuantizer(nn.Module):
    """
    向量量化器，用于实现VQ-VAE中向量量化过程
    参考：https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, quan_bits: int = 4):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings  # 嵌入向量的数量
        self.D = embedding_dim  # 嵌入向量的维度

        # 定义嵌入矩阵
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)  # 初始化嵌入权重

    def forward(self, latents):
        """ 前向传播，执行量化过程 """

        # 如果 latents 是 BaseModelOutputWithPoolingAndCrossAttentions 类型，提取 last_hidden_state
        if isinstance(latents, BaseModelOutputWithPoolingAndCrossAttentions):
            latents = latents.last_hidden_state  # 获取实际的 Tensor

        latents_shape = latents.shape  # 获取 latents 的形状
        flat_latents = latents.view(-1, self.D)  # 展开为二维
        device = latents.device

        # 计算latents与嵌入向量之间的L2距离
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())

        # 找到距离最小的编码索引
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]
        shape = encoding_inds.shape

        encoding_inds = encoding_inds.to(device).reshape(shape)
        encoding_inds = encoding_inds.to(torch.int64)

        # 将编码索引转换为独热编码
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]

        # 量化latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
        quantized_latents = quantized_latents.view(latents_shape)  # 恢复原始形状

        # 量化后的输出
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.contiguous()

# class VectorQuantizer(nn.Module):
#     """
#     向量量化器，用于实现VQ-VAE中向量量化过程
#     参考：https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
#     """
#     def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25,quan_bits: int = 4):
#         super(VectorQuantizer, self).__init__()
#         self.K = num_embeddings  # 嵌入向量的数量
#         self.D = embedding_dim  # 嵌入向量的维度
#         self.beta = beta  # 损失中的权衡参数
#
#         # 定义嵌入矩阵
#         self.embedding = nn.Embedding(self.K, self.D)
#         self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)  # 初始化嵌入权重
#
#     # def forward(self, latents, SNRdB, bit_per_index):
#     def forward(self, latents):
#         """ 前向传播，执行量化过程 """
#         latents_shape =latents.shape
#         flat_latents = latents.view(-1, self.D)  # 展开为二维
#         device = latents.device
#
#         # 计算latents与嵌入向量之间的L2距离
#         dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
#                torch.sum(self.embedding.weight ** 2, dim=1) - \
#                2 * torch.matmul(flat_latents, self.embedding.weight.t())
#
#         # 找到距离最小的编码索引
#         encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]
#         shape = encoding_inds.shape
#
#         # 使用传输模型对编码进行模拟
#         # Rx_signal = transmit(encoding_inds, SNRdB, bit_per_index)
#         # Rx_signal = transmit(encoding_inds)
#         # encoding_inds = torch.from_numpy(Rx_signal).to(device).reshape(shape)
#         encoding_inds = encoding_inds.to(device).reshape(shape)
#         encoding_inds = encoding_inds.to(torch.int64)
#         # 将编码索引转换为独热编码
#         encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
#         encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]
#
#         # 量化latents
#         quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
#         quantized_latents = quantized_latents.view(latents_shape)  # 恢复原始形状
#
#         # 计算量化损失
#         commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
#         embedding_loss = F.mse_loss(quantized_latents, latents.detach())
#         vq_loss = commitment_loss * self.beta + embedding_loss
#
#         # 返回量化后的latents和损失
#         quantized_latents = latents + (quantized_latents - latents).detach()
#         return quantized_latents.contiguous()
            #, vq_loss
#
# class Channels():
#     """ 模拟信道，包括AWGN、Rayleigh和Rician信道 """
#     def AWGN(self, Tx_sig, n_std):
#         """ 加性高斯白噪声（AWGN）信道 """
#         device = Tx_sig.device
#         noise = torch.normal(0, n_std / math.sqrt(2), size=Tx_sig.shape).to(device)  # 生成噪声
#         Rx_sig = Tx_sig + noise  # 叠加噪声
#         return Rx_sig
#
#     def Rayleigh(self, Tx_sig, n_std):
#         """ Rayleigh信道 """
#         device = Tx_sig.device
#         shape = Tx_sig.shape
#         H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)  # Rayleigh信道的实部
#         H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)  # Rayleigh信道的虚部
#         H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)  # 信道矩阵
#         Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)  # 信道传输
#         Rx_sig = self.AWGN(Tx_sig, n_std)  # 添加噪声
#         Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)  # 信道估计
#         return Rx_sig
#
#     def Rician(self, Tx_sig, n_std, K=1):
#         """ Rician信道 """
#         device = Tx_sig.device
#         shape = Tx_sig.shape
#         mean = math.sqrt(K / (K + 1))  # Rician分量的均值
#         std = math.sqrt(1 / (K + 1))  # Rician分量的标准差
#         H_real = torch.normal(mean, std, size=[1]).to(device)
#         H_imag = torch.normal(mean, std, size=[1]).to(device)
#         H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)  # 信道矩阵
#         Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)  # 信道传输
#         Rx_sig = self.AWGN(Tx_sig, n_std)  # 添加噪声
#         Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)  # 信道估计
#         return Rx_sig
