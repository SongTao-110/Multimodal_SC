from functools import partial
from UDeepSC.trans_deocer import Decoder
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from transformers import BertModel
from UDeepSC.model_util import *
from UDeepSC.model_util import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from UDeepSC.base_args import IMGC_NUMCLASS, TEXTC_NUMCLASS, IMGR_LENGTH, TEXTR_NUMCLASS, VQA_NUMCLASS, MSA_NUMCLASS
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

__all__ = [
    'UDeepSC_model']

class UDeepSC(nn.Module):
    def __init__(self, mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0,
                 img_embed_dim=384, text_embed_dim=384, speech_embed_dim=128, img_encoder_depth=4,
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False, num_classes=0):
        # 初始化函数，接收多个超参数，用于定义图像、文本、语音的嵌入维度、编码器深度、解码器设置等。
        super().__init__()

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
        model_name = "bert-base-uncased"  # 使用预训练的BERT模型
        self.text_encoder = BertModel.from_pretrained(model_name)

        # 根据mode来调整文本嵌入维度
        if mode == 'tiny':
            text_embed_dim = 128
        elif mode == 'small':
            text_embed_dim = 512
        else:
            text_embed_dim = 512  # 默认为512

        # 设置符号数量（在本模型中用于映射到解码器）
        self.num_symbols_img = 16
        self.num_symbols_text = 8
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
        self.codebook['img'] = VectorQuantizer(num_embeddings=2 ** self.bit_per_digit,
                                               embedding_dim=img_embed_dim,
                                               quan_bits=self.bit_per_digit)
        self.codebook['text'] = VectorQuantizer(num_embeddings=2 ** self.bit_per_digit,
                                                embedding_dim=text_embed_dim,
                                                quan_bits=self.bit_per_digit)

        # 初始化解码器
        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim,
                               num_heads=decoder_num_heads, dff=mlp_ratio * decoder_embed_dim,
                               drop_rate=drop_rate)

        # 初始化信道层（用于多模态数据的信道处理）
        # self.channel = Channels()

        # 使用Sigmoid激活函数
        self.sigmoid_layer = nn.Sigmoid()

    def _init_weights(self, m):
        # 权重初始化函数，用于初始化模型的权重
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # 对线性层的权重使用Xavier均匀分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 对LayerNorm层的偏置初始化为0
            nn.init.constant_(m.weight, 1.0)  # 对LayerNorm层的权重初始化为1

    def get_num_layers(self):
        # 获取网络的层数（目前未使用，可以根据需要扩展）
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        # 指定哪些参数不需要进行权重衰减（常用于位置编码和某些固定参数）
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, speech=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        # 前向传播函数，处理多模态输入（文本、图像、语音），以及不同的任务（分类、回归、VQA等）
        # 初始化损失字典和状态信息字典
        m_dict, rho_dict, codebook_loss = {}, {}, {}

        # 初始化每个模态（图像、文本、语音）的codebook损失为零
        codebook_loss['img'] = torch.tensor(0.)
        codebook_loss['text'] = torch.tensor(0.)
        # codebook_loss['spe'] = torch.tensor(0.)

        ###### 计算编码器和codebook的映射
        # 根据任务类型（ta_perform）选择不同的模态并进行处理
        if ta_perform.startswith('textc'):
            # 任务是文本分类
            # 使用文本编码器处理文本输入
            x_text, m_dict['text'], rho_dict['text'] = self.text_encoder(text, ta_perform)
            # 使用codebook对文本特征进行量化，并计算损失
            x_text, codebook_loss['text'] = self.codebook['text'](x_text)
            # 将文本特征映射到解码器输入维度
            x_text = self.text_channel_to_decoder(x_text)

        elif ta_perform.startswith('imgc'):
            # 任务是图像分类
            # 使用图像编码器处理图像输入
            x_img, m_dict['img'], rho_dict['img'] = self.img_encoder(img, ta_perform)
            # 使用codebook对图像特征进行量化，并计算损失
            x_img, codebook_loss['img'] = self.codebook['img'](x_img)
            # 将图像特征映射到解码器输入维度
            x_img = self.img_channel_to_decoder(x_img)

        ####### 计算策略向量
        if ta_perform.startswith('img'):
            # 对图像任务，选取图像特征
            x = x_img
            if self.training:
                # 如果是训练模式，生成类别标记（cls_m）并与当前的状态（curr_m）拼接
                cls_m = torch.ones(x.shape[0], 1, 1, dtype=x.dtype, device=x.device)
                curr_m = m_dict['img'][-1]
                policy = torch.cat([cls_m, curr_m], dim=1)

        elif ta_perform.startswith('text'):
            # 对文本任务，选取文本特征
            x = x_text
            if self.training:
                # 如果是训练模式，生成类别标记（cls_m）并与当前的状态（curr_m）拼接
                cls_m = torch.ones(x.shape[0], 1, 1, dtype=x.dtype, device=x.device)
                curr_m = m_dict['text'][-1]
                policy = torch.cat([cls_m, curr_m], dim=1)


        ####### 解码过程
        # 根据任务类型选择相应的任务字典并扩展维度
        query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(x.shape[0], 1, 1)

        # 使用解码器进行处理
        # 如果是训练模式，传入策略向量进行训练；否则只进行推理
        x = self.decoder(query_embed, x, policy, None, None) if self.training else self.decoder(query_embed,
                                                                                                x, None, None,
                                                                                                None)

        # 根据任务类型使用不同的头进行输出
        if ta_perform.startswith('textr'):
            # 如果是文本回归任务，应用回归头
            x = self.head[ta_perform](x)
        else:
            # 对于其他任务，先进行平均池化，然后应用相应的任务头
            x = self.head[ta_perform](x.mean(1))

        # 返回训练模式下的输出，包括预测结果、状态信息和codebook损失；推理模式下只返回预测结果和状态信息
        return (x, m_dict, rho_dict, codebook_loss) if self.training else {'outputs': x, 'rho': rho_dict,
                                                                           'mask': m_dict}


@register_model
def UDeepSC_model(pretrained=False, **kwargs):
    """
    创建UDeepSC模型的函数。该函数通过传入一些超参数配置并根据是否预训练加载模型权重来初始化UDeepSC模型。

    参数：
    - pretrained (bool): 是否加载预训练的模型权重，默认为False。
    - **kwargs: 允许传入额外的模型配置参数。

    返回：
    - model (UDeepSC): 创建的UDeepSC模型。
    """

    # 初始化UDeepSC模型，传入超参数配置
    model = UDeepSC(
        mode='small',  # 模型的规模（小型配置）
        img_size=224,  # 输入图像的尺寸
        patch_size=32,  # 图像切分的patch尺寸
        img_embed_dim=384,  # 图像嵌入的维度
        text_embed_dim=384,  # 文本嵌入的维度
        speech_embed_dim=128,  # 语音嵌入的维度
        img_encoder_depth=6,  # 图像编码器的层数
        text_encoder_depth=4,  # 文本编码器的层数
        speech_encoder_depth=4,  # 语音编码器的层数
        encoder_num_heads=6,  # 编码器多头自注意力机制的头数
        decoder_embed_dim=128,  # 解码器嵌入的维度
        decoder_depth=2,  # 解码器的层数
        decoder_num_heads=4,  # 解码器多头自注意力机制的头数
        mlp_ratio=4,  # MLP（多层感知机）部分的维度比率
        qkv_bias=True,  # 是否使用QKV偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 使用的归一化层
        **kwargs  # 传入的其他超参数
    )

    # 设置模型的默认配置
    model.default_cfg = _cfg()

    # 如果需要加载预训练权重
    if pretrained:
        # 加载预训练模型的权重
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"  # 从指定路径加载模型检查点
        )
        # 加载模型的权重到当前模型
        model.load_state_dict(checkpoint["model"])

    return model  # 返回初始化后的模型
