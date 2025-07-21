import torch
from torch import optim as optim

# 从 timm 库中导入不同类型的优化器
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

# 检查是否支持 apex 优化器
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True  # 表示支持 apex
except ImportError:
    has_apex = False  # 表示不支持 apex

def get_num_layer_for_vit(var_name, num_max_layer):
    """
    根据变量名称推断其所属的 Transformer 层数，用于 ViT 模型的分层学习率。

    参数:
    ----------
    var_name: str
        参数变量的名称。
    num_max_layer: int
        模型中最大层数。

    返回:
    ----------
    layer_id: int
        对应的层编号。
    """
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0  # 特定变量（如 CLS Token）属于第 0 层
    elif var_name.startswith("patch_embed"):
        return 0  # Patch Embedding 模块归为第 0 层
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1  # 相对位置偏置归为最后一层
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])  # 提取 Block 的编号
        return layer_id + 1
    elif var_name.startswith("block_cas"):
        return 10 + 1  # 特殊 block 的层数（固定为 11）
    else:
        return num_max_layer - 1  # 默认归为最后一层


class LayerDecayValueAssigner(object):
    """
    层学习率衰减值分配器，根据层编号分配学习率缩放值。
    """
    def __init__(self, values):
        self.values = values  # 每层的学习率缩放值

    def get_scale(self, layer_id):
        """
        获取指定层的学习率缩放值。

        参数:
        ----------
        layer_id: int
            层编号。

        返回:
        ----------
        scale: float
            该层的学习率缩放值。
        """
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        """
        根据变量名称获取其所属的层编号。

        参数:
        ----------
        var_name: str
            参数变量名称。

        返回:
        ----------
        layer_id: int
            对应的层编号。
        """
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    """
    根据模型参数生成分组，用于实现分层学习率和不同的 weight decay 设置。

    参数:
    ----------
    model: nn.Module
        待优化的模型。
    weight_decay: float, 默认 1e-5
        权重衰减值。
    skip_list: list, 默认 ()
        不进行 weight decay 的参数名称列表。
    get_num_layer: function, 默认 None
        获取层编号的函数。
    get_layer_scale: function, 默认 None
        获取层学习率缩放值的函数。

    返回:
    ----------
    parameter_groups: list
        包含每个参数分组及其优化器设置的列表。
    """
    parameter_group_names = {}  # 记录分组的名称
    parameter_group_vars = {}  # 记录分组对应的变量

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)  # 打印冻结的参数名称
            continue  # 忽略冻结的参数
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"  # 不进行权重衰减的分组
            this_weight_decay = 0.0
        else:
            group_name = "decay"  # 进行权重衰减的分组
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)  # 获取参数所属的层编号
            group_name = f"layer_{layer_id}_{group_name}"  # 添加层编号到分组名称
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)  # 获取学习率缩放值
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)  # 添加参数到分组
        parameter_group_names[group_name]["params"].append(name)  # 添加参数名称到分组

    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    """
    根据用户指定的参数创建优化器。

    参数:
    ----------
    args: argparse.Namespace
        优化器配置参数，包括 `opt`、`lr`、`weight_decay` 等。
    model: nn.Module
        待优化的模型。
    get_num_layer: function, 默认 None
        获取层编号的函数。
    get_layer_scale: function, 默认 None
        获取层学习率缩放值的函数。
    filter_bias_and_bn: bool, 默认 True
        是否过滤掉偏置项和 BatchNorm 层参数的权重衰减。
    skip_list: list, 默认 None
        不进行 weight decay 的参数列表。

    返回:
    ----------
    optimizer: Optimizer
        创建的优化器实例。
    """
    opt_lower = args.opt.lower()  # 转换优化器名称为小写
    weight_decay = args.weight_decay  # 权重衰减
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list  # 用户指定的跳过列表
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()  # 模型中指定不衰减的参数
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.0  # 在分组内设置权重衰减，因此这里设为 0
    else:
        parameters = model.parameters()

    # 如果选择了 fused 优化器，需确保支持 apex 并有 GPU
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX 和 CUDA 是使用 fused 优化器的必要条件'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps  # 添加 eps 参数
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas  # 添加 betas 参数

    print("优化器配置:", opt_args)

    opt_split = opt_lower.split('_')  # 分解优化器名称，支持 Lookahead
    opt_lower = opt_split[-1]

    # 根据优化器名称创建优化器实例
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    # 其他优化器依次添加...
    else:
        raise ValueError("无效的优化器选择")

    # 如果选择了 Lookahead，则将其应用于基础优化器
    if len(opt_split) > 1 and opt_split[0] == 'lookahead':
        optimizer = Lookahead(optimizer)

    return optimizer
