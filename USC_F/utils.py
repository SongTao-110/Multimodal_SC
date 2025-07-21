import io
import os
import csv
import math
import time
import json
import thop
import torch
import datetime
import numpy as np
import torch.distributed as dist

from pathlib import Path
from torch._six import inf
import torch.nn.functional as F
from timm.utils import get_state_dict
from timm.models import create_model
from collections import OrderedDict
from pytorch_msssim import ms_ssim, ssim
from collections import defaultdict, deque
from timm.loss import LabelSmoothingCrossEntropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, f1_score
## 包含的必要包
import torch
from collections import OrderedDict

def sel_criterion_train(args, ta_sel, device):
    """
    根据任务选择训练阶段的损失函数。

    参数:
    ----------
    args: argparse.Namespace, 包含模型训练的相关配置参数。
    ta_sel: list, 包含选择的任务类型，如 'imgc'、'textc'、'vqa' 等。
    device: torch.device, 指定计算设备（CPU 或 GPU）。

    返回:
    ----------
    criterion_group: dict, 包含每个任务对应的损失函数。
    """
    criterion_group = {}
    for ta in ta_sel:  # 遍历任务列表
        if ta.startswith('imgc'):  # 图像分类任务
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
            print(f"criterion for {args.ta_perform} classification = {criterion}")
        elif ta.startswith('textc'):  # 文本分类任务
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
            print(f"criterion for {args.ta_perform} classification = {criterion}")
        elif ta.startswith('imgr'):  # 图像重建任务
            criterion = torch.nn.MSELoss()  # 使用均方误差损失
            print(f"criterion for {args.ta_perform} Reconstruction = {criterion}")
        elif ta.startswith('textr'):  # 文本重建任务
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
            print(f"criterion for {args.ta_perform} classification = {criterion}")
        elif ta.startswith('vqa'):  # 视觉问答任务
            criterion = torch.nn.BCELoss(reduction='sum').to(device)  # 使用二元交叉熵损失
            print(f"criterion for {args.ta_perform} classification = {criterion}")
        elif ta.startswith('msa'):  # 多模态语义任务
            criterion = torch.nn.MSELoss().to(device)
            print(f"criterion for {args.ta_perform} Reconstruction = {criterion}")
        criterion_group[ta] = criterion  # 将损失函数加入字典
    return criterion_group


def sel_criterion_test(args, device):
    """
    根据任务选择测试阶段的损失函数。

    参数:
    ----------
    args: argparse.Namespace, 包含模型训练的相关配置参数。
    device: torch.device, 指定计算设备（CPU 或 GPU）。

    返回:
    ----------
    criterion: torch.nn.Module, 对应任务的损失函数。
    """
    if args.ta_perform.startswith('imgc'):  # 图像分类任务
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        print(f"criterion for {args.ta_perform} classification = {criterion}")
    elif args.ta_perform.startswith('textc'):  # 文本分类任务
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        print(f"criterion for {args.ta_perform} classification = {criterion}")
    elif args.ta_perform.startswith('imgr'):  # 图像重建任务
        criterion = torch.nn.MSELoss().to(device)
        print(f"criterion for {args.ta_perform} Reconstruction = {criterion}")
    elif args.ta_perform.startswith('textr'):  # 文本重建任务
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        print(f"criterion for {args.ta_perform} classification = {criterion}")
    elif args.ta_perform.startswith('vqa'):  # 视觉问答任务
        criterion = torch.nn.BCELoss(reduction='sum').to(device)
        print(f"criterion for {args.ta_perform} classification = {criterion}")
    elif args.ta_perform.startswith('msa'):  # 多模态语义任务
        criterion = torch.nn.MSELoss().to(device)
        print(f"criterion for {args.ta_perform} Reconstruction = {criterion}")
    return criterion


def get_model(args):
    """
    根据参数初始化模型。

    参数:
    ----------
    args: argparse.Namespace, 包含模型的配置参数。

    返回:
    ----------
    model: torch.nn.Module, 初始化的模型实例。
    """
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,  # 模型名称
        pretrained=False,  # 不加载预训练权重
        drop_path_rate=args.drop_path,  # 路径丢弃率
        drop_block_rate=None,  # 块丢弃率（此处未设置）
    )
    # 计算模型的参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=> Number of params: {:.2f} M'.format(n_parameters / 1e6))
    return model


def load_checkpoint(model, args):
    """
    加载模型检查点文件，并处理相关的状态字典。

    参数:
    ----------
    model: torch.nn.Module, 模型实例。
    args: argparse.Namespace, 包含检查点路径等相关参数。

    返回:
    ----------
    checkpoint_model: dict, 模型的状态字典。
    """
    # 加载检查点
    checkpoint = torch.load(args.resume, map_location='cpu')
    print("从指定路径加载检查点")
    checkpoint_model = None

    # 根据模型键提取状态字典
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f"通过 model_key = {model_key} 加载状态字典")
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    state_dict = model.state_dict()  # 获取当前模型的状态字典
    all_keys = list(checkpoint_model.keys())  # 检查点中的所有键
    new_dict = OrderedDict()  # 用于存储更新后的状态字典

    # 处理检查点的键
    for key in all_keys:
        if key.startswith('encoder1.'):  # 对特定编码器的键进行处理
            new_dict['img_' + key] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict
    return checkpoint_model

def is_dist_avail_and_initialized():
    """
    判断分布式模式是否可用并已初始化
    - 如果 `torch.distributed` 不可用，则返回 False
    - 如果分布式模式未初始化，则返回 False
    """
    if not dist.is_available():  # 检查分布式是否可用
        return False
    if not dist.is_initialized():  # 检查分布式是否已初始化
        return False
    return True

def get_world_size():
    """
    获取全局通信中包含的进程数（世界大小）
    - 如果分布式不可用或未初始化，则默认返回 1
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()  # 获取当前分布式通信的总进程数

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    将已经加载的 checkpoint 传入模型的 EMA（指数移动平均）更新
    - 使用内存文件 `io.BytesIO` 模拟文件系统保存和加载
    """
    mem_file = io.BytesIO()  # 创建内存中的二进制文件对象
    torch.save(checkpoint, mem_file)  # 保存 checkpoint 到内存文件
    mem_file.seek(0)  # 将内存文件指针重置到文件开头
    model_ema._load_checkpoint(mem_file)  # 加载 checkpoint

def get_rank():
    """
    获取当前进程的 rank（在分布式通信中的唯一标识）
    - 如果分布式不可用或未初始化，则返回 0（主进程）
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()  # 获取当前进程的 rank

def is_main_process():
    """
    判断当前进程是否为主进程（rank 为 0 的进程）
    """
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    """
    在主进程上保存数据
    - 仅主进程执行 `torch.save`，避免多个进程重复保存
    """
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    为分布式模式设置打印行为
    - 非主进程将禁止打印，避免重复输出
    """
    import builtins as __builtin__  # 引入内建模块
    builtin_print = __builtin__.print  # 保存原始 print 函数

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)  # 检查是否强制打印
        if is_master or force:  # 主进程或设置强制打印时才输出
            builtin_print(*args, **kwargs)

    __builtin__.print = print  # 替换内建 print 函数

def init_distributed_mode(args):
    """
    初始化分布式模式
    - 根据不同的分布式环境设置 rank、world_size 和 gpu
    """
    if args.dist_on_itp:  # 如果使用 ITP 分布式环境
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])  # 获取全局 rank
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])  # 获取进程总数
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])  # 获取本地 GPU 编号
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])  # 设置分布式地址
        os.environ['LOCAL_RANK'] = str(args.gpu)  # 设置本地 rank 环境变量
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:  # 如果通过环境变量设置
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:  # 如果在 SLURM 环境下
        args.rank = int(os.environ['SLURM_PROCID'])  # 获取 rank
        args.gpu = args.rank % torch.cuda.device_count()  # 分配 GPU
    else:  # 未使用分布式模式
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True  # 启用分布式模式

    torch.cuda.set_device(args.gpu)  # 设置当前进程使用的 GPU
    args.dist_backend = 'nccl'  # 使用 NCCL 后端
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)  # 初始化分布式组
    torch.distributed.barrier()  # 等待所有进程同步
    setup_for_distributed(args.rank == 0)  # 配置主进程打印

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    """
    加载模型的状态字典，并处理缺失键和额外键
    - 支持忽略特定的缺失键（通过 ignore_missing 参数指定）
    """
    missing_keys = []  # 存储缺失的键
    unexpected_keys = []  # 存储额外的键
    error_msgs = []  # 存储加载时的错误信息

    metadata = getattr(state_dict, '_metadata', None)  # 获取元数据
    state_dict = state_dict.copy()  # 复制状态字典
    if metadata is not None:
        state_dict._metadata = metadata  # 保留元数据

    def load(module, prefix=''):
        """
        递归加载模块的状态字典
        """
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)  # 加载状态
        for name, child in module._modules.items():  # 遍历子模块
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)  # 开始加载模型

    warn_missing_keys = []  # 警告的缺失键
    ignore_missing_keys = []  # 忽略的缺失键
    for key in missing_keys:  # 遍历缺失的键
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):  # 检查是否需要忽略
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))  # 打印警告信息
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))  # 打印额外键信息
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))  # 打印忽略的键信息
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))  # 打印错误信息
class NativeScalerWithGradNormCount:
    """
    使用 torch.cuda.amp.GradScaler 实现的梯度缩放器（Scaler），
    包含梯度裁剪功能以及支持梯度范数计算。
    """
    state_dict_key = "amp_scaler"  # 状态字典中的键，用于保存和加载 Scaler 的状态

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()  # 初始化 GradScaler

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
        前向传播和反向传播的核心逻辑
        - loss: 损失值
        - optimizer: 优化器
        - clip_grad: 梯度裁剪阈值（可选）
        - parameters: 需要更新的参数（当进行梯度裁剪时必需）
        - create_graph: 是否创建计算图（用于二阶优化等）
        - update_grad: 是否更新梯度
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)  # 缩放损失并进行反向传播
        if update_grad:
            if clip_grad is not None:  # 如果设置了梯度裁剪
                assert parameters is not None  # 参数不能为空
                self._scaler.unscale_(optimizer)  # 取消梯度缩放
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)  # 执行梯度裁剪
            else:
                self._scaler.unscale_(optimizer)  # 取消梯度缩放
                norm = get_grad_norm_(parameters)  # 计算梯度范数
            self._scaler.step(optimizer)  # 执行优化器更新
            self._scaler.update()  # 更新梯度缩放因子
        else:
            norm = None
        return norm

    def state_dict(self):
        """
        返回梯度缩放器的状态字典
        """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """
        加载梯度缩放器的状态字典
        """
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    计算参数的梯度范数
    - parameters: 参数列表
    - norm_type: 范数类型（默认为 L2 范数）
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]  # 如果输入是单个张量，转换为列表
    parameters = [p for p in parameters if p.grad is not None]  # 仅保留有梯度的参数
    norm_type = float(norm_type)  # 转换为浮点数
    if len(parameters) == 0:  # 如果参数为空
        return torch.tensor(0.)  # 返回 0
    device = parameters[0].grad.device  # 获取梯度所在设备
    if norm_type == float('inf'):  # L∞ 范数
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:  # Lp 范数
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    """
    实现余弦学习率调度器
    - base_value: 基础学习率
    - final_value: 最终学习率
    - epochs: 总训练轮数
    - niter_per_ep: 每轮的迭代数
    - warmup_epochs: 热身阶段的轮数
    - start_warmup_value: 热身阶段的起始值
    - warmup_steps: 热身阶段的迭代步数
    """
    warmup_schedule = np.array([])  # 热身阶段的学习率列表
    warmup_iters = warmup_epochs * niter_per_ep  # 热身阶段的迭代次数
    if warmup_steps > 0:
        warmup_iters = warmup_steps  # 如果设置了特定热身步数，则覆盖
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)  # 线性插值生成热身学习率

    iters = np.arange(epochs * niter_per_ep - warmup_iters)  # 非热身阶段的迭代索引
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])  # 余弦调度公式

    schedule = np.concatenate((warmup_schedule, schedule))  # 合并热身和余弦阶段的学习率

    assert len(schedule) == epochs * niter_per_ep  # 检查调度器的总长度
    return schedule


def path_exists_make(path):
    """
    检查路径是否存在，如果不存在则创建
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """
    保存模型的检查点
    - args: 配置参数
    - epoch: 当前轮数
    - model: 模型
    - model_without_ddp: 不含分布式包装的模型
    - optimizer: 优化器
    - loss_scaler: 梯度缩放器
    - model_ema: EMA 模型（可选）
    """
    output_dir = Path(args.output_dir + '/ckpt_' + args.ta_perform)  # 检查点保存路径
    path_exists_make(output_dir)  # 确保路径存在
    epoch_name = str(epoch)  # 当前轮次的名称
    if loss_scaler is not None:  # 如果有梯度缩放器
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]  # 保存路径
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),  # 模型状态
                'optimizer': optimizer.state_dict(),  # 优化器状态
                'epoch': epoch,  # 当前轮次
                'scaler': loss_scaler.state_dict(),  # 梯度缩放器状态
                'args': args,  # 参数
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)  # EMA 模型状态

            save_on_master(to_save, checkpoint_path)  # 主进程保存
    else:  # 不使用梯度缩放器时
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """
    自动加载最新的检查点
    - args: 配置参数
    - model: 模型
    - model_without_ddp: 不含分布式包装的模型
    - optimizer: 优化器
    - loss_scaler: 梯度缩放器
    - model_ema: EMA 模型（可选）
    """
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        if args.auto_resume and len(args.resume) == 0:  # 如果启用了自动恢复且未指定检查点
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))  # 搜索所有检查点
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]  # 提取轮次编号
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:  # 找到最新的检查点
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:  # 如果有检查点
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')  # 加载检查点
            model_without_ddp.load_state_dict(checkpoint['model'])  # 加载模型状态
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
                args.start_epoch = checkpoint['epoch'] + 1  # 设置起始轮次
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])  # 加载 EMA 状态
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])  # 加载梯度缩放器状态
                print("With optim & sched!")
    else:  # 使用 DeepSpeed 时的检查点加载
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))  # 搜索所有检查点
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, client_states['model_ema'])  # 加载 EMA 状态
def tensor2cuda(tensor):
    """
    将张量（Tensor）转移到 CUDA 设备上
    - 如果 CUDA 可用，则调用 .cuda() 将张量转移到 GPU
    - 否则，保持张量不变
    """
    if torch.cuda.is_available():  # 检查是否支持 CUDA
        tensor = tensor.cuda()  # 将张量转移到 CUDA
    return tensor  # 返回张量


def create_ds_config(args):
    """
    创建 DeepSpeed 配置文件
    - args: 输入参数，包括批量大小、学习率、权重衰减等
    """
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")  # 配置文件路径
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),  # 全局批量大小
            "train_micro_batch_size_per_gpu": args.batch_size,  # 每个 GPU 的微批量大小
            "steps_per_print": 1000,  # 每隔多少步打印一次日志
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,  # 启用 AdamW 优化
                "params": {
                    "lr": args.lr,  # 学习率
                    "weight_decay": args.weight_decay,  # 权重衰减
                    "bias_correction": True,  # 启用偏差校正
                    "betas": [0.9, 0.999],  # Adam 的 beta 参数
                    "eps": 1e-8  # Adam 的 epsilon
                }
            },
            "fp16": {  # 使用混合精度（FP16）训练
                "enabled": True,  # 启用混合精度
                "loss_scale": 0,  # 自动 loss scale
                "initial_scale_power": 7,  # 初始 loss scale 的指数
                "loss_scale_window": 128  # 更新 loss scale 的窗口大小
            }
        }
        writer.write(json.dumps(ds_config, indent=2))  # 将配置写入文件（格式化为 JSON）


def batch_index_select(x, idx):
    """
    根据索引批量选择张量的子集
    - x: 输入张量
    - idx: 索引张量
    - 支持 2D 和 3D 张量的索引操作
    """
    if len(x.size()) == 3:  # 3D 张量
        B, N, C = x.size()  # B: 批量大小, N: 序列长度, C: 特征维度
        N_new = idx.size(1)  # 新的序列长度
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N  # 计算偏移量
        idx = idx + offset  # 索引加偏移量，变为全局索引
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)  # 按索引选取并重塑张量
        return out
    elif len(x.size()) == 2:  # 2D 张量
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError  # 对于其他维度的张量，抛出未实现的异常


def psnr(img1, img2):
    """
    计算 PSNR（峰值信噪比）
    - img1: 原始图像
    - img2: 预测图像
    """
    mse = torch.mean((img1 - img2) ** 2.0, dtype=torch.float32)  # 计算均方误差
    if mse == 0:  # 如果误差为 0，返回一个高值
        return torch.tensor([100.0])
    PIXEL_MAX = 255.0  # 像素值的最大值
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))  # 根据公式计算 PSNR


def get_imagenet_list(path):
    """
    从文件中读取 ImageNet 数据集的文件名列表
    - path: 包含文件名的 CSV 文件路径
    """
    fns = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)  # 使用 CSV 读取器
        for row in reader:
            fns.append(row[0])  # 将文件名加入列表
    return fns  # 返回文件名列表


def complex_sig(shape, device):
    """
    生成复数信号（标准正态分布）
    - shape: 信号的形状
    - device: 信号所在设备
    """
    sig_real = torch.randn(*shape)  # 实部
    sig_imag = torch.randn(*shape)  # 虚部
    return (torch.complex(sig_real, sig_imag) / np.sqrt(2)).to(device)  # 标准化并转移到设备


def pwr_normalize(sig):
    """
    对信号进行功率归一化
    - sig: 输入信号张量
    """
    _, num_ele = sig.shape[0], torch.numel(sig[0])  # 获取信号元素个数
    pwr_sig = torch.sum(torch.abs(sig) ** 2, dim=-1) / num_ele  # 计算信号功率
    sig = sig / torch.sqrt(pwr_sig.unsqueeze(-1))  # 对信号归一化
    return sig


def np_to_torch(img):
    """
    将 NumPy 图像转换为 PyTorch 张量
    - img: NumPy 图像数据（格式为 W, H, C）
    """
    img = np.swapaxes(img, 0, 1)  # 转换为 H, W, C
    img = np.swapaxes(img, 0, 2)  # 转换为 C, H, W
    return torch.from_numpy(img).float()  # 转换为浮点型张量


def to_chan_last(img):
    """
    将通道顺序转换为最后一维（用于图像处理）
    - img: 输入张量（C, H, W）
    """
    img = img.transpose(1, 2)  # 交换 H 和 W
    img = img.transpose(2, 3)  # 交换 W 和 C
    return img


def as_img_array(image):
    """
    将张量转换为图像数组（像素值在 0-255 之间）
    """
    image = image.clamp(0, 1) * 255.0  # 限制值域并缩放
    return torch.round(image)  # 四舍五入为整数


def calc_psnr(predictions, targets):
    """
    批量计算 PSNR 指标
    - predictions: 预测结果列表
    - targets: 真实目标列表
    """
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])  # 转换为图像数组
        compare = as_img_array(pred)
        val = psnr(original, compare)  # 计算 PSNR
        metric.append(val)
    return metric


def calc_msssim(predictions, targets):
    """
    批量计算 MS-SSIM 指标
    - predictions: 预测结果列表
    - targets: 真实目标列表
    """
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        val = ms_ssim(original, compare, data_range=255, win_size=3, size_average=True)  # 调用 MS-SSIM 函数
        metric.append(val)
    return metric


def calc_ssim(predictions, targets):
    """
    批量计算 SSIM 指标
    - predictions: 预测结果列表
    - targets: 真实目标列表
    """
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        val = ssim(original, compare, data_range=255, size_average=True)  # 调用 SSIM 函数
        metric.append(val)
    return metric


import nltk
from pytorch_transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokens2sentence(outputs):
    """
    将 token 转换为句子
    - outputs: 一个二维列表，每个子列表包含多个 token
    """
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = tokenizer.decode([int(token)])  # 解码 token 为单词
            if word == '[PAD]':  # 如果遇到填充标记 [PAD]，停止解析
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences


def computebleu(sentences, targets):
    """
    计算 BLEU 分数
    - sentences: 生成的句子列表
    - targets: 目标句子列表
    """
    score = 0
    assert (len(sentences) == len(targets))  # 确保生成和目标句子数量一致

    def cut_token(sentence):
        """
        切分单词或处理特殊标记 [UNK]
        """
        tmp = []
        for token in sentence:
            if token == '[UNK]':  # 未知标记直接保留
                tmp.append(token)
            else:
                tmp += [word for word in token]  # 分解单词
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))  # 计算 BLEU-1 分数
    return score


def adjust_learning_rate(param_groups, init_lr, min_lr, step, max_step, warming_up_step=2,
                         warmup_predictor=False, base_multi=0.1):
    """
    动态调整学习率
    - param_groups: 参数组
    - init_lr: 初始学习率
    - min_lr: 最小学习率
    - step: 当前步数
    - max_step: 总步数
    - warming_up_step: 热身步数
    - warmup_predictor: 是否启用预测器的热身阶段
    """
    cos_lr = (math.cos(step / max_step * math.pi) + 1) * 0.5  # 余弦学习率调整
    cos_lr = min_lr + cos_lr * (init_lr - min_lr)  # 计算当前学习率
    if warmup_predictor and step < 1:  # 初始预测器热身
        cos_lr = init_lr * 0.1
    if step < warming_up_step:  # 主网络热身阶段
        backbone_lr = 0
    else:
        backbone_lr = min(init_lr * 0.1, cos_lr)  # 主网络使用较小的学习率
    print('## Using lr  %.7f for BACKBONE, cosine lr = %.7f for PREDICTOR' % (backbone_lr, cos_lr))
    for param_group in param_groups:
        if param_group['name'] == 'mask_generator':  # 为 mask 生成器设置学习率
            param_group['lr'] = cos_lr
        else:
            param_group['lr'] = backbone_lr  # 为主网络设置学习率


def calc_metrics(y_true, y_pred, mode=None, to_print=True):
    """
    计算评价指标，包括 MAE、相关系数、多类准确率等
    - y_true: 真实值
    - y_pred: 预测值
    - mode: 可选的评估模式（暂未使用）
    - to_print: 是否打印详细报告
    """

    def multiclass_acc(preds, truths):
        """
        计算多类别准确率
        - preds: 预测值数组
        - truths: 真实值数组
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    test_preds = y_pred
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])  # 筛选非零数据

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)  # 裁剪预测值到 [-3, 3]
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)  # 裁剪预测值到 [-2, 2]
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # 计算 MAE（平均绝对误差）
    corr = np.corrcoef(test_preds, test_truth)[0][1]  # 计算相关系数
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)  # 多类准确率（a7）
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)  # 多类准确率（a5）

    # pos - neg 二分类评估
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    if to_print:
        print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))  # 打印正负分类准确率

        # non-neg - neg 二分类评估
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        if to_print:
            print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))  # 打印非负分类准确率

        return accuracy_score(binary_truth, binary_preds)  # 返回准确率


def get_param_groups(model, weight_decay):
    """
    获取模型参数组，用于优化器设置
    - model: 模型
    - weight_decay: 权重衰减系数
    """
    decay = []  # 需要权重衰减的参数
    no_decay = []  # 不需要权重衰减的参数
    mask_generator = []  # mask 生成器的参数
    for name, param in model.named_parameters():
        if 'mask_generator' in name:  # mask 生成器参数
            mask_generator.append(param)
        elif not param.requires_grad:  # 冻结的参数，跳过
            continue
        elif 'cls_token' in name or 'pos_embed' in name:  # 冻结的特殊参数
            continue
        elif len(param.shape) == 1 or name.endswith(".bias"):  # 不需要衰减的参数（如偏置项）
            no_decay.append(param)
        else:  # 需要权重衰减的参数
            decay.append(param)
    return [
        {'params': mask_generator, 'weight_decay': weight_decay, 'name': 'predictor'},  # mask 生成器组
        {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay'},  # 无权重衰减组
        {'params': decay, 'weight_decay': weight_decay, 'name': 'base_decay'}  # 权重衰减组
    ]


class DiffPruningLoss(torch.nn.Module):
    """
    差分剪枝损失函数
    - 该类用于实现一种自定义的损失函数，包括主损失、监督损失、掩码损失、稀疏性约束和码本损失等
    """
    def __init__(self, base_criterion: torch.nn.Module, dynamic=True, ratio_weight=2.0, main_weight=1.):
        """
        初始化差分剪枝损失函数
        - base_criterion: 基础损失函数，如交叉熵或均方误差
        - dynamic: 是否启用动态损失模式
        - ratio_weight: 比例权重（未使用）
        - main_weight: 主损失的权重
        """
        super().__init__()
        self.base_criterion = base_criterion  # 基础损失函数
        self.main_weight = 1.  # 主损失的权重
        self.surp_weight = 120  # 监督损失的权重
        self.rho_weight = 0  # 稀疏性约束的权重
        self.mask_weight = 0.0  # 掩码损失的权重
        self.codebook_weight = 2.0  # 码本损失的权重
        self.print_mode = True  # 是否打印调试信息

        self.count = 0  # 用于记录打印的周期

        # 损失记录变量，用于统计平均损失
        self.main_loss_record = 0.
        self.surp_loss_record = 0.
        self.codebook_loss_record_img = 0.
        self.codebook_loss_record_text = 0.
        self.codebook_loss_record_spe = 0.
        self.mask_loss_record = 0.
        self.rho_record_img = 0.
        self.rho_record_text = 0.
        self.keep_ratio_record_img = 0.
        self.keep_ratio_record_text = 0.

        self.dynamic = dynamic
        if self.dynamic:
            print('using dynamic loss')  # 如果启用了动态损失，打印信息

    def forward(self, outputs, labels, ta_perform):
        """
        前向传播，计算损失值
        - outputs: 模型的输出，包括预测值、掩码信息、稀疏性信息和码本损失
        - labels: 真实标签
        - ta_perform: 当前任务类型（如 'img', 'text', 'vqa' 等）
        """
        pred, m_dict, rho_dict, codebook_loss_dict = outputs
        surp_loss = 0.0  # 监督损失
        mask_loss = 0.0  # 掩码损失
        keep_ratio_text = np.array([0.])  # 文本的保留比率
        keep_ratio_img = np.array([0.])  # 图像的保留比率
        rho_img, rho_text = 0., 0.  # 稀疏性约束

        # 码本损失（包含图像、文本和语音的损失权重）
        codebook_loss = codebook_loss_dict['img'] + codebook_loss_dict['text'] * 0.8 + codebook_loss_dict['spe']

        if ta_perform.startswith('vqa'):  # 如果是 VQA 任务
            for i, mask_m in enumerate(m_dict['text']):
                mask_m = mask_m.squeeze(2)  # 去除多余的维度
                keep_ratio_text = mask_m.mean(1)  # 计算每个样本的保留比率
                surp_loss += ((keep_ratio_text - rho_dict['text'][i]) ** 2).mean()  # 监督损失
                mask_loss += keep_ratio_text.mean()  # 累加掩码损失
            for i, mask_m in enumerate(m_dict['img']):
                mask_m = mask_m.squeeze(2)
                keep_ratio_img = mask_m.mean(1)
                surp_loss += ((keep_ratio_img - rho_dict['img'][i]) ** 2).mean()
                mask_loss += keep_ratio_img.mean()
            main_loss = self.base_criterion(pred, labels)  # 计算主损失（如重构损失）
            loss = self.main_weight * main_loss + \
                self.surp_weight * surp_loss + \
                self.codebook_weight * codebook_loss  # 加权合成总损失
            rho_text, rho_img = rho_dict['text'][-1], rho_dict['img'][-1]

        elif ta_perform.startswith('img'):  # 如果是图像任务
            for i, mask_m in enumerate(m_dict['img']):
                mask_m = mask_m.squeeze(2)
                keep_ratio_img = mask_m.mean(1)
                surp_loss += ((keep_ratio_img - rho_dict['img'][i]) ** 2).mean()
                mask_loss += keep_ratio_img.mean()
            main_loss = self.base_criterion(pred, labels)
            loss = self.main_weight * main_loss + \
                self.surp_weight * surp_loss + \
                self.codebook_weight * codebook_loss
            rho_img = rho_dict['img'][-1]

        elif ta_perform.startswith('text'):  # 如果是文本任务
            for i, mask_m in enumerate(m_dict['text']):
                mask_m = mask_m.squeeze(2)
                keep_ratio_text = mask_m.mean(1)
                surp_loss += ((keep_ratio_text - rho_dict['text'][i]) ** 2).mean()
                mask_loss += keep_ratio_text.mean()
            main_loss = self.base_criterion(pred, labels)
            loss = self.main_weight * main_loss + \
                self.surp_weight * surp_loss + \
                self.codebook_weight * codebook_loss
            rho_text = rho_dict['text'][-1]

        if self.print_mode:  # 打印调试信息
            self.main_loss_record += 0
            self.surp_loss_record += surp_loss.item()
            self.codebook_loss_record_img += codebook_loss_dict['img'].item()
            self.codebook_loss_record_text += codebook_loss_dict['text'].item()
            self.keep_ratio_record_img += keep_ratio_img.mean().item()
            self.keep_ratio_record_text += keep_ratio_text.mean().item()
            self.rho_record_img += rho_img
            self.rho_record_text += rho_text
            self.mask_loss_record += mask_loss.item()
            self.count += 1
            if self.count == 50:  # 每 50 个批次打印一次信息
                print('loss info: task=%s | main_loss=%.4f, surp_loss=%.4f, cl_i=%.4f, cl_t=%.4f, kr_i=%.4f,kr_t=%.4f, rho_i=%.4f, rho_t=%.4f, mask_loss=%.4f'
                        % ( ta_perform,
                           self.main_loss_record / self.count,
                           self.surp_loss_record / self.count,
                           self.codebook_loss_record_img / self.count,
                           self.codebook_loss_record_text / self.count,
                           self.keep_ratio_record_img / self.count,
                           self.keep_ratio_record_text / self.count,
                           self.rho_record_img / self.count,
                           self.rho_record_text / self.count,
                           self.mask_loss_record / self.count))
                # 重置记录变量
                self.main_loss_record = 0
                self.surp_loss_record = 0
                self.codebook_loss_record_img = 0
                self.codebook_loss_record_text = 0
                self.keep_ratio_record_img = 0
                self.keep_ratio_record_text = 0
                self.rho_record_img = 0
                self.rho_record_text = 0
                self.mask_loss_record = 0
                self.count = 0

        return loss  # 返回总损失
