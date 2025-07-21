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
## Including pakages
from UDeepSC_FSM.utils import _load_checkpoint_for_ema


def sel_criterion_train(args, ta_sel, device):
    """
    根据任务类型选择对应的训练损失函数。
    """
    criterion_group = {}  # 用于存储不同任务的损失函数
    for ta in ta_sel:  # 遍历所有任务
        if ta.startswith('imgc'):  # 图像分类任务
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)  # 使用标签平滑的交叉熵损失
            print("criterion for %s classification = %s" % (args.ta_perform, str(criterion)))
        elif ta.startswith('textc'):  # 文本分类任务
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
            print("criterion for %s classification = %s" % (args.ta_perform, str(criterion)))
        elif ta.startswith('imgr'):  # 图像重建任务
            criterion = torch.nn.MSELoss()  # 使用均方误差损失
            print("criterion for %s Reconstruction = %s" % (args.ta_perform, str(criterion)))
        elif ta.startswith('textr'):  # 文本生成任务
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
            print("criterion for %s classification = %s" % (args.ta_perform, str(criterion)))
        elif ta.startswith('vqa'):  # VQA任务
            criterion = torch.nn.BCELoss(reduction='sum').to(device)  # 使用二元交叉熵损失
            print("criterion for %s classification = %s" % (args.ta_perform, str(criterion)))
        elif ta.startswith('msa'):  # 多模态任务
            criterion = torch.nn.MSELoss().to(device)  # 使用均方误差损失
            print("criterion for %s Reconstruction = %s" % (args.ta_perform, str(criterion)))
        criterion_group[ta] = criterion  # 将损失函数加入对应任务的字典
    return criterion_group  # 返回任务损失函数字典


def sel_criterion_test(args, device):
    """
    根据任务类型选择对应的测试损失函数。
    """
    if args.ta_perform.startswith('imgc'):
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        print("criterion for %s classification = %s" % (args.ta_perform, str(criterion)))
    elif args.ta_perform.startswith('textc'):
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        print("criterion for %s classification = %s" % (args.ta_perform, str(criterion)))
    elif args.ta_perform.startswith('imgr'):
        criterion = torch.nn.MSELoss().to(device)
        print("criterion for %s Reconstruction = %s" % (args.ta_perform, str(criterion)))
    elif args.ta_perform.startswith('textr'):
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        print("criterion for %s classification = %s" % (args.ta_perform, str(criterion)))
    elif args.ta_perform.startswith('vqa'):
        criterion = torch.nn.BCELoss(reduction='sum').to(device)
        print("criterion for %s classification = %s" % (args.ta_perform, str(criterion)))
    elif args.ta_perform.startswith('msa'):
        criterion = torch.nn.MSELoss().to(device)
        print("criterion for %s Reconstruction = %s" % (args.ta_perform, str(criterion)))
    return criterion  # 返回测试任务对应的损失函数


def get_model(args):
    """
    创建模型并打印模型参数数量。
    """
    print(f"Creating model: {args.model}")  # 打印模型名称
    model = create_model(  # 调用create_model函数创建模型
        args.model,
        pretrained=False,  # 不加载预训练权重
        drop_path_rate=args.drop_path,  # 设置路径丢弃率
        drop_block_rate=None,  # 不使用块丢弃
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计可训练参数数量
    print('=> Number of params: {} M'.format(n_parameters / 1e6))  # 将参数数量转换为百万单位输出

    return model  # 返回创建的模型


def load_checkpoint(model, args):
    """
    加载模型检查点，并处理对应权重。
    """
    checkpoint = torch.load(args.resume, map_location='cpu')  # 从指定路径加载检查点

    print("Load ckpt from the place")  # 打印加载提示
    checkpoint_model = None
    for model_key in args.model_key.split('|'):  # 遍历可能的模型键
        if model_key in checkpoint:  # 如果键在检查点中
            checkpoint_model = checkpoint[model_key]  # 加载对应的模型权重
            print("Load state_dict by model_key = %s" % model_key)  # 打印加载的键
            break
    if checkpoint_model is None:  # 如果未找到对应的键
        checkpoint_model = checkpoint  # 使用检查点本身
    state_dict = model.state_dict()  # 获取模型的当前状态字典

    # 处理权重的前缀，重新整理权重字典
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()  # 创建有序字典
    for key in all_keys:
        if key.startswith('encoder.'):  # 如果权重键以"encoder."开头
            new_dict['img_' + key] = checkpoint_model[key]  # 在键名前添加"img_"前缀
        else:
            new_dict[key] = checkpoint_model[key]  # 保留原键值对
    checkpoint_model = new_dict  # 更新检查点模型
    return checkpoint_model  # 返回处理后的权重


def is_dist_avail_and_initialized():
    """
    检查分布式训练是否可用并已初始化。
    """
    if not dist.is_available():  # 如果分布式模块不可用
        return False
    if not dist.is_initialized():  # 如果分布式未初始化
        return False
    return True  # 分布式可用且已初始化


def get_world_size():
    """
    获取分布式训练的世界大小（即总进程数）。
    """
    if not is_dist_avail_and_initialized():
        return 1  # 如果分布式不可用，返回1
    return dist.get_world_size()  # 返回当前分布式的世界大小


def get_rank():
    """
    获取当前进程的rank。
    """
    if not is_dist_avail_and_initialized():
        return 0  # 如果分布式不可用，返回0（默认主进程）
    return dist.get_rank()  # 返回当前进程的rank


def is_main_process():
    """
    判断是否是主进程。
    """
    return get_rank() == 0  # 如果rank为0，则是主进程


def save_on_master(*args, **kwargs):
    """
    仅在主进程上保存文件。
    """
    if is_main_process():  # 如果是主进程
        torch.save(*args, **kwargs)  # 保存文件

def setup_for_distributed(is_master):
    """
    仅允许主进程打印信息，用于分布式模式下抑制其他进程的打印输出。
    """
    import builtins as __builtin__  # 引入Python内置模块
    builtin_print = __builtin__.print  # 保存原始print函数

    def print(*args, **kwargs):
        """
        覆盖内置print函数，仅在主进程或强制打印时输出。
        """
        force = kwargs.pop('force', False)  # 提取强制打印参数
        if is_master or force:  # 如果是主进程或强制打印
            builtin_print(*args, **kwargs)  # 调用原始print函数

    __builtin__.print = print  # 替换内置print函数


def init_distributed_mode(args):
    """
    初始化分布式训练模式。
    """
    if args.dist_on_itp:  # 如果在ITP环境中
        # 获取OMPI提供的分布式相关环境变量
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        # 设置LOCAL_RANK、RANK和WORLD_SIZE环境变量
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 如果环境变量中有RANK和WORLD_SIZE，直接读取
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # 如果在SLURM环境中，使用SLURM的进程ID
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # 如果没有分布式相关环境变量，打印提示并退出分布式模式
        print('Not using distributed mode')
        args.distributed = False
        return

    # 标记为分布式模式
    args.distributed = True

    # 设置当前进程使用的GPU
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 使用NCCL作为通信后端
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    # 初始化分布式进程组
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()  # 同步所有进程
    setup_for_distributed(args.rank == 0)  # 设置主进程打印


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    """
    加载模型权重，并处理缺失或多余的权重。
    """
    missing_keys = []  # 用于存储缺失的键
    unexpected_keys = []  # 用于存储多余的键
    error_msgs = []  # 用于存储错误信息

    # 拷贝state_dict，避免直接修改
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        """
        递归加载模块的权重。
        """
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        # 调用PyTorch的内置方法加载权重
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():  # 遍历子模块
            if child is not None:
                load(child, prefix + name + '.')  # 递归加载子模块权重

    load(model, prefix=prefix)  # 从指定前缀开始加载权重

    warn_missing_keys = []  # 用于存储需要警告的缺失键
    ignore_missing_keys = []  # 用于存储需要忽略的缺失键
    for key in missing_keys:  # 遍历所有缺失的键
        keep_flag = True  # 标记是否保留该键
        for ignore_key in ignore_missing.split('|'):  # 遍历所有忽略模式
            if ignore_key in key:  # 如果键包含忽略模式
                keep_flag = False  # 不保留该键
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys  # 更新缺失键列表

    # 打印未初始化的权重信息
    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    # 打印未使用的预训练权重信息
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    # 打印被忽略的权重信息
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    # 打印加载错误信息
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

class NativeScalerWithGradNormCount:
    """
    一个用于梯度缩放和梯度裁剪的类，支持混合精度训练。
    """
    state_dict_key = "amp_scaler"

    def __init__(self):
        # 初始化AMP的GradScaler
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
        执行反向传播并更新梯度。

        参数:
        loss - 损失值
        optimizer - 优化器
        clip_grad - 梯度裁剪阈值
        parameters - 模型参数
        create_graph - 是否创建计算图（支持高阶导数）
        update_grad - 是否更新梯度
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)  # 缩放损失并反向传播
        if update_grad:  # 如果需要更新梯度
            if clip_grad is not None:  # 如果启用了梯度裁剪
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # 取消缩放
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)  # 裁剪梯度
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)  # 获取梯度范数
            self._scaler.step(optimizer)  # 执行优化器的步进
            self._scaler.update()  # 更新缩放因子
        else:
            norm = None
        return norm

    def state_dict(self):
        # 返回内部缩放器的状态字典
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        # 加载缩放器的状态
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    计算参数的梯度范数。

    参数:
    parameters - 模型参数
    norm_type - 范数类型（默认为L2范数）
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    """
    生成一个余弦学习率调度器。

    参数:
    base_value - 初始学习率
    final_value - 最终学习率
    epochs - 总训练轮数
    niter_per_ep - 每轮的迭代次数
    warmup_epochs - 学习率预热的轮数
    start_warmup_value - 预热阶段的起始值
    warmup_steps - 预热的步数
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def path_exists_make(path):
    """
    如果路径不存在，则创建路径。
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """
    保存模型的检查点。

    参数:
    args - 配置参数
    epoch - 当前训练轮数
    model - 模型对象
    model_without_ddp - 未包装的模型
    optimizer - 优化器
    loss_scaler - AMP缩放器
    model_ema - 指数移动平均模型
    """
    output_dir = Path(args.output_dir+'/ckpt_'+args.ta_perform)
    path_exists_make(output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """
    自动加载最新的模型检查点。

    参数:
    args - 配置参数
    model - 模型
    model_without_ddp - 未包装的模型
    optimizer - 优化器
    loss_scaler - AMP缩放器
    model_ema - 指数移动平均模型
    """
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # 使用torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed模式，仅支持'--auto_resume'
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
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
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def tensor2cuda(tensor):
    """
    将张量移动到CUDA设备（如果可用）。
    """
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def create_ds_config(args):
    """
    创建DeepSpeed配置文件，并写入到指定路径。

    参数:
    args - 包含配置信息的参数对象
    """
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),  # 全局批量大小
            "train_micro_batch_size_per_gpu": args.batch_size,  # 每个GPU的微批量大小
            "steps_per_print": 1000,  # 每隔多少步打印日志
            "optimizer": {  # 优化器配置
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,  # 学习率
                    "weight_decay": args.weight_decay,  # 权重衰减
                    "bias_correction": True,
                    "betas": [0.9, 0.999],  # Adam的超参数
                    "eps": 1e-8  # 防止分母为零的平滑项
                }
            },
            "fp16": {  # 混合精度训练配置
                "enabled": True,
                "loss_scale": 0,  # 动态损失缩放
                "initial_scale_power": 7,  # 初始损失缩放值
                "loss_scale_window": 128  # 动态缩放窗口
            }
        }
        writer.write(json.dumps(ds_config, indent=2))  # 写入JSON文件


def batch_index_select(x, idx):
    """
    根据索引选择张量的特定元素，支持2D和3D张量。

    参数:
    x - 输入张量
    idx - 索引张量
    """
    if len(x.size()) == 3:  # 如果是3D张量
        B, N, C = x.size()
        N_new = idx.size(1)  # 新的索引维度
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N  # 计算批次偏移
        idx = idx + offset  # 加入偏移后的索引
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)  # 按索引提取元素
        return out
    elif len(x.size()) == 2:  # 如果是2D张量
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError("仅支持2D和3D张量")


def psnr(img1, img2):
    """
    计算两张图像之间的峰值信噪比(PSNR)。

    参数:
    img1, img2 - 输入图像张量
    """
    mse = torch.mean((img1 - img2) ** 2.0, dtype=torch.float32)  # 计算均方误差
    if mse == 0:
        return torch.tensor([100.0])  # 如果没有误差，则返回最大PSNR值
    PIXEL_MAX = 255.0  # 像素最大值
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))  # 计算PSNR公式


def get_imagenet_list(path):
    """
    从指定路径读取ImageNet文件列表。

    参数:
    path - 文件路径
    """
    fns = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fns.append(row[0])  # 添加文件名
    return fns


def complex_sig(shape, device):
    """
    生成复数信号，其中实部和虚部均为正态分布。

    参数:
    shape - 信号的形状
    device - 设备（如CUDA）
    """
    sig_real = torch.randn(*shape)  # 实部
    sig_imag = torch.randn(*shape)  # 虚部
    return (torch.complex(sig_real, sig_imag) / np.sqrt(2)).to(device)  # 标准化后的复数信号


def pwr_normalize(sig):
    """
    对信号进行功率归一化。

    参数:
    sig - 输入信号张量
    """
    _, num_ele = sig.shape[0], torch.numel(sig[0])  # 计算每个信号的元素数量
    pwr_sig = torch.sum(torch.abs(sig) ** 2, dim=-1) / num_ele  # 计算每个信号的功率
    sig = sig / torch.sqrt(pwr_sig.unsqueeze(-1))  # 归一化信号
    return sig


def np_to_torch(img):
    """
    将Numpy数组转换为PyTorch张量。

    参数:
    img - 输入的Numpy数组
    """
    img = np.swapaxes(img, 0, 1)  # 转换维度
    img = np.swapaxes(img, 0, 2)
    return torch.from_numpy(img).float()  # 转换为PyTorch张量并设为浮点型


def to_chan_last(img):
    """
    将图像转换为通道在最后的格式。

    参数:
    img - 输入图像张量
    """
    img = img.transpose(1, 2)  # 转换通道位置
    img = img.transpose(2, 3)
    return img


def as_img_array(image):
    """
    将图像值限制在[0, 255]范围内，并四舍五入为整数。

    参数:
    image - 输入图像张量
    """
    image = image.clamp(0, 1) * 255.0  # 限制值范围并放大
    return torch.round(image)  # 四舍五入


def calc_psnr(predictions, targets):
    """
    计算预测结果与目标之间的PSNR。

    参数:
    predictions - 预测结果
    targets - 目标结果
    """
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])  # 转换为整数
        compare = as_img_array(pred)
        val = psnr(original, compare)  # 计算PSNR
        metric.append(val)
    return metric


def calc_msssim(predictions, targets):
    """
    计算预测和目标之间的多尺度结构相似性（MS-SSIM）。
    参数:
    predictions - 模型的预测值
    targets - 实际目标值
    """
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])  # 将目标值转换为图像数组
        compare = as_img_array(pred)  # 将预测值转换为图像数组
        val = ms_ssim(original, compare, data_range=255,  # 计算MS-SSIM
                      win_size=3, size_average=True)
        metric.append(val)  # 将结果加入列表
    return metric


def calc_ssim(predictions, targets):
    """
    计算预测和目标之间的结构相似性（SSIM）。
    参数:
    predictions - 模型的预测值
    targets - 实际目标值
    """
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])  # 将目标值转换为图像数组
        compare = as_img_array(pred)  # 将预测值转换为图像数组
        val = ssim(original, compare, data_range=255, size_average=True)  # 计算SSIM
        metric.append(val)  # 将结果加入列表
    return metric


import nltk  # 导入自然语言处理库
from pytorch_transformers import BertTokenizer  # 导入BERT分词器
from nltk.translate.bleu_score import sentence_bleu  # 用于BLEU分数的计算

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # 初始化BERT分词器


def tokens2sentence(outputs):
    """
    将模型输出的token转换为人类可读的句子。
    参数:
    outputs - 模型的输出tokens
    """
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = tokenizer.decode([int(token)])  # 解码token为对应的单词
            if word == '[PAD]':  # 如果遇到填充符号，则停止解析
                break
            sentence.append(word)  # 添加解码的单词到句子中
        sentences.append(sentence)  # 将句子加入列表
    return sentences


def computebleu(sentences, targets):
    """
    计算预测句子和目标句子之间的BLEU分数。
    参数:
    sentences - 模型预测的句子
    targets - 实际目标句子
    """
    score = 0
    assert (len(sentences) == len(targets))  # 检查句子数量是否一致

    def cut_token(sentence):
        """
        切分句子token。
        参数:
        sentence - 输入的句子
        """
        tmp = []
        for token in sentence:
            if token == '[UNK]':  # 对未知单词单独处理
                tmp.append(token)
            else:
                tmp += [word for word in token]  # 将句子分解为单词
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)  # 切分预测句子
        target = cut_token(target)  # 切分目标句子
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))  # 计算BLEU分数
    return score


def calc_metrics(y_true, y_pred, mode=None, to_print=True):
    """
    计算多种评估指标，包括准确率和回归相关性。
    参数:
    y_true - 实际值
    y_pred - 模型预测值
    mode - 模式，用于指定评估方式
    to_print - 是否打印评估结果
    """

    def multiclass_acc(preds, truths):
        """
        计算多分类准确率。
        参数:
        preds - 预测类别
        truths - 实际类别
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))  # 计算分类准确率

    test_preds = y_pred
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])  # 筛选非零项

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)  # 将预测值裁剪到[-3, 3]
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)  # 将实际值裁剪到[-3, 3]
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)  # 将预测值裁剪到[-2, 2]
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)  # 将实际值裁剪到[-2, 2]

    mae = np.mean(np.absolute(test_preds - test_truth))  # 计算平均绝对误差
    corr = np.corrcoef(test_preds, test_truth)[0][1]  # 计算预测和实际值的相关系数
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)  # 多分类准确率（裁剪到a7范围）
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)  # 多分类准确率（裁剪到a5范围）

    binary_truth = (test_truth[non_zeros] > 0)  # 将实际值转换为二分类
    binary_preds = (test_preds[non_zeros] > 0)  # 将预测值转换为二分类

    if to_print:
        print("Classification Report (pos/neg) :")  # 打印正负分类报告
        print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))  # 打印正负分类准确率

        binary_truth = (test_truth >= 0)  # 将实际值转换为非负/负分类
        binary_preds = (test_preds >= 0)  # 将预测值转换为非负/负分类

        if to_print:
            print("Classification Report (non-neg/neg) :")  # 打印非负/负分类报告
            print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))  # 打印非负/负分类准确率

        return accuracy_score(binary_truth, binary_preds)  # 返回准确率


class DiffPruningLoss(torch.nn.Module):
    """
    差分剪枝损失类，用于动态优化模型。
    """

    def __init__(self, base_criterion: torch.nn.Module, dynamic=True, ratio_weight=2.0, main_weight=1.):
        super().__init__()
        self.base_criterion = base_criterion  # 基础损失函数
        self.main_weight = main_weight  # 主损失权重
        self.surp_weight = 0.022  # 剩余项损失权重
        self.rho_weight = 0.01  # 稀疏正则化权重
        self.vq_weight = 2.0  # 向量量化损失权重
        self.print_mode = True  # 是否打印损失信息

        # 初始化统计变量
        self.count = 0
        self.main_loss_record = 0.
        self.surp_loss_record = 0.
        self.vq_loss_record = 0.
        self.keep_ratio_record = 0.

        self.dynamic = dynamic
        if self.dynamic:
            print('using dynamic loss')  # 如果启用动态损失，打印提示

    def forward(self, outputs, labels):
        """
        计算总损失，包括主损失、稀疏损失和量化损失。
        """
        pred, mask_m, rho, vq_loss = outputs  # 解压模型输出
        surp_loss = 0.0
        score = mask_m
        keep_ratio = score.mean(1)  # 计算保持的稀疏比例

        surp_loss = surp_loss + ((keep_ratio - rho) ** 2).mean()  # 稀疏正则化损失
        main_loss = self.base_criterion(pred, labels)  # 主损失（重构误差）

        # 总损失 = 主损失 + 剩余损失 + 稀疏损失 + 向量量化损失
        loss = self.main_weight * main_loss + \
               self.surp_weight * surp_loss + \
               self.rho_weight * rho + self.vq_weight * vq_loss

        if self.print_mode:  # 如果启用打印模式，更新记录并定期输出统计信息
            self.main_loss_record += main_loss.item()
            self.surp_loss_record += surp_loss.item()
            self.vq_loss_record += vq_loss.item()
            self.keep_ratio_record += keep_ratio.mean().item()
            self.count += 1
            if self.count == 100:  # 每100步打印统计
                print('loss info: main_loss=%.4f, surp_loss=%.4f, vq_loss=%.4f, keep ratio=%.4f'
                      % (self.main_loss_record / self.count,
                         self.surp_loss_record / self.count,
                         self.vq_loss_record / self.count,
                         self.keep_ratio_record / self.count))
                self.main_loss_record = 0
                self.surp_loss_record = 0
                self.vq_loss_record = 0
                self.keep_ratio_record = 0
                self.count = 0
        return loss  # 返回总损失
