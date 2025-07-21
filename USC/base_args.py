import argparse
IMGC_NUMCLASS = 10  # CIFAR Data  # CIFAR 数据集的类别数
IMGR_LENGTH = 48  # CIFAR Data patch4/48   patch2/12  # CIFAR 数据的图像分块大小，48 表示特征长度
TEXTC_NUMCLASS = 2  # SST Data  # SST 数据的分类类别数（如情感分类：正面和负面）
TEXTR_NUMCLASS = 34000  # Size of vacab  # 文本词汇表的大小
VQA_NUMCLASS = 3129  # number of VQA class  # VQA（视觉问答）任务的类别数
MSA_NUMCLASS = 1  # number of MSA class  # MSA（多模态语义对齐）任务的类别数


def get_args():
    parser = argparse.ArgumentParser('U-DeepSC training script', add_help=False)  # 定义一个参数解析器
    parser.add_argument('--batch_size', default=64, type=int)  # 批量大小，默认值为 64
    parser.add_argument('--epochs', default=300, type=int)  # 训练总轮数，默认值为 300
    parser.add_argument('--save_freq', default=15, type=int)  # 模型保存频率，每 15 个 epoch 保存一次
    parser.add_argument('--update_freq', default=1, type=int)  # 梯度更新频率，默认每次迭代都更新
    parser.add_argument('--chep', default='', type=str,  # 模型检查点的路径
                        help='chceckpint path')

    # Dataset parameters
    parser.add_argument('--num_samples', default=50000, type=int,  # 每个 epoch 的数据样本数
                        help='number of data samples per epoch')
    parser.add_argument('--data_path', default='data/', type=str,  # 数据集路径，默认为 "data/"
                        help='dataset path')
    parser.add_argument('--input_size', default=32, type=int,  # 输入图像的尺寸，默认 32x32
                        help='images input size for data')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',  # DropPath 的比例，用于正则化
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',  # 优化器，默认使用 AdamW
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',  # 优化器 epsilon 值
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',  # 优化器 beta 参数
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',  # 梯度裁剪阈值，默认不裁剪
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',  # 动量优化器的动量，默认 0.9
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,  # 权重衰减值，默认 0.01
                        help='weight decay (default: 0.02)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")  # 最终权重衰减值
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',  # 初始学习率，默认 5e-4
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',  # 最小学习率，默认 1e-5
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, metavar='LR',  # 学习率预热的起始值
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',  # 学习率预热所需的 epoch 数
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',  # 学习率预热步数
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--model', default='UDeepSC_Model', type=str, metavar='MODEL',  # 模型名称
                        help='Name of model to train')
    parser.add_argument('--model_key', default='model|module', type=str)  # 模型的主键，默认 "model|module"
    parser.add_argument('--model_prefix', default='', type=str)  # 模型前缀
    parser.add_argument('--output_dir', default='',  # 模型保存路径
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',  # 使用的设备，默认 GPU
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1000, type=int)  # 随机种子值，默认 1000
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # 从检查点恢复训练
    parser.add_argument('--auto_resume', action='store_true')  # 自动恢复训练开关
    parser.set_defaults(auto_resume=False)  # 默认关闭自动恢复训练

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',  # 起始训练轮数
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',  # 是否只进行评估
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)  # 数据加载的线程数，默认 4
    parser.add_argument('--pin_mem', action='store_true',  # 是否使用固定内存
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,  # 分布式训练的进程数
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)  # 本地训练的 GPU 序号
    parser.add_argument('--dist_on_itp', action='store_true')  # 是否在分布式训练中使用 IT 环境
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')  # 分布式 URL
    parser.add_argument('--dist_eval', action='store_true', default=False,  # 是否启用分布式评估
                        help='Enabling distributed evaluation')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1,  # 标签平滑系数
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',  # 训练数据插值方法
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--save_ckpt', action='store_true')  # 是否保存检查点
    parser.set_defaults(save_ckpt=True)  # 默认保存检查点

    parser.add_argument('--ta_perform', default='', choices=['imgc', 'textc', 'vqa', 'imgr', 'textr', 'msa'],
                        type=str, help='Eval Data')  # 指定评估任务（多模态任务）

    return parser.parse_args()  # 返回解析的参数
