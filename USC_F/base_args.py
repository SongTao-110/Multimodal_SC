import argparse

# 定义与各种数据集相关的常量
IMGC_NUMCLASS = 10  # CIFAR数据的类别数
IMGR_LENGTH = 48  # CIFAR数据的长度（例如：patch4/48, patch2/12）
TEXTC_NUMCLASS = 2  # SST数据的类别数
TEXTR_NUMCLASS = 34000  # 文本数据的词汇量大小
VQA_NUMCLASS = 3129  # VQA（视觉问答）数据的类别数
MSA_NUMCLASS = 1  # MSA（多模态情感分析）数据的类别数


def get_args():
    # 初始化参数解析器，描述训练脚本
    parser = argparse.ArgumentParser('U-DeepSC训练脚本', add_help=False)

    # 通用训练参数
    parser.add_argument('--batch_size', default=64, type=int, help='每个批次的样本数量（默认值：64）')
    parser.add_argument('--epochs', default=300, type=int, help='训练的轮次（默认值：300）')
    parser.add_argument('--save_freq', default=15, type=int, help='保存检查点的频率（以轮次为单位）（默认值：15）')
    parser.add_argument('--update_freq', default=1, type=int, help='参数更新的频率（默认值：1）')
    parser.add_argument('--chep', default='', type=str, help='检查点文件路径（默认值：空）')

    # 数据集参数
    parser.add_argument('--num_samples', default=50000, type=int, help='每轮的数据样本数量（默认值：50000）')
    parser.add_argument('--data_path', default='data/', type=str, help='数据集路径（默认值："data/"）')
    parser.add_argument('--input_size', default=32, type=int, help='数据集图像的输入大小（默认值：32）')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='路径丢弃率（默认值：0.1）')

    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='使用的优化器（默认值："adamw"）')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='优化器的Epsilon值（默认值：1e-8）')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='优化器的Beta值（默认值：None，使用优化器的默认值）')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='梯度裁剪的最大梯度范数（默认值：None，无梯度裁剪）')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD优化器的动量值（默认值：0.9）')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='优化器的权重衰减值（默认值：0.01）')
    parser.add_argument('--weight_decay_end', type=float, default=None,
                        help='使用余弦调度的最终权重衰减值（默认值：与--weight_decay相同）')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='学习率（默认值：5e-4）')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='学习率的下限（默认值：1e-5）')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, metavar='LR', help='学习率预热的初始值（默认值：1e-4）')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N', help='学习率预热的轮次（默认值：1）')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='学习率预热的步数，如果适用（默认值：-1）')

    # 模型参数
    parser.add_argument('--model', default='UDeepSC_model', type=str, metavar='MODEL',
                        help='要训练的模型名称（默认值："UDeepSC_model"）')
    parser.add_argument('--model_key', default='model|module', type=str, help='加载模型参数的键（默认值："model|module"）')
    parser.add_argument('--model_prefix', default='', type=str, help='模型参数的前缀（默认值：空）')

    # 输出和设备设置
    parser.add_argument('--output_dir', default='', help='保存输出的路径，留空表示不保存')
    parser.add_argument('--device', default='cuda', help='用于训练/测试的设备（默认值："cuda"）')
    parser.add_argument('--seed', default=100, type=int, help='随机种子，用于结果复现（默认值：100）')
    parser.add_argument('--resume', default='', help='恢复检查点的路径（默认值：空）')
    parser.add_argument('--auto_resume', action='store_true', help='自动从最新的检查点恢复')
    parser.set_defaults(auto_resume=False)

    # 训练管理
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='恢复训练的起始轮次（默认值：0）')
    parser.add_argument('--eval', action='store_true', help='仅进行评估')
    parser.add_argument('--num_workers', default=4, type=int, help='数据加载的工作线程数（默认值：4）')
    parser.add_argument('--pin_mem', action='store_true', help='在DataLoader中固定CPU内存，以更高效地传输到GPU')

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int, help='分布式进程的数量（默认值：1）')
    parser.add_argument('--local_rank', default=-1, type=int, help='分布式训练的本地进程号（默认值：-1）')
    parser.add_argument('--dist_on_itp', action='store_true', help='在初始化时设置分布式训练')
    parser.add_argument('--dist_url', default='env://', help='用于设置分布式训练的URL（默认值："env://"）')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='启用分布式评估（默认值：False）')

    # 数据增强参数
    parser.add_argument('--smoothing', type=float, default=0.1, help='标签平滑值（默认值：0.1）')
    parser.add_argument('--train_interpolation', type=str, default='bicubic', help='训练插值方法（默认值："bicubic"）')

    # 检查点保存
    parser.add_argument('--save_ckpt', action='store_true', help='启用检查点保存')
    parser.set_defaults(save_ckpt=True)

    # 任务特定的评估参数
    parser.add_argument('--ta_perform', default='', choices=['imgc', 'textc', 'vqa', 'imgr', 'textr', 'msa'],
                        type=str, help='指定评估数据集（选项：imgc, textc, vqa, imgr, textr, msa）')

    # 解析并返回参数
    return parser.parse_args()

# 该脚本定义了用于配置训练设置的命令行参数。
# 这些参数包括数据集、模型、优化器、分布式训练、数据增强等，
# 以方便在不同场景下训练U-DeepSC。
