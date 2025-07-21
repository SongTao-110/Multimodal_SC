import datetime
import numpy as np
import time
import torch
import utils
import model
import torch.backends.cudnn as cudnn

from engine import *
from pathlib import Path
from base_args import get_args
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import get_model, sel_criterion_train, sel_criterion_test, load_checkpoint
from datasets import build_dataset_train, build_dataset_test, BatchSchedulerSampler, collate_fn, \
    build_dataloader


############################################################
def seed_initial(seed=0):
    """
    初始化随机种子，确保结果的可重复性。

    参数:
    ----------
    seed: int, 随机种子值。
    """
    seed += utils.get_rank()  # 考虑分布式训练时的进程编号
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 自动调优
    torch.backends.cudnn.deterministic = True  # 确保每次运行结果一致


def main(args):
    """
    主函数，用于初始化模型、数据集、优化器，并执行训练和评估。

    参数:
    ----------
    args: argparse.Namespace, 从命令行或配置文件中获取的参数。
    """
    ### 配置分布式训练
    utils.init_distributed_mode(args)
    device = torch.device(args.device)  # 指定设备（CPU/GPU）
    seed_initial(seed=args.seed)  # 设置随机种子

    ####################################### 获取模型
    model = get_model(args)  # 加载模型
    if args.resume:  # 如果指定了断点加载
        print(args.resume)
        checkpoint_model = load_checkpoint(model, args)  # 加载断点
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)  # 恢复模型状态

    model.to(device)  # 将模型转移到指定设备
    model_without_ddp = model  # 非分布式版本
    if args.distributed:  # 如果启用了分布式训练
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    print("------------------------------------------------------")

    ############## 加载训练数据集和数据加载器
    ta_sel = ['imgc', 'textc']  # 任务选择
    # ta_sel = ['imgc', 'vqa', 'textc']  # 任务选择
    trainset_group = build_dataset_train(is_train=True, ta_sel=ta_sel, args=args)  # 构建训练集
    trainloader_group = build_dataloader(ta_sel, trainset_group, args=args)  # 构建训练数据加载器

    ############################################## 加载测试数据集和数据加载器
    valset = None
    if args.ta_perform:
        valset = build_dataset_test(is_train=False, args=args)  # 构建测试集
        sampler_val = torch.utils.data.SequentialSampler(valset)  # 顺序采样器
    else:
        valset = None

    if valset is not None:
        Collate_fn = collate_fn if args.ta_perform.startswith('msa') else None
        dataloader_val = torch.utils.data.DataLoader(
            valset, sampler=sampler_val, batch_size=int(1.0 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False, collate_fn=Collate_fn
        )
    else:
        dataloader_val = None

    ############################# 初始化优化器和训练相关设置
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()  # 总批次大小
    num_training_steps_per_epoch = args.num_samples // total_batch_size  # 每轮的训练步数

    parameter_group = get_param_groups(model_without_ddp, args.weight_decay)  # 参数分组
    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)  # 优化器参数
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    optimizer = torch.optim.AdamW(parameter_group, **opt_args)  # AdamW 优化器
    loss_scaler = NativeScaler()  # 混合精度训练的损失缩放器

    print("使用逐步学习率和权重衰减调度器！")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )  # 余弦学习率调度器
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch
    )  # 余弦权重衰减调度器
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    ###################################################### 获取损失函数
    criterion_train = sel_criterion_train(args, ta_sel, device)  # 选择训练损失函数
    criterion_test = sel_criterion_test(args, device)  # 选择测试损失函数
    # 添加差分剪枝损失
    criterion_train['vqa'] = DiffPruningLoss(criterion_train['vqa'])
    criterion_train['imgc'] = DiffPruningLoss(criterion_train['imgc'])
    criterion_train['textc'] = DiffPruningLoss(criterion_train['textc'])

    ################################## 如果是评估模式，直接评估
    if args.eval:
        if args.ta_perform.startswith('img') or args.ta_perform.startswith('text'):
            test_stats = evaluate(
                ta_perform=args.ta_perform, net=model, dataloader=dataloader_val,
                device=device, criterion=criterion_test
            )
            if args.ta_perform.startswith('imgc') or args.ta_perform.startswith('textc'):
                print(f"测试集上的准确率: {test_stats['acc'] * 100:.3f}")
            elif args.ta_perform.startswith('imgr'):
                print(f"测试集上的平均 PSNR: {test_stats['psnr']:.3f} dB")
            elif args.ta_perform.startswith('textr'):
                print(f"测试集上的平均 BLEU: {test_stats['bleu']:.3f}")
        exit(0)

    ################################## 开始训练
    print(f"开始训练，共 {args.epochs} 轮")
    start_time = time.time()  # 记录训练开始时间
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # 设置分布式训练的采样器
            for trainloader in trainloader_group.values():
                trainloader.sampler.set_epoch(epoch)

        # 调整学习率
        adjust_learning_rate(
            optimizer.param_groups, args.lr, args.min_lr, epoch, args.epochs,
            warmup_predictor=False, warming_up_step=0, base_multi=0.1
        )

        # 单轮训练
        train_stats = train_epoch_uni(
            model, criterion_train, trainloader_group, optimizer, device, epoch, loss_scaler,
            ta_sel, args.clip_grad, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq
        )

        # 保存检查点
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=None
                )

        # 评估模型性能
        if dataloader_val is not None:
            test_stats = evaluate(
                ta_perform=args.ta_perform, net=model, dataloader=dataloader_val,
                device=device, criterion=criterion_test
            )
            print(f"测试集上的准确率: {test_stats['acc'] * 100:.3f}")

    total_time = time.time() - start_time
    print('总训练时间: {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)  # 创建输出目录
    main(opts)
