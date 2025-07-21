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
from datasets import build_dataset_train, build_dataset_test, BatchSchedulerSampler, collate_fn, build_dataloader


def seed_initial(seed=0):
    """
    初始化随机种子以确保实验结果的可重复性。
    """
    seed += utils.get_rank()  # 获取当前进程的rank（在分布式训练中每个进程的唯一标识），并加到种子中
    np.random.seed(seed)  # 使用该种子初始化numpy随机数生成器
    torch.manual_seed(seed)  # 使用该种子初始化PyTorch随机数生成器
    torch.cuda.manual_seed(seed)  # 使用该种子初始化CUDA随机数生成器（当前设备）
    torch.cuda.manual_seed_all(seed)  # 使用该种子初始化所有GPU上的CUDA随机数生成器
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN的自动优化功能
    torch.backends.cudnn.deterministic = True  # 强制CuDNN以确定性方式进行计算，保证结果可重复


def main(args):
    """
    主程序入口：包括分布式训练的初始化、模型加载、数据加载、训练和评估逻辑。
    """
    utils.init_distributed_mode(args)  # 初始化分布式训练模式
    device = torch.device(args.device)  # 根据传入的设备参数设置计算设备（CPU或GPU）
    seed_initial(seed=args.seed)  # 使用指定的随机种子初始化随机数生成器

    ### 加载模型
    model = get_model(args)  # 调用自定义函数，根据参数加载模型
    if args.resume:  # 如果提供了恢复训练的路径
        print(args.resume)  # 输出恢复路径
        checkpoint_model = load_checkpoint(model, args)  # 加载指定路径下的模型检查点
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)  # 加载权重到当前模型中

    model.to(device)  # 将模型加载到指定设备
    model_without_ddp = model  # 保存一个不包含分布式包装的模型对象
    if args.distributed:  # 如果是分布式训练模式
        model = torch.nn.parallel.DistributedDataParallel(  # 使用分布式数据并行包装模型
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module  # 提取出原始模型

    print("------------------------------------------------------")  # 分隔符，便于日志观察

    ### 加载数据集和数据加载器
    ta_sel = ['msa', 'textr']  # 选择需要训练的任务类型
    trainset_group = build_dataset_train(is_train=True, ta_sel=ta_sel, args=args)  # 构建训练数据集
    trainloader_group = build_dataloader(ta_sel, trainset_group, args=args)  # 构建训练数据加载器

    ### 加载验证数据加载器
    valset = None  # 初始化验证集变量
    if args.ta_perform:  # 如果指定了验证任务类型
        valset = build_dataset_test(is_train=False, args=args)  # 构建测试数据集
        sampler_val = torch.utils.data.SequentialSampler(valset)  # 构建顺序采样器
    else:
        valset = None  # 如果未指定任务类型，则验证集设置为None

    if valset is not None:  # 如果存在验证集
        Collate_fn = collate_fn if args.ta_perform.startswith('msa') else None  # 根据任务类型选择是否使用Collate函数
        dataloader_val = torch.utils.data.DataLoader(  # 构建验证数据加载器
            valset, sampler=sampler_val, batch_size=int(1.0 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False, collate_fn=Collate_fn)
    else:
        dataloader_val = None  # 如果验证集不存在，验证加载器设置为None

    ### 设置优化器和训练相关参数
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()  # 计算总批量大小
    num_training_steps_per_epoch = args.num_samples // total_batch_size  # 每轮训练的总步数

    optimizer = create_optimizer(args, model)  # 根据参数创建优化器
    loss_scaler = NativeScaler()  # 创建混合精度训练的损失缩放器

    print("使用逐步学习率和权重衰减调度器!")  # 提示使用调度器
    lr_schedule_values = utils.cosine_scheduler(  # 生成学习率调度计划
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:  # 如果未指定最终权重衰减值
        args.weight_decay_end = args.weight_decay  # 默认使用初始权重衰减值
    wd_schedule_values = utils.cosine_scheduler(  # 生成权重衰减调度计划
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("最大权重衰减=%.7f, 最小权重衰减=%.7f" % (max(wd_schedule_values), min(wd_schedule_values)))  # 打印调度范围

    ### 获取损失函数
    criterion_train = sel_criterion_train(args, ta_sel, device)  # 根据任务选择训练损失函数
    criterion_test = sel_criterion_test(args, device)  # 根据任务选择验证损失函数

    ### 如果是评估模式
    if args.eval:
        if args.ta_perform.startswith('img') or args.ta_perform.startswith('text'):  # 图像或文本任务
            test_stats = evaluate(ta_perform=args.ta_perform,  # 调用评估函数
                                  net=model, dataloader=dataloader_val,
                                  device=device, criterion=criterion_test)
            if args.ta_perform.startswith('imgc') or args.ta_perform.startswith('textc'):
                print(f"测试集准确率: {test_stats['acc'] * 100:.3f}%")  # 打印分类任务的准确率
            elif args.ta_perform.startswith('imgr'):
                print(f"平均PSNR: {test_stats['psnr']:.3f} dB")  # 打印图像重建任务的PSNR
            elif args.ta_perform.startswith('textr'):
                print(f"平均BLEU得分: {test_stats['bleu']:.3f}")  # 打印文本生成任务的BLEU分数
        elif args.ta_perform.startswith('msa'):  # 多模态任务
            test_stats = evaluate_msa(ta_perform=args.ta_perform,
                                      net=model, dataloader=dataloader_val,
                                      device=device, criterion=criterion_test)
            print(f"测试集准确率: {test_stats['acc'] * 100:.3f}%")  # 打印多模态任务的准确率
        elif args.ta_perform.startswith('vqa'):  # VQA任务
            test_stats = evaluate_vqa(ta_perform=args.ta_perform,
                                      net=model, dataloader=dataloader_val,
                                      device=device, criterion=criterion_test)
            print("总体准确率: %.02f" % (test_stats['overall']))  # 打印总体准确率
            print("每个答案类型的准确率:")  # 打印每种答案类型的准确率
            for ansType in test_stats['perAnswerType']:
                print("%s : %.02f" % (ansType, test_stats['perAnswerType'][ansType]))
        exit(0)  # 退出程序

    ### 开始训练
    print(f"开始训练，共 {args.epochs} 轮")
    max_accuracy = 0.0  # 初始化最高准确率
    start_time = time.time()  # 记录起始时间
    for epoch in range(args.start_epoch, args.epochs):  # 遍历每个epoch
        if args.distributed:  # 如果是分布式训练
            for trainloader in trainloader_group.values():
                trainloader.sampler.set_epoch(epoch)  # 设置采样器的epoch

        train_stats = train_epoch_uni(  # 调用统一训练函数
            model, criterion_train, trainloader_group, optimizer, device, epoch, loss_scaler,
            ta_sel, args.clip_grad, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq)

        inter_time = time.time() - start_time  # 计算中间时间
        inter_time_str = str(datetime.timedelta(seconds=int(inter_time)))  # 转换为可读时间格式
        print('当前训练时间 {}'.format(inter_time_str))  # 打印当前训练时间

        if args.output_dir and args.save_ckpt:  # 如果需要保存检查点
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(  # 调用保存模型函数
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=None)

        if dataloader_val is not None:  # 如果有验证集
            if args.ta_perform.startswith('img') or args.ta_perform.startswith('text'):
                test_stats = evaluate(ta_perform=args.ta_perform,  # 评估模型
                                      net=model, dataloader=dataloader_val,
                                      device=device, criterion=criterion_test)
            elif args.ta_perform.startswith('vqa'):
                test_stats = evaluate_vqa(ta_perform=args.ta_perform,  # 评估VQA任务
                                          net=model, dataloader=dataloader_val,
                                          device=device, criterion=criterion_test)
            else:
                test_stats = evaluate_msa(ta_perform=args.ta_perform,  # 评估多模态任务
                                          net=model, dataloader=dataloader_val,
                                          device=device, criterion=criterion_test)

    total_time = time.time() - start_time  # 计算总时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # 转换为可读时间格式
    print('总训练时间 {}'.format(total_time_str))  # 打印总训练时间


if __name__ == '__main__':
    opts = get_args()  # 获取命令行参数
    if opts.output_dir:  # 如果指定了输出目录
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
    main(opts)  # 调用主函数
