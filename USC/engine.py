import torch
import math
import nltk
import torch.nn as nn
import sys

from utils import *
from tqdm import tqdm
from timm.data import Mixup
from einops import rearrange
from typing import Iterable, Optional
from vqa_utils import VQA_Tool, VQA_Eval
from timm.utils import accuracy, AverageMeter
from nltk.translate.bleu_score import sentence_bleu
####################################
# 获取DeepSpeed优化器中的loss scale
def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer  # 获取模型的优化器
    # 如果优化器中有"loss_scale"属性，则返回该值，否则返回"cur_scale"
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


@torch.no_grad()  # 禁用梯度计算以提高评估效率
def evaluate(ta_perform: str, net: torch.nn.Module, dataloader: Iterable,
             device: torch.device, criterion: torch.nn.Module, print_freq=10):
    net.eval()  # 将模型设置为评估模式，禁用dropout等正则化
    if ta_perform.startswith('imgc'):  # 如果任务是图像分类
        acc_meter = AverageMeter()  # 初始化用于记录准确率的计数器
        loss_meter = AverageMeter()  # 初始化用于记录损失的计数器
        with torch.no_grad():  # 禁用梯度计算以提高效率
            for batch_idx, (imgs, targets) in enumerate(dataloader):  # 遍历数据加载器
                imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到指定设备
                outputs = net(img=imgs, ta_perform=ta_perform)  # 使用模型进行前向传播
                loss = criterion(outputs, targets)  # 计算损失
                batch_size = targets.size(0)  # 获取当前批次大小
                idx, predicted = outputs.max(1)  # 获取每个样本的预测类别
                acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)  # 更新准确率
                loss_meter.update(loss.item(), 1)  # 更新损失
                if batch_idx % print_freq == 0:  # 每隔print_freq批次打印一次结果
                    print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100]' % (batch_idx * batch_size,
                                                                         len(dataloader.dataset),
                                                                         loss_meter.avg, acc_meter.avg * 100))
        test_stat = {'loss': loss_meter.avg,  # 返回损失和准确率的统计结果
                     'acc': acc_meter.avg}
        return test_stat

    elif ta_perform.startswith('imgr'):  # 如果任务是图像重建
        psnr_meter = AverageMeter()  # 初始化PSNR计数器
        loss_meter = AverageMeter()  # 初始化损失计数器
        psnr_list = []  # 用于存储PSNR值的列表
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (imgs, targets) in enumerate(dataloader):  # 遍历数据加载器
                imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到设备
                outputs = net(img=imgs, ta_perform=ta_perform)  # 模型预测输出
                # 调整输出的张量形状以符合重建任务的格式
                outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
                outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
                loss = criterion(outputs, targets)  # 计算损失
                batch_size = targets.shape[0]  # 获取当前批次大小
                # 分离预测值和真实值以计算PSNR
                predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
                targets = torch.chunk(imgs, chunks=imgs.size(0), dim=0)
                psnr_vals = calc_psnr(predictions, targets)  # 计算PSNR值
                psnr_list.extend(psnr_vals)  # 将PSNR值存入列表
                psnr_meter.update(torch.mean(torch.tensor(psnr_vals)).item(), n=batch_size)  # 更新PSNR计数器
                loss_meter.update(loss.item(), 1)  # 更新损失计数器
                if batch_idx % print_freq == 0:  # 每隔print_freq批次打印一次结果
                    print('Test %d/%d: [loss: %.4f] [psnr: %.3f dB]' % (batch_idx * batch_size,
                                                                        len(dataloader.dataset),
                                                                        loss_meter.avg, psnr_meter.avg))
        test_stat = {'loss': loss_meter.avg,  # 返回损失和PSNR统计结果
                     'psnr': psnr_meter.avg}
        return test_stat

    elif ta_perform.startswith('textc'):  # 如果任务是文本分类
        acc_meter = AverageMeter()  # 初始化准确率计数器
        loss_meter = AverageMeter()  # 初始化损失计数器
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (texts, targets) in enumerate(dataloader):  # 遍历数据加载器
                texts, targets = texts.to(device), targets.to(device)  # 将数据移动到设备
                outputs = net(text=texts, ta_perform=ta_perform)  # 模型预测输出
                loss = criterion(outputs, targets)  # 计算损失
                batch_size = targets.size(0)  # 获取当前批次大小
                idx, predicted = outputs.max(1)  # 获取预测的类别
                acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)  # 更新准确率计数器
                loss_meter.update(loss.item(), 1)  # 更新损失计数器
                if batch_idx % print_freq == 0:  # 每隔print_freq批次打印结果
                    print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100]' % (batch_idx * batch_size,
                                                                         len(dataloader.dataset),
                                                                         loss_meter.avg, acc_meter.avg * 100))
        test_stat = {'loss': loss_meter.avg,  # 返回损失和准确率统计结果
                     'acc': acc_meter.avg}
        return test_stat

    elif ta_perform.startswith('textr'):  # 如果任务是文本重建
        bleu_meter = AverageMeter()  # 初始化BLEU计数器
        loss_meter = AverageMeter()  # 初始化损失计数器
        result = []  # 初始化存储预测结果的列表
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (texts, targets) in enumerate(dataloader):  # 遍历数据加载器
                loss = 0  # 初始化损失为0
                texts, targets = texts.to(device), targets.to(device)  # 将数据移动到设备
                targets = targets[:, 1:]  # 忽略targets的第一个时间步
                outputs = net(text=texts, ta_perform=ta_perform)  # 模型预测输出
                batch_size = targets.size(0)  # 获取当前批次大小
                preds = torch.zeros_like(targets)  # 初始化预测张量
                for i in range(outputs.shape[1]):  # 遍历时间步
                    loss += criterion(outputs, targets[:, ])  # 逐时间步计算损失
                    preds[:, i] = outputs[:, i].max(-1)[-1]  # 获取预测的类别
                preds = tokens2sentence(preds)  # 将预测的token转换为句子
                targets = tokens2sentence(targets)  # 将目标token转换为句子
                for pred, target in zip(preds, targets):  # 遍历预测和目标
                    result.append((pred, target))  # 将结果追加到列表中
                bleu_meter.update(computebleu(preds, targets) / batch_size, n=batch_size)  # 更新BLEU计数器
                loss_meter.update(loss.item(), 1)  # 更新损失计数器
                if batch_idx % print_freq == 0:  # 每隔print_freq批次打印结果
                    print('Test %d/%d: [loss: %.4f] [bleu: %.3f]' % (batch_idx * batch_size,
                                                                     len(dataloader.dataset), loss_meter.avg,
                                                                     bleu_meter.avg))
        test_stat = {'loss': loss_meter.avg,  # 返回损失和BLEU统计结果
                     'bleu': bleu_meter.avg}
        return test_stat


@torch.no_grad()  # 禁用梯度计算
def evaluate_vqa(ta_perform: str, net: torch.nn.Module, dataloader: Iterable,
                 device: torch.device, criterion: torch.nn.Module, print_freq=500):
    net.eval()  # 将模型设置为评估模式
    dataset = dataloader.dataset  # 获取数据集
    qid_list = [ques['question_id'] for ques in dataset.ques_list]  # 提取问题ID列表
    ans_ix_list = []  # 初始化答案索引列表
    i = 0  # 初始化计数器
    for batch_idx, (imgs, texts, targets) in enumerate(dataloader):  # 遍历数据加载器
        imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)  # 数据转移到设备
        batch_size = imgs.shape[0]  # 获取当前批次大小
        i += batch_size  # 累计批次大小
        outputs = net(img=imgs, text=texts, ta_perform=ta_perform)  # 模型预测输出
        pred_np = outputs.cpu().data.numpy()  # 转为numpy数组
        pred_argmax = np.argmax(pred_np, axis=1)  # 获取最大值索引
        if pred_argmax.shape[0] != dataset.configs.eval_batch_size:  # 如果批次大小不匹配
            pred_argmax = np.pad(  # 用-1填充到目标批次大小
                pred_argmax, (0, dataset.configs.eval_batch_size - pred_argmax.shape[0]),
                mode='constant', constant_values=-1)
        ans_ix_list.append(pred_argmax)  # 添加预测结果索引
        if batch_idx % print_freq == 0:  # 每隔print_freq批次打印结果
            print('Test %d/%d:' % (batch_idx * batch_size,
                                   len(dataloader.dataset)))

    ans_ix_list = np.array(ans_ix_list).reshape(-1)  # 转换为一维数组
    result = [{  # 生成结果字典列表
        'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # 从索引到答案的映射
        'question_id': int(qid_list[qix])} for qix in range(qid_list.__len__())]

    result_eval_file = 'vqaeval_result/result_run_' + dataset.configs.version + '.json'  # 定义结果文件路径
    print('Save the result to file: {}'.format(result_eval_file))  # 打印保存路径
    json.dump(result, open(result_eval_file, 'w'))  # 将结果保存为JSON文件

    # 创建VQA对象和评估对象
    ques_file_path = dataset.configs.question_path['val']  # 验证集问题路径
    ans_file_path = dataset.configs.answer_path['val']  # 验证集答案路径
    vqa = VQA_Tool(ans_file_path, ques_file_path)  # 加载VQA工具
    vqaRes = vqa.loadRes(result_eval_file, ques_file_path)  # 加载预测结果
    vqaEval = VQA_Eval(vqa, vqaRes, n=2)  # 初始化VQA评估器
    vqaEval.evaluate()  # 评估结果

    return vqaEval.accuracy  # 返回评估的准确率


# 定义单任务的训练函数，处理不同任务类型的输入
def train_class_batch_uni(ta_perform, model, sel_batch, targets, criterion):
    loss = 0  # 初始化损失值
    imgs, texts, speechs = sel_batch  # 从选择的批次中获取图像、文本和语音数据
    # 图像分类任务
    if ta_perform.startswith('imgc'):
        outputs = model(img=imgs, ta_perform=ta_perform)  # 模型前向传播
        loss = criterion[ta_perform](outputs, targets) * 1  # 计算损失
    # 图像重建任务
    elif ta_perform.startswith('imgr'):
        outputs = model(img=imgs, ta_perform=ta_perform)  # 模型前向传播
        # 重建任务需要调整目标张量的形状
        targets = rearrange(targets, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
        loss = criterion[ta_perform](outputs, targets) * 30  # 计算损失
    # 文本分类任务
    elif ta_perform.startswith('textc'):
        outputs = model(text=texts, ta_perform=ta_perform)  # 模型前向传播
        loss = criterion[ta_perform](outputs, targets) * 0.6  # 计算损失
    # 文本重建任务
    elif ta_perform.startswith('textr'):
        outputs = model(text=texts, ta_perform=ta_perform) * 1  # 模型前向传播
        targets = targets[:, 1:]  # 忽略targets的第一个时间步
        for i in range(outputs.shape[1]):  # 遍历时间步
            loss += criterion[ta_perform](outputs[:, i], targets[:, i]) * 5  # 累积损失
    # VQA任务
    elif ta_perform.startswith('vqa'):
        outputs = model(img=imgs, text=texts, ta_perform=ta_perform)  # 模型前向传播
        loss = criterion[ta_perform](outputs, targets) * 3  # 计算损失
    # 多模态语义对齐任务
    elif ta_perform.startswith('msa'):
        outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform)  # 模型前向传播
        loss = criterion[ta_perform](outputs, targets) * 8  # 计算损失
    return loss, outputs  # 返回损失和模型输出


# 初始化评估指标
def meter(ta_sel):
    acc_meter_dict = {}  # 准确率评估指标字典
    acc_meter_dict['imgc'] = AverageMeter()  # 图像分类准确率
    acc_meter_dict['textc'] = AverageMeter()  # 文本分类准确率
    acc_meter_dict['vqa'] = AverageMeter()  # VQA准确率

    loss_meter_dict = {}  # 损失评估指标字典
    for ta in ta_sel:  # 为每个任务初始化损失计数器
        loss_meter_dict[ta] = AverageMeter()
    psnr_meter = AverageMeter()  # PSNR指标（用于图像重建）
    return acc_meter_dict, loss_meter_dict, psnr_meter


# 训练一个epoch
def train_epoch_uni(model: torch.nn.Module, criterion: dict,
                    data_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, ta_sel, max_norm: float = 0,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    update_freq=None, print_freq=10):
    model.train(True)  # 设置模型为训练模式
    acc_meter_dict, loss_meter_dict, psnr_meter = meter(ta_sel)  # 初始化评估指标

    # 如果没有使用loss scaler，初始化模型的梯度
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    num_samples = 5000  # 总样本数
    data_iter_step = 0  # 当前数据迭代步
    num_tasks = len(data_dict)  # 任务数量
    data_tuple = [data_loader for data_loader in data_dict.values()]  # 获取任务的数据加载器

    # 遍历每批数据
    for data_batch in zip(data_tuple[0], data_tuple[1]):
        step = data_iter_step // update_freq  # 计算当前步
        it = start_steps + step  # 当前全局步数
        # 根据学习率调度表更新学习率和权重衰减
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # 初始化任务的输入数据
        imgs, texts, speechs, targets = None, None, None, None
        ta_index = np.random.randint(num_tasks)  # 随机选择任务索引
        ta = ta_sel[ta_index]  # 当前任务类型
        data = data_batch[ta_index]  # 获取对应任务的数据
        # 根据任务类型处理输入
        if ta.startswith('img'):
            imgs = data[0].to(device, non_blocking=True)
            targets = data[1].to(device, non_blocking=True)
        elif ta.startswith('text'):
            texts = data[0].to(device, non_blocking=True)
            targets = data[1].to(device, non_blocking=True)
        elif ta.startswith('vqa'):
            imgs = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            targets = data[2].to(device, non_blocking=True)
        elif ta.startswith('msa'):
            imgs = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            speechs = data[2].to(device, non_blocking=True)
            targets = data[3].to(device, non_blocking=True)
        else:
            raise NotImplementedError()  # 如果任务未实现则抛出错误

        batch_size = targets.shape[0]  # 获取批次大小
        sel_batch = [imgs, texts, speechs]  # 选中的批次数据
        # 训练当前批次
        loss, outputs = train_class_batch_uni(
            ta, model, sel_batch, targets, criterion)
        loss_value = loss.item()  # 获取损失值
        # 如果损失值无效，则停止训练
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        # 根据是否使用loss scaler更新模型
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)  # 反向传播
            model.step()  # 参数更新
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()  # 同步GPU
        data_iter_step += 1  # 更新数据迭代步数

        # 更新学习率最小值和最大值
        min_lr, max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr, max_lr = min(min_lr, group["lr"]), max(max_lr, group["lr"])

        # 根据任务类型更新指标
        if ta.endswith('c'):  # 分类任务
            acc_meter_dict[ta].update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('imgr'):  # 图像重建任务
            outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
            outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
            tr_imgs = torch.tensor((imgs * 255).detach().cpu().numpy().astype(int).clip(0, 255)).float()
            re_imgs = torch.tensor((outputs * 255).detach().cpu().numpy().astype(int).clip(0, 255)).float()
            mse_cal = nn.MSELoss()  # 均方误差损失
            psnr_meter.update(10 * math.log10(255.0 ** 2 / (mse_cal(tr_imgs, re_imgs))), n=1)
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('textr'):  # 文本重建任务
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('vqa'):  # VQA任务
            acc_meter_dict[ta].update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(),
                                      n=batch_size)
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('msa'):  # 多模态任务
            loss_meter_dict[ta].update(loss_value, 1)

        # 定期打印训练状态
        if data_iter_step % print_freq == 0:
            if ta.startswith('imgc'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]'
                      % (epoch, ta, batch_size * data_iter_step, num_samples,
                         loss_meter_dict[ta].avg, acc_meter_dict[ta].avg * 100, max_lr))
            elif ta.startswith('imgr'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [psnr: %.3f dB] [lr: %.3e]'
                      % (epoch, ta, batch_size * data_iter_step, num_samples,
                         loss_meter_dict[ta].avg, psnr_meter.avg, max_lr))
            elif ta.startswith('textc'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]'
                      % (epoch, ta, batch_size * data_iter_step, num_samples,
                         loss_meter_dict[ta].avg, acc_meter_dict[ta].avg * 100, max_lr))
            elif ta.startswith('textr'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [lr: %.3e]'
                      % (epoch, ta, batch_size * data_iter_step, num_samples,
                         loss_meter_dict[ta].avg, max_lr))
            elif ta.startswith('vqa'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]'
                      % (epoch, ta, batch_size * data_iter_step, num_samples,
                         loss_meter_dict[ta].avg, acc_meter_dict[ta].avg * 100, max_lr))
            elif ta.startswith('msa'):
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [lr: %.3e]'
                      % (epoch, ta, batch_size * data_iter_step, num_samples,
                         loss_meter_dict[ta].avg, max_lr))

    train_stat = None  # 训练统计信息
    return train_stat


# 定义VQA（Visual Question Answering）任务的训练函数
def train_class_batch_vqa(ta_perform, model, imgs, texts, targets, criterion):
    if ta_perform.startswith('vqa'):  # 检查任务是否为VQA任务
        outputs = model(img=imgs, text=texts, ta_perform=ta_perform)  # 使用模型前向传播
        loss = criterion(outputs, targets)  # 计算损失
    return loss, outputs  # 返回损失和模型输出


# 训练一个epoch的VQA任务
def train_epoch_vqa(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, ta_perform, max_norm: float = 0,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    update_freq=None, print_freq=500):
    model.train(True)  # 设置模型为训练模式
    acc_meter = AverageMeter()  # 准确率计数器
    loss_meter = AverageMeter()  # 损失计数器

    # 初始化优化器或模型的梯度
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # 遍历数据加载器
    for data_iter_step, (imgs, texts, targets) in enumerate(data_loader):
        step = data_iter_step // update_freq  # 当前迭代步数的更新步
        it = start_steps + step  # 全局迭代步数
        # 动态调整学习率和权重衰减
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:  # 更新学习率
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:  # 更新权重衰减
                    param_group["weight_decay"] = wd_schedule_values[it]

        # 将数据移动到指定设备
        imgs = imgs.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = imgs.size(0)  # 获取当前批次大小

        # 训练当前批次
        loss, outputs = train_class_batch_vqa(
            ta_perform, model, imgs, texts, targets, criterion)
        loss_value = loss.item()  # 获取损失值

        # 检查损失是否为有限值，避免训练失败
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 更新模型参数
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)  # 反向传播
            model.step()  # 参数更新
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()  # 同步设备以确保更新完成

        # 更新学习率范围
        min_lr, max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr, max_lr = min(min_lr, group["lr"]), max(max_lr, group["lr"])

        # 计算任务的评估指标
        if ta_perform.startswith('vqa'):
            acc_meter.update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
            loss_meter.update(loss_value, 1)

        # 打印训练状态
        if data_iter_step % print_freq == 0:
            if ta_perform.startswith('vqa'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]'
                      % (epoch, batch_size * data_iter_step, len(data_loader.dataset),
                         loss_meter.avg, acc_meter.avg * 100, max_lr))

    # 返回训练统计结果
    train_stat = {'loss': loss_meter.avg,
                  'acc': acc_meter.avg}
    return train_stat


@torch.no_grad()
# 定义多模态语义对齐任务（MSA）的评估函数
def evaluate_msa(ta_perform: str, net: torch.nn.Module, dataloader: Iterable,
                 device: torch.device, criterion: torch.nn.Module, print_freq=10):
    net.eval()  # 设置模型为评估模式
    loss_meter = AverageMeter()  # 损失计数器
    y_true, y_pred = [], []  # 初始化真实值和预测值
    with torch.no_grad():  # 禁用梯度计算以提高效率
        # 遍历数据加载器
        for batch_idx, (imgs, texts, speechs, targets) in enumerate(dataloader):
            # 将数据移动到指定设备
            imgs, texts, speechs, targets = imgs.to(device), texts.to(device), speechs.to(device), targets.to(
                device)
            outputs = net(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform)  # 模型预测
            loss = criterion(outputs, targets)  # 计算损失
            y_pred.append(outputs.detach().cpu().numpy())  # 保存预测值
            y_true.append(targets.detach().cpu().numpy())  # 保存真实值
            loss_meter.update(loss.item(), 1)  # 更新损失计数器
    y_true = np.concatenate(y_true, axis=0).squeeze()  # 合并真实值数组
    y_pred = np.concatenate(y_pred, axis=0).squeeze()  # 合并预测值数组
    acc = calc_metrics(y_true, y_pred)  # 计算评价指标
    test_stat = {'loss': loss_meter.avg,  # 保存测试统计结果
                 'acc': acc}
    return test_stat  # 返回测试统计结果


# 定义多模态语义对齐（MSA）任务的训练函数
def train_class_batch_msa(ta_perform, model, imgs, texts, speechs, targets, criterion):
    if ta_perform.startswith('msa'):  # 判断任务是否为MSA任务
        # 前向传播：模型处理图像、文本和语音输入
        outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform)
        loss = criterion(outputs, targets)  # 计算损失
    return loss, outputs  # 返回损失和模型输出


# 定义一个epoch的MSA任务训练过程
def train_epoch_msa(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, ta_perform, max_norm: float = 0,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    update_freq=None, print_freq=5):
    model.train(True)  # 设置模型为训练模式
    acc_meter = AverageMeter()  # 初始化准确率计数器
    loss_meter = AverageMeter()  # 初始化损失计数器

    # 初始化优化器或模型的梯度
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # 遍历数据加载器
    for data_iter_step, (imgs, texts, speechs, targets) in enumerate(data_loader):
        step = data_iter_step // update_freq  # 当前步数的更新频率
        it = start_steps + step  # 当前全局步数

        # 动态调整学习率和权重衰减
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:  # 更新学习率
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:  # 更新权重衰减
                    param_group["weight_decay"] = wd_schedule_values[it]

        # 将数据移动到指定设备
        imgs = imgs.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)
        speechs = speechs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = imgs.size(0)  # 获取当前批次大小

        # 自动混合精度（AMP）训练，减少显存占用
        with torch.cuda.amp.autocast():
            loss, outputs = train_class_batch_msa(
                ta_perform, model, imgs, texts, speechs, targets, criterion)  # 训练当前批次
        loss_value = loss.item()  # 获取损失值

        # 检查损失是否为有限值，避免训练失败
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 更新模型参数
        if loss_scaler is None:
            loss /= update_freq  # 将损失平均到更新频率
            model.backward(loss)  # 反向传播
            model.step()  # 参数更新
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq  # 平均损失
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:  # 每隔更新频率清零梯度
                optimizer.zero_grad()

        torch.cuda.synchronize()  # 同步设备，确保所有操作完成

        # 更新学习率范围
        min_lr, max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr, max_lr = min(min_lr, group["lr"]), max(max_lr, group["lr"])

        # 记录损失
        if ta_perform.startswith('msa'):
            loss_meter.update(loss_value, 1)  # 更新损失计数器

        # 定期打印训练状态
        if data_iter_step % print_freq == 0:
            if ta_perform.startswith('msa'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [lr: %.3e]'
                      % (epoch, batch_size * data_iter_step, len(data_loader.dataset),
                         loss_meter.avg, max_lr))

    # 返回训练统计结果
    train_stat = {'loss': loss_meter.avg,  # 平均损失
                  'acc': acc_meter.avg}  # 平均准确率

    return train_stat  # 返回训练结果
