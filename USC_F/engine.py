import torch
import math
import nltk
import torch.nn as nn
import sys

# 导入必要的工具和模块
from utils import *  # 自定义工具函数
from tqdm import tqdm  # 用于显示进度条
from timm.data import Mixup  # 数据混合工具，用于数据增强
from einops import rearrange  # 用于张量维度重排
from typing import Iterable, Optional  # 用于类型提示
from vqa_utils import VQA_Tool, VQA_Eval  # 用于VQA任务的工具和评估
from timm.utils import accuracy, AverageMeter  # 用于计算精度和维护平均值
from nltk.translate.bleu_score import sentence_bleu  # 用于BLEU分数计算（多语言翻译质量评估）


####################################

# 获取DeepSpeed优化器的损失缩放因子
def get_loss_scale_for_deepspeed(model):
    """
    返回DeepSpeed优化器的损失缩放因子。如果优化器包含`loss_scale`属性，则返回该值。
    否则，返回优化器的当前缩放值（cur_scale）。
    """
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


# 定义全局变量，用于记录图像和文本的统计数据
mu_record_img, mu_record_text = [], []


# 评估函数
@torch.no_grad()  # 禁用梯度计算（提高评估速度，减少显存占用）
def evaluate(ta_perform: str, net: torch.nn.Module, dataloader: Iterable,
             device: torch.device, criterion: torch.nn.Module, print_freq=10, test_snr=10):
    """
    评估函数，根据任务类型（ta_perform），对图像分类、图像重建和文本分类任务进行评估。

    参数：
    - ta_perform: str，任务类型标识（如'imgc'表示图像分类）。
    - net: torch.nn.Module，待评估的神经网络模型。
    - dataloader: Iterable，数据加载器。
    - device: torch.device，设备类型（如'cuda'或'cpu'）。
    - criterion: torch.nn.Module，损失函数。
    - print_freq: int，打印日志的频率。
    - test_snr: int，信噪比参数，用于设置噪声强度。

    返回：
    - test_stat: dict，评估统计数据，包括损失值和精度/PSNR。
    """
    net.eval()  # 设置模型为评估模式（冻结批归一化和Dropout层）

    # 测试不同信噪比下模型对噪声的鲁棒性
    for snr in range(-6, 10, 1):  # 信噪比范围从-6到9（步长为1）
        noise_std = torch.FloatTensor([1]) * 10 ** (-snr / 20)  # 计算噪声标准差
        # 打印模型在不同信噪比下的响应值
        print(net.img_encoder.RHO_Dict['imgc']((noise_std).cuda()))
        print(net.text_encoder.RHO_Dict['textc'](noise_std.cuda()))

    # 如果任务是图像分类（imgc）
    if ta_perform.startswith('imgc'):
        acc_meter = AverageMeter()  # 用于记录准确率
        loss_meter = AverageMeter()  # 用于记录损失值
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到指定设备
                outputs_dict = net(img=imgs, ta_perform=ta_perform)  # 前向传播得到预测结果

                outputs = outputs_dict['outputs']  # 模型输出
                rho = outputs_dict['rho']['img']  # 获取图像的稀疏性因子
                loss = criterion(outputs, targets)  # 计算损失
                batch_size = targets.size(0)  # 获取当前批次大小
                idx, predicted = outputs.max(1)  # 获取预测类别
                acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)  # 更新准确率
                loss_meter.update(loss.item(), 1)  # 更新损失
                if batch_idx % print_freq == 0:  # 每隔一定频率打印日志
                    print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100] [keep number: %d]' %
                          (batch_idx * batch_size, len(dataloader.dataset),
                           loss_meter.avg, acc_meter.avg * 100, np.round(rho[-1].item() * 65)))

        # 记录图像任务的统计量
        mu_record_img.append(rho[-1].cpu().numpy().item())
        print(mu_record_img)
        test_stat = {'loss': loss_meter.avg,  # 平均损失
                     'acc': acc_meter.avg}  # 平均准确率
        return test_stat

    # 如果任务是图像重建（imgr）
    elif ta_perform.startswith('imgr'):
        psnr_meter = AverageMeter()  # 用于记录PSNR
        loss_meter = AverageMeter()  # 用于记录损失
        psnr_list = []  # 用于保存PSNR值
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到指定设备
                outputs_dict = net(img=imgs, ta_perform=ta_perform)  # 前向传播
                outputs = outputs_dict['outputs']  # 模型输出
                rho = outputs_dict['rho']['img']  # 获取图像的稀疏性因子
                # 重排模型输出形状
                outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
                outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
                loss = criterion(outputs, targets)  # 计算损失
                batch_size = targets.shape[0]  # 当前批次大小
                ######  计算PSNR ######
                predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)  # 将输出分块
                targets = torch.chunk(imgs, chunks=imgs.size(0), dim=0)  # 将目标分块
                psnr_vals = calc_psnr(predictions, targets)  # 计算PSNR值
                psnr_list.extend(psnr_vals)  # 添加到PSNR列表中
                psnr_meter.update(torch.mean(torch.tensor(psnr_vals)).item(), n=batch_size)  # 更新平均PSNR
                loss_meter.update(loss.item(), 1)  # 更新平均损失
                if batch_idx % print_freq == 0:  # 打印日志
                    print('Test %d/%d: [loss: %.4f] [psnr: %.3f dB] [[cr ratio: %.4f]]' %
                          (batch_idx * batch_size, len(dataloader.dataset),
                           loss_meter.avg, psnr_meter.avg, rho.item()))

        test_stat = {'loss': loss_meter.avg,  # 平均损失
                     'psnr': psnr_meter.avg}  # 平均PSNR
        return test_stat

    # 如果任务是文本分类（textc）
    elif ta_perform.startswith('textc'):
        acc_meter = AverageMeter()  # 用于记录准确率
        loss_meter = AverageMeter()  # 用于记录损失值
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (texts, targets) in enumerate(dataloader):
                texts, targets = texts.to(device), targets.to(device)  # 将数据移动到指定设备
                outputs_dict = net(text=texts, ta_perform=ta_perform)  # 前向传播
                outputs = outputs_dict['outputs']  # 模型输出
                rho = outputs_dict['rho']['text']  # 获取文本的稀疏性因子
                loss = criterion(outputs, targets)  # 计算损失
                batch_size = targets.size(0)  # 当前批次大小
                idx, predicted = outputs.max(1)  # 获取预测类别
                acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)  # 更新准确率
                loss_meter.update(loss.item(), 1)  # 更新损失
                if batch_idx % print_freq == 0:  # 打印日志
                    print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100] [keep number: %d]' %
                          (batch_idx * batch_size, len(dataloader.dataset),
                           loss_meter.avg, acc_meter.avg * 100, np.round(rho[-1].item() * 66)))

        # 记录文本任务的统计量
        mu_record_text.append(rho[-1].cpu().numpy().item())
        print(mu_record_text)
        test_stat = {'loss': loss_meter.avg,  # 平均损失
                     'acc': acc_meter.avg}  # 平均准确率
        return test_stat


@torch.no_grad()  # 禁用梯度计算（提高评估速度，减少显存占用）
def evaluate_vqa(ta_perform: str, net: torch.nn.Module, dataloader: Iterable,
                 device: torch.device, criterion: torch.nn.Module, print_freq=500):
    """
    VQA（Visual Question Answering）任务的评估函数。

    参数：
    - ta_perform: str，任务类型标识。
    - net: torch.nn.Module，待评估的神经网络模型。
    - dataloader: Iterable，数据加载器，提供图像、文本和目标数据。
    - device: torch.device，设备类型（如'cuda'或'cpu'）。
    - criterion: torch.nn.Module，损失函数（未直接使用，仅为统一接口）。
    - print_freq: int，打印日志的频率（默认500）。

    返回：
    - vqaEval.accuracy: float，VQA任务的评估准确率。
    """
    net.eval()  # 设置模型为评估模式（冻结批归一化和Dropout层）
    dataset = dataloader.dataset  # 获取数据集对象
    qid_list = [ques['question_id'] for ques in dataset.ques_list]  # 获取问题的ID列表
    ans_ix_list = []  # 用于存储模型预测的答案索引
    i = 0  # 初始化累积计数器，用于跟踪评估进度

    # 遍历dataloader，逐批评估
    for batch_idx, (imgs, texts, targets) in enumerate(dataloader):
        # 将图像、文本和目标数据移动到指定设备
        imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
        batch_size = imgs.shape[0]  # 获取当前批次的样本数
        i += batch_size  # 更新累积样本计数
        outputs_dict = net(img=imgs, text=texts, ta_perform=ta_perform)  # 前向传播，获取预测结果
        outputs = outputs_dict['outputs']  # 获取模型输出
        rho_img = outputs_dict['rho']['img']  # 获取图像稀疏性因子
        rho_text = outputs_dict['rho']['text']  # 获取文本稀疏性因子
        pred_np = outputs.cpu().data.numpy()  # 将预测结果转为NumPy数组
        pred_argmax = np.argmax(pred_np, axis=1)  # 获取预测的类别索引

        # 如果当前批次的预测结果数量小于预期批次大小，进行填充（避免评估出错）
        if pred_argmax.shape[0] != dataset.configs.eval_batch_size:
            pred_argmax = np.pad(
                pred_argmax,
                (0, dataset.configs.eval_batch_size - pred_argmax.shape[0]),
                mode='constant', constant_values=-1  # 填充值为-1
            )
        ans_ix_list.append(pred_argmax)  # 将当前批次的预测结果添加到答案索引列表中
        if batch_idx % print_freq == 0:  # 每隔指定频率打印评估进度
            print('Test %d/%d:' % (batch_idx * batch_size, len(dataloader.dataset)))

    # 将答案索引列表展平为一维数组
    ans_ix_list = np.array(ans_ix_list).reshape(-1)
    # 构造评估结果的字典列表
    result = [{
        'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # 将索引映射为答案文本
        'question_id': int(qid_list[qix])  # 添加对应的问题ID
    } for qix in range(len(qid_list))]

    # 将评估结果保存为JSON文件
    result_eval_file = 'vqaeval_result/result_run_' + dataset.configs.version + '.json'
    print('Save the result to file: {}'.format(result_eval_file))
    json.dump(result, open(result_eval_file, 'w'))  # 将结果写入文件

    # 创建VQA工具对象和评估对象
    ques_file_path = dataset.configs.question_path['val']  # 问题文件路径
    ans_file_path = dataset.configs.answer_path['val']  # 答案文件路径
    vqa = VQA_Tool(ans_file_path, ques_file_path)  # 初始化VQA工具
    vqaRes = vqa.loadRes(result_eval_file, ques_file_path)  # 加载评估结果
    vqaEval = VQA_Eval(vqa, vqaRes, n=2)  # 初始化VQA评估工具（n=2表示2次评估）
    vqaEval.evaluate()  # 进行VQA评估

    # 打印模型的稀疏性因子
    print('[image keep number: %d] [text keep number: %d]' %
          (np.round(rho_img[-1].item() * 101), np.round(rho_text[-1].item() * 16)))

    return vqaEval.accuracy  # 返回评估的准确率

def train_class_batch_uni(ta_perform, model, sel_batch, targets, criterion):
    """
    训练单个批次的通用任务函数。

    参数：
    - ta_perform: str，任务类型标识，如 'imgc' 表示图像分类，'vqa' 表示视觉问答。
    - model: torch.nn.Module，训练的神经网络模型。
    - sel_batch: list，包含图像、文本和语音数据的列表。
    - targets: Tensor，目标标签。
    - criterion: dict，各任务对应的损失函数。

    返回：
    - loss: Tensor，计算的损失值。
    - outputs: Tensor，模型输出。
    """
    loss = 0  # 初始化损失
    imgs, texts, speechs = sel_batch  # 从批次中解包图像、文本、语音数据

    # 图像分类任务
    if ta_perform.startswith('imgc'):
        outputs = model(img=imgs, ta_perform=ta_perform)  # 获取模型输出
        loss = criterion[ta_perform](outputs, targets, ta_perform) * 1  # 计算损失

    # 图像重建任务
    elif ta_perform.startswith('imgr'):
        outputs = model(img=imgs, ta_perform=ta_perform)  # 获取模型输出
        # 重排目标张量形状以匹配输出
        targets = rearrange(targets, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
        loss = criterion[ta_perform](outputs, targets, ta_perform) * 300  # 计算损失（加权因子为300）

    # 文本分类任务
    elif ta_perform.startswith('textc'):
        outputs = model(text=texts, ta_perform=ta_perform)  # 获取模型输出
        loss = criterion[ta_perform](outputs, targets, ta_perform) * 0.5  # 计算损失（加权因子为0.5）

    # 文本生成任务
    elif ta_perform.startswith('textr'):
        outputs = model(text=texts, ta_perform=ta_perform)  # 获取模型输出
        targets = targets[:, 1:]  # 去掉序列的第一个时间步
        for i in range(outputs.shape[1]):  # 遍历时间步计算损失
            loss += criterion[ta_perform](outputs[:, i], targets[:, i], ta_perform) * 5

    # 视觉问答任务
    elif ta_perform.startswith('vqa'):
        outputs = model(img=imgs, text=texts, ta_perform=ta_perform)  # 获取模型输出
        loss = criterion[ta_perform](outputs, targets, ta_perform) * 1

    # 多模态任务
    elif ta_perform.startswith('msa'):
        outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform)  # 获取模型输出
        loss = criterion[ta_perform](outputs, targets, ta_perform) * 1

    return loss, outputs  # 返回损失和模型输出
def meter(ta_sel):
    """
    初始化度量器，用于记录任务的损失和准确率等指标。

    参数：
    - ta_sel: list，选择的任务类型列表。

    返回：
    - acc_meter_dict: dict，每个分类任务对应的准确率记录器。
    - loss_meter_dict: dict，每个任务对应的损失记录器。
    - psnr_meter: AverageMeter，用于记录图像质量的PSNR值。
    """
    acc_meter_dict = {}  # 准确率记录器字典
    acc_meter_dict['imgc'] = AverageMeter()  # 图像分类任务
    acc_meter_dict['textc'] = AverageMeter()  # 文本分类任务
    acc_meter_dict['vqa'] = AverageMeter()  # 视觉问答任务

    loss_meter_dict = {}  # 损失记录器字典
    for ta in ta_sel:
        loss_meter_dict[ta] = AverageMeter()  # 为每个任务初始化记录器

    psnr_meter = AverageMeter()  # 初始化PSNR记录器
    return acc_meter_dict, loss_meter_dict, psnr_meter


def train_epoch_uni(model: torch.nn.Module, criterion: dict,
                    data_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, ta_sel, max_norm: float = 0,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    update_freq=None, print_freq=10):
    """
    通用训练函数，支持多任务训练，处理单个训练周期。

    参数：
    - model: torch.nn.Module，待训练的模型。
    - criterion: dict，各任务的损失函数。
    - data_dict: dict，包含每个任务的数据加载器。
    - optimizer: torch.optim.Optimizer，优化器。
    - device: torch.device，设备类型（如'cuda'或'cpu'）。
    - epoch: int，当前训练的轮数。
    - loss_scaler: 用于动态损失缩放的工具（混合精度训练）。
    - ta_sel: list，选择的任务类型列表。
    - max_norm: float，梯度裁剪的阈值。
    - start_steps: int，训练的起始步数（可选）。
    - lr_schedule_values: list，学习率调度计划（可选）。
    - wd_schedule_values: list，权重衰减调度计划（可选）。
    - update_freq: int，梯度累积的更新频率。
    - print_freq: int，打印日志的频率。

    返回：
    - train_stat: dict，包含训练统计信息（例如损失和准确率）。
    """
    model.train(True)  # 设置模型为训练模式
    acc_meter_dict, loss_meter_dict, psnr_meter = meter(ta_sel)  # 初始化度量器

    # 清零优化器的梯度或模型的内部状态
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    num_samples = 500  # 样本总数（默认值）
    data_iter_step = 0  # 初始化迭代计数器
    num_tasks = len(data_dict)  # 获取任务数量
    data_tuple = [data_loader for data_loader in data_dict.values()]  # 提取数据加载器

    # 遍历每个批次的任务数据
    for data_batch in zip(data_tuple[0], data_tuple[1], data_tuple[2]):
        step = data_iter_step // update_freq  # 当前全局步数
        it = start_steps + step  # 更新累计步数

        # 更新学习率和权重衰减（如果设置了调度计划）
        # if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
        #     for i, param_group in enumerate(optimizer.param_groups):
        #         if lr_schedule_values is not None:
        #             param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
        #         if wd_schedule_values is not None and param_group["weight_decay"] > 0:
        #             param_group["weight_decay"] = wd_schedule_values[it]

        # 初始化每批次的数据
        imgs, texts, speechs, targets = None, None, None, None
        ta_index = np.random.randint(num_tasks)  # 随机选择任务索引
        ta = ta_sel[ta_index]  # 获取当前任务的类型
        data = data_batch[ta_index]  # 获取当前任务的数据批次

        # 根据任务类型加载数据到对应变量
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
            raise NotImplementedError("未实现的任务类型")  # 如果任务类型未知，抛出错误

        batch_size = targets.shape[0]  # 当前批次的样本数量
        sel_batch = [imgs, texts, speechs]  # 将当前批次数据打包
        loss, outputs = train_class_batch_uni(ta, model, sel_batch, targets, criterion)  # 计算损失和输出

        # 模型输出解包
        outputs, mask_m, rho, vq_loss = outputs
        loss_value = loss.item()  # 获取当前损失值

        # 梯度更新（考虑混合精度训练）
        if loss_scaler is None:
            loss /= update_freq  # 累积损失
            model.backward(loss)  # 反向传播
            model.step()  # 参数更新
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()  # 清零优化器梯度

        torch.cuda.synchronize()  # 同步GPU运算
        data_iter_step += 1  # 增加迭代计数器

        # 更新学习率最小值和最大值
        min_lr, max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr, max_lr = min(min_lr, group["lr"]), max(max_lr, group["lr"])

        # 根据任务类型记录损失和度量指标
        if ta.endswith('c'):  # 分类任务
            acc_meter_dict[ta].update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('imgr'):  # 图像重建任务
            outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
            outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
            tr_imgs = torch.tensor((imgs * 255).detach().cpu().numpy().astype(int).clip(0, 255)).float()
            re_imgs = torch.tensor((outputs * 255).detach().cpu().numpy().astype(int).clip(0, 255)).float()
            mse_cal = nn.MSELoss()  # 使用均方误差
            psnr_meter.update(10 * math.log10(255.0**2 / (mse_cal(tr_imgs, re_imgs))), n=1)  # 计算PSNR
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('textr'):  # 文本生成任务
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('vqa'):  # 视觉问答任务
            acc_meter_dict[ta].update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
            loss_meter_dict[ta].update(loss_value, 1)
        elif ta.startswith('msa'):  # 多模态任务
            loss_meter_dict[ta].update(loss_value, 1)

        # 打印日志
        if data_iter_step % print_freq == 0:
            if ta.startswith('imgc'):  # 图像分类任务
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' %
                      (epoch, ta, batch_size * data_iter_step, 5000,
                       loss_meter_dict[ta].avg, acc_meter_dict[ta].avg * 100, max_lr))
            elif ta.startswith('imgr'):  # 图像重建任务
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [psnr: %.3f dB] [lr: %.3e]' %
                      (epoch, ta, batch_size * data_iter_step, 5000,
                       loss_meter_dict[ta].avg, psnr_meter.avg, max_lr))
            elif ta.startswith('textc'):  # 文本分类任务
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' %
                      (epoch, ta, batch_size * data_iter_step, num_samples,
                       loss_meter_dict[ta].avg, acc_meter_dict[ta].avg * 100, max_lr))
            elif ta.startswith('textr'):  # 文本生成任务
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [lr: %.3e]' %
                      (epoch, ta, batch_size * data_iter_step, num_samples,
                       loss_meter_dict[ta].avg, max_lr))
            elif ta.startswith('vqa'):  # 视觉问答任务
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' %
                      (epoch, ta, batch_size * data_iter_step, 5000,
                       loss_meter_dict[ta].avg, acc_meter_dict[ta].avg * 100, max_lr))
            elif ta.startswith('msa'):  # 多模态任务
                print('Epoch:[%d] [%s] %d/%d: [loss: %.3f] [lr: %.3e]' %
                      (epoch, ta, batch_size * data_iter_step, 5000,
                       loss_meter_dict[ta].avg, max_lr))

    train_stat = None  # 训练统计信息（可根据需要填充）
    return train_stat



@torch.no_grad()
def evaluate_msa(ta_perform: str, net: torch.nn.Module, dataloader: Iterable,
                 device: torch.device, criterion: torch.nn.Module, print_freq=10):
    """
    评估多模态任务（MSA）的函数。

    参数：
    - ta_perform: str，任务类型标识。
    - net: torch.nn.Module，待评估的神经网络模型。
    - dataloader: Iterable，数据加载器，提供图像、文本、语音和目标标签。
    - device: torch.device，设备类型（如'cuda'或'cpu'）。
    - criterion: torch.nn.Module，损失函数。
    - print_freq: int，打印日志的频率（默认值为10）。

    返回：
    - test_stat: dict，评估统计信息，包括平均损失和准确率。
    """
    net.eval()  # 设置模型为评估模式（冻结批归一化和Dropout层）
    loss_meter = AverageMeter()  # 初始化损失记录器，用于计算平均损失
    y_true, y_pred = [], []  # 用于存储真实标签和预测结果

    # 禁用梯度计算，提高评估速度，减少显存占用
    with torch.no_grad():
        for batch_idx, (imgs, texts, speechs, targets) in enumerate(dataloader):
            # 将批次数据加载到指定设备
            imgs = imgs.to(device)
            texts = texts.to(device)
            speechs = speechs.to(device)
            targets = targets.to(device)

            # 模型前向传播，获取预测输出
            outputs = net(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform)

            # 计算当前批次的损失
            loss = criterion(outputs, targets)

            # 将预测结果和真实标签保存到列表中
            y_pred.append(outputs.detach().cpu().numpy())  # 将预测结果从GPU转到CPU，并转换为NumPy格式
            y_true.append(targets.detach().cpu().numpy())  # 同样处理目标标签

            # 更新损失记录器
            loss_meter.update(loss.item(), 1)  # 将当前损失添加到平均损失记录器中

    # 将所有批次的预测结果和真实标签拼接成完整的数组
    y_true = np.concatenate(y_true, axis=0).squeeze()  # 合并所有真实标签
    y_pred = np.concatenate(y_pred, axis=0).squeeze()  # 合并所有预测结果

    # 计算准确率或其他评估指标（根据任务自定义）
    acc = calc_metrics(y_true, y_pred)  # 调用自定义评估函数计算准确率

    # 构造评估统计信息字典
    test_stat = {
        'loss': loss_meter.avg,  # 平均损失
        'acc': acc  # 准确率
    }

    return test_stat  # 返回评估统计信息
