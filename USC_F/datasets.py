import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

from transformers import BertTokenizer
from data import CIFAR_CR, SST_CR
from timm.data import create_transform
from vqa_utils import VQA2, Config_VQA
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
from msa_utils import PAD, Config_MSA, MSA
from torch.utils.data.sampler import RandomSampler

# 加载BERT分词器，用于处理文本数据
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 定义Batch调度采样器，用于在每个小批量中提供不同任务的数据
class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    迭代任务，并在每个小批量中为每个任务提供一个随机批次。
    用于处理多任务学习中不同任务的数据调度。
    """

    def __init__(self, dataset, batch_size, number_samp=5000):
        # dataset: 输入的数据集
        # batch_size: 每个批次的大小
        # number_samp: 每个epoch的采样数量
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_samp = number_samp
        self.largest_dataset_size = number_samp

    def __len__(self):
        return self.number_samp

    def __iter__(self):
        # 使用随机采样器对数据集进行采样
        sampler = RandomSampler(self.dataset)
        sampler_iterator = sampler.__iter__()
        step = self.batch_size
        samples_to_grab = self.batch_size
        epoch_samples = self.number_samp
        final_samples_list = []
        # 创建索引列表，用于组合数据集
        for es in range(0, epoch_samples, step):
            cur_batch_sampler = sampler_iterator
            cur_samples = []
            for eg in range(samples_to_grab):
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org
                    cur_samples.append(cur_sample)
                except StopIteration:
                    # 到达采样器末尾，重新启动采样器继续获取样本
                    sampler_iterator = sampler.__iter__()
                    cur_batch_sampler = sampler_iterator
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org
                    cur_samples.append(cur_sample)
            final_samples_list.extend(cur_samples)
        return iter(final_samples_list)

    def set_epoch(self, epoch: int) -> None:
        # 设置当前epoch，用于多任务训练中的采样调整
        self.epoch = epoch


# 构建数据加载器
# ta_sel: 选择的任务列表, trainsets: 训练数据集字典, args: 参数对象
# 返回每个任务对应的数据加载器
def build_dataloader(ta_sel, trainsets, args):
    trainloaders = {}
    for ta in ta_sel:
        trainset = trainsets[ta]
        # 如果任务是多模态数据（例如MSA），使用自定义的collate函数
        Collate_fn = collate_fn if ta.startswith('msa') else None
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                  sampler=BatchSchedulerSampler(dataset=trainset,
                                                                                batch_size=args.batch_size,
                                                                                number_samp=15000 * len(
                                                                                    ta_sel)),
                                                  num_workers=args.num_workers, pin_memory=True,
                                                  batch_size=args.batch_size, shuffle=False,
                                                  collate_fn=Collate_fn,
                                                  drop_last=True)
        trainloaders[ta] = trainloader
    return trainloaders


# 构建测试数据集
# is_train: 是否为训练数据, args: 参数对象
# 返回构建的测试数据集
def build_dataset_test(is_train, args):
    if args.ta_perform.startswith('img'):
        # 如果任务是图像相关，构建图像转换操作
        transform = build_img_transform(is_train, args)
        print("Transform = ")
        if isinstance(transform, tuple):
            for trans in transform:
                print(" - - - - - - - - - - ")
                for t in trans.transforms:
                    print(t)
        else:
            for t in transform.transforms:
                print(t)
        print("------------------------------------------------------")

    # 根据任务类型选择不同的数据集构建
    if args.ta_perform.startswith('imgc'):
        dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform,
                           download=True, if_class=True)
    elif args.ta_perform.startswith('imgr'):
        dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform,
                           download=True, if_class=False)
    elif args.ta_perform.startswith('textc'):
        dataset = SST_CR(root=False, train=is_train, binary=True, if_class=True)
    elif args.ta_perform.startswith('textr'):
        dataset = SST_CR(root=True, train=is_train, binary=True, if_class=False)
    elif args.ta_perform.startswith('vqa'):
        config_vqa = Config_VQA()
        config_vqa.proc(args)
        dataset = VQA2(config_vqa, train=is_train)
    elif args.ta_perform.startswith('msa'):
        config_msa = Config_MSA()
        dataset = MSA(config_msa, train=is_train)
    else:
        raise NotImplementedError("未实现的任务类型")

    return dataset


# 构建训练数据集
# is_train: 是否为训练数据, ta_sel: 选择的任务列表, args: 参数对象
# 返回构建的训练数据集字典
def build_dataset_train(is_train, ta_sel, args):
    # 如果任务是图像相关，构建图像转换操作
    transform = build_img_transform(is_train, args)
    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("------------------------------------------------------")

    datasets = {}
    for ta in ta_sel:
        # 根据任务类型选择不同的数据集构建
        if ta.startswith('imgc'):
            dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform,
                               download=True, if_class=True)
        elif ta.startswith('imgr'):
            dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform,
                               download=True, if_class=False)
        elif ta.startswith('textc'):
            dataset = SST_CR(root=False, train=is_train, binary=True, if_class=True)
        elif ta.startswith('textr'):
            dataset = SST_CR(root=True, train=is_train, binary=True, if_class=False)
        elif ta.startswith('vqa'):
            config_vqa = Config_VQA()
            config_vqa.proc(args)
            dataset = VQA2(config_vqa, train=is_train)
        elif ta.startswith('msa'):
            config_msa = Config_MSA()
            dataset = MSA(config_msa, train=is_train)
        else:
            raise NotImplementedError("未实现的任务类型")

        datasets[ta] = dataset

    return datasets


# 构建图像数据集的转换操作
# is_train: 是否为训练数据, args: 参数对象
# 返回构建的图像转换操作
def build_img_transform(is_train, args):
    # 判断是否需要调整图像大小
    resize_im = args.input_size > 32
    mean = (0., 0., 0.)
    std = (1., 1., 1.)

    if is_train:
        # 如果是训练数据，创建训练用的图像转换操作
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            interpolation=args.train_interpolation,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # 如果不需要调整大小，则使用随机裁剪
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    # 如果是测试数据，创建测试用的图像转换操作
    t = []
    if resize_im:
        crop_pct = 1
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # 保持与224图像相同的比例
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())  # 将图像转换为Tensor
    t.append(transforms.Normalize(mean, std))  # 归一化图像
    return transforms.Compose(t)


# 自定义collate函数，用于将不同模态的数据批量处理
# batch: 输入的批量数据
# 返回处理后的批量图像、文本、语音和目标数据
def collate_fn(batch):
    # 根据文本长度对batch进行排序，便于后续的处理
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    targets = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    texts = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    images = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    speechs = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    SENT_LEN = texts.size(0)
    # 使用BERT分词器将文本转换为BERT的输入索引
    bert_details = []
    for sample in batch:
        text = " ".join(sample[0][3])
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, max_length=SENT_LEN + 2, add_special_tokens=True, pad_to_max_length=True, truncation=True)
        bert_details.append(encoded_bert_sent)

    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    texts = bert_sentences

    return images.permute(1, 0, 2), texts, speechs.permute(1, 0, 2), targets
