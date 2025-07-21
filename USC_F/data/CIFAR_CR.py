import os
import os.path
import numpy as np
import pickle
import torch
from PIL import Image

from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# 自定义数据集类CIFAR_CR，继承自torchvision的VisionDataset
class CIFAR_CR(VisionDataset):
    # 基础配置参数，包括文件夹名称、下载链接、文件名及校验码等
    base_folder = 'cifar'  # 数据集的基础文件夹名称
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"  # 数据集下载链接
    filename = "cifar-10-python.tar.gz"  # 下载的压缩包文件名
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'  # 压缩包的MD5校验码，用于验证文件完整性

    # 训练集和测试集的文件列表及其MD5校验码
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [['test_batch', '40351d587109b95175f43aff81a1287e'],]

    # 元数据文件的信息，包括文件名、键值和MD5校验码
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    # 初始化函数
    # root: 数据集的根目录
    # train: 是否加载训练集，True为训练集，False为测试集
    # transform: 图像的转换函数（如数据增强）
    # target_transform: 标签的转换函数
    # download: 是否下载数据集
    # if_class: 是否返回图像和标签，True返回图像和标签，False返回图像和图像
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, if_class: bool = False) -> None:
        # 调用父类的初始化函数
        super(CIFAR_CR, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # 设置是否为训练集
        self.if_class = if_class  # 设置是否返回类别标签

        # 如果download为True，则下载数据集
        if download:
            self.download()

        # 检查数据集的完整性，如果数据集不存在或损坏则抛出异常
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # 根据train参数选择加载训练集或测试集
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []  # 存储图像数据
        self.targets = []  # 存储对应的标签

        # 加载下载的numpy数组文件
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)  # 获取文件路径
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')  # 加载文件，使用latin1编码以支持特殊字符
                self.data.append(entry['data'])  # 读取图像数据
                # 读取图像对应的标签
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        # 将数据转换为合适的形状
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # 将数据堆叠并调整为 (N, 3, 32, 32) 形状
        self.data = self.data.transpose((0, 2, 3, 1))  # 转换为 (N, 32, 32, 3) 的 HWC 形式
        self._load_meta()  # 加载元数据，获取类别信息

    # 加载元数据的方法，用于获取类别标签信息
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])  # 获取元数据文件路径
        # 检查元数据文件的完整性
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')  # 加载元数据
            self.classes = data[self.meta['key']]  # 获取类别名称
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}  # 类别名称到索引的映射

    # 获取单个数据的方法
    # index: 数据的索引
    # 返回图像及其对应的标签（或图像自身）
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.if_class:
            img, target = self.data[index], self.targets[index]  # 获取图像和标签
            img = Image.fromarray(img)  # 将numpy数组转换为PIL图像
            # 如果设置了transform，则对图像进行转换
            if self.transform is not None:
                img = self.transform(img)
            # 如果设置了target_transform，则对标签进行转换
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target  # 返回图像和标签
        else:
            img = self.data[index]  # 获取图像
            img = Image.fromarray(img)  # 将numpy数组转换为PIL图像
            # 如果设置了transform，则对图像进行转换
            if self.transform is not None:
                img = self.transform(img)
            return img, img  # 返回图像和图像本身（无标签）

    # 返回数据集的长度
    def __len__(self) -> int:
        return len(self.data)  # 返回数据集的长度

    # 检查数据集文件的完整性
    def _check_integrity(self) -> bool:
        root = self.root  # 数据集的根目录
        # 检查所有训练和测试文件的完整性
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)  # 获取文件路径
            # 如果文件不存在或校验失败，则返回False
            if not check_integrity(fpath, md5):
                return False
        return True  # 所有文件检查通过，返回True

    # 下载并解压数据集的方法
    def download(self) -> None:
        # 如果数据集已经存在且完整，则跳过下载
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        # 下载并解压数据集
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
