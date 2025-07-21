import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def qam_mod(M):
    """
    生成 M-QAM 调制的映射表和解映射表。

    参数:
    ----------
    M: int
        调制阶数，必须是 2 的幂且为完全平方数，或为特殊值 8 和 32。

    返回:
    ----------
    map_table: dict
        M-QAM 调制映射表，从整数（Gray 编码）到星座点的映射。
    demap_table: dict
        M-QAM 解调映射表，从星座点到整数（Gray 编码）的映射。
    """
    sqrtM = int(math.sqrt(M))  # 计算调制阶数的平方根
    assert (sqrtM ** 2 == M and M & (M - 1) == 0) or (M == 32) or (M == 8), \
        "M 必须是 2 的幂且为完全平方数，或者为特殊值 8 和 32。"

    if M == 8:  # 针对特殊情况 8-QAM
        graycode = np.array([0, 1, 3, 7, 5, 4, 6, 2])  # Gray 编码序列
        constellation = [(-2 - 2j), (-2 + 0j), (-2 + 2j), (0 + 2j), (2 + 2j), (2 + 0j), (2 - 2j),
                         (0 - 2j)]  # 星座点
    elif M == 32:  # 针对特殊情况 32-QAM
        temp1 = np.bitwise_xor(np.arange(8), np.right_shift(np.arange(8), 1))  # Gray 编码生成
        temp2 = np.bitwise_xor(np.arange(4), np.right_shift(np.arange(4), 1))
        graycode = np.zeros(M, dtype=int)
        num = 0
        for i in temp1:
            for j in temp2:
                graycode[num] = 4 * i + j  # 二维构造 Gray 编码
                num += 1
        constellation = [(-7 - 3j) + 2 * (x + y * 1j) for x, y in np.ndindex(8, 4)]  # 星座点生成
    else:  # 一般情况 M-QAM
        temp = np.bitwise_xor(np.arange(sqrtM), np.right_shift(np.arange(sqrtM), 1))  # Gray 编码生成
        graycode = np.zeros(M, dtype=int)
        num = 0
        for i in temp:
            for j in temp:
                graycode[num] = sqrtM * i + j  # 二维 Gray 编码
                num += 1
        constellation = [-(sqrtM - 1) * (1 + 1j) + 2 * (x + y * 1j) for x, y in
                         np.ndindex(sqrtM, sqrtM)]  # 星座点生成

    map_table = dict(zip(graycode, constellation))  # 构建调制映射表
    demap_table = {v: k for k, v in map_table.items()}  # 构建解调映射表
    return map_table, demap_table


def qam_mapper(bits, map_table):
    """
    使用 M-QAM 技术将编码比特映射为符号。

    参数:
    ----------
    bits: array(num_bit,)
        待调制的比特序列。
    map_table: dict
        M-QAM 映射表。

    返回:
    ----------
    syms: array(num_symbol,)
        调制后的符号，用于传输。
    """
    M = len(map_table)  # 调制阶数
    bits = np.reshape(bits, (-1,))  # 展平比特序列
    nbits = int(math.log2(M))  # 每个符号所需的比特数

    if len(bits) % nbits != 0:
        # 如果比特数不是 nbits 的倍数，则填充比特为 0
        bits = np.pad(bits, (0, nbits - len(bits) % nbits), constant_values=(0, 0))

    bit_blocks = np.reshape(bits, (-1, nbits))  # 将比特分块
    blocks_bin = [''.join(str(_) for _ in block) for block in bit_blocks]  # 转换为字符串表示
    blocks_dec = [int(block, 2) for block in blocks_bin]  # 二进制转十进制
    syms = np.array([map_table[block] for block in blocks_dec])  # 使用映射表进行调制
    return syms


def qam_demapper(syms, demap_table):
    """
    根据 M-QAM 映射表将接收到的符号解调为比特。

    参数:
    ----------
    syms: array(num_symbols,)
        接收到的符号（可能包含噪声）。
    demap_table: dict
        M-QAM 解映射表。

    返回:
    ----------
    bits: array(num_bit,)
        解调后的比特序列。
    """
    M = len(demap_table)  # 调制阶数
    nbits = int(math.log2(M))  # 每个符号所需的比特数
    constellation = np.array([x for x in demap_table.keys()])  # 提取星座点
    dists = np.abs(syms.reshape(-1, 1) - constellation.reshape(1, -1))  # 计算符号与星座点的欧氏距离
    const_index = dists.argmin(axis=1)  # 找到距离最小的星座点索引
    hardDecision = constellation[const_index]  # 取最近的星座点作为硬判决
    bit_blocks = [bin(demap_table[C])[2:].rjust(nbits, '0') for C in hardDecision]  # 映射为比特块
    bits_str = ''.join(block for block in bit_blocks)  # 合并比特块为字符串
    bits = np.array([int(_) for _ in bits_str])  # 转换为比特数组
    return bits


def channel_Awgn(tx_signal, snr, output_power=False):
    """
    AWGN 信道模型。

    参数:
    ----------
    tx_signal: array(num_symbols,)
        待传输的信号。
    snr: int
        接收端的信噪比。
    output_power: bool, 默认 False
        是否输出信号功率和噪声功率。

    返回:
    ----------
    rx_signal: array(num_symbols,)
        加入噪声后的接收信号。
    """
    signal_power = np.mean(abs(tx_signal ** 2))  # 信号功率
    n_var = signal_power * 10 ** (-snr / 10)  # 噪声功率
    if output_power:
        print(f"RX Signal power: {signal_power: .4f}. Noise power: {n_var: .4f}")
    # 生成复高斯噪声
    noise = math.sqrt(n_var / 2) * (
                np.random.randn(*tx_signal.shape) + 1j * np.random.randn(*tx_signal.shape))
    return tx_signal + noise  # 加入噪声


def channel_Rayleigh(tx_signal, snr, output_power=False):
    """
    瑞利信道模型。

    参数:
    ----------
    tx_signal: array(num_symbols,)
        待传输的信号。
    snr: int
        接收端的信噪比。
    output_power: bool, 默认 False
        是否输出信号功率和噪声功率。

    返回:
    ----------
    rx_signal: array(num_symbols,)
        加入信道和噪声后的接收信号。
    """
    shape = tx_signal.shape
    sigma = math.sqrt(1 / 2)
    H = np.random.normal(0.0, sigma, size=shape) + 1j * np.random.normal(0.0, sigma, size=shape)  # 信道系数
    Tx_sig = tx_signal * H  # 通过信道
    Rx_sig = channel_Awgn(Tx_sig, snr, output_power=output_power)  # 加入 AWGN 噪声
    Rx_sig = Rx_sig / H  # 信道估计
    return Rx_sig


def channel_Rician(tx_signal, snr, output_power=False, K=1):
    """
    瑞利信道模型。

    参数:
    ----------
    tx_signal: array(num_symbols,)
        待传输的信号。
    snr: int
        接收端的信噪比。
    output_power: bool, 默认 False
        是否输出信号功率和噪声功率。
    K: int, 默认 1
        Rician 因子，表示直达路径与散射路径的功率比。

    返回:
    ----------
    rx_signal: array(num_symbols,)
        加入信道和噪声后的接收信号。
    """
    shape = tx_signal.shape
    mean = math.sqrt(K / (K + 1))  # 直达路径均值
    std = math.sqrt(1 / (K + 1))  # 散射路径标准差
    H = np.random.normal(mean, std, size=shape) + 1j * np.random.normal(mean, std, size=shape)  # 信道系数
    Tx_sig = tx_signal * H  # 通过信道
    Rx_sig = channel_Awgn(Tx_sig, snr, output_power=output_power)  # 加入 AWGN 噪声
    Rx_sig = Rx_sig / H  # 信道估计
    return Rx_sig


def bit_error_rate(tx_bits, rx_bits):
    """
    计算比特错误率 (BER)。

    参数:
    ----------
    tx_bits: array(num_bit,)
        传输端的比特序列。
    rx_bits: array(num_bit,)
        接收端的比特序列。

    返回:
    ----------
    ber: float
        比特错误率。
    """
    return np.sum(abs(tx_bits - rx_bits)) / len(tx_bits)


def q_func(x):
    """
    Q 函数，用于计算高斯分布右尾概率。

    参数:
    ----------
    x: float
        输入值。

    返回:
    ----------
    Qx: float
        高斯右尾概率值。
    """
    Qx = 0.5 * math.erfc(x / math.sqrt(2))
    return Qx
