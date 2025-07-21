import math
import torch
import numpy as np

# 定义每个符号的比特数，这里为64QAM调制，因此每个符号包含4比特
bit_per_symbol = 4  # bits per symbol (64QAM)

# 映射表，将4比特映射到相应的复数调制符号
mapping_table = {
    (1, 0, 1, 0): -3 - 3j,
    (1, 0, 1, 1): -3 - 1j,
    (1, 0, 0, 1): -3 + 1j,
    (1, 0, 0, 0): -3 + 3j,
    (1, 1, 1, 0): -1 - 3j,
    (1, 1, 1, 1): -1 - 1j,
    (1, 1, 0, 1): -1 + 1j,
    (1, 1, 0, 0): -1 + 3j,
    (0, 1, 1, 0): 1 - 3j,
    (0, 1, 1, 1): 1 - 1j,
    (0, 1, 0, 1): 1 + 1j,
    (0, 1, 0, 0): 1 + 3j,
    (0, 0, 1, 0): 3 - 3j,
    (0, 0, 1, 1): 3 - 1j,
    (0, 0, 0, 1): 3 + 1j,
    (0, 0, 0, 0): 3 + 3j,
}
# 解调表，将调制符号映射回4比特数据
demapping_table = {v: k for k, v in mapping_table.items()}


# 将字符串拆分为字符列表
def split(word):
    return [char for char in word]


# 将比特按每4个分组，用于QAM调制
# bitc: 输入的比特流，bity: 按每bit_per_symbol比特分组后的结果
def group_bits(bitc):
    bity = []
    x = 0
    for i in range((len(bitc) // bit_per_symbol)):
        bity.append(bitc[x:x + bit_per_symbol])
        x = x + bit_per_symbol
    return bity


# 高斯信道模拟函数
# signal: 输入信号, SNRdb: 信噪比, output_power: 是否输出功率信息
# 返回加入噪声后的接收信号
def channel(signal, SNRdb, ouput_power=False):
    signal_power = np.mean(abs(signal ** 2))
    sigma2 = signal_power * 10 ** (-SNRdb / 10)  # 根据信号功率和SNR计算噪声功率
    if ouput_power:
        print("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
        # 生成符合噪声功率的复数噪声
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise


# 瑞利衰落信道模拟
# H为瑞利衰落系数，包含实部和虚部，Rx_sig为接收信号
def channel_Rayleigh(signal, SNRdb, ouput_power=False):
    shape = signal.shape
    sigma = math.sqrt(1 / 2)
    H = np.random.normal(0.0, sigma, size=[1]) + 1j * np.random.normal(0.0, sigma, size=[1])
    Tx_sig = signal * H

    Rx_sig = channel(Tx_sig, SNRdb, ouput_power=False)
    # 信道估计
    Rx_sig = Rx_sig / H
    return Rx_sig


# 莱斯衰落信道模拟
# K为莱斯因子，H为莱斯衰落系数，Tx_sig为发送信号，Rx_sig为接收信号
def channel_Rician(signal, SNRdb, ouput_power=False, K=1):
    shape = signal.shape
    mean = math.sqrt(K / (K + 1))
    std = math.sqrt(1 / (K + 1))
    H = np.random.normal(mean, std, size=[1]) + 1j * np.random.normal(mean, std, size=[1])
    Tx_sig = signal * H

    Rx_sig = channel(Tx_sig, SNRdb, ouput_power=False)
    # 信道估计
    Rx_sig = Rx_sig / H
    return Rx_sig


# 解调函数，将接收到的QAM符号解调为比特
# 返回解调后的比特流和硬判决结果
def Demapping(QAM):
    # 获取可能的星座点
    constellation = np.array([x for x in demapping_table.keys()])

    # 计算每个接收点到每个可能星座点的距离
    dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))

    # 为每个QAM点选择最近的星座点的索引
    const_index = dists.argmin(axis=1)

    # 获取实际的星座点
    hardDecision = constellation[const_index]

    # 将星座点转换为比特组
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision


# 数据传输函数，模拟数据在信道中的传输
# data: 输入数据, SNRdb: 信噪比, bits_per_digit: 每个数值的比特数
# 返回接收到的数据
def transmit(data, SNRdb, bits_per_digit):
    TX_signal = data[:].cpu().numpy().flatten()
    Tx_data_binary = []

    # 生成二进制数据
    for i in TX_signal:
        Tx_data_binary.append('{0:b}'.format(i).zfill(bits_per_digit))
    Tx_data = []
    Tx_data_ready = []
    for i in Tx_data_binary:
        Tx_data.append(split(i))
    img_for_trans1 = np.vstack(Tx_data)
    for i in img_for_trans1:
        for j in range(bits_per_digit):
            Tx_data_ready.append(int(i[j]))
    Tx_data_ready = np.array(Tx_data_ready)

    ori_len = len(Tx_data_ready)
    padding_len = ori_len

    # 如果数据长度不是bit_per_symbol的倍数，则补齐
    if ori_len % 4 != 0:
        padding_len = ori_len + (bit_per_symbol - (ori_len % bit_per_symbol))

    Whole_tx_data = np.zeros(padding_len, dtype=int)
    Whole_tx_data[:ori_len] = Tx_data_ready

    bit_group = group_bits(Whole_tx_data)
    bit_group = np.array(bit_group)

    # 将比特组映射为QAM符号
    QAM_symbols = []
    for bits in bit_group:
        symbol = mapping_table[tuple(bits)]
        QAM_symbols.append(symbol)
    QAM_symbols = np.array(QAM_symbols)

    # 通过高斯信道传输
    Rx_symbols = channel(QAM_symbols, SNRdb)  # Pass the Guassian Channel
    Rx_bits, hardDecision = Demapping(Rx_symbols)

    # 使用接收的比特重新构造数据
    data_rea = []
    Rx_long = Rx_bits.reshape(-1, )[0:ori_len]
    k = 0
    for i in range(Rx_long.shape[0] // bits_per_digit):
        data_rea.append(Rx_long[k:k + bits_per_digit])
        k += bits_per_digit
    data_done = []
    for i in data_rea[:]:
        x = []
        for j in range(len(i)):
            x.append(str(i[j]))
        data_done.append(x)
    sep = ''
    data_fin = []
    for i in data_done:
        data_fin.append(sep.join(i))
    data_dec = []
    for i in data_fin:
        data_dec.append(i[0:bits_per_digit])
    data_dec = np.array(data_dec)
    data_back = []
    for i in range(len(Tx_data_binary)):
        data_back.append(int(data_dec[i], 2))
    data_back = np.array(data_back)

    return data_back


# 信号功率归一化函数，用于归一化batch中的信号
# signal: 输入信号, power: 目标功率
# 返回归一化后的信号

def power_norm_batchwise(signal, power=1):
    batchsize, num_elements = signal.shape[0], len(signal[0].flatten())
    num_complex = num_elements // 2
    signal_shape = signal.shape
    signal = signal.view(batchsize, num_complex, 2)
    signal_power = torch.sum((signal[:, :, 0] ** 2 + signal[:, :, 1] ** 2), dim=-1) / num_complex

    signal = signal * math.sqrt(power) / torch.sqrt(signal_power.unsqueeze(-1).unsqueeze(-1))
    signal = signal.view(signal_shape)
    return signal
