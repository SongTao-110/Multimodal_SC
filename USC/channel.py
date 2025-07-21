import math  # 导入数学库
import torch  # 导入 PyTorch 库
import numpy as np  # 导入 NumPy 库

bit_per_symbol = 4  # 每个符号的比特数 (64QAM 使用 4 位)
mapping_table = {  # QAM 映射表，将比特组映射到复数符号
    (1,0,1,0) : -3-3j,
    (1,0,1,1) : -3-1j,
    (1,0,0,1) : -3+1j,
    (1,0,0,0) : -3+3j,
    (1,1,1,0) : -1-3j,
    (1,1,1,1) : -1-1j,
    (1,1,0,1) : -1+1j,
    (1,1,0,0) : -1+3j,
    (0,1,1,0) : 1-3j,
    (0,1,1,1) : 1-1j,
    (0,1,0,1) : 1+1j,
    (0,1,0,0) : 1+3j,
    (0,0,1,0) : 3-3j,
    (0,0,1,1) : 3-1j,
    (0,0,0,1) : 3+1j,
    (0,0,0,0) : 3+3j,
}
demapping_table = {v : k for k, v in mapping_table.items()}  # 解映射表，将 QAM 符号映射回比特组

def split(word):
    return [char for char in word]  # 将字符串拆分为字符列表

def group_bits(bitc):
    bity = []  # 用于存储分组后的比特
    x = 0  # 当前索引
    for i in range((len(bitc)//bit_per_symbol)):  # 每 `bit_per_symbol` 个比特分为一组
        bity.append(bitc[x:x+bit_per_symbol])  # 添加一组比特
        x = x + bit_per_symbol  # 更新索引
    return bity  # 返回分组后的比特列表

def channel(signal, SNRdb, ouput_power=False):
    signal_power = np.mean(abs(signal**2))  # 计算信号功率
    sigma2 = signal_power * 10**(-SNRdb/10)  # 根据 SNR 计算噪声功率
    if ouput_power:
        print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))  # 输出信号和噪声功率
    noise = np.sqrt(sigma2/2) * (np.random.randn(*signal.shape)+1j*np.random.randn(*signal.shape))  # 生成复高斯噪声
    return signal + noise  # 返回加噪后的信号

def channel_Rayleigh(signal, SNRdb, ouput_power=False):
    shape = signal.shape  # 获取信号形状
    sigma = math.sqrt(1/2)  # Rayleigh 分布标准差
    H = np.random.normal(0.0, sigma , size=[1]) + 1j*np.random.normal(0.0, sigma, size=[1])  # 生成 Rayleigh 渠道系数
    Tx_sig = signal * H  # 信号经过 Rayleigh 渠道
    Rx_sig = channel(Tx_sig, SNRdb, ouput_power=False)  # 通过高斯信道
    Rx_sig = Rx_sig / H  # 信道均衡
    return Rx_sig  # 返回均衡后的信号

def channel_Rician(signal, SNRdb, ouput_power=False, K=1):
    shape = signal.shape  # 获取信号形状
    mean = math.sqrt(K / (K + 1))  # Rician 渠道的直达分量均值
    std = math.sqrt(1 / (K + 1))  # Rician 渠道的绕射分量标准差
    H = np.random.normal(mean, std , size=[1]) + 1j*np.random.normal(mean, std, size=[1])  # 生成 Rician 渠道系数
    Tx_sig = signal * H  # 信号经过 Rician 渠道
    Rx_sig = channel(Tx_sig, SNRdb, ouput_power=False)  # 通过高斯信道
    Rx_sig = Rx_sig / H  # 信道均衡
    return Rx_sig  # 返回均衡后的信号

def Demapping(QAM):
    constellation = np.array([x for x in demapping_table.keys()])  # 获取可能的星座点
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))  # 计算接收点与星座点的距离
    const_index = dists.argmin(axis=1)  # 找到最近的星座点索引
    hardDecision = constellation[const_index]  # 获取最近的星座点
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision  # 返回解映射的比特组和星座点

# def transmit(data, SNRdb, bits_per_digit):
def transmit(data):
    TX_signal = data[:].cpu().numpy().flatten()  # 转换数据为 NumPy 数组并展平
    Tx_data_binary = []  # 用于存储二进制数据

    # # 将数据转换为二进制字符串
    # for i in TX_signal:
    #     Tx_data_binary.append('{0:b}'.format(i).zfill(bits_per_digit))  # 转换为二进制并填充到 `bits_per_digit` 位

    Tx_data = []  # 存储比特
    Tx_data_ready = []
    for i in Tx_data_binary:
        Tx_data.append(split(i))  # 将二进制字符串拆分为比特
    img_for_trans1 = np.vstack(Tx_data)  # 将比特转为 NumPy 数组
    for i in img_for_trans1:
        # for j in range(bits_per_digit):
        #     Tx_data_ready.append(int(i[j]))  # 转换为整型比特
        Tx_data_ready.append(int(i))  # 转换为整型比特
    Tx_data_ready = np.array(Tx_data_ready)  # 转换为 NumPy 数组

    ori_len = len(Tx_data_ready)  # 原始比特长度
    padding_len = ori_len

    if ori_len % 4 != 0:  # 如果比特长度不是 4 的倍数，进行填充
        padding_len = ori_len + (bit_per_symbol - (ori_len % bit_per_symbol))

    Whole_tx_data = np.zeros(padding_len, dtype=int)  # 创建填充后的比特数组
    Whole_tx_data[:ori_len] = Tx_data_ready  # 填充原始比特

    bit_group = group_bits(Whole_tx_data)  # 将比特分组
    bit_group = np.array(bit_group)

    # 将比特映射到 QAM 符号
    QAM_symbols = []
    for bits in bit_group:
        symbol = mapping_table[tuple(bits)]  # 通过映射表映射符号
        QAM_symbols.append(symbol)
    QAM_symbols = np.array(QAM_symbols)

    # Rx_symbols = channel(QAM_symbols, SNRdb)  # 通过高斯信道
    Rx_symbols = channel(QAM_symbols)  # 通过高斯信道
    Rx_bits, hardDecision = Demapping(Rx_symbols)  # 解映射得到接收比特和星座点

    # 重构接收比特
    data_rea = []
    Rx_long = Rx_bits.reshape(-1,)[0:ori_len]
    k = 0
    # for i in range(Rx_long.shape[0]//bits_per_digit):
    #     data_rea.append(Rx_long[k:k+bits_per_digit])
    #     k += bits_per_digit
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
    # for i in data_fin:
    #     data_dec.append(i[0:bits_per_digit])
    data_dec = np.array(data_dec)
    data_back = []
    for i in range(len(Tx_data_binary)):
        data_back.append(int(data_dec[i],2))
    data_back = np.array(data_back)
    return data_back  # 返回重构后的数据

def power_norm_batchwise(signal, power=1):
    batchsize , num_elements = signal.shape[0], len(signal[0].flatten())  # 获取批量大小和元素数量
    num_complex = num_elements // 2  # 计算复数数量
    signal_shape = signal.shape  # 保存信号形状
    signal = signal.view(batchsize, num_complex, 2)  # 将信号重塑为复数形式
    signal_power = torch.sum((signal[:,:,0]**2 + signal[:,:,1]**2), dim=-1) / num_complex  # 计算信号功率

    signal = signal * math.sqrt(power) / torch.sqrt(signal_power.unsqueeze(-1).unsqueeze(-1))  # 归一化信号功率
    signal = signal.view(signal_shape)  # 恢复信号形状
    return signal  # 返回归一化后的信号
