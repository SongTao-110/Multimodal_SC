import os
import numpy as np
import matplotlib.pyplot as plt


def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + 'is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + 'create successfully!')

def smooth_curve(data, weight=0.99, method='ema', window_size=100):
    """
    平滑曲线函数，支持指数移动平均 (EMA) 和滑动窗口平均 (Moving Average)。

    参数:
    - data: 原始数据 (list or numpy array)。
    - weight: 平滑系数 (float, 0-1之间, EMA 模式下使用)。
    - method: 平滑方法 ('ema' 或 'moving_avg')。
    - window_size: 滑动窗口大小 (int, Moving Average 模式下使用)。

    返回:
    - 平滑后的数据 (numpy array)。
    """
    if method == 'ema':
        smoothed = []
        last = data[0]
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return np.array(smoothed)
    elif method == 'moving_avg':
        # 滑动窗口平均
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    else:
        return np.array(data)


def plot_curve(x, y, title='', ylabel='', figure_file='',
               avg_color='blue', avg_linewidth=2.5, figsize=(10, 6),
               grid=True, smooth_weight=0.99, method='ema', window_size=100):
    """
    绘制奖励曲线，支持不同平滑方法。
    """
    # 平滑奖励曲线
    smoothed_y = smooth_curve(y, weight=smooth_weight, method=method, window_size=window_size)

    # 确保 x 和 smoothed_y 的长度一致
    if len(x) > len(smoothed_y):
        x = x[-len(smoothed_y):]
    else:
        smoothed_y = smoothed_y[-len(x):]

    # 累积平均线
    cumulative_avg = np.cumsum(y) / np.arange(1, len(y) + 1)

    # 创建图表
    plt.figure(figsize=figsize)

    # 绘制平滑奖励曲线
    plt.plot(x, smoothed_y, color=avg_color, linewidth=avg_linewidth, label='Smoothed Curve')

    # 绘制累积平均线
    # plt.plot(x, cumulative_avg[len(cumulative_avg) - len(x):], color='orange', linestyle='--',
    #          label='Cumulative Avg')

    # 设置标题和坐标轴标签
    plt.title(title, fontsize=16)
    plt.xlabel('Episodes (or Million Steps)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # 显示网格
    if grid:
        plt.grid(alpha=0.3, linestyle='--')

    # 添加图例
    plt.legend(fontsize=12)

    # 调整布局并保存图表
    plt.tight_layout()
    plt.savefig(figure_file, dpi=300)
    plt.close()