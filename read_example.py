import numpy as np
import os
from os.path import dirname, abspath, join
import pickle
from scipy.signal import butter, sosfiltfilt
import pywt
import matplotlib.pyplot as plt
from datetime import datetime

DATA_TYPE = "Calibration_Test"
RESOURCE_PATH = dirname(abspath(__file__))
DATA_FOLDER = os.path.join(RESOURCE_PATH, 'data', DATA_TYPE)
block_files = [
    os.path.join(DATA_FOLDER, fname)
    for fname in sorted(os.listdir(DATA_FOLDER))
    if fname.endswith(".pkl")
]

# 1 读取raw EEG并归类
def load_eeg_data(window_size=1250, step_size=250):
    """
    加窗切分EEG数据
    Args:
        window_size: int, 每个滑动窗口的长度（采样点）
        step_size: int, 每次滑动的步长（采样点）
    Returns:
        X: ndArray, shape = [num_samples, num_channels, window_size]
        y: ndArray, shape = [num_samples]
    """
    sub_dict = {id: [] for id in range(1, 24)}
    for block_file in block_files:
        with open(block_file, "rb") as f:
            _data: np.ndarray = pickle.load(f)
        _sig = _data[:-1, :]
        triggers = _data[-1, :]
        start_indices = np.where((triggers >= 1) & (triggers <= 23))[0]
        end_indices = np.where(triggers == 241)[0]
        for start, end in zip(start_indices, end_indices):
            sub_id = int(triggers[start])
            sub_dict[sub_id].append(_sig[:, start : end + 1])

    X_list = []
    y_list = []
    for id in sub_dict.keys():
        for trial in sub_dict[id]:
            num_channels, total_len = trial.shape
            # 滑窗切分
            for s in range(0, total_len - window_size + 1, step_size):
                window = trial[:, s : s + window_size]  # shape: [channels, window_size]
                if window.shape[1] == window_size:
                    X_list.append(window)
                    y_list.append(id)  # 下标从0开始

    X = np.stack(X_list)  # [num_samples, num_channels, window_size]
    y = np.array(y_list)
    print("subid", np.unique(y))
    # X = torch.tensor(X, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.long)
    print(f"Step1: load and slide aug: X.shape={X.shape}, y.shape={y.shape}")
    return X, y

# 2 滤波处理
def bandpass_filter_eeg(eeg_data: np.ndarray,
                        sfreq: float = 250,
                        lowcut: float = 4.0,
                        highcut: float = 40.0,
                        order: int = 4) -> np.ndarray:
    """
    对形状为 (N, C, T) 的 EEG 数据做 4–30 Hz 带通滤波。

    参数:
        eeg_data: np.ndarray, shape = (N, C, T)，原始 EEG 时域数据
        sfreq: float，采样频率 (Hz)
        lowcut: float，低截止频率 (默认为 4 Hz)
        highcut: float，高截止频率 (默认为 30 Hz)
        order: int，Butterworth 滤波器阶数 (默认为 4)

    返回:
        filtered: np.ndarray, shape = (N, C, T)，滤波后 EEG 数据
    """
    # 1. 计算奈奎斯特频率
    nyquist = sfreq / 2.0

    # 2. 归一化截止频率 (归一化到 [0, 1]，1 对应奈奎斯特频率)
    low = lowcut / nyquist
    high = highcut / nyquist

    # 3. 生成二阶节（Second-Order Sections）形式的带通 Butterworth 滤波器
    sos = butter(order, [low, high], btype='band', output='sos')

    # 4. 将 (N, C, T) 的数据先 reshape 为 (N*C, T)，以便一次性滤波
    N, C, T = eeg_data.shape
    flat_data = eeg_data.reshape(-1, T)  # 变为 (N*C, T)

    # 5. 对每条时间序列做零相位双向滤波
    #    sosfiltfilt 会在时轴上 (axis=1) 对每一行做前向+反向滤波，消除相位失真
    filtered_flat = sosfiltfilt(sos, flat_data, axis=1)

    # 6. reshape 回原始形状 (N, C, T)
    filtered = filtered_flat.reshape(N, C, T)
    print(f"Step2: bandfilter: X.shape={filtered.shape}, filterbands={lowcut}-{highcut}")
    return filtered

# 3 进行CWT处理，提取时频特征

def cwt_eeg_data(eeg_data: np.ndarray,
                 sfreq: float,
                 freqs: np.ndarray,
                 wavelet: str = 'cmor1.5-1.0') -> np.ndarray:
    """
    对形状为 (N, C, T) 的 EEG 数据做连续小波变换（CWT）。

    参数:
        eeg_data: np.ndarray, shape = (N, C, T)，原始 EEG 时域数据
        sfreq: float，采样频率 (Hz)
        freqs: np.ndarray, 一维数组，想要分析的频率列表 (Hz)
        wavelet: str，小波名称（默认为复 Morlet 小波 'cmor1.5-1.0'）

    返回:
        coeffs: np.ndarray, shape = (N, C, F, T)，CWT 系数
                其中 F = len(freqs)，系数为复数，axis=2 对应不同频率，axis=3 对应时间点
    """
    # 1. 计算采样周期
    dt = 1.0 / sfreq

    # 2. 由所给频率计算对应的尺度
    #    对于给定小波，尺度 a 与频率 f 的近似关系 a = center_freq / (f * dt)
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq / (freqs * dt)

    N, C, T = eeg_data.shape
    F = len(scales)
    coeffs = np.zeros((N, C, F, T), dtype=np.complex64)

    # 3. 将前两维打平，批量处理每条一维信号
    flat_data = eeg_data.reshape(-1, T)  # 变为 (N*C, T)

    for idx, sig in enumerate(flat_data):
        # sig: shape (T,)
        cwt_matrix, _ = pywt.cwt(sig, scales, wavelet, sampling_period=dt)
        # cwt_matrix: shape (F, T)
        n = idx // C
        c = idx % C
        coeffs[n, c, :, :] = cwt_matrix

    print(f"Step3: cwt: X_cwt.shape={coeffs.shape}")
    return coeffs

def get_cwt_data():
    X, y = load_eeg_data(window_size=1250, step_size=250)
    filtered_X = bandpass_filter_eeg(X, sfreq=250, lowcut=4.0, highcut=40.0, order=4)
    freqs = np.linspace(4.0, 40.0, num=50)
    X_cwt = cwt_eeg_data(filtered_X, sfreq=250, freqs=freqs, wavelet='cmor1.5-1.0')
    return X_cwt, y

# ========================================================可视化==========================================================
# def plot_cwt_trials(X_cwt: np.ndarray,
#                     freqs: np.ndarray,
#                     sfreq: float,
#                     trial_list: list):
#     """
#     在一个图（figure）里，最多显示 5 个 trial 的时频图。若 trial 数量超过 5，只取前 5；
#     若少于 5，则剩余子图留空，并在最下排标记 x 轴数值。

#     参数:
#         X_cwt: np.ndarray, shape = (N, C, F, T)，CWT 系数（复数）
#                N = trial 数量, C = 通道数, F = 频率点数, T = 时间点数
#         freqs: np.ndarray, shape = (F,)，对应每个尺度的频率 (Hz)
#         sfreq: float，采样率 (Hz)
#         trial_list: list of int，要显示的 trial 索引列表（0-based）
#     """
#     N, C, F, T = X_cwt.shape
#     times = np.arange(T) / sfreq        # 时间轴 (秒)
#     max_display = 5                      # 最多显示 5 个 trial

#     # 保留合法索引，并截取前 5 个
#     sel = [t for t in trial_list if 0 <= t < N][:max_display]
#     num_sel = len(sel)

#     # 创建 5×C 的子图网格
#     fig, axes = plt.subplots(nrows=max_display, ncols=C,
#                              figsize=(4 * C, 2.5 * max_display),
#                              sharex=True, sharey=True)

#     # 保证 axes 为 2D 数组
#     if max_display == 1 and C == 1:
#         axes = np.array([[axes]])
#     elif max_display == 1:
#         axes = axes[np.newaxis, :]
#     elif C == 1:
#         axes = axes[:, np.newaxis]

#     # 遍历每一行（对应一个 trial slot）
#     for row in range(max_display):
#         if row < num_sel:
#             trial_idx = sel[row]
#             coeffs_trial = X_cwt[trial_idx]  # shape = (C, F, T)

#             for ch in range(C):
#                 ax = axes[row, ch]
#                 magnitude = np.abs(coeffs_trial[ch])  # shape = (F, T)

#                 im = ax.imshow(
#                     magnitude,
#                     aspect='auto',
#                     origin='lower',
#                     extent=[times[0], times[-1], freqs[0], freqs[-1]]
#                 )

#                 # 左边第一列显示 trial 编号
#                 if ch == 0:
#                     ax.set_ylabel(f"Trial {trial_idx}", fontsize=10)

#                 # 最上排显示通道编号
#                 if row == 0:
#                     ax.set_title(f"Ch {ch}", fontsize=10)

#                 # 最下排显示时间轴标签并标记刻度
#                 if row == max_display - 1:
#                     ax.set_xlabel("Time (s)", fontsize=9)
#                     # 设置 x 轴刻度为 5 个等间隔值
#                     xticks = np.linspace(times[0], times[-1], num=5)
#                     ax.set_xticks(xticks)
#                     ax.set_xticklabels([f"{x:.2f}" for x in xticks])
#         else:
#             # 如果没有第 row 个 trial，就隐藏所有对应子图
#             for ch in range(C):
#                 axes[row, ch].axis('off')

#     fig.suptitle("Selected Trials: CWT Time–Frequency", fontsize=14, y=0.92)
#     fig.tight_layout(rect=[0, 0, 1, 0.90])

#     # 保存图片，文件名带时间戳
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     fig.savefig(f"figs/cwt_time_frequency_{ts}.png", dpi=300, bbox_inches='tight')

#     plt.show()

def plot_cwt_trials(X_cwt: np.ndarray, trial_list: list):
    """
    显示任意数量的 trial，每行显示一个 trial 的所有通道（共 C 列），
    每个单元格为某通道的 CWT 时频图（复数模值）。

    参数:
        X_cwt: np.ndarray, shape = (N, C, F, T)
        trial_list: list of int，trial 索引（0-based）
    """
    N, C, F, T = X_cwt.shape

    # 保留合法 trial 索引
    sel = [t for t in trial_list if 0 <= t < N]
    num_sel = len(sel)

    # 创建 num_sel × C 的子图网格
    fig, axes = plt.subplots(nrows=num_sel, ncols=C,
                             figsize=(4 * C, 2.5 * num_sel),
                             sharex=False, sharey=False)

    # 保证 axes 为 2D 数组
    if num_sel == 1 and C == 1:
        axes = np.array([[axes]])
    elif num_sel == 1:
        axes = axes[np.newaxis, :]
    elif C == 1:
        axes = axes[:, np.newaxis]

    for row in range(num_sel):
        trial_idx = sel[row]
        coeffs_trial = X_cwt[trial_idx]  # shape = (C, F, T)

        for ch in range(C):
            ax = axes[row, ch]
            magnitude = np.abs(coeffs_trial[ch])  # shape = (F, T)

            im = ax.imshow(
                magnitude,
                aspect='auto',
                origin='lower'
            )

            # 第一列显示 trial 编号
            if ch == 0:
                ax.set_ylabel(f"Trial {trial_idx}", fontsize=10)

            # 最上排显示通道编号
            if row == 0:
                ax.set_title(f"Ch {ch}", fontsize=10)

            # 最下排显示时间索引
            if row == num_sel - 1:
                ax.set_xlabel("Time Index", fontsize=9)
                ax.set_xticks(np.linspace(0, T - 1, 5, dtype=int))
                ax.set_xticklabels([str(int(x)) for x in np.linspace(0, T - 1, 5)])

            # 第一列显示频率索引
            if ch == 0:
                ax.set_yticks(np.linspace(0, F - 1, 5, dtype=int))
                ax.set_yticklabels([str(int(y)) for y in np.linspace(0, F - 1, 5)])

    fig.suptitle("Selected Trials: CWT Magnitude (Index-Based)", fontsize=14, y=0.92)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # 保存图片
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f"figs/cwt_time_frequency_indices_{ts}.png", dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    # # 功能1：生成数据。除非需要改变滑窗，获修改bandfilter、cwt参数，否则直接读取
    # X_cwt, y = get_cwt_data()
    # np.savez('data/cwt_data.npz', X_cwt=X_cwt, y=y)
    
    # # 功能2：绘制时频图
    # data = np.load('data/cwt_data.npz')
    # X_cwt = data['X_cwt']
    # y = data['y']
    
    # trials_to_plot_base = [0, 10, 20, 30, 39]
    # trials_to_plot = [x + 80 for x in trials_to_plot_base]
    # freqs = np.linspace(4.0, 40.0, num=50)
    # sfreq = 250
    # plot_cwt_trials(X_cwt, trials_to_plot)
    
    # 功能3：对比真实数据与生成数据t-sne
    from tsne import compare_generated_with_real_tsne
    generated_data = np.load('generated_data/generated_eeg_20trials.npy')
    real_data, _ = load_eeg_data()
    real_data_sub1 = real_data[:20, :, :]
    compare_generated_with_real_tsne(generated_data, real_data_sub1)