import numpy as np
import os
from os.path import dirname, abspath, join
import pickle
from scipy.signal import butter, sosfiltfilt, stft
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
    加窗切分EEG数据：先把同一个id的所有trial按时间拼接，再整体滑窗

    Args:
        window_size: int, 每个滑动窗口的长度（采样点）
        step_size: int, 每次滑动的步长（采样点）
    Returns:
        X: np.ndarray, shape = [num_samples, num_channels, window_size]
        y: np.ndarray, shape = [num_samples]
    """
    # 按被试ID收集所有trial
    sub_dict = {sid: [] for sid in range(1, 24)}
    for block_file in block_files:
        with open(block_file, "rb") as f:
            _data: np.ndarray = pickle.load(f)
        _sig = _data[:-1, :]       # EEG信号  (channels, time)
        triggers = _data[-1, :]    # 触发标记
        start_indices = np.where((triggers >= 1) & (triggers <= 23))[0]
        end_indices = np.where(triggers == 241)[0]
        for start, end in zip(start_indices, end_indices):
            sid = int(triggers[start])
            sub_dict[sid].append(_sig[:, start:end+1])

    X_list, y_list = [], []
    # 对每个ID，将所有trial拼接后整体滑窗
    for sid, trials in sub_dict.items():
        if not trials:
            continue
        # 将该ID所有trial按时间轴拼接
        concat_sig = np.concatenate(trials, axis=1)  # (channels, total_length)
        total_len = concat_sig.shape[1]
        # 滑窗切分
        for s in range(0, total_len - window_size + 1, step_size):
            window = concat_sig[:, s:s + window_size]
            # 保证窗口长度
            if window.shape[1] == window_size:
                X_list.append(window)
                y_list.append(sid)

    X = np.stack(X_list, axis=0)  # (num_samples, channels, window_size)
    y = np.array(y_list, dtype=int)
    print("subid", np.unique(y))
    print(f"Step1: concat-by-id and slide window: X.shape={X.shape}, y.shape={y.shape}")
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

# 3 stft时频分析
def stft_eeg_data(
    eeg_data: np.ndarray,
    sfreq: float,
    nperseg: int = 250,
    noverlap: int = 125,
    window: str = 'hann'
):
    """
    对形状为 (N, C, T) 的 EEG 数据做短时傅里叶变换（STFT），
    并对时频系数做归一化、最后取幅值。

    返回:
        f: np.ndarray, shape = (F,), 频率轴
        t: np.ndarray, shape = (T_w,), 时间帧中心点（秒）
        coeffs: np.ndarray, shape = (N, C, F, T_w)，归一化后的幅值（float32）
    """
    if noverlap is None:
        noverlap = nperseg // 2

    N, C, T = eeg_data.shape

    # 用第一条通道确定输出维度
    f, t, Z0 = stft(
        eeg_data[0, 0],
        fs=sfreq,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )
    F, T_w = Z0.shape

    # 分配复数系数数组
    coeffs = np.zeros((N, C, F, T_w), dtype=np.complex64)

    # 逐 trial、逐通道做 STFT
    for n in range(N):
        for c in range(C):
            _, _, Z = stft(
                eeg_data[n, c],
                fs=sfreq,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap
            )
            coeffs[n, c] = Z

    # 计算幅值与相位
    mag   = np.abs(coeffs)                  # 幅值
    phase = np.angle(coeffs)                # 相位信息

    # 归一化
    abs_max   = np.max(mag,   axis=(2,3), keepdims=True)
    mag_norm  = (mag / (abs_max + 1e-8)).astype(np.float32)

    # 保存 abs_max 和 phase 到磁盘
    np.save("abs_max.npy",  abs_max)
    np.save("phase.npy",    phase)

    print(f"Step3: stft + normalize + magnitude: coeffs.shape={mag_norm.shape}")
    return f, t, mag_norm

def get_tf_data():
    X, y = load_eeg_data(window_size=1250, step_size=1250)
    # filtered_X = bandpass_filter_eeg(X, sfreq=250, lowcut=4.0, highcut=40.0, order=4)

    # 做STFT
    f, t, X_tf = stft_eeg_data(
        X,
        sfreq=250,
        nperseg=126,
        noverlap=106,
        window='hann',
    )

    return X_tf, y

# ========================================================可视化==========================================================
def plot_tf_trials(X_tf: np.ndarray, trial_list: list):
    """
    显示任意数量的 trial，每行显示一个 trial 的所有通道（共 C 列），
    每个单元格为某通道的时频图（复数模值）。

    参数:
        X_tf: np.ndarray, shape = (N, C, F, T)
        trial_list: list of int，trial 索引（0-based）
    """
    N, C, F, T = X_tf.shape

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
        coeffs_trial = X_tf[trial_idx]  # shape = (C, F, T)

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

    fig.suptitle("Selected Trials: TF Magnitude (Index-Based)", fontsize=14, y=0.92)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # 保存图片
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f"figs/time_frequency_indices_{ts}.png", dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    # # 功能1：生成数据。除非需要改变滑窗，获修改bandfilter、tf参数，否则直接读取
    # X_tf, y = get_tf_data()
    # np.savez('data/tf_data.npz', X_tf=X_tf, y=y)

    # # 功能2：绘制时频图
    # data = np.load('data/tf_data.npz')
    # X_tf = data['X_tf']
    # y = data['y']

    # trials_to_plot_base = [0, 10, 20, 30, 39]
    # trials_to_plot = [x + 0 for x in trials_to_plot_base]

    # # X_tf = np.load('generated_data/Tosato_generated_eeg_stft.npy')
    # # trials_to_plot = [0, 1, 2, 3, 4]

    # plot_tf_trials(X_tf, trials_to_plot)

    # 功能3：对比真实数据与生成数据t-sne
    from tsne import compare_generated_with_real_tsne
    # 对比raw EEG
    subid = 3
    generated_data = np.load(f'generated_data/generated_eeg_40trials_sub{subid}.npy')
    real_data, _ = load_eeg_data()

    # # 对比tf归一化幅值特征
    # generated_data = np.load('generated_data/generated_tf.npy')
    # data = np.load('data/tf_data.npz')
    # real_data = data['X_tf']

    real_data_sub1 = real_data[(subid - 1) * 40: subid * 40, :, :]
    compare_generated_with_real_tsne(generated_data, real_data_sub1)