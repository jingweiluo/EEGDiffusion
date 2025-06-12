import torch
import numpy as np
from diffusers import DDPMPipeline
from scipy.signal import istft
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def istft_eeg_data(
    stft_coeffs:       np.ndarray,
    start:             int,
    sfreq:             float,
    nperseg:           int   = 126,
    noverlap:          int   = 106,
    window:            str   = 'hann',
    original_length:   int   = None,
    abs_max_path:      str   = "abs_max.npy",
    phase_path:        str   = "phase.npy"
) -> np.ndarray:
    """
    将形状为 (N_gen, C, F, T_w) 的归一化幅值 stft_coeffs
    和正向保存的 abs_max、phase，逆变换回时域信号。

    返回:
        X_rec: shape = (N_gen, C, original_length)
    """
    # 1. 载入正向保存的 abs_max 和 phase
    abs_max = np.load(abs_max_path)  # shape (N_orig, C, 1, 1)
    phase   = np.load(phase_path)    # shape (N_orig, C, F, T_w)

    # 2. 按实际生成数量切片
    N_gen, C, F, T_w = stft_coeffs.shape
    abs_max = abs_max[start:start+N_gen, :, :, :]    # -> (N_gen, C, 1, 1)
    phase   = phase[start:start+N_gen,   :, :, :]    # -> (N_gen, C, F, T_w)

    # 3. 反归一化：恢复真实幅值
    mag = stft_coeffs * (abs_max + 1e-8)   # shape (N_gen, C, F, T_w)

    # 4. 准备好输出数组长度
    if original_length is None:
        original_length = nperseg + (T_w - 1) * (nperseg - noverlap)
    X_rec = np.zeros((N_gen, C, original_length), dtype=np.float32)

    # 5. 逐条 trial/通道做逆 STFT
    for n in range(N_gen):
        for c in range(C):
            # 5.1 构造复数谱：幅值 * exp(j * phase)
            Zxx = mag[n, c] * np.exp(1j * phase[n, c])

            # 5.2 逆 STFT，boundary='zeros' 以满足 NOLA 条件
            _, x_rec = istft(
                Zxx,
                fs       = sfreq,
                window   = window,
                nperseg  = nperseg,
                noverlap = noverlap,
                boundary = 'zeros'
            )

            if x_rec is None:
                raise RuntimeError(f"ISTFT failed at trial={n}, channel={c}")

            # 5.3 截断或补零到 original_length
            L = x_rec.shape[-1]
            if L >= original_length:
                X_rec[n, c, :] = x_rec[:original_length]
            else:
                X_rec[n, c, :L] = x_rec

    # 6. 返回重建的时域 EEG
    return X_rec

# ----------- 主生成函数 -----------
def generate_eeg_from_diffusion(model_dir: str, num_trials: int, seed: int = 42) -> np.ndarray:
    # 1. 加载最佳模型
    pipeline = DDPMPipeline.from_pretrained(f"{model_dir}").to(device)

    # 2. 从噪声中采样生成图像
    images = pipeline(
        batch_size=num_trials,
        generator=torch.Generator(device=device).manual_seed(seed),
        output_type="np.array"
    ).images  # shape = (N, H, W, C)

    print("生成图像原始 shape:", images.shape)  # e.g., (N, H, W, 1)

    # 3. 转换为 (N, C, F, T)，即 (trial, channel, frequency, time)
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).numpy()  # (N, C, H, W)

    # 4. 构造复数系数（假设模值 + 相位为0）
    X_cwt_fake = images_tensor.astype(np.complex64)
    # 逆向 ISTFT，重建回 (N, 5, 1250)
    X_eeg = istft_eeg_data(
        X_cwt_fake,
        start=300, # 使用start:start+generate_num的原始数据的abs_max和phase来生成新数据
        sfreq=250,
        nperseg=126,
        noverlap=106,
        window='hann',
        original_length=1250
    )

    print("生成的 EEG 时域数据 shape:", X_eeg.shape)  # (N, C, T)
    return X_eeg


# ----------- 入口示例 -----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="results/best_model", help="模型保存目录")
    parser.add_argument("--num_trials", type=int, default=200, help="要生成的 trial 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    eeg_data = generate_eeg_from_diffusion(
        model_dir=args.model_dir,
        num_trials=args.num_trials,
        seed=args.seed
    )

    # 可选择保存
    np.save(f"generated_data/generated_eeg_{args.num_trials}trials.npy", eeg_data)
    print(f"已保存到: {args.model_dir}/generated_eeg_{args.num_trials}trials.npy")
