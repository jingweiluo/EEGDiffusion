import torch
import numpy as np
import pywt
from diffusers import DDPMPipeline
from scipy.ndimage import zoom
from typing import Tuple

# ----------- 参数设置 -----------
SFREQ = 250
FREQ_LOW = 4.0
FREQ_HIGH = 40.0
NUM_FREQS = 50
WAVELET = 'cmor1.5-1.0'
C = 5        # 通道数
T = 1250     # 目标时间长度
F = NUM_FREQS


# ----------- 生成频率数组 -----------
def get_freqs():
    return np.linspace(FREQ_LOW, FREQ_HIGH, num=NUM_FREQS)


# ----------- inverse CWT -----------
def icwt_eeg_data(cwt_data: np.ndarray,
                  sfreq: float,
                  freqs: np.ndarray,
                  wavelet: str = 'cmor1.5-1.0') -> np.ndarray:
    dt = 1.0 / sfreq
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq / (freqs * dt)

    N, C, F, T = cwt_data.shape
    eeg_reconstructed = np.zeros((N, C, T), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            coeff = cwt_data[n, c, :, :]  # shape: (F, T)
            rec = pywt.icwt(coeff, scales, wavelet, sampling_period=dt)
            eeg_reconstructed[n, c, :] = np.real(rec[:T])
    return eeg_reconstructed


# ----------- 主生成函数 -----------
def generate_eeg_from_diffusion(model_dir: str, num_trials: int, seed: int = 42) -> np.ndarray:
    # 1. 加载最佳模型
    pipeline = DDPMPipeline.from_pretrained(f"{model_dir}/best_model")

    # 2. 从噪声中采样生成图像
    images = pipeline(
        batch_size=num_trials,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np.array"
    ).images  # shape = (N, H, W, C)

    print("生成图像原始 shape:", images.shape)  # e.g., (N, H, W, 1)

    # 3. 转换为 (N, C, F, T)，即 (trial, channel, frequency, time)
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).numpy()  # (N, C, H, W)

    # 4. resize 到 (N, C, F, T)
    N, C_img, H, W = images_tensor.shape
    if C_img != C:
        raise ValueError(f"模型生成图像的通道数为 {C_img}，与期望通道数 {C} 不一致，请检查模型输入或训练过程")

    # 假设 H = 当前频率轴长度，W = 当前时间点数
    images_resized = zoom(images_tensor, (1, 1, F / H, T / W))  # 线性插值缩放

    print("调整大小后的图像 shape:", images_resized.shape)  # (N, C, F, T)

    # 5. 构造复数系数（假设模值 + 相位为0）
    X_cwt_fake = images_resized.astype(np.complex64)

    # 6. inverse CWT 恢复 EEG
    freqs = get_freqs()
    X_eeg = icwt_eeg_data(
        cwt_data=X_cwt_fake,
        sfreq=SFREQ,
        freqs=freqs,
        wavelet=WAVELET
    )

    print("生成的 EEG 时域数据 shape:", X_eeg.shape)  # (N, C, T)
    return X_eeg


# ----------- 入口示例 -----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="results/best_model", required=True, help="模型保存目录")
    parser.add_argument("--num_trials", type=int, default=20, help="要生成的 trial 数量")
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
