import torch
import numpy as np
from scipy.signal import istft
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import trange
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def istft_eeg_data(
    stft_coeffs:       np.ndarray,
    subid:             int,
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
    N_orig = abs_max.shape[0]
    trials_per_sub = N_orig // 23
    start = (subid - 1) * trials_per_sub
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


@torch.no_grad()  # 1. 关闭梯度
def generate_tf_from_diffusion(
    model_dir: str,
    num_trials: int,
    seed: int = 42,
    class_value: int = 3,
    num_inference_steps: int = 500  # 如果你想少走几步
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 UNet 和 Scheduler，使用 BFloat16
    unet = UNet2DModel.from_pretrained(
        model_dir,
        # torch_dtype=torch.bfloat16  # 4. 半精度
    ).to(device)
    scheduler = DDPMScheduler.from_pretrained(model_dir)

    # 2. 切到 eval 模式，启用 attention slicing
    unet.eval()                                # 2. eval 模式

    # 3. (可选) 少走采样步
    scheduler.set_timesteps(num_inference_steps)

    # 4. 准备 class_labels
    eval_labels = torch.full(
        (num_trials,),
        fill_value=class_value,
        device=device,
        dtype=torch.int64
    )

    # 5. 计算初始化噪声的 shape
    sample_size = unet.config.sample_size
    if isinstance(sample_size, (list, tuple)):
        H, W = sample_size
    else:
        H = W = sample_size
    shape = (num_trials, unet.config.in_channels, H, W)

    # 6. 随机噪声初始化，并统一类型
    generator = torch.Generator(device=device).manual_seed(seed)
    imgs = torch.randn(shape, device=device, generator=generator, dtype=unet.dtype)

    # 7. 反向扩散循环
    for t in trange(len(scheduler.timesteps), desc="Generating EEG"):
        step_t = scheduler.timesteps[t]
        # 模型前向（BFloat16 下也能跑）
        noise_pred = unet(
            imgs,
            torch.full((num_trials,), step_t, device=device),
            class_labels=eval_labels,
            return_dict=False
        )[0]
        # 更新
        step = scheduler.step(noise_pred, step_t, imgs)
        imgs = step.prev_sample

    # 8. 转 NumPy (N, C, H, W) 即 (trial, channel, freq, time)
    return imgs.cpu().numpy()

# ----------- 主生成函数 -----------
def generate_eeg_from_tf(tf_np, subid) -> np.ndarray:
    # 构造复数系数（假设模值 + 相位为0）
    X_cwt_fake = tf_np.astype(np.complex64)
    # 逆向 ISTFT，重建回 (N, 5, 1250)
    X_eeg = istft_eeg_data(
        X_cwt_fake,
        subid, # 使用start:start+generate_num的原始数据的abs_max和phase来生成新数据
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
    parser.add_argument("--num_trials", type=int, default=40, help="要生成的 trial 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--subid", type=int, default=1, help="1-23")

    args = parser.parse_args()

    tf_np = generate_tf_from_diffusion(
        model_dir=args.model_dir,
        num_trials=args.num_trials,
        seed=args.seed,
        class_value=args.subid  # 固定类别索引，可按需修改
    )

    # 保存单一被试数据
    # eeg_data = generate_eeg_from_tf(tf_np, args.subid)
    # np.save(f"generated_data/generated_eeg_{args.num_trials}trials_sub{args.subid}.npy", eeg_data)
    # print(f"已保存到: generated_eeg_{args.num_trials}trials_sub{args.subid}.npy")




    eeg_dict = {}
    for sub in trange(1, 24):
        # 依次为每个子 ID 生成 EEG 时–频数据
        tf_np = generate_tf_from_diffusion(
            model_dir=args.model_dir,
            num_trials=args.num_trials,
            seed=args.seed,
            class_value=sub  # 固定类别索引，可按需修改
        )
        eeg_data = generate_eeg_from_tf(tf_np, sub)
        eeg_dict[sub] = eeg_data

    # 把字典保存到磁盘
    output_path = "generated_data/gen_data_dict.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(eeg_dict, f)

    print(f"已保存 23 个子 ID 的生成数据到: {output_path}")
