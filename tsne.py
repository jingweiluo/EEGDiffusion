import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def compare_generated_with_real_tsne(generated_data: np.ndarray, real_data: np.ndarray, title="t-SNE Comparison"):
    """
    读取生成数据，与给定真实 EEG 数据一起进行 t-SNE 可视化。

    参数:
        generated_path: str，保存生成数据的 .npy 文件路径
        real_data: np.ndarray，真实 EEG 数据，shape=(N, C, T)
        title: str，图标题
    """
    # 1. 加载生成的数据
    print(f"生成数据 shape: {generated_data.shape}")
    print(f"真实数据 shape: {real_data.shape}")

    # 2. 样本数对齐（可选：取 min(N1, N2)）
    N = min(generated_data.shape[0], real_data.shape[0])
    generated_data = generated_data[:N]
    real_data = real_data[:N]

    # 3. 压缩为特征向量（示例：通道平均后展开时间轴）
    def flatten_eeg(X):  # shape (N, C, T) → (N, T)
        return X.mean(axis=1)  # or X.reshape(N, -1)

    gen_feat = flatten_eeg(generated_data)  # shape (N, T)
    real_feat = flatten_eeg(real_data)      # shape (N, T)

    # 4. 拼接并归一化
    all_feat = np.vstack([real_feat, gen_feat])  # shape (2N, T)
    all_feat = StandardScaler().fit_transform(all_feat)

    # 5. t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(all_feat)

    # 6. 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:N, 0], tsne_result[:N, 1], c='blue', label='Real EEG', alpha=0.6)
    plt.scatter(tsne_result[N:, 0], tsne_result[N:, 1], c='red', label='Generated EEG', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/tsne_eeg_comparison.png", dpi=300)
    plt.show()
