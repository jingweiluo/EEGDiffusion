import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from tsne import compare_generated_with_real_tsne
from read_example import load_eeg_data

# 1 对比某个sub的raw EEG和gen EEG的tsne
def compare_tsne(subid):
    generated_data = np.load(f'generated_data/generated_eeg_40trials_sub{subid}.npy')
    real_data, _ = load_eeg_data()

    per_sub_length = real_data.shape[0] // 23
    real_data_sub = real_data[(subid - 1) * per_sub_length: subid * per_sub_length, :, :]

    compare_generated_with_real_tsne(generated_data, real_data_sub)

    # gen1 = np.load(f'generated_data/generated_eeg_40trials_sub1.npy')
    # gen2 = np.load(f'generated_data/generated_eeg_40trials_sub2.npy')
    # gen3 = np.load(f'generated_data/generated_eeg_40trials_sub3.npy')
    # generated_data = np.concatenate([gen1, gen2, gen3], axis=0)
    # print('生成shape', generated_data.shape)

    # real_data, _ = load_eeg_data()

    # per_sub_length = real_data.shape[0] // 23
    # real_data_sub = real_data[:120, :, :]

    # compare_generated_with_real_tsne(generated_data, real_data_sub)



def compare_eeg_statistics(subid):
    """
    对比真实和生成的 EEG 数据统计特征。

    Args:
        real_data: np.ndarray, shape = (N, C, T)，真实数据
        gen_data:  np.ndarray, shape = (N, C, T)，生成数据

    Returns:
        real_df: pd.DataFrame, 每个 trial 的统计特征
        gen_df:  pd.DataFrame, 每个 trial 的统计特征
        agg_df:  pd.DataFrame, 真实/生成数据的整体统计特征对比
    """
    gen_data = np.load(f'generated_data/generated_eeg_40trials_sub{subid}.npy')
    real_data_all, _ = load_eeg_data()

    per_sub_length = real_data_all.shape[0] // 23
    real_data = real_data_all[(subid - 1) * per_sub_length: subid * per_sub_length, :, :]

    features = ['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']

    def trial_stats(arr):
        x = arr.flatten()
        return {
            'mean': x.mean(),
            'std': x.std(),
            'min': x.min(),
            'max': x.max(),
            'skewness': skew(x),
            'kurtosis': kurtosis(x)
        }

    # 1. 每个 trial 提取统计特征
    real_list = [{'trial': i, **trial_stats(real_data[i])} for i in range(real_data.shape[0])]
    gen_list  = [{'trial': i, **trial_stats(gen_data[i])}  for i in range(gen_data.shape[0])]

    real_df = pd.DataFrame(real_list)
    gen_df  = pd.DataFrame(gen_list)

    # 2. 聚合整体统计
    agg_records = []
    for label, df in [('real', real_df), ('gen', gen_df)]:
        agg = df[features].agg(['mean', 'std']).T.reset_index()
        agg.columns = ['feature', 'agg_mean', 'agg_std']
        agg['dataset'] = label
        agg_records.append(agg)
    agg_df = pd.concat(agg_records, ignore_index=True)

    # 3. 透视表合并真实与生成
    pivot = agg_df.pivot(index='feature', columns='dataset', values=['agg_mean', 'agg_std'])
    pivot.columns = [f"{stat}_{ds}" for stat, ds in pivot.columns]
    pivot = pivot.reset_index()

    return real_df, gen_df, pivot





if __name__ == "__main__":
    subid = 4

    # 比较统计特征
    real_df, gen_df, pivot_df = compare_eeg_statistics(subid)
    # 打印并记录结果
    print(pivot_df.to_markdown(index=False))
    with open(f"logs/sub{subid}_comparison.txt", "w") as f:
        # 方式一：写成 Markdown 表格
        f.write(pivot_df.to_markdown(index=False))

    # 比较tsne
    compare_tsne(subid)






