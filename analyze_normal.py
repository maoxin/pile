import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(prefix='gridn_normal_transformed', ground_normal=np.array([-0.00161291, 0.000182169, 0.999999])):
    values = []
    for i in range(4):
        df = pd.read_csv(f'build/{prefix}{i}.pcd', skiprows=11, header=None, delim_whitespace=True)
        ar_value = df.values[:, :3]
        values.append(pd.Series(np.arccos((ar_value * ground_normal).sum(axis=1)) / np.pi * 180))
    df = pd.read_csv(f'build/last_cluster_normal.pcd', skiprows=11, header=None, delim_whitespace=True)
    ar_value = df.values[:, :3]
    values.append(pd.Series(np.arccos((ar_value * ground_normal).sum(axis=1)) / np.pi * 180))
    df_normal = pd.DataFrame({'main_pile': values[0], 'ground1': values[1], 'ground2': values[2], 'pile_part': values[3], 'last_cluster': values[4]})
    print(df_normal.describe())

    df_normal.plot(kind='kde')
    plt.title('kde for degree against z of clusters')
    plt.savefig(f'pile_analysis/by_part/kde_normal_{prefix}.pdf')

    return df_normal

def last(ground_normal=np.array([-0.00161291, 0.000182169, 0.999999])):
    values = []
    df = pd.read_csv(f'build/last_cluster_normal.pcd', skiprows=11, header=None, delim_whitespace=True)
    ar_value = df.values[:, :3]
    values.append(pd.Series(np.arccos((ar_value * ground_normal).sum(axis=1)) / np.pi * 180))
    df_normal = pd.DataFrame({'last_cluster': values[0]})
    print(df_normal.describe())

    df_normal.plot(kind='kde')
    plt.title('kde for degree against z of the last cluster')
    plt.savefig(f'pile_analysis/by_part/kde_normal_last_cluster.pdf')

    return df_normal


if __name__ == '__main__':
    df_normal = main()
    # df_normal = last()
