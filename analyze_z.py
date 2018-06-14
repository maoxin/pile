import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(prefix='gridn_cloud_transformed_x'):
    dfs = []
    for i in range(4):
        df = pd.read_csv(f'build/{prefix}{i}.pcd', skiprows=11, header=None, delim_whitespace=True)
        df.columns = ['x', 'y', 'z']
        dfs.append(df)
    df = pd.read_csv(f'build/last_cluster_cloud.pcd', skiprows=11, header=None, delim_whitespace=True)
    df.columns = ['x', 'y', 'z']
    dfs.append(df)
    dfz = pd.DataFrame({'main_pile': dfs[0]['z'], 'ground1': dfs[1]['z'], 'ground2': dfs[2]['z'], 'pile_part': dfs[3]['z'], 'last_cluster': dfs[4]['z']})
    print(dfz.describe())

    dfz.plot(kind='kde')
    plt.title('kde for z value of clusters')
    plt.savefig(f'pile_analysis/by_part/kde_z_{prefix}.pdf')

    return dfz


if __name__ == '__main__':
    dfz = main()
