import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.style.use('ggplot')

if __name__ == '__main__':
    path = "/home/mathieu/Dataset/DeepTrack/model/mixed_skull/scores"
    scores_file = [f for f in os.listdir(path) if "score" in f]
    df = pd.read_csv(os.path.join(path, scores_file[0])).T
    sequences_data = pd.DataFrame(columns=df.iloc[0])
    for file in scores_file:
        index = int(re.findall(r'\d+', file)[0])
        df = pd.read_csv(os.path.join(path, file), index_col=0).T
        sequences_data.loc[index] = list(df.loc["mean"])

    sequences_data.sort_index(axis=0, inplace=True)
    x = sequences_data.index.values
    y = list(sequences_data.columns)
    fig, axes = plt.subplots(nrows=2, ncols=1)
    sequences_data.plot(ax=axes[0], x=x, y=y[:3])
    sequences_data.plot(ax=axes[1], x=x, y=y[3:])
    fig = plt.gcf()
    fig.savefig(os.path.join(path, "sequences_stats.png"))