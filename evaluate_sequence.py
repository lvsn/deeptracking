import pandas as pd
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.style.use('ggplot')

if __name__ == '__main__':
    path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/models/schubert/outputs/deeptracking/models/dragon_squeeze/scores/occlusion_eval"
    files = [f for f in os.listdir(path) if f[-3:] == "csv"]

    figure_path = os.path.join(path, "figs")
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    analytic_path = os.path.join(path, "analytic")
    if not os.path.exists(analytic_path):
        os.mkdir(analytic_path)

    for name in files:
        path_csv = os.path.join(path, name)
        name = name.split(".")[0]

        df = pd.read_csv(path_csv)
        N = len(df)
        column_tags = df.columns.values

        # Frequency analysis
        #signal = df.as_matrix(columns=[column_tags[2]])
        #yf = np.fft.fft(signal)/N
        #yf = np.fft.fftshift(yf)
        #xf = np.fft.fftfreq(N, 0.06)
        #axes[2].plot(xf, yf)
        #axes[2].set_ylim(0, 0.00003)

        # Diff analysis
        plt.figure(0)
        fig, axes = plt.subplots(nrows=2, ncols=1)
        ax1 = df.plot(ax=axes[0], x=np.arange(N), y=column_tags[:3])
        ax2 = df.plot(ax=axes[1], x=np.arange(N), y=column_tags[3:])
        ax1.set_ylim(0, 0.02)
        ax2.set_ylim(0, 20)
        fig = plt.gcf()
        fig.savefig(os.path.join(figure_path, "Sequence_Diff_{}.png".format(name)))
        #plt.show()

        # Diff mean Translation/Rotation
        plt.figure(1)
        fig, axes = plt.subplots(nrows=2, ncols=1)
        #t_mean = df[['Tx', 'Ty', 'Tz']].mean(axis=1).as_matrix()
        t_mean = np.sqrt(df[['Tx']].as_matrix() ** 2 + df[['Ty']].as_matrix() ** 2 + df[['Tz']].as_matrix() ** 2)
        r_mean = df[['Rx', 'Ry', 'Rz']].mean(axis=1).as_matrix()
        axes[0].plot(np.arange(N), t_mean)
        axes[1].plot(np.arange(N), r_mean)
        axes[0].set_ylim(0, 0.08)
        axes[0].set_xlim(0, len(t_mean))
        axes[1].set_ylim(0, 20)
        axes[1].set_xlim(0, len(t_mean))
        fig.savefig(os.path.join(figure_path, "Mean_Diff_{}.svg".format(name)), transparent=True)


        # Compute scores data
        info = pd.DataFrame()
        info["mean"] = df.mean(axis=0)
        info["std"] = df.std(axis=0)

        error_ratio_translation = df[["Tx", "Ty", "Tz"]] > 0.02
        error_ratio_rotation = df[["Rx", "Ry", "Rz"]] > 10
        info["critical ratio"] = pd.concat([error_ratio_translation, error_ratio_rotation]).mean(axis=0)
        error_ratio_translation = df[["Tx", "Ty", "Tz"]] > 0.04
        error_ratio_rotation = df[["Rx", "Ry", "Rz"]] > 20
        info["fail ratio"] = pd.concat([error_ratio_translation, error_ratio_rotation]).mean(axis=0)

        # critical and fail are different behavior, so we quantify them differently
        # critical : huge offset in tracking    fail : no tracking anymore
        info["critical ratio"] = info["critical ratio"] - info["fail ratio"]


        print(name)
        print(info)
        info.to_csv(os.path.join(analytic_path, "{}.csv".format(name)), index=True, encoding='utf-8')
