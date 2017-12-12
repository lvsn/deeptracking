import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

if __name__ == '__main__':
    path = "/home/mathieu/Dataset/DeepTrack/model/skull/"
    files = ["Minibatch.csv", "Grad_Translation.csv", "Grad_Rotation.csv", "Epoch.csv"]

    for filename in files:
        file_path = os.path.join(path, filename)
        df = pd.read_csv(file_path)
        column_tags = df.columns.values
        df.plot(x=np.arange(len(df)), y=column_tags)
        plt.show()