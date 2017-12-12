import pandas as pd
import os


class DataLogger:
    def __init__(self):
        self.data_frames = {}

    def get_dataframes_id(self):
        return list(self.data_frames.keys())

    def get_dataframe_columns(self, id):
        return list(self.data_frames[id].columns.values)

    def create_dataframe(self, id, tags):
        self.data_frames[id] = pd.DataFrame(columns=tags)

    def add_row(self, id, data):
        df_shape = self.data_frames[id].shape
        if len(data) != df_shape[1]:
            raise IndexError("Dataframe and input data has different size : df:{} input:{}".format(df_shape[1], len(data)))
        self.data_frames[id].loc[df_shape[0]] = data

    def add_row_from_dict(self, id, data):
        cols = self.get_dataframe_columns(id)
        self.add_row(id, [data[x] for x in cols])

    def get_as_numpy(self, id):
        return self.data_frames[id].as_matrix()

    def get_dataframe_as_strings(self):
        string_list = []
        for df in self.data_frames:
            string_list.append(df.to_string())
        return string_list

    def save(self, path):
        for key, df in self.data_frames.items():
            filename = os.path.join(path, key + ".csv")
            df.to_csv(filename, index=False)

    def load(self, path):
        files = [f for f in os.listdir(path) if os.path.splitext(os.path.join(path, f))[1] == ".csv"]
        for file in files:
            name, _ = os.path.splitext(file)
            df = pd.read_csv(os.path.join(path, file))
            self.data_frames[name] = df

    def clear_csv(self, path):
        files = [f for f in os.listdir(path) if os.path.splitext(os.path.join(path, f))[1] == ".csv"]
        for file in files:
            os.remove(os.path.join(path, file))
