import os.path
from typing import Mapping

import numpy as np
import pandas
import pandas as pd
import torch
import scipy.fft
from torch.utils.data import Dataset
from tqdm import tqdm

from utils_wisdm import train_test_split, make_split


class WISDMDataset(Dataset):
    """
    A PyTorch Dataset class for the WISDM dataset.
    """

    def __init__(self, data):
        """
        Initialize the dataset with data mapping.
        Args:
            data (Mapping[str, list[np.ndarray | int]]): A dictionary containing the data and targets.
        """
        self.data = np.array(data[0], dtype=np.float32)
        self.targets = np.array(data[1], dtype=np.float32)

    def __getitem__(self, index):
        """
        Get an item from the dataset by index.
        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data and target tensors for the specified index.
        """
        return self.data[index], self.targets[index]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data[1])


def define_cols(df: pandas.DataFrame, prefix='acc'):
    """
    Define columns in the DataFrame and drop the 'null' column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        prefix (str, optional): The prefix for the x, y, and z columns. Defaults to 'acc'.

    Returns:
        pandas.DataFrame: The DataFrame with columns renamed and the 'null' column dropped.
    """
    columns = ['subject', 'activity', 'timestamp', f'x_{prefix}', f'y_{prefix}', f'z_{prefix}', 'null']
    df.columns = columns
    df = df.drop('null', axis=1)
    return df


def filter_merge_interval(dfa: pandas.DataFrame, dfg: pandas.DataFrame, act_df: pandas.DataFrame):
    """
    Filter and merge accelerometer and gyroscope DataFrames based on timestamps and activity codes.

    Args:
        dfa (pandas.DataFrame): The accelerometer DataFrame.
        dfg (pandas.DataFrame): The gyroscope DataFrame.
        act_df (pandas.DataFrame): The activity DataFrame.

    Returns:
        pandas.DataFrame: The merged and filtered DataFrame.
    """
    t0_a = dfa['timestamp'].min()
    t0_g = dfg['timestamp'].min()
    t1_a = dfa['timestamp'].max()
    t1_g = dfg['timestamp'].max()

    t0 = max(t0_a, t0_g)
    t1 = min(t1_a, t1_g)
    dfa = dfa[(t0 <= dfa['timestamp']) & (dfa['timestamp'] <= t1)]
    dfg = dfg[(t0 <= dfg['timestamp']) & (dfg['timestamp'] <= t1)]

    df = dfa.merge(dfg.drop(dfg.columns[[0, 1]], axis=1), how='inner', on='timestamp')
    df = df.sort_values(by='timestamp')
    df = df.dropna()
    codes = act_df.code.unique()
    df = df[df.activity.isin(codes)]
    replace_codes = zip(act_df.code, act_df.fcode)
    for code, replacement_code in replace_codes:
        df['activity'] = df.activity.replace(code, replacement_code)
    return df


def process_dataset(act_df: pandas.DataFrame, data_path: str, modality='watch'):
    """
    Process the WISDM dataset by reading accelerometer and gyroscope data and merging them.

    Args:
        act_df (pandas.DataFrame): The activity DataFrame.
        data_path (str): The path to the directory containing the dataset.

    Returns:
        pandas.DataFrame: The concatenated and merged DataFrame of accelerometer and gyroscope data.
    """
    dfs = []
    data_path = "/home/claudio/Documentos/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset"
    for i in tqdm(range(1600, 1651)):
        df_wa = define_cols(
            pd.read_csv(f'{data_path}/raw/{modality}/accel/data_{i}_accel_{modality}.txt', header=None, sep=',|;',
                        engine='python'))
        df_wg = define_cols(
            pd.read_csv(f'{data_path}/raw/{modality}/gyro/data_{i}_gyro_{modality}.txt', header=None, sep=',|;',
                        engine='python'),
            prefix='gyro')
        dfs.append(filter_merge_interval(df_wa, df_wg, act_df))
    return pd.concat(dfs)


def normalize_data(df):
    """
    Normalize the data in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The normalized DataFrame.
    """
    cols = [f'{axis}_{sensor}' for axis in ['x', 'y', 'z'] for sensor in ['acc', 'gyro']]
    for col in tqdm(cols):
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def get_processed_dataframe(reprocess=False, modality='watch'):
    """
    Load or reprocess the processed WISDM dataset.

    Args:
        reprocess (bool, optional): Whether to reprocess the dataset. Defaults to False.

    Returns:
        pandas.DataFrame: The processed DataFrame.
    """
    dir_path = "WISDM/"
    # if os.path.exists(dir_path + f'processed_{modality}.csv') and not reprocess:
    #     return pd.read_csv(dir_path + f'processed_{modality}.csv', index_col=0)
    act_df = pd.read_csv(dir_path + 'activity_key_filtered.txt', index_col=0)
    processed_df = process_dataset(act_df, dir_path + "wisdm-dataset", modality=modality)
    print("ooee", len(processed_df['activity'].unique()))
    processed_df = normalize_data(processed_df)
    processed_df.to_csv(dir_path + f'processed_{modality}.csv')
    return processed_df


def create_dataset(df, clients=None, window=200, overlap=0.7):
    """
    Create a dataset from the input DataFrame based on the specified parameters.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        clients (list, optional): The list of client ids. Defaults to None.
        window (int, optional): The window size for segmenting data. Defaults to 200.
        overlap (float, optional): The overlap ratio between windows. Defaults to 0.5.

    Returns:
        tuple: A tuple containing a dictionary with 'X' and 'Y' keys, and a dictionary with client indices.
    """
    if clients is None:
        clients = list(range(1600, 1651))
    c_idxs = {}
    idx = 0
    X = []
    Y = []
    for client in tqdm(clients):
        c_idxs[client] = []
        # data = df[df.subject == client].sort_values(by='timestamp').sample(frac=0.5, random_state=None)
        size_selected = int(len(df[df.subject == client]))
        data = df[df.subject == client].sort_values(by='timestamp').head(n=size_selected)
        activities = data.activity.unique()
        for activity in activities:
            df_f = data[data.activity == activity]
            for i in range(window, len(df_f), int(window * overlap)):
                if i + window > len(df_f):
                    continue
                # x_data = df_f[df_f.columns[3:10]].iloc[i:i + window].to_numpy().tolist()
                # print(x_data)
                # print()
                # print(x_data[2])
                # print(x_data[5])
                fft_columns = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']
                T = df_f['timestamp'].max() - df_f['timestamp'].min()
                n = len(df_f)
                delta_t = T/n
                
                # for column in fft_columns:
                #
                #     fourier = np.absolute(scipy.fft.fft(df_f[column].to_numpy()))
                #     # N = len(df_f[column].to_numpy())
                #     # fourier = np.abs(fourier[0:N])
                #     # print(pd.Series(fourier).describe())
                #     # print(len(fourier), len(df_f))
                #     # result = seasonal_decompose(df_f[column], model='additive', period=1)
                #     # trend = result.trend
                #     # seasonal = result.seasonal
                #     # residual = result.resid
                #     # print(seasonal, type(seasonal))
                #     df_f[column] = fourier
                # print("colunas: ", df_f.columns)
                # # ['subject', 'activity', 'timestamp', 'x_acc', 'y_acc', 'z_acc', 'x_gyro',
                # #        'y_gyro', 'z_gyro'],
                # print(" iloc ", df_f.iloc[0:50])
                # exit()
                X.append(df_f[df_f.columns[3:10]].iloc[i:i + window].to_numpy().tolist())
                Y.append(activity)
                c_idxs[client].append(idx)
                idx += 1

    print("classes: ", len(np.unique(Y)))

    return (X, Y), c_idxs


def split_dataset(data: dict, client_mapping_train: dict, client_mapping_test: dict):
    """
    Split the dataset into train and test sets based on the client mappings.

    Args:
        data (dict): The input dataset as a dictionary with 'X' and 'Y' keys.
        client_mapping_train (dict): A dictionary containing the client indices for the training set.
        client_mapping_test (dict): A dictionary containing the client indices for the test set.

    Returns:
        tuple: A tuple containing the train and test WISDMDatasets, and a dictionary with train and test mappings.
    """
    all_train, mapping_train = make_split(client_mapping_train)
    all_test, mapping_test = make_split(client_mapping_test)
    x = data[0]
    y = data[1]
    train_data = ([x[i] for i in all_train], [y[i] for i in all_train])
    test_data = ([x[i] for i in all_test], [y[i] for i in all_test])
    return WISDMDataset(train_data), WISDMDataset(test_data), {'train': mapping_train, 'test': mapping_test}


def load_dataset(window=200, overlap=0.5, reprocess=True, split=0.8, modality='watch'):
    # antes window 25 rsmprop
    """
    Load the WISDM dataset, either from disk or by reprocessing it based on the specified parameters.

    Args:
        window (int, optional): The window size for segmenting data. Defaults to 200.
        overlap (float, optional): The overlap ratio between windows. Defaults to 0.5.
        reprocess (bool, optional): Whether to reprocess the dataset. Defaults to True.
        split (float, optional): The ratio for the train/test split. Defaults to 0.8.
        modality (str, optional): The modality to use. Defaults to 'watch'.

    Returns:
        dict: A dictionary containing the full dataset, train and test datasets, client mapping, and split.
    """
    dir_path = "/dataset_utils/data/WISDM/"
    # if os.path.exists(dir_path+f'wisdm_{modality}.dt') and not reprocess:
    #     return torch.load(dir_path + f'wisdm_{modality}.dt')

    if reprocess or not os.path.exists(dir_path + f'wisdm_{modality}.dt'):
        processed_df = get_processed_dataframe(reprocess=reprocess, modality=modality)
        print(processed_df)
        clients = list(range(1600, 1651))
        data, idx = create_dataset(processed_df, clients=clients, window=window, overlap=overlap)
        pd.DataFrame({"x": data[0], "y": data[1]}).to_csv(
            f"WISDM/wisdm_{modality}_sequences_window={window}_overlap={overlap}.csv", index=False)
        # print(data['X'][0].shape, data['X'][0][0])
        # exit()
        dataset = WISDMDataset(data)
        client_mapping_train, client_mapping_test = train_test_split(idx, split)
        train_dataset, test_dataset, split = split_dataset(data, client_mapping_train, client_mapping_test)
        print("Count treino: ", np.unique(train_dataset.targets, return_counts=True))
        print("Count teste: ", np.unique(test_dataset.targets, return_counts=True))
        # exit()

        torch.save({
            'full_dataset': dataset,
            'train': train_dataset,
            'test': test_dataset,
            'client_mapping': idx,
            'split': split
        }, "WISDM/" + f'wisdm_{modality}.dt')
    print("salvar: ", "WISDM/" + f'wisdm_{modality}.dt')
    data = torch.load("WISDM/"+ f'wisdm_{modality}.dt', weights_only=False)
    # print("ler data")
    # exit()
    return data


if __name__ == '__main__':
    dt = load_dataset()
    print(len(dt['train']))
