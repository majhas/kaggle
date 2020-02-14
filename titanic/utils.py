import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    def __init__(self, data, labels, to_normalize=False):
        self.data = data
        self.labels = labels
        if to_normalize:
            self.data = normalize(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        data = torch.tensor(data)
        labels = torch.tensor(labels)

        return (data, labels)

    def __len__(self):
        return len(self.data)

class Utils():
    def __init__(self):
        pass

    @staticmethod
    def process_data(file):
        df = pd.read_csv(file)
        df = df.drop('Cabin', axis=1)
        df = df.drop('Ticket', axis=1)
        df = df.drop('Name', axis=1)

        for i in range(df.shape[0]):
            if df.iloc[i]['Sex'].strip() == 'male':
                df.at[i,'Sex'] = 0
            else:
                df.at[i,'Sex'] = 1

            if df.iloc[i]['Embarked'] == 'S':
                df.at[i, 'Embarked'] = 0
            elif df.iloc[i]['Embarked'] == 'C':
                df.at[i, 'Embarked'] = 1
            else:
                df.at[i, 'Embarked'] = 2

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['Alone'] = 1
        df['Alone'].loc[df['FamilySize']>1] = 0
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

        data = df.to_numpy()
        data = data.astype('double')

        return data

    @staticmethod
    def split_data(data, split=0.8):

        val_split = int(split*len(data))
        val_data = data[val_split:]
        train_data = data[:val_split]


        train_ys = train_data[:, 1]
        train_ys = train_ys.reshape(train_ys.shape[0],1)
        train_data = np.delete(train_data, 1, 1)

        val_ys = val_data[:, 1]
        val_ys = val_ys.reshape(val_ys.shape[0],1)
        val_data = np.delete(val_data, 1, 1)

        return train_data, train_ys, val_data, val_ys

    @staticmethod
    def calculate_acc(y_pred, y_true):
        acc = 0
        for yp, yt in zip(y_pred, y_true):
            if yp == yt:
                acc += 1

        acc /= len(y_pred)
        return acc
