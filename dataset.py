import config
import torch
import pandas as pd

class GlaucomaSurgeryDataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset which is consistent with the input requirement of the Trainer
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_glaucoma_surgery_train_test_val(
        data_url=config.data_url):
    """
    This function read in the glaucoma surgery data with the specified data url.
    The first 500 are the held-out test dataset, next 400 are the val dataset and the rest are for the train.
    This function will return the train, val, test data
    """
    glaucoma_surgery_data = pd.read_csv(data_url, header=None)
    test_data = glaucoma_surgery_data.iloc[:500, :]
    val_data = glaucoma_surgery_data.iloc[500:900, :]
    train_data = glaucoma_surgery_data.iloc[900:, :]

    return train_data, val_data, test_data


def get_texts_labels_split(data):
    """
    This function split the glaucoma surgey data into text and labels.
    In the glaucoma surgey data, text are in col ind2, label is in the last col
    """

    data = data.copy()

    texts = data.iloc[:, 2].values.tolist()
    labels = data.iloc[:, -1].values.tolist()

    return texts, labels