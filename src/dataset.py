import torch


class InforDataset:
    def __init__(self, target, features):
        self.reviews = target
        self.sentiment = features

    def __len__(self):
        # length of the dataset
        return self.target.shape[0]

    def __getitem__(self, idx):
        # convert each features, sentiment to tensors
        return {
            "target": torch.tensor(self.target[idx], dtype=torch.float),
            "features": torch.tensor(self.features[idx, :], dtype=torch.float),
        }