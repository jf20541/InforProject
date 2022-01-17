import pandas as pd
import config
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import mean_squared_error
from dataset import InforDataset
from engine import Engine


def train():
    # import cleaned csv file
    df = pd.read_csv(config.TRAINING_FILE_CLEAN)
    targets = df["target"].values
    features = df.drop("target", axis=1).values

    # no shuffling on time-series data,  trainingset 80% and testingset 20%
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, shuffle=False
    )
    print(x_train.shape), print(x_test.shape), print(y_train.shape), print(y_test.shape)

    # initiate custom dataset and feed to dataloader
    train_dataset = InforDataset(features=x_train, targets=y_train)
    test_dataset = InforDataset(features=x_test, targets=y_test)

    # Initialize DataLoader, each iteration returns a batch of features and labels
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    # couldnt think of a model (I left it empty on the model.py module)
    model = ()

    # Adam as my optimzier
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # initiate Enginer class (model, optimzier, loss function)
    eng = Engine(model, optimizer)

    for epochs in range(config.EPOCHS):
        # initiating training and evaluation function
        train_targets, train_outputs = eng.train_fn(train_loader)
        eval_targets, eval_outputs = eng.eval_fn(test_loader)

        train_metric = mean_squared_error(train_targets, train_outputs)
        eval_metric = mean_squared_error(eval_targets, eval_outputs)
        print(
            f"Epoch:{epochs+1}/{config.EPOCHS}, Train MSE: {train_metric:.4f}, Eval MSE: {eval_metric:.4f}"
        )


if __name__ == "__main__":
    train()
