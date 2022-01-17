import torch


class Engine:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    # MSE Loss function 
    def loss_fn(self, outputs, targets):
        return torch.nn.MSELoss(outputs, targets)

    # training function for the train_loader
    def train_fn(self, dataloader):
        self.model.train()
        final_targets, final_outputs = [], []
        # for data in dataloader:
        for data in dataloader:
            # define features and target tensors
            features = data["features"]
            targets = data["target"]
            # compute forward pass through the model
            outputs = self.model(features)
            # compute loss Binary Cross Entropy function
            loss = self.loss_fn(outputs, targets)
            # set gradients to 0
            self.optimizer.zero_grad()
            # compute gradient of loss w.r.t all the parameters
            loss.backward()
            # optimizer iterate over all parameters (updates parameters)
            self.optimizer.step()
            # append to empty list and conver to numpy array  to list
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(outputs.cpu().detach().numpy().tolist())
        return final_targets, final_outputs

    # evaluation function for the test_loader
    def eval_fn(self, dataloader):
        self.model.eval()
        final_targets, final_outputs = [], []
        # disabling tracking of gradients
        with torch.no_grad():
            for data in dataloader:
                features = data["features"]
                targets = data["target"]
                outputs = self.model(features)
                # append to empty list and conver to numpy array  to list
                final_targets.extend(targets.cpu().detach().numpy().tolist())
                final_outputs.extend(outputs.cpu().detach().numpy().tolist())
        return final_targets, final_outputs
