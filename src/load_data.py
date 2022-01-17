import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch import optim

data = np.loadtxt(fyle, delimiter=",")
ynp = data[:, 0].astype(np.float32)
Xnp = data[:, 1:15].astype(np.float32)

# Select 2 inputs.

# So we want to train a model for
# y(t) = a_1y(t-1) + a_2y(t-2) + b_1x_1(t)+ b_2x_2(t)

# tips: (you may ignore)
# if a.shape = (N,) then a can be re-sized to (N,1) using a.reshape(-1,1)
# to concatenate several variables np.concatenate((v1,v2,v3),axis=1) might be useful.
# to print the loss convert from torch to numpy using: print('loss: ' + str(loss.detach().numpy()))
