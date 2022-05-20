import torch
from torch import nn
from torch.nn import functional as F
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

import utils
from models import LinearModel, SimpleConv


train_input, train_target, test_input, test_target = \
    utils.load_data(cifar=1, one_hot_labels = True, normalize = True, flatten = False)
    

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs=100):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = eta)
    loss_arr = []
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            loss_arr.append(loss.item())
            acc_loss = acc_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(e, acc_loss)
    return np.array(loss_arr)

def compute_nb_errors(model, input, target, mini_batch_size):
    # input: nxd
    # target nxc
    
    nb_errors = 0
    
    for b in range(0, input.size(0), mini_batch_size):
        output_labels = torch.argmax(model(input.narrow(0, b, mini_batch_size)), axis=1)
        for k in range(mini_batch_size):
            if target[b+k, output_labels[k]] <= 0:
                nb_errors += 1
    return nb_errors
    

model = SimpleConv()
mini_batch_size = 100
for i in range(80):
    train_loss = train_model(model, train_input, train_target, mini_batch_size, nb_epochs=100)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test acc Net {:0.2f}% {:d}/{:d}'.format((100 * (test_input.size(0)-nb_test_errors)) / test_input.size(0),
                                                        (test_input.size(0)-nb_test_errors), test_input.size(0)))