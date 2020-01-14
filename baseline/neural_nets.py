import csv
from random import random

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from baseline.utils import normalise, get_normalised_data

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, model_save_file, loss_save_file='data/nn_train_loss.txt',
                 load_checkpoint=True):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size, True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1, True)
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        self.curepoch = 0
        self.save_file = model_save_file
        self.loss_file = open(loss_save_file, "a+")
        if load_checkpoint:
            checkpoint = torch.load(self.save_file)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.curepoch = checkpoint['epoch']

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

    def train_network(self, xs, ys, epoch=10000):
        self.train()
        for i in tqdm(range(epoch - self.curepoch)):
            running_loss = 0
            self.curepoch += 1
            for x, y in zip(xs, ys):
                y_pred = self(x)
                self.optimizer.zero_grad()
                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()
                running_loss += loss
            av_loss = running_loss / len(ys)
            print('Epoch {}: train loss: {}'.format(self.curepoch, av_loss))
            self.loss_file.write("{},".format(av_loss))
            self.loss_file.flush()
            torch.save({
                'epoch': self.curepoch,
                'model_state_dict': model.state_dict()
            }, self.save_file)


if __name__ == "__main__":
    x_train, y_train = get_normalised_data('data/nn_train_quater.csv')

    inputs = list(map(lambda s: Variable(torch.Tensor([s])), x_train))
    targets = list(map(lambda s: Variable(torch.Tensor([s])), y_train))

    # model.eval()
    # y_pred = model(x_test)
    # before_train = model.criterion(y_pred.squeeze(), y_test)
    # print('Test loss before training', before_train.item())

    model = Feedforward(5, 10, "models/nn.pth", load_checkpoint=False)

    print("Starting Training")
    model.train_network(inputs, targets)
    # y_pred = model(x_test)
    # after_train = model.criterion(y_pred.squeeze(), y_test)
    # print('Test loss after Training', after_train.item())
