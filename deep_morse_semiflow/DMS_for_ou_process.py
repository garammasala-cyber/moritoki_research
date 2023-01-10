# パッケージのインポート
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as func
from tqdm import tqdm

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 他ファイルのインポート
import json

with open('parameter.json') as parameter:
    parameters = json.load(parameter)

num_epoch = parameters["Deep_Learning"]["num_epoch"]
num_data = parameters["Deep_Learning"]["num_data"]
num_partition = parameters["Deep_Learning"]["num_partition"]
learning_rate = parameters["Deep_Learning"]["learning_rate"]
num_testdata = parameters["Deep_Learning"]["num_testdata"]

epsilon = parameters["Hyperparameter"]["epsilon"]

strike_price = parameters["Option"]["strike_price"]

expiration = parameters["OU_Process"]["T"]
alpha = parameters["OU_Process"]["alpha"]
sigma = parameters["OU_Process"]["sigma"]
initial_value = parameters["OU_Process"]["initial_value"]

len_partition = expiration / num_partition

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))
        x = func.relu(self.fc5(x))
        x = self.fc6(x)
        return x


def payoffs_function(model_values):
    payoffs = func.relu(torch.exp(model_values) - strike_price).requires_grad_(True)
    return payoffs


variation = 1e-5


def first_differential_operator_ou(f, x):
    return (sigma / np.sqrt(2)) * (f(x + variation) - f(x - variation)) / (2 * variation)


def data_loader(x, y):
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    return dataloader


class CustomLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, diff, phi, penalty):
        return torch.mean((target - output) ** 2 / len_partition - diff ** 2
                          + (1 / penalty) ** 2 * func.relu(output - phi) ** 2)


def main(penalty=0.01):
    model = Net()
    for i in tqdm(range(1, num_partition + 1)):
        input_data = torch.tensor(np.random.normal(loc=0, scale=sigma ** 2 / (2 * alpha), size=num_data)
                                  .reshape(num_data, 1), requires_grad=False, dtype=torch.float32).to(device)
        if i == 0:
            train_data_loader = data_loader(input_data, payoffs_function(input_data))
        else:
            model = model.to(device)
            train_data_loader = data_loader(input_data, model(input_data))

        model = Net()
        model = model.to(device)
        criterion = CustomLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for n in range(num_epoch):
            for inputs, target in train_data_loader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)

                diff = first_differential_operator_ou(model, inputs)
                phi = payoffs_function(inputs).to(device)

                loss = criterion(output, target, diff, phi, penalty)
                loss.backward(retain_graph=True)
                optimizer.step()

    test_datas = torch.tensor(np.array([[j / 10] for j in range(-50, 50 + 1)]),
                              requires_grad=False, dtype=torch.float32).to(device)
    initial_variables = test_datas.cpu().clone().detach().numpy().flatten().tolist()
    value = ((-1) * model(test_datas)).cpu().clone().detach().numpy().flatten().tolist()
    return initial_variables, value, 'epsilon={}'.format(penalty)


if __name__ == '__main__':
    main(0.1)
    main(0.01)
    main(0.001)
    main(0.0001)
    plt.legend()
    plt.show()
