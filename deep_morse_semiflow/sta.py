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
num_data = int(parameters["Deep_Learning"]["num_data"])
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

# 関数Phi (e^x - K)_+
def payoffs_function(input_data):
#    return torch.maximum(torch.exp(input_data) - strike_price, torch.zeros(num_data).reshape(num_data, 1))
    return torch.exp(- input_data**2) - 0.1


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

    def forward(self, output, target, diff, penalty):
        return torch.mean(diff ** 2 - (1 / penalty) * func.relu(output - target) ** 2)


def main(penalty=0.01):
    model = Net()
    input_data = torch.tensor(np.random.normal(loc=0, scale=sigma ** 2 / (2 * alpha), size=num_data)
                              .reshape(num_data, 1), requires_grad=False, dtype=torch.float32).to(device)

    train_data_loader = data_loader(input_data, payoffs_function(input_data))

    model = model.to(device)
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for n in tqdm(range(num_epoch)):
        for inputs, target in train_data_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            output = model(inputs)

            diff = first_differential_operator_ou(model, inputs)

            loss = criterion(output, target, diff, penalty)
            loss.backward()
            optimizer.step()

    test_datas = torch.tensor(np.array([[j / 100] for j in range(-500, 500 + 1)]),
                              requires_grad=False, dtype=torch.float32).to(device)
    initial_variables = test_datas.cpu().clone().detach().numpy().flatten().tolist()
    value = (- model(test_datas)).cpu().clone().detach().numpy().flatten().tolist()
    return initial_variables, value, 'penalty={}'.format(penalty)


"""if __name__ == '__main__':
    main(0.1)
    main(0.01)
    main(0.001)
    main(0.0001)
    plt.legend()
    plt.show()"""
