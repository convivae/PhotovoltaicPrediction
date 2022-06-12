import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import parse_data


# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


def draw(x, y, x_label, y_label):
    plt.plot(x, y, '.')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def draw_comp(y1, y2, y1_label, y2_label, x_label, y_label):
    x = np.arange(len(y1))

    plt.plot(x, y1, label=y1_label)
    plt.plot(x, y2, label=y2_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.show()


# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df).float().to(device)


def get_data_loader(batch_size, is_train=True):
    if is_train:
        file_path = parse_data.new_train_path
    else:
        file_path = parse_data.new_test_path
    x, y, _ = parse_data.load_data(file_path)
    x_tensor = df_to_tensor(x)
    y_tensor = df_to_tensor(y)
    data_set = TensorDataset(x_tensor, y_tensor)
    return DataLoader(data_set, batch_size, shuffle=is_train), x_tensor, y_tensor


def liner_module(data_iter, features, labels, num_epochs):
    net = nn.Sequential(nn.Linear(6, 1)).to(get_device())
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    x_epochs = np.array([])
    y_bias = np.array([])

    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        x_epochs = np.append(x_epochs, epoch + 1)
        y_bias = np.append(y_bias, l.cpu().detach().numpy())
        print(f'epoch {epoch + 1}, loss {l:f}')

    torch.save(net.state_dict(), "linear.pth")
    draw(x_epochs, y_bias, "epoch", "train loss")
    for parameters in net.parameters():
        print("parameters: ", parameters)


def print_test_res(true_value, predict_value):
    for i in range(len(true_value)):
        print("predict: {}, corresponding: {}".format(predict_value[i], true_value[i]))


def cal_r2(true_value, predict_value):
    a, b = 0.0, 0.0
    for i in range(len(true_value)):
        a += (predict_value[i] - parse_data.mean_values) ** 2
        b += (true_value[i] - parse_data.mean_values) ** 2
    # 计算 R^2
    R_2 = a / b
    print("R^2: ", R_2)


def test(model_path="linear.pth", print_res=False):
    net = nn.Sequential(nn.Linear(6, 1)).to(get_device())
    net.load_state_dict(torch.load(model_path))

    # 计算 R^2, 读取 train.csv 进行计算
    features, _, true_power = parse_data.load_data(parse_data.new_train_path)
    prediction = net(df_to_tensor(features))
    prediction = (prediction * (parse_data.max_values - parse_data.min_values) + parse_data.mean_values)
    prediction = prediction.cpu().detach().numpy()
    cal_r2(true_power, prediction)

    # 进行预测， 读取 test.csv 进行计算
    features, _, true_power = parse_data.load_data(parse_data.new_test_path)
    prediction = net(df_to_tensor(features))
    prediction = (prediction * (parse_data.max_values - parse_data.min_values) + parse_data.mean_values)
    prediction = prediction.cpu().detach().numpy()

    draw_comp(true_power, prediction, "true", "predict", "", "compare")

    if print_res:
        print_test_res(true_power, prediction)


def train():
    batch_size = 500
    num_epochs = 200
    data_iter, x, y = get_data_loader(batch_size)
    liner_module(data_iter, x, y, num_epochs)


if __name__ == '__main__':
    # train()
    test(print_res=False)
