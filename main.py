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


def score(true_value, predict_value):
    length = len(true_value)
    mse_loss = 0
    for i in range(length):
        print("Inference result is {}, the corresponding label is {}".format(predict_value[i], true_value[i]))
        mse_loss += (true_value[i] - predict_value[i]) ** 2
    mse_loss /= length
    print("mse loss is: ", mse_loss)


def test(model_path="linear.pth"):
    net = nn.Sequential(nn.Linear(6, 1)).to(get_device())
    net.load_state_dict(torch.load(model_path))
    features, _, true_power = parse_data.load_data(parse_data.new_test_path)

    prediction = net(df_to_tensor(features))
    prediction = (prediction * (parse_data.max_values - parse_data.min_values) + parse_data.mean_values)

    prediction = prediction.cpu().detach().numpy()

    draw(true_power, prediction, "true", "predict")
    score(true_power, prediction)


def train():
    batch_size = 500
    num_epochs = 200
    data_iter, x, y = get_data_loader(batch_size)
    liner_module(data_iter, x, y, num_epochs)


if __name__ == '__main__':
    # train()
    test()
