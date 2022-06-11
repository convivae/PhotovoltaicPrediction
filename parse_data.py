import time

import pandas as pd

pd.set_option('display.max_columns', 15)

old_train_path = 'data/Training/PV_data_all.csv'
new_train_path = 'data/ProcessedData/train/train.csv'
old_test_path = 'data/Download/PV_data_sample.csv'
new_test_path = 'data/ProcessedData/test/test.csv'

global max_values
global min_values
global mean_values


def write_lines(filename, lines):
    # lines = [line + '\n' for line in lines]
    with open(filename, 'w') as f:
        f.writelines(lines)


# 时间不是数字，先进行处理
def pre_process_data(old_file_path, new_file_path):
    lines = []
    with open(old_file_path) as f:
        for line in f:
            t = line.split(",")[0]
            other = ",".join(line.split(",")[1:])
            if ':' in t:
                t = time.mktime(time.strptime(t, "%Y/%m/%d %H:%M"))
            else:
                other = other.replace("Current_V", "Current")
            lines.append(str(t) + ',' + other)
    write_lines(new_file_path, lines)


def load_data(file_path):
    light_frame = pd.read_csv(file_path, sep=',')
    print(light_frame.describe())
    param_columns = ['t', 'AmbiTemp', 'Irradiance', 'ModuleTemp', 'InclAngle', 'Current', 'Voltage', 'Humidity']

    value = light_frame.pop('TruePower')

    params = light_frame[param_columns].copy()

    # 去掉两列完全相同的列：InclAngle、Humidity
    params.pop('InclAngle')
    params.pop('Humidity')

    global max_values
    global min_values
    global mean_values

    max_values, min_values, mean_values = value.max(axis=0), value.min(axis=0), value.mean(axis=0)

    # 对输入数据进行标准化
    params = (params - params.mean()) / (params.max() - params.min())

    original_val = value

    value = (value - value.mean()) / (value.max() - value.min())
    params = params.values
    value = value.values.reshape(-1, 1)

    return params, value, original_val


def process():
    pre_process_data(old_train_path, new_train_path)
    pre_process_data(old_test_path, new_test_path)


if __name__ == '__main__':
    # process()
    load_data(new_train_path)
    # load_data(new_test_path)
