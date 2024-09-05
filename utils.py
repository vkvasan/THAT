import os
import csv
import yaml
import numpy as np
from torch.utils.data.dataset import Dataset

class BaseDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.array(x).astype(np.float32)
        self.labels = y

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

def split_valid(args, xtrain, ytrain):
    np.random.seed(args.seed)
    np.random.shuffle(xtrain)
    np.random.seed(args.seed)
    np.random.shuffle(ytrain)
    lens = len(xtrain)*9//10
    xvalid = xtrain[lens:]
    yvalid = ytrain[lens:]
    xtrain = xtrain[:lens]
    ytrain = ytrain[:lens]
    return xtrain, ytrain, xvalid, yvalid

def read_data(args):
    path = os.path.join(args.data_path , args.dataset)
    x_train = np.load(os.path.join(path, 'x_train.npy'))
    y_train = np.load(os.path.join(path, 'y_train.npy')).astype('int64').tolist()
    x_test = np.load(os.path.join(path, 'x_test.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy')).astype('int64').tolist()
    np.random.seed(args.seed)
    
    dict_map = {}
    cnt = 0
    for i in range(len(y_train)):
        if y_train[i] not in dict_map:
            dict_map[y_train[i]] = cnt
            cnt += 1
    print(dict_map)
    for i in range(len(y_train)):
        y_train[i] = dict_map[y_train[i]]
    for i in range(len(y_test)):
        y_test[i] = dict_map[y_test[i]]
    args.num_labels = max(y_train) + 1

    summary = [0 for i in range(args.num_labels)]
    for i in y_train:
        summary[i] += 1
    #args.log("Label num cnt: "+ str(summary))
    #args.log("Training size: " + str(len(y_train)))
    #args.log("Testing size: " + str(len(y_test)))
    return list(x_train), y_train, list(x_test), y_test

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))

def logging(file):
    def write_log(s, printing = True):
        if printing:
            print(s)
        with open(file, 'a') as f:
            f.write(s+'\n')
    return write_log

def set_up_logging(args):
    log = logging(os.path.join(args.log_path, args.model+'.txt'))
    log(str(args))
    return log

