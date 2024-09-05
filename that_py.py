import torch
from torch.utils.data import DataLoader
from utils import *
import argparse

class Config:
    def __init__(self):
        self.data_path = '/home/xiyuan/keerthi/TARnet_inference/TARnet/'
        self.dataset = 'easy_imu_phone'
        self.model = 'THAT'
        self.seed = 0
        self.log = 'log'
        self.n_gpu = 0
        self.epochs = 50
        self.lr = 1e-3
        self.batch_size = 16

    def __repr__(self):
        return (f"Config(data_path={self.data_path}, dataset={self.dataset}, model={self.model}, seed={self.seed}, "
                f"log={self.log}, n_gpu={self.n_gpu}, epochs={self.epochs}, lr={self.lr}, batch_size={self.batch_size})")


config = Config()
xtrain, ytrain, xtest, ytest = read_data(config)
TestDataset = BaseDataset(xtest, ytest)
model = torch.load('train_model_THAT.pth')
accuracy = torch.load('accuracy_tensor.pt')

TestLoader = DataLoader(TestDataset, batch_size=16, shuffle=True)


import time
import torch

def test_run(model, Loader):
    y_pred = []
    y_true = []

    batch_times = []
    sample_times = []
    device = torch.device('cpu')
    
    with torch.no_grad():
        model.eval()
        model.to(device)
        model.cpu()
        for i, (x, y) in enumerate(Loader): 
            y_true += y.tolist()
            
            x = x.to(device)
            batch_size = x.size(0)

            start_time = time.time()
            out = model(x)
            pred = torch.argmax(out, dim=-1)
            y_pred += pred.cpu().tolist()
            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)

            # Calculate per-sample time for the current batch
            per_sample_time = batch_time / batch_size
            sample_times.append(per_sample_time)
    
    # Calculate average batch time and average per-sample time
    average_batch_time = sum(batch_times) / len(batch_times)
    average_per_sample_time = sum(sample_times) / len(sample_times)

    return average_batch_time, average_per_sample_time

average_batch_time, average_per_sample_time = test_run(model, TestLoader)
print(f'Test_accuracy: {accuracy:.6f} ')
print(f"Average Batch Time: {average_batch_time:.6f} seconds")
print(f"Average Per-Sample Time: {average_per_sample_time:.8f} seconds")