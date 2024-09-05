import argparse
import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
from utils import *
from model import *
from sklearn.metrics import recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='train and test')
    parser.add_argument('--data_path', default = '/home/xiyuan/keerthi/TARnet_inference/TARnet/', type = str)
    parser.add_argument('--dataset', default = 'easy_imu_phone', type = str,
                        choices=['easy_imu_phone', 'easy_imu_all', 'hard_imu_phone', 'hard_imu_all'])
    parser.add_argument('--model', default='THAT', type=str,
                        choices=['THAT'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log', default='log', type=str,
                        help="Log directory")
    parser.add_argument('--n_gpu', default=0, type =int)
    
    parser.add_argument('--epochs', default = 50, type = int)
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--batch_size', default = 64, type = int)

    args = parser.parse_args()
    #print(args.n_gpu)
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    args.log_path = os.path.join(args.log, args.dataset)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    #torch.cuda.set_device(args.n_gpu)

    if 'imu_all' in args.dataset:
        args.input_channel = 12
        args.input_size = 150
        args.hheads = 3
        args.SENSOR_AXIS = 1
    elif 'imu_phone' in args.dataset:
        args.input_channel = 9
        args.input_size = 150
        args.hheads = 3
        args.SENSOR_AXIS = 1

    return args
args = parse_args()
#log = set_up_logging(args)
#args.log = log


def test(model, Loader):
    y_pred = []
    y_true = []

    batch_times = []
    device = torch.device('cpu')
    
    #start_time = time.time()
    with torch.no_grad():
        model.eval()
        model.to(device)
        model.cpu()
        for i, (x, y) in enumerate(Loader): 
            y_true += y
            
            x = x.to(device)
            start_time = time.time()
            out = model(x)
            pred = torch.argmax(out, dim = -1)
            y_pred += pred.cpu().tolist()
            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)
            #print(f"Batch {i + 1} completed in {batch_time:.4f} seconds")
    
    
    average_batch_time = sum(batch_times) / len(batch_times)
    #elapsed_times.append(average_batch_time)

    acc = accuracy_score(y_true, y_pred)
    #log("Accuracy : " + str(accuracy_score(y_true, y_pred)) +
    #    "\nMacro F1 : " + str(f1_score(y_true, y_pred, labels=list(range(args.num_labels)),average='macro')) )
    return average_batch_time, acc

device = 'cpu'

def main():
    #log("Start time:" + time.asctime( time.localtime(time.time())) )
    xtrain, ytrain, xtest, ytest = read_data(args)
    xtrain, ytrain, xvalid, yvalid = split_valid(args, xtrain, ytrain) 
    print(np.array(xtrain).shape)
    
    if args.model == 'THAT':
        args.hlayers = 5
        args.vlayers = 1
        args.vheads = 10
        args.K = 10
        args.sample = 3
        model = HARTrans(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    '''
    if args.model == 'THAT':
        args.hlayers = 10
        args.vlayers = 5
        args.vheads = 10
        args.K = 10
        args.sample = 3
        model = HARTrans(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    '''
    total_params = sum(p.numel() for p in model.parameters())
    #log('Total parameters: ' + str(total_params))

    TrainDataset = BaseDataset(xtrain, ytrain)
    TrainLoader = DataLoader(TrainDataset, batch_size = args.batch_size, shuffle = True)
    TestDataset = BaseDataset(xtest, ytest)
    TestLoader = DataLoader(TestDataset, batch_size=args.batch_size, shuffle=True)
    ValidDataset = BaseDataset(xvalid, yvalid)
    ValidLoader = DataLoader(ValidDataset, batch_size=args.batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    cur_acc = 0
    elapsed_times= []
    for ep in range(args.epochs):
        model.train()
        epoch_loss = 0
        #log("Training epoch : " + str(ep))
        for i, (x, y) in enumerate(TrainLoader):  
            x = x.to(device)
            y = y.to(device)
            #print(x.size())                 
            out = model(x)
            loss = loss_func(out, y)
            epoch_loss += loss.cpu().item()

            optimizer.zero_grad()           
            loss.backward()
            optimizer.step()

        _,valid_acc = test(model, ValidLoader)
        #log(f"Epoch : {ep+1} / {args.epochs}, loss : {epoch_loss / i}")
        if valid_acc > cur_acc:
            cur_acc = valid_acc
            average_batch_time, test_acc = test(model, TestLoader)
            elapsed_times.append(average_batch_time)
            return_acc = test_acc
            #log(f"Return acc now is {return_acc:.4f} (valid:{valid_acc:.4f})")
        #log("----------------------------")
    torch.save(model,'train_model_THAT.pth')
    avg_inference_time  = sum(elapsed_times)/len(elapsed_times)
    return avg_inference_time,return_acc


if __name__ == '__main__':
    avg_inference_time,return_acc = main()
    #torch.save(model,'train_model_THAT.pth')
    print("Accuracy:")
    print( return_acc )
    torch.save(return_acc, 'accuracy_tensor.pt')
    #print('avg_inference_time(per batch)')
    #print( avg_inference_time )
    #print( 'avg_inference)time per sample')
    #print( avg_inference_time/args.batch_size)
