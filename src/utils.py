import numpy as np
import torch
import os

from torch.utils.data import TensorDataset
from ignite.metrics import Accuracy, Precision, Recall

from matplotlib import pyplot as plt


def train(model, trainloader, criterion, optimizer, epoches=10, device='cpu', run_name='model_mlstm_fcn'):
    print("Training started on device: {}".format(device))
    
    best_loss = np.inf
    
    losses = []
    state = {}
    
    model.train()
    for epoch in range(epoches):
        train_loss = 0.0
        for inputs, labels, in trainloader:
            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        losses.append(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            print("Epoch: {}/{}.. ".format(epoch + 1, epoches), "Training Loss: {:.6f}.. ".format(train_loss))
            state = model.state_dict()
    
    torch.save(state, f'weights/{run_name}.pt')       
    return losses
            
     
def test(model, testloader, criterion, device='cuda:0'):
    acc_metric = Accuracy()
    prec_metric = Precision(average=True)
    rec_metric = Recall(average=True)
    test_loss = 0.0
    model.eval()
    for inputs, labels in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels).item()
        test_loss += loss
        y_pred = outputs
        acc_metric.update((y_pred, labels))
        prec_metric.update((y_pred, labels))
        rec_metric.update((y_pred, labels))
    
    acc = acc_metric.compute()
    prec = prec_metric.compute()
    rec = rec_metric.compute()
    f1 = 2 * prec * rec / (prec + rec)
    return test_loss, acc, prec, rec, f1


def load_dataset(dataset):
    data_path = os.path.join('./data', dataset)
    
    x_train = np.load(os.path.join(data_path, 'x_train.npy')).astype(np.float64)
    y_train = np.load(os.path.join(data_path, 'y_train.npy')).astype(np.int64)
    x_test = np.load(os.path.join(data_path, 'x_test.npy')).astype(np.float64)
    y_test = np.load(os.path.join(data_path, 'y_test.npy')).astype(np.int64)
    
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    return train_dataset, test_dataset


def plot_loss(losses):
    length = len(losses)
    x = np.arange(1, length+1)
    y = np.array(losses)
    plt.plot(x, y)
    plt.savefig('loss')
    