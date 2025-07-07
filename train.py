import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import MLSTM_FCN
from src.utils import train, test, load_dataset, plot_loss
from src.constants import NUM_CLASSES, MAX_SEQ_LEN, NUM_FEATURES, KERNELS

import argparse


def main(args):
    dataset = args.dataset
    
    train_dataset, test_dataset = load_dataset(dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    
    mlstm_fcn_model = MLSTM_FCN(num_classes=NUM_CLASSES[dataset], max_seq_len=MAX_SEQ_LEN[dataset], num_features=NUM_FEATURES[dataset], kernels=KERNELS[dataset])
    mlstm_fcn_model.to(device)

    optimizer = optim.SGD(mlstm_fcn_model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()
    
    losses = train(mlstm_fcn_model, train_dataloader, criterion, optimizer, args.epoches, device, run_name=args.name)
    
    plot_loss(losses)
    
    test_loss, acc, prec, rec, f1 = test(mlstm_fcn_model, train_dataloader, criterion, device)
    print("Test loss: {:.6f}.. Test Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(test_loss, acc*100, prec*100, rec*100, f1*100))
    
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--epoches", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--name", type=str, default="model_mlstm_fcn")
    p.add_argument("--dataset", type=str, default="AF")
    args = p.parse_args()
    main(args)