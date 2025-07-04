import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import MLSTM_FCN
from src.utils import train, load_dataset, plot_loss

import argparse


def main(args):
    dataset = args.dataset
    
    train_dataset, test_dataset = load_dataset(dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=6)
    test_dataloader = DataLoader(test_dataset, batch_size=6)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    
    mlstm_fcn_model = MLSTM_FCN(3,640,2)
    mlstm_fcn_model.to(device)

    optimizer = optim.SGD(mlstm_fcn_model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()
    
    losses = train(mlstm_fcn_model, train_dataloader, criterion, optimizer, args.epoches, device, run_name=args.name)
    
    plot_loss(losses)
    
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--epoches", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--name", type=str, default="model_mlstm_fcn")
    p.add_argument("--dataset", type=str, default="AF")
    args = p.parse_args()
    main(args)