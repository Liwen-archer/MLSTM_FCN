import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import MLSTM_FCN
from src.utils import test, load_dataset

import argparse


def main(args):
    dataset = args.dataset
    _, test_dataset = load_dataset(dataset)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    
    mlstm_fcn_model = MLSTM_FCN(3, 640, 2)
    mlstm_fcn_model.load_state_dict(torch.load('weights/'+args.weights))
    mlstm_fcn_model.to(device)

    criterion = nn.NLLLoss()
    
    test_loss, acc, prec, rec, f1 = test(mlstm_fcn_model, test_loader, criterion, device)
    print("Test loss: {:.6f}.. Test Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(test_loss, acc*100, prec*100, rec*100, f1*100))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--weights", type=str, default="model_mlstm_fcn.pt")
    p.add_argument("--dataset", type=str, default="AF")
    args = p.parse_args()
    main(args)