import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import MLSTM_FCN
from src.utils import test, load_dataset
from src.constants import NUM_CLASSES, MAX_SEQ_LEN, NUM_FEATURES, KERNELS

import argparse


def main(args):
    dataset = args.dataset
    train_dataset, test_dataset = load_dataset(dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    if args.train == 'train':
        print("load train dataloader")
        loader = train_loader
    elif args.train == 'test':
        print("load test dataloader")
        loader = test_loader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    
    mlstm_fcn_model = MLSTM_FCN(num_classes=NUM_CLASSES[dataset], max_seq_len=MAX_SEQ_LEN[dataset], num_features=NUM_FEATURES[dataset], kernels=KERNELS[dataset])
    mlstm_fcn_model.load_state_dict(torch.load('weights/'+args.weights))
    mlstm_fcn_model.to(device)

    criterion = nn.NLLLoss()
    
    test_loss, acc, prec, rec, f1 = test(mlstm_fcn_model, loader, criterion, device)
    print("Test loss: {:.6f}.. Test Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(test_loss, acc*100, prec*100, rec*100, f1*100))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--weights", type=str, default="model_mlstm_fcn.pt")
    p.add_argument("--dataset", type=str, default="AF")
    p.add_argument("--train", type=str, default='train')
    args = p.parse_args()
    main(args)