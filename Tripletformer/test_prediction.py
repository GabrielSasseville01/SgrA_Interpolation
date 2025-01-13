import argparse
from collections import Counter
import os
import numpy as np
import torch
import torch.optim as optim
from random import SystemRandom
import models
import utils
import sys
import matplotlib.pyplot as plt
from scipy.stats import mode

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=4)
parser.add_argument('--dec-num-heads', type=int, default=4)
parser.add_argument('--dataset', type=str, default='sgrA')
parser.add_argument('--net', type=str, default='triple')
parser.add_argument('--sample-tp', type=float, default=0.1)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--mse-weight', type=float, default=1.0)
parser.add_argument('--imab-dim', type=int, default=64)
parser.add_argument('--cab-dim', type=int, default=256)
parser.add_argument('--decoder-dim', type=int, default=128)
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--sample-type', type=str, default='random')
parser.add_argument('--experiment-id', type=str, default=None)
parser.add_argument('--max-early-stop', type=int, default=30)

args = parser.parse_args()
print(' '.join(sys.argv))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_obj = utils.get_dataset(args.batch_size, args.dataset, test_batch_size=1)
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]
    net = models.load_network(args, dim, device=device).to(device)
    
    chp = torch.load(f'./saved_models/{args.dataset}_{args.experiment_id}.h5')
    net.load_state_dict(chp['state_dict'])

    utils.plot_test(net, dim, test_loader, 30)
    