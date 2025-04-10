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
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=4)
parser.add_argument('--dec-num-heads', type=int, default=4)
parser.add_argument('--dataset', type=str, default='sgra')
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
parser.add_argument('--modelpath', type=str, default="")

args = parser.parse_args()
print(' '.join(sys.argv))

# Define file paths for checkpoint and loss files
experiment_id = args.experiment_id or str(int(SystemRandom().random() * 10000000))
# checkpoint_path = f'./saved_models/{args.dataset}_{experiment_id}.h5'
if args.modelpath == "":
    best_model_path = f'./saved_models/best_model_{args.dataset}_{experiment_id}.h5'
else:
    best_model_path = f'./saved_models/{args.modelpath}'
metrics_path = f'./metrics/test_metrics_{args.dataset}_{experiment_id}.npz'


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_obj = utils.get_dataset(args.batch_size, args.dataset, test_batch_size=1)
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]
    net = models.load_network(args, dim, device=device).to(device)
    
    chp = torch.load(best_model_path)
    
    net.load_state_dict(chp['state_dict'])

    keys = ["X", 'NIR', "IR", "Sub-mm"]

    mse, crps, num_samples = utils.evaluate_model(net, dim, test_loader, keys, 'cuda')

    avg_mse = mse/num_samples
    avg_crps = crps/num_samples

    for i, key in enumerate(keys):
        print(f'\nWavelength {key}: MSE is {avg_mse[i]} and CRPS is {avg_crps[i]}')

    print(f'\nTotal average MSE is {np.sum(avg_mse)/dim}')
    print(f'Total average CRPS is {np.sum(avg_crps)/dim}')

    np.savez(metrics_path, mse=avg_mse, crps=avg_crps)

    
