import argparse
import torch
import datetime
import json
import yaml
import os
from pathlib import Path
import numpy as np

from dataset_real import create_real_data_dataloader
from main_model import CSDI_Sgra
from dataset_sgra import get_dataloader
from utils import train, plot_test_for_example

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--gp_noise", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--real_data_path", type=str, default="real_data.npz") #Added real data path.

args = parser.parse_args()
print(args)

path = Path(__file__).parents[0].resolve() / 'config' / args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

# Load the real dataset
real_data = np.load(args.real_data_path)['real_data']

# Create a DataLoader for the real data
real_data_loader = create_real_data_dataloader(real_data, batch_size=1)

model = CSDI_Sgra(config, args.device, gp_noise=args.gp_noise, gp_sigma=0.02).to(args.device)

foldername = "./save/" + args.modelfolder + "/"
print(f'Loading pre-trained model from folder: {"./save/" + args.modelfolder + "/model.pth"}')
# Load model and epoch
checkpoint = torch.load("./save/" + args.modelfolder + "/model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

print('Evaluating Model')
plot_test_for_example(model, real_data_loader, example_idx=0, nsample=100, scaler=1, foldername=foldername, y_labels=['X', 'NIR', 'IR', 'Sub-mm'], save_path='real_results.npz')
