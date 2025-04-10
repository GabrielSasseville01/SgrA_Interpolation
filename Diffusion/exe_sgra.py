import argparse
import torch
import datetime
import json
import yaml
import os
from pathlib import Path

from main_model import CSDI_Sgra
from dataset_sgra import get_dataloader
from utils import train, evaluate, plot_test_for_example

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

args = parser.parse_args()
print(args)

path = Path(__file__).parents[0].resolve() / 'config' / args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = CSDI_Sgra(config, args.device, gp_noise=args.gp_noise, gp_sigma=0.02).to(args.device)

if args.modelfolder == "":
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/sgra_fold" + str(args.nfold) + "_" + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
    print('No pre-trained model found, training from scratch')
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    foldername = "./save/" + args.modelfolder + "/"
    print(f'Loading pre-trained model from folder: {"./save/" + args.modelfolder + "/model.pth"}')
    # Load model and epoch
    checkpoint = torch.load("./save/" + args.modelfolder + "/model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    current_epoch = checkpoint['epoch'] + 1  # Get the saved epoch number
    # best_valid_loss = checkpoint['loss']  # If you saved it
    if current_epoch <= config["train"]["epochs"]:
        print(f'Training incomplete, starting training from epoch: {current_epoch}')
        train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )


print('Evaluating Model')
# evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
plot_test_for_example(model, test_loader, example_idx=150, nsample=100, scaler=1, foldername=foldername, y_labels=['X', 'NIR', 'IR', 'Sub-mm'], save_path='model_results.npz')
# saved_data = save_data_for_example(model, test_loader, example_idx=150, nsample=100, scaler=1, mean_scaler=0, foldername=foldername, y_labels=['X', 'NIR', 'IR', 'Sub-mm'])
# plot_saved_data(saved_data, scaler=1, y_labels=['X', 'NIR', 'IR', 'Sub-mm'])
