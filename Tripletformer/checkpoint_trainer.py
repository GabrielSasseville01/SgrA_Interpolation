import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from random import SystemRandom
import models
import utils
import sys

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

# Define file paths for checkpoint and loss files
experiment_id = args.experiment_id or str(int(SystemRandom().random() * 10000000))
checkpoint_path = f'./saved_models/{args.dataset}_{experiment_id}.h5'
# train_loss_path = f'./losses/train_losses_{args.dataset}_{experiment_id}.npy'
# val_loss_path = f'./losses/val_losses_{args.dataset}_{experiment_id}.npy'
# nll_path = f'./nll/nll_{args.dataset}_{experiment_id}.npy'
# mse_path = f'./mse/mse_{args.dataset}_{experiment_id}.npy'
metrics_path = f'./metrics/metrics_{args.dataset}_{experiment_id}'

# Load losses from checkpoint if they exist
# if os.path.isfile(train_loss_path) and os.path.isfile(val_loss_path):
#     train_losses = np.load(train_loss_path)
#     val_losses = np.load(val_loss_path)
#     nlls = np.load(nll_path)
#     mses = np.load(mse_path)
# else:
#     train_losses = np.empty(0)
#     val_losses = np.empty(0)
#     nlls = np.empty(0)
#     mses = np.empty(0)

# Load metrics from file if they exist
if os.path.isfile(metrics_path):
    loaded_data = np.load(metrics_path, allow_pickle=True)
    metrics = loaded_data['metrics'].item()  # Load the dictionary from the npz file
else:
    metrics = {
        "train": {
            "loss": [],
            "nll": [],
            "mse": []
        },
        "val": {
            "loss": [],
            "nll": [],
            "mse": []
        }
    }

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_obj = utils.get_dataset(args.batch_size, args.dataset)
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    net = models.load_network(args, dim, device=device).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=0.00001, verbose=True)

    # Track best validation loss and starting epoch
    best_val_loss = 10000
    start_epoch = 1
    max_early_stop = args.max_early_stop
    early_stop = 0

    # Load checkpoint if it exists
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        print(f'Resuming training from checkpoint at Epoch {start_epoch}')
    else:
        print('No checkpoint file found. Training from Epoch 1')

    print(f'Experiment ID {experiment_id}')
    for itr in range(start_epoch, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_loglik, mse, mae = 0, 0, 0

        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            train_batch = train_batch.to(device)

            # Create context and reconstruction masks as per the original code
            original_mask = torch.ones(train_batch[:, :, dim:2 * dim].shape).to(device)
            subsampled_mask = train_batch[:, :, dim:2 * dim]
            recon_mask = original_mask - subsampled_mask
            context_y = torch.cat((train_batch[:, :, :dim] * subsampled_mask, subsampled_mask), -1)

            # Compute unsupervised loss and update
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, :, -1], # Time progression indicator
                context_y,             # Observed values and mask. 0's correspond to masked.
                train_batch[:, :, -1], # Time progression indicator
                torch.cat((train_batch[:, :, :dim] * recon_mask, recon_mask), -1) # Ground truth for masked values and mask. 1's correspond to masked.
            )
            optimizer.zero_grad()
            loss_info.composite_loss.backward() # Composite: NLL + MSE
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            train_loss += loss_info.composite_loss.item() * batch_len 
            avg_loglik += loss_info.loglik.item() * batch_len
            mse += loss_info.mse.item() * batch_len
            mae += loss_info.mae.item() * batch_len
            train_n += batch_len

        # # Log and save training loss for this epoch
        # train_losses = np.append(train_losses, train_loss / train_n)
        # np.save(train_loss_path, train_losses)

        # # Log and save metrics for this epoch
        # nlls = np.append(nlls, -avg_loglik / train_n)
        # np.save(nll_path, nlls)
        # mses = np.append(mses, mse / train_n)
        # np.save(mse_path, mses)

        # Log and save training metrics for this epoch
        metrics["train"]["loss"].append(train_loss / train_n)
        metrics["train"]["nll"].append(-avg_loglik / train_n)
        metrics["train"]["mse"].append(mse / train_n)
        
        print('\nEpoch {} completed'.format(itr))
        # print('Training loss: {:.4f}'.format(-avg_loglik / train_n))
        print('Training loss: {:.4f}'.format(train_loss / train_n))
        print('NLL: {:.4f}'.format(-avg_loglik / train_n))
        print('MSE: {:.4f}'.format(mse / train_n))

        # Validation and checkpointing
        if itr % 1 == 0:
            val_loss, val_nll, val_mse = utils.test_result(net, dim, val_loader, args.sample_type, args.sample_tp, shuffle=False, k_iwae=1)
        
            # val_losses = np.append(val_losses, val_loss)
            # np.save(val_loss_path, val_losses)

            # Log and save validation metrics for this epoch
            metrics["val"]["loss"].append(val_loss)
            metrics["val"]["nll"].append(val_nll)
            metrics["val"]["mse"].append(val_mse)

            print('Validation loss: {:.4f}'.format(val_loss))

            # Checkpoint if validation improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'args': args,
                    'epoch': itr,
                    'state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss / train_n,
                }, checkpoint_path)
                early_stop = 0
            else:
                early_stop += 1
            if early_stop == max_early_stop:
                print("Early stopping due to no improvement in validation metric for 30 epochs.")
                break
            scheduler.step(val_loss)
        
        # Save metrics to file
        np.savez(metrics_path, metrics=metrics)

    # Final model evaluation
    # chp = torch.load(checkpoint_path)
    # net.load_state_dict(chp['state_dict'])
    # test_loss = utils.test_result(net, dim, test_loader, args.sample_type, args.sample_tp, shuffle=False, k_iwae=1)
    # print(f'best_val_loss: {best_val_loss}, test_loss: {test_loss.cpu().detach().numpy()}')
    # utils.get_prediction(net, dim, test_loader)
