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

# # Define file paths for checkpoint and loss files
# experiment_id = args.experiment_id or str(int(SystemRandom().random() * 10000000))
# checkpoint_path = f'./saved_models/{args.dataset}_{experiment_id}.h5'
# train_loss_path = f'./losses/train_losses_{args.dataset}_{experiment_id}.npy'
# val_loss_path = f'./losses/val_losses_{args.dataset}_{experiment_id}.npy'

# # Load losses from checkpoint if they exist
# if os.path.isfile(train_loss_path) and os.path.isfile(val_loss_path):
#     train_losses = np.load(train_loss_path)
#     val_losses = np.load(val_loss_path)
# else:
#     train_losses = np.empty(0)
#     val_losses = np.empty(0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_obj = utils.get_dataset(args.batch_size, args.dataset, test_batch_size=1)
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]
    net = models.load_network(args, dim, device=device).to(device)
    
    chp = torch.load(f'./saved_models/{args.dataset}_{args.experiment_id}.h5')
    # chp = torch.load('saved_models/sgrA_982081.h5')
    net.load_state_dict(chp['state_dict'])

    means, stds, time_indices, channel_indices, test_batch = utils.batch_prediction(net, dim, test_loader)
    print('Length of means', len(means))
    print('Length of channel indices', len(channel_indices))

    # Create subplots for each channel
    fig, axs = plt.subplots(dim, 1, figsize=(10, 2 * dim), sharex=True)
    timesteps = np.arange(1, test_batch.size(0) + 1)
    total_pred_points = 0
    for mychan in range(dim):
        print('Interpolation of Channel: ', mychan)
        ax = axs[mychan]
        
        # Plot observed values in orange
        ax.scatter(
            timesteps[np.where(test_batch[:, mychan + dim] == 1)],
            test_batch[:, mychan][np.where(test_batch[:, mychan + dim] == 1)],
            s=0.5,
            color='blue',
            label='Observed'
        )
        
        # Plot unobserved values in blue
        ax.scatter(
            timesteps[np.where(test_batch[:, mychan + dim] == 0)],
            test_batch[:, mychan][np.where(test_batch[:, mychan + dim] == 0)],
            s=0.5,
            color='orange',
            label='Masked'
        )
        
        # Plot predicted means and uncertainty bounds for the current channel
        indices = np.where(channel_indices == mychan)
        total_pred_points += len(indices[0])
        print('Amount of points to predict: ', len(indices[0]), 'for channel: ', mychan)
        times = time_indices[indices]
        mymeans = means[indices]

        print('Length of means: ', len(mymeans))
        # Flatten the array in case it's multidimensional
        mymeans_flat = mymeans.flatten()

        # Find the most common value
        unique_values, counts = np.unique(mymeans_flat, return_counts=True)
        most_frequent_value = unique_values[np.argmax(counts)]

        print("Most frequent value:", most_frequent_value)
        # print("Count:", counts[np.argmax(counts)])
        
        # total_pred_points += counts[np.argmax(counts)]
        otheridx = np.where(np.abs(most_frequent_value - mymeans) > 1e-2)
        print('Count:', len(mymeans) - len(mymeans[otheridx]))
        print('Mean of the means: ', np.mean(np.array(mymeans[otheridx])))
        # print(otheridx)
        # ax.scatter(
        #     time_indices[indices],
        #     means[indices],
        #     linewidth=1,
        #     color='red',
        #     label='Predicted Mean'
        # )
        print('Time indices of artefacts: ', np.setdiff1d(times, times[otheridx]))
        print('Length of this array of indices: ', len(np.setdiff1d(times, times[otheridx])))
        # ax.scatter(
        #     times[otheridx],
        #     mymeans[otheridx],
        #     linewidth=1,
        #     color='red',
        #     label='Predicted Mean',
        #     s=0.5
        # )
        ax.scatter(
            times,
            mymeans,
            linewidth=1,
            color='red',
            label='Predicted Mean',
            s=0.5
        )
        # if mychan == 3:
        #     print(mymeans)
        # total_pred_points += len(mymeans[otheridx])
        # if mychan == 3:
        #     print(means[indices])
        
        # Fill area for ±1 standard deviation
        # ax.fill_between(
        #     time_indices[indices],
        #     (means[indices] - stds[indices]).flatten(),
        #     (means[indices] + stds[indices]).flatten(),
        #     alpha=0.2,
        #     color='red',
        #     label='±1 Std Dev'
        # )
        # mystds = stds[indices]
        # ax.fill_between(
        #     times[otheridx],
        #     (mymeans[otheridx] - mystds[otheridx]).flatten(),
        #     (mymeans[otheridx] + mystds[otheridx]).flatten(),
        #     alpha=0.2,
        #     color='red',
        #     label='±1 Std Dev'
        # )
        mystds = stds[indices]
        ax.fill_between(
            times,
            (mymeans - mystds).flatten(),
            (mymeans + mystds).flatten(),
            alpha=0.2,
            color='red',
            label='±1 Std Dev'
        )
        
        # Set labels and title for each subplot
        ax.set_ylabel(f"Channel {mychan + 1}")
        ax.legend(loc="upper right")

    # Set x-axis label and overall title
    axs[-1].set_xlabel("Timesteps")
    plt.suptitle("Predictions and Observed Data for Each Channel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # counts = np.bincount(means[indices].cpu())
    # print(len(np.where(channel_indices.cpu() == mychan)[0]))
    # most_frequent_value = np.argmax(counts)
    # print(means)
    # print(most_frequent_value)
    # print(np.where(means == 0))
    # most_frequent = mode(means).mode[0]

    # print(means.shape)
    print('Total pred points is:', total_pred_points)

    plt.savefig('figures/test.png')
    # print(timesteps)
