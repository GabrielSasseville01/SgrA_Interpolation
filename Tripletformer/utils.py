import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def mse_inference(y, preds):
    return (np.sum(np.square(y - preds))) / len(y)


def mean_absolute_error(orig, pred, mask):
    error = torch.abs(orig - pred)
    error = error * mask
    return error.sum() / mask.sum()


def crps_norm(y, mu, sigma):
    w = (y-mu)/sigma
    return np.sum(sigma * (w*(2*norm.cdf(w)-1) + 2 * norm.pdf(w) - 1/np.sqrt(np.pi))) / len(y)


def get_dataset(batch_size, dataset, test_batch_size=1, filter_anomalies=True):
    """
    Load and prepare a specified dataset for training, validation, and testing, returning data loaders for each.

    Args:
        batch_size (int): Batch size for training and validation data loaders.
        dataset (str): Name of the dataset to load. Options are 'physionet', 'mimiciii', 'PenDigits', 
                       'physionet2019', and 'PhonemeSpectra'.
        test_batch_size (int, optional): Batch size for the test data loader. Default is 5.
        filter_anomalies (bool, optional): Placeholder for future use, currently does not affect data loading.

    Returns:
        dict: A dictionary containing:
            - "train_dataloader": DataLoader for the training set.
            - "val_dataloader": DataLoader for the validation set.
            - "test_dataloader": DataLoader for the test set.
            - "input_dim": Integer representing the input dimension for the data model.

    Raises:
        FileNotFoundError: If the specified dataset is not found in the file paths.
        ValueError: If an unrecognized dataset name is provided.

    Example:
        data = get_dataset(batch_size=32, dataset='physionet')
        train_loader = data['train_dataloader']
    """
    if dataset == 'physionet':
        x = np.load("./data_lib/physionet.npz")
    elif dataset == 'sgra':
        x = np.load("./data_lib/sgra_triplet.npz")
    elif dataset == 'noise_10':
        x = np.load("./data_lib/noise_10.npz")
    elif dataset == 'noise_30':
        x = np.load("./data_lib/noise_30.npz")
    elif dataset == 'noise_50':
        x = np.load("./data_lib/noise_50.npz")
    elif dataset == 'noise_70':
        x = np.load("./data_lib/noise_70.npz")
    elif dataset == 'noise_90':
        x = np.load("./data_lib/noise_90.npz")
    elif dataset == 'noise_95':
        x = np.load("./data_lib/noise_95.npz")
    elif dataset == 'randomwalk':
        x = np.load("./data_lib/simulated_random_walk_data.npz")
    elif dataset == 'xray':
        x = np.load("./data_lib/xray_triplet.npz")
    elif dataset == 'noxray':
        x = np.load("./data_lib/noxray_triplet.npz")
    elif dataset == 'nomask':
        x = np.load("./data_lib/sgra_no_mask_triplet.npz")
    elif dataset == 'noxray_no_mask':
        x = np.load("./data_lib/noxray_no_mask_triplet.npz")
    elif dataset == 'nomask_50':
        x = np.load("./data_lib/nomask_50_noise_triplet.npz")
    elif dataset == 'mimiciii':
        x = np.load("~/Desktop/codes_github/tripletformer/data_lib/mimiciii.npz")
    elif dataset == 'PenDigits':
        x = np.load("~/Desktop/codes_github/tripletformer/data_lib/PenDigits.npz")
    elif dataset == 'physionet2019':
        x = np.load("~/Desktop/codes_github/tripletformer/data_lib/physionet2019.npz")
    elif dataset == 'PhonemeSpectra':
        x = np.load("~/Desktop/codes_github/tripletformer/data_lib/PhonemeSpectra.npz")
    else:
        print("No dataset found")
    input_dim = (x['train'].shape[-1] - 1)//2
    train_data, val_data, test_data = x['train'], x['val'], x['test']

    # Shape is Number of Samples, Number of Timesteps, Number of Variables (Dimensions) x 2
    # Last dimension is x2 because first half is the value and second half is the mask (0 for unobserved, 1 for observed)
    print(train_data.shape, val_data.shape, test_data.shape)

    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()
    test_data = torch.from_numpy(test_data).float()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    
    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": input_dim
    }
    return data_objects



def subsample_timepoints(mask, percentage_tp_to_sample=None, shuffle=False):
    # Subsample percentage of points from each time series
    if not shuffle:
        seed = 0
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 100000)
        np.random.seed(seed)
    for i in range(mask.size(0)):
        # mask is of shape Batch size, timesteps, features
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu() # sums over mask dimension, resulting in an array of length Timesteps where each entry is the number of unmasked features for given timestep
        non_missing_tp = np.where(current_mask > 0)[0] # indices where the previous is nonzero
        n_tp_current = len(non_missing_tp) # amount of non-zeros
        n_to_sample = max(1, int(n_tp_current * percentage_tp_to_sample)) # number of points to subsample either 1 or 10% of the number of non-zeros
        n_to_sample = min((n_tp_current - 1), n_to_sample) # makes sure the previous value is not greater than the number of non-zeros
        subsampled_idx = sorted(
            np.random.choice(non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.
    return mask

def subsample_bursts(mask, percentage_tp_to_sample=None, shuffle=False):
    # Subsample percentage of points from each time series
    if not shuffle:
        seed = 0
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 100000)
        np.random.seed(seed)
    asd = mask.cpu()
    for i in range(asd.shape[0]):
        # pdb.set_trace()
        total_times = asd[i].sum(-1).to(torch.bool).sum()
        #total_times = current_mask.sum().cpu()
        n_tp_to_sample = max(1, total_times*(1-percentage_tp_to_sample))
        n_tp_to_sample = min((total_times - 1), n_tp_to_sample)
        start_times = total_times - n_tp_to_sample
        start_tp = np.random.randint(start_times+1)
        missing_tp = np.arange(start_tp, start_tp+n_tp_to_sample)
        if mask is not None:
            mask[i, missing_tp] = 0
    return mask


def test_result(
    net,
    dim,
    train_loader,
    sample_type='random',
    sample_tp=0.5,
    shuffle=False,
    k_iwae=1,
    device='cuda'):
    # torch.manual_seed(seed=0)
    # np.random.seed(seed=0)
    train_n = 0
    train_loss = 0
    avg_loglik, mse, mae, crps = 0, 0, 0, 0
    mean_mae, mean_mse = 0, 0
    with torch.no_grad():
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            # In our case the original mask is a np.ones of the same size, because we have "observed" (simulated) all data points
            original_mask = torch.ones(train_batch[:, :, dim:2 * dim].shape).to(device)
    
            subsampled_mask = train_batch[:, :, dim:2 * dim]

            recon_mask = original_mask - subsampled_mask
            
            context_y = torch.cat((
                train_batch[:, :, :dim] * subsampled_mask, subsampled_mask
            ), -1)

            loss_info = net.compute_unsupervised_loss(
                train_batch[:, :, -1],
                context_y,
                train_batch[:, :, -1],
                torch.cat((train_batch[:, :, :dim] * recon_mask, recon_mask), -1),
                num_samples=k_iwae
            )
            num_context_points = recon_mask.sum().item()
            train_loss += loss_info.composite_loss.item() * num_context_points 
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            avg_loglik += loss_info.loglik * num_context_points
            train_n += num_context_points

    return train_loss/train_n, (-avg_loglik/train_n).item(), (mse/train_n).item()


def evaluate_model(
    net,
    dim,
    test_loader,
    keys,  # List of channel names
    device='cuda'):
    
    mse = np.zeros(dim)
    crps = np.zeros(dim)
    
    with torch.no_grad():
        for batch_idx, test_batch in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print('Evaluating Sample:', batch_idx)
            test_batch = test_batch.to(device)
            
            # Create context and reconstruction masks
            original_mask = torch.ones_like(test_batch[:, :, dim:2 * dim])
            subsampled_mask = test_batch[:, :, dim:2 * dim]
            recon_mask = original_mask - subsampled_mask
            
            context_y = torch.cat((test_batch[:, :, :dim] * subsampled_mask, subsampled_mask), -1)
            
            # Perform inference
            px, time_indices, channel_indices = net.inference(
                test_batch[:, :, -1],  # Time progression indicator
                context_y,             # Observed values and mask. 0's correspond to masked.
                test_batch[:, :, -1],  # Time progression indicator
                torch.cat((test_batch[:, :, :dim] * recon_mask, recon_mask), -1)  # Ground truth for masked values and mask. 1's correspond to masked.
            )
            
            means = px.mean
            logvars = px.logvar
            std = torch.sqrt(torch.exp(logvars))

            means = means.squeeze().cpu()
            stds = std.squeeze().cpu()
            time_indices = time_indices.squeeze().cpu()
            channel_indices = channel_indices.squeeze().cpu()
            test_batch = test_batch[:, :, :-1].squeeze().cpu()


            for chan in range(dim):

                y = test_batch[:, chan][np.where(test_batch[:, chan + dim] == 0)]
                indices = np.where(channel_indices == chan)
                preds = means[indices]
                sigmas = stds[indices]

                y = y.cpu().numpy()
                preds = preds.cpu().numpy()
                sigmas = sigmas.cpu().numpy()

                mse[chan] += mse_inference(y, preds)
                crps[chan] += crps_norm(y, preds, sigmas)
        
        return mse, crps, batch_idx


# def plot_test(
#     net,
#     dim,
#     test_loader,
#     sample,
#     device='cuda',
#     y_labels=None):
    
#     with torch.no_grad():
#         for batch_idx, test_batch in enumerate(test_loader):
#             if batch_idx == sample:
#                 test_batch = test_batch.to(device)

#                 # Create context and reconstruction masks as per the original code
#                 original_mask = torch.ones(test_batch[:, :, dim:2 * dim].shape).to(device)
#                 subsampled_mask = test_batch[:, :, dim:2 * dim]
#                 recon_mask = original_mask - subsampled_mask
#                 context_y = torch.cat((test_batch[:, :, :dim] * subsampled_mask, subsampled_mask), -1)

#                 # Compute unsupervised loss and update
#                 px, time_indices, channel_indices = net.inference(
#                     test_batch[:, :, -1], # Time progression indicator
#                     context_y,             # Observed values and mask. 0's correspond to masked.
#                     test_batch[:, :, -1], # Time progression indicator
#                     torch.cat((test_batch[:, :, :dim] * recon_mask, recon_mask), -1) # Ground truth for masked values and mask. 1's correspond to masked.
#                 )
                
#                 means = px.mean
#                 logvars = px.logvar
#                 std = torch.sqrt(torch.exp(logvars))

#                 means, stds, time_indices, channel_indices, test_batch = means.squeeze().cpu(), std.squeeze().cpu(), time_indices.squeeze().cpu(), channel_indices.squeeze().cpu(), test_batch[:, :, :-1].squeeze().cpu()

#                 # Create subplots for each channel
#                 fig, axs = plt.subplots(dim, 1, figsize=(10, 2 * dim), sharex=True, gridspec_kw={'hspace': 0})
#                 timesteps = np.arange(1, test_batch.size(0) + 1)
#                 total_pred_points = 0
                
#                 if dim == 1:
#                     axs = [axs]

#                 for chan in range(dim):
#                     ax = axs[chan]
                    
#                     # Plot observed values in orange
#                     ax.scatter(
#                         timesteps[np.where(test_batch[:, chan + dim] == 1)],
#                         test_batch[:, chan][np.where(test_batch[:, chan + dim] == 1)],
#                         s=3,
#                         color='#E6C229',
#                         label='Observed'
#                     )
                    
#                     # Plot unobserved values in blue
#                     ax.scatter(
#                         timesteps[np.where(test_batch[:, chan + dim] == 0)],
#                         test_batch[:, chan][np.where(test_batch[:, chan + dim] == 0)],
#                         s=3,
#                         color='#1B998B',
#                         label='Masked'
#                     )
                    
#                     # Plot predicted means and uncertainty bounds for the current channel
#                     indices = np.where(channel_indices == chan)
#                     total_pred_points += len(indices[0])

#                     times = time_indices[indices]
#                     mymeans = means[indices]

#                     ax.scatter(
#                         times,
#                         mymeans,
#                         linewidth=1,
#                         color='#DF2935',
#                         label='Predicted Mean',
#                         s=3
#                     )

#                     mystds = stds[indices]
#                     ax.fill_between(
#                         times,
#                         (mymeans - 2*mystds).flatten(),
#                         (mymeans + 2*mystds).flatten(),
#                         alpha=0.2,
#                         color='#DF2935',
#                         label=r'2-$\sigma$'
#                     )

#                     tmp_mse = (np.sum(np.square(
#     test_batch[:, chan][np.where(test_batch[:, chan + dim] == 0)].cpu().numpy()
#     - mymeans.cpu().numpy()
# ))) / len(test_batch[:, chan][np.where(test_batch[:, chan + dim] == 0)])
#                     print(f'MSE for key {y_labels[chan]} is: {tmp_mse}')

#                     # Set labels and title for each subplot
#                     if y_labels is not None:
#                         ax.set_ylabel(y_labels[chan])
#                     else:
#                         ax.set_ylabel(f"Channel {chan + 1}")

#                     ax.set_ylim(-3.5, 7.5)
#                     if chan == 0:
#                         ax.legend(loc="upper right")
#                     ax.grid(True)

#                 # Set x-axis label and overall title
#                 axs[-1].set_xlabel("Timesteps")
#                 plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#                 plt.savefig('figures/final.png')
                    
def plot_test(
    net,
    dim,
    test_loader,
    sample,
    device='cuda',
    y_labels=None,
    save_path="tripletformer_results.npz"):
    
    with torch.no_grad():
        for batch_idx, test_batch in enumerate(test_loader):
            if batch_idx == sample:
                test_batch = test_batch.to(device)

                # Create context and reconstruction masks
                original_mask = torch.ones(test_batch[:, :, dim:2 * dim].shape).to(device)
                subsampled_mask = test_batch[:, :, dim:2 * dim]
                recon_mask = original_mask - subsampled_mask
                context_y = torch.cat((test_batch[:, :, :dim] * subsampled_mask, subsampled_mask), -1)

                # Compute predictions
                px, time_indices, channel_indices = net.inference(
                    test_batch[:, :, -1],  # Time progression indicator
                    context_y,  
                    test_batch[:, :, -1],  
                    torch.cat((test_batch[:, :, :dim] * recon_mask, recon_mask), -1) 
                )
                
                means = px.mean
                logvars = px.logvar
                std = torch.sqrt(torch.exp(logvars))

                # Move everything to CPU for processing
                means = means.squeeze().cpu().numpy()
                stds = std.squeeze().cpu().numpy()
                time_indices = time_indices.squeeze().cpu().numpy()
                channel_indices = channel_indices.squeeze().cpu().numpy()
                test_batch = test_batch[:, :, :-1].squeeze().cpu().numpy()

                # Save data
                saved_data = {}

                # Create subplots
                fig, axs = plt.subplots(dim, 1, figsize=(10, 2 * dim), sharex=True, gridspec_kw={'hspace': 0})
                timesteps = np.arange(1, test_batch.shape[0] + 1)

                if dim == 1:
                    axs = [axs]

                for chan in range(dim):
                    ax = axs[chan]
                    
                    # Observed values
                    obs_indices = np.where(test_batch[:, chan + dim] == 1)
                    obs_x = timesteps[obs_indices]
                    obs_y = test_batch[:, chan][obs_indices]
                    
                    # Masked values
                    masked_indices = np.where(test_batch[:, chan + dim] == 0)
                    masked_x = timesteps[masked_indices]
                    masked_y = test_batch[:, chan][masked_indices]

                    # Predictions
                    pred_indices = np.where(channel_indices == chan)
                    pred_times = time_indices[pred_indices]
                    pred_means = means[pred_indices]
                    pred_stds = stds[pred_indices]
                    pred_lower = pred_means - 2 * pred_stds
                    pred_upper = pred_means + 2 * pred_stds

                    # Save arrays
                    key_label = y_labels[chan] if y_labels else f"Channel_{chan+1}"
                    saved_data[f"{key_label}_train_x"] = obs_x
                    saved_data[f"{key_label}_train_y"] = obs_y
                    saved_data[f"{key_label}_test_x"] = masked_x
                    saved_data[f"{key_label}_test_y"] = masked_y
                    saved_data[f"{key_label}_predicted_means"] = pred_means
                    saved_data[f"{key_label}_lower_bound"] = pred_lower
                    saved_data[f"{key_label}_upper_bound"] = pred_upper

                    # Plot
                    ax.scatter(obs_x, obs_y, s=3, color='#E6C229', label='Observed')
                    ax.scatter(masked_x, masked_y, s=3, color='#1B998B', label='Masked')
                    ax.scatter(pred_times, pred_means, linewidth=1, color='#DF2935', label='Predicted Mean', s=3)
                    ax.fill_between(pred_times, pred_lower, pred_upper, alpha=0.2, color='#DF2935', label=r'2-$\sigma$')

                    # Compute and print MSE
                    tmp_mse = np.mean((masked_y - pred_means) ** 2)
                    print(f'MSE for key {key_label} is: {tmp_mse:.5f}')

                    # Labels
                    ax.set_ylabel(key_label)
                    ax.set_ylim(-3.5, 7.5)
                    if chan == 0:
                        ax.legend(loc="upper right")
                    ax.grid(True)

                axs[-1].set_xlabel("Timesteps")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig('figures/final.png')

                # Save the results
                np.savez(save_path, **saved_data)
                print(f"Results saved to {save_path}")

