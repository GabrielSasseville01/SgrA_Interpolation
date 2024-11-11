# pylint: disable=E1101
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import model_selection
import pdb
import torch.nn.functional as F



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    # pdb.set_trace()
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask



def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def mean_absolute_error(orig, pred, mask):
    error = torch.abs(orig - pred)
    error = error * mask
    return error.sum() / mask.sum()


def evaluate_model(
    net,
    dim,
    train_loader,
    sample_tp=0.5,
    shuffle=False,
    k_iwae=1,
    device='cuda',
):
    # torch.manual_seed(seed=0)
    # np.random.seed(seed=0)
    train_n = 0
    avg_loglik, mse, mae = 0, 0, 0
    mean_mae, mean_mse = 0, 0
    with torch.no_grad():
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            subsampled_mask = subsample_timepoints(
                train_batch[:, :, dim:2 * dim].clone(),
                sample_tp,
                shuffle=shuffle,
            )
            recon_mask = train_batch[:, :, dim:2 * dim] - subsampled_mask
            context_y = torch.cat((
                train_batch[:, :, :dim] * subsampled_mask, subsampled_mask
            ), -1)
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, :, -1],
                context_y,
                train_batch[:, :, -1],
                torch.cat((
                    train_batch[:, :, :dim] * recon_mask, recon_mask
                ), -1),
                num_samples=k_iwae,
            )
            num_context_points = recon_mask.sum().item()
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            mean_mse += loss_info.mean_mse * num_context_points
            mean_mae += loss_info.mean_mae * num_context_points
            avg_loglik += loss_info.mogloglik * num_context_points
            train_n += num_context_points
    print(
        'nll: {:.4f}, mse: {:.4f}, mae: {:.4f}, '
        'mean_mse: {:.4f}, mean_mae: {:.4f}'.format(
            - avg_loglik / train_n,
            mse / train_n,
            mae / train_n,
            mean_mse / train_n,
            mean_mae / train_n
        )
    )


def get_dataset(batch_size, dataset, test_batch_size=2, filter_anomalies=True):
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
    elif dataset == 'sgrA':
        x = np.load("./data_lib/sgrA.npz")
    elif dataset == 'randomwalk':
        x = np.load("./data_lib/simulated_random_walk_data.npz")
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
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            avg_loglik += loss_info.loglik * num_context_points
            train_n += num_context_points
    # print(
    #     'nll: {:.4f}, mse: {:.4f}, mae: {:.4f},'.format(
    #         - avg_loglik / train_n,
    #         mse / train_n,
    #         mae / train_n,
    #     )
    # )
    return -avg_loglik/train_n


def batch_prediction(
    net,
    dim,
    test_loader,
    device='cuda'):
    tmp = 0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)

            # Create context and reconstruction masks as per the original code
            original_mask = torch.ones(test_batch[:, :, dim:2 * dim].shape).to(device)
            subsampled_mask = test_batch[:, :, dim:2 * dim]
            recon_mask = original_mask - subsampled_mask
            context_y = torch.cat((test_batch[:, :, :dim] * subsampled_mask, subsampled_mask), -1)

            # Compute unsupervised loss and update
            px, time_indices, channel_indices = net.inference(
                test_batch[:, :, -1], # Time progression indicator
                context_y,             # Observed values and mask. 0's correspond to masked.
                test_batch[:, :, -1], # Time progression indicator
                torch.cat((test_batch[:, :, :dim] * recon_mask, recon_mask), -1) # Ground truth for masked values and mask. 1's correspond to masked.
            )
           
            means = px.mean
            logvars = px.logvar
            std = torch.sqrt(torch.exp(logvars))
            if tmp == 50:
                return means.squeeze().cpu(), std.squeeze().cpu(), time_indices.squeeze().cpu(), channel_indices.squeeze().cpu(), test_batch[:, :, :-1].squeeze().cpu()
            tmp += 1
        
    

