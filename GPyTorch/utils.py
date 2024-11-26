import torch
import numpy as np
from scipy.stats import norm

def get_dataset(dataset):
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
    x = np.load(f'data_lib/{dataset}_mogp.npz')

    data = torch.from_numpy(x['data']).float()

    return data

def process_data(sample, keys):
    dim = len(keys)

    train_x = []
    train_y = []
    train_idx = []
    test_x = []
    test_y = []
    test_idx = []

    for i in range(dim):
        train_xi = torch.where((sample[:, dim + i].int() == 1))[0] # get the indices
        train_yi = sample[:, i][train_xi]
        train_idx_i = torch.full((train_xi.shape[0], 1), dtype=torch.long, fill_value=0)

        test_xi = torch.where((sample[:, dim + i].int() == 0))[0]
        test_yi = sample[:, i][test_xi]
        test_idx_i = torch.full((test_xi.shape[0], 1), dtype=torch.long, fill_value=0)

        train_x.append(train_xi)
        train_y.append(train_yi)
        train_idx.append(train_idx_i)

        test_x.append(test_xi)
        test_y.append(test_yi)
        test_idx.append(test_idx_i)

    return train_x, train_idx, train_y, test_x, test_y, test_idx

def mean_squared_error(y, preds):
    return (np.sum(np.square(y - preds))) / len(y)

def crps_norm(y, mu, sigma):
    w = (y-mu)/sigma
    return np.sum(sigma * (w*(2*norm.cdf(w)-1) + 2 * norm.pdf(w) - 1/np.sqrt(np.pi))) / len(y)


# Define plotting function for the observed and predicted values
def ax_plot(ax, train_y, train_x, test_y, test_x, means, lower, upper, title):
    # Plot the training data as black stars
    ax.scatter(train_x, train_y, color='blue', s=5)
    # Plot test data as black stars
    ax.scatter(test_x, test_y, color='orange', s=5)
    # Plot the predictive mean as a blue line
    ax.plot(test_x, means, 'red')
    # Shade in the confidence region
    ax.fill_between(test_x, lower, upper, alpha=0.25, color='red')
    # ax.set_ylim([-3, 3])  # Set limits for the y-axis
    ax.legend(['Observed Data', 'Mean', r'2-$\sigma$'])
    ax.set_title(title)