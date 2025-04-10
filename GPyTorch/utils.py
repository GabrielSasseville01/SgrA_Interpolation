import torch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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

    data = torch.from_numpy(x['test']).float()

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
def ax_plot(ax, train_y, train_x, test_y, test_x, means, lower, upper, label):
    # Plot the training data as black stars
    ax.scatter(train_x, train_y, color='#E6C229', s=3, label='Observed')
    # Plot test data as black stars
    ax.scatter(test_x, test_y, color='#1B998B', s=3, label='Masked')
    # Plot the predictive mean as a blue line
    ax.scatter(test_x, means, linewidth=1, color='#DF2935', s=3, label='Predicted Mean')
    # Shade in the confidence region
    ax.fill_between(test_x, lower, upper, alpha=0.2, color='#DF2935', label=r'2-$\sigma$')

    ax.set_ylabel(label)
    ax.set_ylim(-3.5, 7.5)
    ax.grid(True)
    # ax.set_ylim([-3, 3])  # Set limits for the y-axis
    # ax.legend(['Observed Data', 'Mean', r'2-$\sigma$'])
    # ax.set_title(title)


def plot_example(data, labels):
    dim = len(labels)

    fig, axs = plt.subplots(dim, 1, figsize=(10, 2 * dim), sharex=True, gridspec_kw={'hspace': 0})

    if dim == 1:
        axs = [axs]

    for i, tmp_dict in enumerate(data):
        ax = axs[i]

        train_x = tmp_dict['train_x']
        train_y = tmp_dict['train_y']
        test_x = tmp_dict['test_x']
        test_y = tmp_dict['test_y']
        means = tmp_dict['means']
        lower = tmp_dict['lower']
        upper = tmp_dict['upper']

        # Plot the training data as black stars
        ax.scatter(train_x, train_y, color='#E6C229', s=3, label='Observed')
        # Plot test data as black stars
        ax.scatter(test_x, test_y, color='#1B998B', s=3, label='Masked')
        # Plot the predictive mean as a blue line
        ax.scatter(test_x, means, linewidth=1, color='#DF2935', s=3, label='Predicted Mean')
        # Shade in the confidence region
        ax.fill_between(test_x, lower, upper, alpha=0.2, color='#DF2935', label=r'2-$\sigma$')

        # Set labels and title for each subplot
        ax.set_ylabel(labels[i])

        ax.set_ylim(-3.5, 7.5)
        ax.grid(True)

        if i == 0:
            ax.legend(loc="upper right")

    # Set x-axis label and overall title
    axs[-1].set_xlabel("Timesteps")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('plots/test.png')



