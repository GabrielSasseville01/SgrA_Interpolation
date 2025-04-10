import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset for Real Data
class RealDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (numpy array): Array of shape (samples, timesteps, dim * 2 + 1).
                                The first `dim` entries are the ground truth values.
                                The next `dim` entries are the ground truth mask (1 for observed, 0 for masked).
                                The last entry (dim * 2 + 1) is ignored.
        """
        self.ground_truth = data[:, :, : data.shape[2] // 2]  # First dim columns
        self.gt_mask = data[:, :, data.shape[2] // 2 : -1]  # Next dim columns
        self.timepoints = np.arange(data.shape[1])  # Create a range for timesteps
        self.times = np.arange(data.shape[1])

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        observed_data = self.ground_truth[idx]
        observed_mask = np.ones_like(observed_data)  # All ones, since all data is observed
        gt_mask = self.gt_mask[idx]
        timepoints = self.timepoints
        times = self.times

        return {
            "observed_data": torch.tensor(observed_data, dtype=torch.float32),
            "observed_mask": torch.tensor(observed_mask, dtype=torch.float32),
            "gt_mask": torch.tensor(gt_mask, dtype=torch.float32),
            "timepoints": torch.tensor(timepoints, dtype=torch.float32),
            "times": torch.tensor(times, dtype=torch.float32)
        }
    # def __init__(self, real_data):
    #     """
    #     Args:
    #         real_data (numpy array): Array of shape (1, timesteps, dim * 2 + 1).
    #                                  The first `dim` entries are the observed values.
    #                                  The next `dim` entries are the observed mask (1 for observed, 0 for masked).
    #                                  The last entry (dim * 2 + 1) is the timepoints.
    #     """
    #     self.observed_data = real_data[:, :, : real_data.shape[2] // 2]  # First dim columns
    #     self.observed_mask = real_data[:, :, real_data.shape[2] // 2 : -1]  # Next dim columns
    #     self.timepoints = real_data[:, :, -1]
    #     self.times = np.arange(real_data.shape[1])  # Create a range for timesteps

    # def __len__(self):
    #     return self.observed_data.shape[0]

    # def __getitem__(self, idx):
    #     observed_data = self.observed_data[idx]
    #     observed_mask = self.observed_mask[idx]
    #     timepoints = self.timepoints[idx]
    #     times = self.times

    #     return {
    #         "observed_data": torch.tensor(observed_data, dtype=torch.float32),
    #         "observed_mask": torch.tensor(observed_mask, dtype=torch.float32),
    #         "gt_mask": torch.tensor(observed_mask, dtype=torch.float32), #gt_mask is set to observed_mask for real data.
    #         "timepoints": torch.tensor(timepoints, dtype=torch.float32),
    #         "times": torch.tensor(times, dtype=torch.float32)
    #     }

def create_real_data_dataloader(real_data, batch_size=1):
    """
    Creates a DataLoader for the real dataset, mimicking the structure of the test DataLoader.

    Args:
        real_data (numpy.ndarray): The prepared real data.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the real dataset.
    """

    real_dataset = RealDataset(real_data)
    real_data_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    return real_data_loader
